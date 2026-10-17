"""A cluster wide view of the work already admitted, for drop policies that decide
against the queues rather than against an empty cluster.

EARLY asks whether a job could still meet its deadline if it were the only thing
running. That is the right question for a lightly loaded cluster and the wrong one
for a congested pipeline, where a job misses its deadline waiting behind other jobs
rather than executing. LOOKAHEAD asks the question this module answers instead: how
long will each stage of the pipeline take once the work already queued and running
ahead of it has been served. "Ahead of it" is the queue's own ordering: a task
waits for the tasks that outrank it, not for those queued behind it.
"""

import bisect

import core.configs.gen_config as gcfg

from schedulers.algo.capacity_algo import get_instance_throughput
from queue_management.queued_task import refresh_queue


class ClusterLoadView:
    """Aggregates, per model, the work the cluster has already accepted: the tasks
    queued for that model anywhere, plus the tasks in batches currently executing
    on it.

    One instance is shared by the scheduler and every worker, because a worker
    holds only its own queue and a job's later stages almost always run somewhere
    else. In a real deployment this view would have to be gathered and would be
    stale; here it is exact, so a LOOKAHEAD run measures the policy rather than
    the quality of the load information behind it.

    Results are cached per simulation timestamp. Every queued task of every
    sweep is tested against the policy, and many of those tests fall on the same
    timestamp, so without the cache the scan over workers would dominate the run.
    """

    def __init__(self, workers: dict, workflows: dict = None):
        """
        Args:
            workers: Map of worker ID -> worker object
            workflows: Map of workflow ID -> workflow, used to cap each model's
            throughput by the largest batch its stages may form under NEXUS SLOs.
            Omitted, no cap is applied.
        """
        self.workers = workers

        # objects holding the task queues, set once the scheduler exists: the
        # scheduler itself when it manages queues, otherwise the workers
        self._queue_holders: list = list(workers.values())

        # under NEXUS a stage's SLO caps the batch it may form, which lowers the
        # throughput its model can sustain. Matches the cap the admission controller
        # sizes against, and the one enforced at batching
        self._max_batch_sizes: dict[int, int] = {}
        if workflows and gcfg.SLO_TYPE == "NEXUS":
            for workflow in workflows.values():
                for task_id, bsize in workflow.task_max_batch_sizes.items():
                    model_id = workflow.tasks[task_id].model_data.id
                    self._max_batch_sizes[model_id] = min(
                        self._max_batch_sizes.get(model_id, bsize), bsize)

        self._capacities: dict[int, float] = {}
        self._capacities_time = None

        # model ID -> (sorted priorities of the tasks queued for it, number of
        # tasks in batches already executing on it)
        self._backlog: dict[int, tuple[list[float], int]] = {}
        self._backlog_time = None

        # (model ID, worker size, batch cap) -> jobs/ms for one instance
        self._instance_throughputs: dict[tuple, float] = {}

    def set_queue_holders(self, holders: list):
        """Sets the objects whose [queues] hold the cluster's pending tasks.

        Args:
            holders: Objects exposing a [queues] dict of model ID -> PriorityQueue
        """
        self._queue_holders = list(holders)

    def _instance_throughput(self, model_data, worker_size: int) -> float:
        key = (model_data.id, worker_size, self._max_batch_sizes.get(model_data.id))
        if key not in self._instance_throughputs:
            self._instance_throughputs[key] = get_instance_throughput(
                model_data, worker_size, self._max_batch_sizes.get(model_data.id))

        return self._instance_throughputs[key]

    def get_capacities(self, time: float) -> dict[int, float]:
        """Returns model ID -> throughput (jobs/ms) summed over that model's
        instances, as placed at [time].
        """
        if self._capacities_time == time:
            return self._capacities

        capacities: dict[int, float] = {}
        for worker in self.workers.values():
            for state in worker.GPU_state.state_at(time):
                model_data = state.model.data
                capacities[model_data.id] = capacities.get(model_data.id, 0) + \
                    self._instance_throughput(model_data, worker.total_memory_gb)

        self._capacities, self._capacities_time = capacities, time
        return capacities

    def _get_backlog(self, time: float) -> dict[int, tuple[list[float], int]]:
        """Returns model ID -> (sorted queue priorities of the tasks waiting for
        that model, number of tasks in batches already executing on it).

        The priorities are kept rather than only their count so that a task can be
        charged for the part of the backlog that will be served before it, which
        is what its queue's ordering decides. Lower priority is served first, so
        the tasks ahead of priority p are those sorted before it.
        """
        if self._backlog_time == time:
            return self._backlog

        queued: dict[int, list[float]] = {}
        for holder in self._queue_holders:
            for model_id, queue in getattr(holder, "queues", {}).items():
                # every queue is read at [time], so that priorities from different
                # queues are comparable under a time varying ordering
                refresh_queue(time, queue)
                # read the heap without disturbing it; the sort below puts it in
                # the order the queue will actually serve
                queued.setdefault(model_id, []).extend(qt.priority for qt in queue.queue)

        executing: dict[int, int] = {}
        for worker in self.workers.values():
            for state in worker.GPU_state.state_at(time):
                batch = state.reserved_batch
                if batch is None:
                    continue

                model_id = state.model.data.id
                executing[model_id] = executing.get(model_id, 0) + len(batch.tasks)

        self._backlog = {
            model_id: (sorted(queued.get(model_id, [])), executing.get(model_id, 0))
            for model_id in set(queued) | set(executing)}
        self._backlog_time = time

        return self._backlog

    def get_pending_tasks(self, time: float) -> dict[int, int]:
        """Returns model ID -> number of tasks already accepted for that model and
        not yet finished: everything queued for it, plus the tasks of whatever
        batch is executing on each of its instances.
        """
        return {model_id: len(priorities) + executing
                for model_id, (priorities, executing) in self._get_backlog(time).items()}

    def prime(self, time: float):
        """Computes and caches the backlog at [time] before the caller mutates any
        queue.

        A sweep that drops from a queue empties it as it goes, so an estimate taken
        part way through would see a backlog that is missing the very tasks being
        judged. Priming first fixes the view at what the cluster held when the
        sweep began, which is also what makes every decision within one sweep see
        the same cluster.
        """
        self._get_backlog(time)

    def get_queue_delay(self, model_id: int, time: float,
                        priority: float | None = None) -> float:
        """Returns how long (ms) a task of [model_id] will wait at [time] before
        the model can start on it, given the work already accepted.

        A queued task waits only for the work its queue serves first, so passing
        its [priority] charges it for the tasks that outrank it plus whatever is
        already executing, and not for the tasks queued behind it, which it will
        not wait for. Omitting [priority] charges the model's whole backlog, which
        is the right question for a task that is not queued yet: a later stage of a
        pipeline, or one arriving at the scheduler, reaches the back of the queue
        and so does wait for all of it.

        The backlog is divided by the model's aggregate throughput, so the estimate
        accounts for every instance serving it and for the batching those instances
        can do. It stays optimistic on that count, assuming the backlog batches as
        efficiently as the model allows.

        Args:
            model_id: Model whose backlog to measure
            time: Time at which to measure
            priority: Queue priority of the waiting task, if it is already queued.
            Lower is served first. Ignored under LOOKAHEAD_WHOLE_QUEUE_DELAY

        Returns:
            queue_delay: Time to clear the work ahead (ms), or infinity if the
            model has no capacity at all
        """
        priorities, executing = self._get_backlog(time).get(model_id, ([], 0))

        if priority is None or gcfg.LOOKAHEAD_WHOLE_QUEUE_DELAY:
            backlog = len(priorities) + executing
        else:
            # ties are counted as behind, keeping the estimate on the optimistic
            # side the rest of this module takes
            backlog = bisect.bisect_left(priorities, priority) + executing

        if backlog == 0:
            return 0.0

        capacity = self.get_capacities(time).get(model_id, 0)
        if capacity <= 0:
            return float("inf")

        return backlog / capacity
