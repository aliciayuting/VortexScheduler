"""A cluster wide view of the work already admitted, for drop policies that decide
against the queues rather than against an empty cluster.

EARLY asks whether a job could still meet its deadline if it were the only thing
running. That is the right question for a lightly loaded cluster and the wrong one
for a congested pipeline, where a job misses its deadline waiting behind other jobs
rather than executing. LOOKAHEAD asks the question this module answers instead: how
long will each stage of the pipeline take once the work already queued and running
ahead of it has been served.
"""

import core.configs.gen_config as gcfg

from schedulers.algo.capacity_algo import get_instance_throughput


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
            throughput by the largest batch its stages may form under per-stage
            (NEXUS) SLOs. Omitted, no cap is applied.
        """
        self.workers = workers

        # objects holding the task queues, set once the scheduler exists: the
        # scheduler itself when it manages queues, otherwise the workers
        self._queue_holders: list = list(workers.values())

        # a stage's SLO caps the batch it may form, which lowers the throughput its
        # model can sustain. Matches the cap the admission controller sizes against
        self._max_batch_sizes: dict[int, int] = {}
        if workflows and gcfg.SLO_TYPE == "NEXUS":
            for workflow in workflows.values():
                for task_id, bsize in workflow.task_max_batch_sizes.items():
                    model_id = workflow.tasks[task_id].model_data.id
                    self._max_batch_sizes[model_id] = min(
                        self._max_batch_sizes.get(model_id, bsize), bsize)

        self._capacities: dict[int, float] = {}
        self._capacities_time = None

        self._pending: dict[int, int] = {}
        self._pending_time = None

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

    def get_pending_tasks(self, time: float) -> dict[int, int]:
        """Returns model ID -> number of tasks already accepted for that model and
        not yet finished: everything queued for it, plus the tasks of whatever
        batch is executing on each of its instances.
        """
        if self._pending_time == time:
            return self._pending

        pending: dict[int, int] = {}

        for holder in self._queue_holders:
            for model_id, queue in getattr(holder, "queues", {}).items():
                pending[model_id] = pending.get(model_id, 0) + queue.qsize()

        for worker in self.workers.values():
            for state in worker.GPU_state.state_at(time):
                batch = state.reserved_batch
                if batch is None:
                    continue

                model_id = state.model.data.id
                pending[model_id] = pending.get(model_id, 0) + len(batch.tasks)

        self._pending, self._pending_time = pending, time
        return pending

    def prime(self, time: float):
        """Computes and caches the backlog at [time] before the caller mutates any
        queue.

        A sweep that drops from a queue empties it as it goes, so an estimate taken
        part way through would see a backlog that is missing the very tasks being
        judged. Priming first fixes the view at what the cluster held when the
        sweep began, which is also what makes every decision within one sweep see
        the same cluster.
        """
        self.get_pending_tasks(time)

    def get_queue_delay(self, model_id: int, time: float) -> float:
        """Returns how long (ms) the work already accepted for [model_id] will take
        to clear at [time], i.e. what a task arriving now would wait before the
        model could start on it.

        The backlog is divided by the model's aggregate throughput, so the estimate
        accounts for every instance serving it and for the batching those instances
        can do. It is an optimistic figure on both counts: it assumes the backlog
        batches as efficiently as the model allows, and it charges a task for the
        whole backlog of its model rather than only the part of it that outranks
        the task in queue order.

        Args:
            model_id: Model whose backlog to measure
            time: Time at which to measure

        Returns:
            queue_delay: Time to clear the existing backlog (ms), or infinity if
            the model has no capacity at all
        """
        backlog = self.get_pending_tasks(time).get(model_id, 0)
        if backlog == 0:
            return 0.0

        capacity = self.get_capacities(time).get(model_id, 0)
        if capacity <= 0:
            return float("inf")

        return backlog / capacity
