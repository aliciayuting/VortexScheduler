from queue import PriorityQueue

import core.configs.gen_config as gcfg

from core.task import Task
from queue_management.queued_task import refresh_queue
from schedulers.algo.boost_algo import get_min_exec_time


# NONE:      never drop
# LAZY:      drop once the job's deadline has already passed
# EARLY:     drop as soon as the queued task and the stages downstream of it can no
#            longer finish by the job's deadline, were it the only job in the cluster
# LOOKAHEAD: drop as soon as they can no longer do so given the work the cluster has
#            already accepted, i.e. counting the time those stages will spend queued
#            behind jobs that are already in the system
DROP_POLICIES = ["NONE", "LAZY", "EARLY", "LOOKAHEAD"]

# policies that need a ClusterLoadView to decide
_LOAD_AWARE_POLICIES = {"LOOKAHEAD"}

# dispatch policies whose scheduler holds the task queues and forms batches itself
_SCHEDULER_QUEUE_POLICIES = {"SHEPHERD"}

# dispatch policies that dispatch straight to workers, which queue and batch
_WORKER_QUEUE_POLICIES = {"ROUND_ROBIN"}


def scheduler_manages_queues() -> bool:
    """Whether the configured scheduler holds the task queues and forms batches
    itself, rather than dispatching straight to worker queues.

    Jobs are dropped wherever queueing happens, since that is the only place a
    task waits long enough to go stale and the only place it can be removed before
    a batch is formed around it. Both the scheduler and the workers read this, so
    exactly one side sweeps.
    """
    if gcfg.DISPATCH_POLICY in _SCHEDULER_QUEUE_POLICIES:
        return True
    elif gcfg.DISPATCH_POLICY in _WORKER_QUEUE_POLICIES:
        return False

    raise ValueError(f"Unknown dispatch policy {gcfg.DISPATCH_POLICY}; cannot tell "
                     "whether queues are managed at the scheduler or at the workers")


class _Stage:
    """A task of a workflow as the remaining time walks see it: its model and the
    stages that follow it. Tasks on the same model with the same stages after them
    take equally long to finish from, so they share one _Stage, which keeps the
    walk over a wide fan-out as short as the walk over a single branch of it.
    """

    def __init__(self, model_data, next_stages: tuple["_Stage", ...]):
        self.model_data = model_data
        self.next_stages = next_stages

        # every stage downstream of this one, each after all the stages that follow
        # it, so that one pass can accumulate time from the sinks backwards
        self.downstream: list[_Stage] = []
        seen: set[int] = set()
        for stage in next_stages:
            for s in stage.downstream + [stage]:
                if id(s) not in seen:
                    seen.add(id(s))
                    self.downstream.append(s)

        # time to finish from here at batch size 1 without queueing: this stage plus
        # the longest path of stages after it
        self.processing_time: float = get_min_exec_time(model_data) + max(
            [s.processing_time for s in next_stages], default=0.0)


def _build_stages(workflow) -> dict[int, _Stage]:
    """Returns task ID -> _Stage for every task of [workflow]."""
    stages: dict[int, _Stage] = {}
    shared: dict[tuple, _Stage] = {}

    def _build(task) -> _Stage:
        if task.id not in stages:
            next_stages = {id(s): s for s in map(_build, task.next_tasks)}
            key = (task.model_data.id, frozenset(next_stages))
            if key not in shared:
                shared[key] = _Stage(task.model_data, tuple(next_stages.values()))
            stages[task.id] = shared[key]

        return stages[task.id]

    for task in workflow.tasks.values():
        _build(task)

    return stages


def _get_stage(task: Task) -> _Stage:
    """Returns the _Stage of [task]. A workflow's DAG never changes, and everything
    downstream of a task that has not run yet is itself yet to run, so what remains
    from [task] on depends on neither the job nor its progress and is built once
    per workflow.
    """
    workflow = task.job.workflow
    if workflow.stages_cache is None:
        workflow.stages_cache = _build_stages(workflow)

    return workflow.stages_cache[task.task_id]


def get_remaining_processing_time(task: Task, worker_size: int | None = None) -> float:
    """Returns the minimum time still needed before [task]'s deadline can be met.

    Under job-level SLOs the deadline is the end of the pipeline, so what remains
    is this task followed by the longest path through the stages downstream of it,
    assuming batch size 1 and no queueing. Stages on parallel paths are left out,
    even when a join downstream will wait for them: each is judged on its own path
    while it is queued itself. Under per-stage (NEXUS) SLOs each stage carries its own
    deadline that already budgets for the stages behind it, so the only work left
    to fit is this stage's own execution.

    Args:
        task: Task whose remaining work to measure
        worker_size: Memory size (GB) of the worker holding the task, whose batch
        exec times to read. Callers that do not know where the task will run pass
        None, and the slowest worker size the model is profiled for is assumed.
    """
    if gcfg.SLO_TYPE != "JOB_LEVEL":
        exec_times = task.model_data.batch_exec_times
        if worker_size is not None and worker_size in exec_times:
            return exec_times[worker_size][1]

        return get_min_exec_time(task.model_data)

    return _get_stage(task).processing_time


def get_remaining_time_with_queueing(time: float, task: Task, load_view,
                                     worker_size: int | None = None,
                                     priority: float | None = None) -> float:
    """Returns how long [task]'s job still needs before its deadline can be met,
    counting the time its remaining stages will spend waiting behind work the
    cluster has already accepted.

    This is the queue aware counterpart of [get_remaining_processing_time], and
    splits the same way. Under per-stage (NEXUS) SLOs the deadline being tested is
    this stage's, which already budgets for the stages behind it, so only this
    stage's wait and execution are counted. Under job-level SLOs the deadline is
    the end of the pipeline, so this stage is counted along with the longest path
    through the stages downstream of it, each stage charged for the backlog standing
    in front of it. Stages on parallel paths are left out, as they are there.

    [task] is charged only for the work its own queue serves ahead of it, since
    that is all it waits for. Its later stages are charged their model's whole
    backlog, which is the right question for a task that has not been queued yet:
    it will arrive at the back of that queue.

    Args:
        time: Time at which the estimate is made
        task: Task whose job's remaining work to measure
        load_view: ClusterLoadView describing the work already in the system
        worker_size: Memory size (GB) of the worker holding the task, whose batch
        exec times to read for this stage. None assumes the slowest worker size
        the model is profiled for.
        priority: Queue priority [task] holds in the queue it is waiting in, which
        fixes how much of that queue it waits for. None if it is not queued yet,
        and is therefore behind all of it

    Returns:
        remaining: Time still needed, including queueing (ms)
    """
    # this stage: its own place in line, on the worker holding it if that is known
    exec_times = task.model_data.batch_exec_times
    if worker_size is not None and worker_size in exec_times:
        exec_time = exec_times[worker_size][1]
    else:
        exec_time = get_min_exec_time(task.model_data)

    remaining = load_view.get_queue_delay(task.model_data.id, time, priority) + exec_time

    if gcfg.SLO_TYPE != "JOB_LEVEL":
        return remaining

    # the stages after it, accumulated from the sinks backwards: the time to finish
    # from a stage is its own plus that of the slowest stage following it
    stage = _get_stage(task)
    finish_times: dict[int, float] = {}
    for s in stage.downstream:
        finish_times[id(s)] = (
            load_view.get_queue_delay(s.model_data.id, time)
            + get_min_exec_time(s.model_data)
            + max([finish_times[id(ns)] for ns in s.next_stages], default=0.0))

    return remaining + max([finish_times[id(ns)] for ns in stage.next_stages],
                           default=0.0)


def _drops_task(policy: str, time: float, task: Task,
                worker_size: int | None = None, load_view=None,
                priority: float | None = None) -> bool:
    """Decides whether [policy] would drop the job owning [task] at [time].

    The applicable deadline comes from [task.get_task_deadline]: the job deadline
    under JOB_LEVEL SLOs, this stage's deadline under NEXUS SLOs. LAZY drops only
    once that deadline has already elapsed. EARLY drops as soon as this task cannot
    be expected to meet it: when [time] plus the remaining processing time of this
    task and the stages downstream of it already exceeds it. Since remaining processing time is non-negative, EARLY drops
    whatever LAZY would, and never later than LAZY does.

    LOOKAHEAD tests the same inequality with a remaining time that also counts
    queueing behind the work the cluster has already accepted. That estimate is
    never below EARLY's, so faced with the same job in the same cluster LOOKAHEAD
    drops whatever EARLY would and never later. What it buys is catching a job
    whose own execution would fit but whose wait will not, which EARLY only
    discovers once the job has sat in a queue long enough to run out of slack, by
    which point the stages it already ran are wasted.

    That per-decision ordering does not mean a LOOKAHEAD run sheds more overall.
    Dropping sooner frees the capacity the doomed jobs would have consumed, so the
    queues it judges against are shorter than the ones EARLY ends up with, and the
    two runs diverge from the first decision they disagree on.

    Args:
        policy: One of DROP_POLICIES
        time: Time at which the drop decision is made
        task: Task whose job is considered for dropping
        worker_size: Memory size (GB) of the worker holding the task, if known (see
        [get_remaining_processing_time])
        load_view: ClusterLoadView describing the work already in the system.
        Required by the policies in [_LOAD_AWARE_POLICIES], unused by the others
        priority: Queue priority of [task] where it is queued, if it is (see
        [get_remaining_time_with_queueing]). Unused by the other policies

    Returns:
        should_drop: True if the policy would drop the job
    """
    if policy == "NONE":
        return False

    deadline = task.get_task_deadline()

    if policy == "LAZY":
        return time > deadline
    elif policy == "EARLY":
        return (time + get_remaining_processing_time(task, worker_size)) > deadline
    elif policy == "LOOKAHEAD":
        assert(load_view is not None), \
            "LOOKAHEAD needs a ClusterLoadView to see the work already in the system"
        remaining = get_remaining_time_with_queueing(time, task, load_view,
                                                     worker_size, priority)
        return (time + remaining) > deadline

    raise ValueError(f"Unrecognized drop policy {policy}")


def needs_load_view(policy: str) -> bool:
    """Whether [policy] can only decide with a ClusterLoadView."""
    return policy in _LOAD_AWARE_POLICIES


def _decides_per_job(policy: str) -> bool:
    """Whether [policy] reaches the same verdict for every task of a job at one
    instant, so that a sweep need only test each job once.

    Under job-level SLOs, LAZY holds only the job's own deadline against the time.
    EARLY and LOOKAHEAD measure the work from the task that asks onwards, and
    per-stage SLOs give each stage its own deadline, so none of those can be decided
    once for the whole job.
    """
    return gcfg.SLO_TYPE == "JOB_LEVEL" and policy == "LAZY"


def should_drop_task(time: float, task: Task, worker_size: int | None = None,
                     load_view=None, priority: float | None = None) -> bool:
    """Whether the job owning [task] should be dropped at [time] under DROP_POLICY,
    i.e. whether it is actually removed from the cluster. See [_drops_task].
    """
    return _drops_task(gcfg.DROP_POLICY, time, task, worker_size, load_view, priority)


def should_shadow_drop_task(time: float, task: Task, worker_size: int | None = None,
                            load_view=None, priority: float | None = None) -> bool:
    """Whether SHADOW_DROP_POLICY would have dropped the job owning [task] at
    [time]. The job keeps running either way; the answer is only recorded. See
    [_drops_task].
    """
    return _drops_task(gcfg.SHADOW_DROP_POLICY, time, task, worker_size, load_view,
                       priority)


def drop_from_queue(time: float, task_queue: PriorityQueue,
                    already_dropped: set[int],
                    worker_size: int | None = None, load_view=None) -> list[int]:
    """Removes every queued task belonging to a job that should be dropped at
    [time], as well as tasks of jobs that were already dropped elsewhere.

    Args:
        time: Time at which the drop decision is made
        task_queue: Queue of QueuedTask to filter in place
        already_dropped: IDs of jobs known to be dropped already
        worker_size: Memory size (GB) of the worker holding the queue, if known (see
        [get_remaining_processing_time])
        load_view: ClusterLoadView, required by the load aware policies

    Returns:
        newly_dropped: IDs of jobs dropped by this call, excluding jobs in
        [already_dropped]
    """
    # a queued task's place in line, which LOOKAHEAD charges it for, is read at
    # [time] under a time varying ordering
    refresh_queue(time, task_queue)

    if load_view is not None and needs_load_view(gcfg.DROP_POLICY):
        # the loop below empties the queue as it decides, so fix the backlog now
        load_view.prime(time)

    kept = []
    newly_dropped: list[int] = []
    # jobs already kept by this sweep, whose other tasks need no second test when
    # the verdict is the same for all of them
    per_job = _decides_per_job(gcfg.DROP_POLICY)
    kept_job_ids: set[int] = set()

    while task_queue.qsize() > 0:
        qt = task_queue.get()
        job = qt.task.job

        if job.id in already_dropped or job.id in newly_dropped:
            continue

        if job.id in kept_job_ids:
            kept.append(qt)
            continue

        if should_drop_task(time, qt.task, worker_size, load_view, qt.priority):
            newly_dropped.append(job.id)
            continue

        kept.append(qt)
        if per_job:
            kept_job_ids.add(job.id)

    for qt in kept:
        task_queue.put(qt)

    return newly_dropped


def shadow_drops_from_queue(time: float, task_queue: PriorityQueue,
                            already_shadow_dropped: set[int],
                            worker_size: int | None = None,
                            load_view=None) -> list[tuple[int, int]]:
    """Finds the queued tasks whose jobs SHADOW_DROP_POLICY would have dropped at
    [time], leaving the queue untouched.

    The queue is only read, never filtered: shadow drops must not change what the
    simulation does, so a job that would have been dropped stays queued and is
    reported once, the first time any policy check rejects it.

    Args:
        time: Time at which the drop decision is made
        task_queue: Queue of QueuedTask to scan
        already_shadow_dropped: IDs of jobs already reported as shadow dropped
        worker_size: Memory size (GB) of the worker holding the queue, if known (see
        [get_remaining_processing_time])

    Returns:
        newly_shadow_dropped: (job ID, task ID) of the jobs first shadow dropped by
        this call, the task ID being the stage whose deadline they failed
    """
    if gcfg.SHADOW_DROP_POLICY == "NONE":
        return []

    refresh_queue(time, task_queue)

    seen_here: set[int] = set()
    newly_shadow_dropped: list[tuple[int, int]] = []
    # jobs this scan found would be kept, skipped thereafter when the verdict is
    # the same for every task of a job
    per_job = _decides_per_job(gcfg.SHADOW_DROP_POLICY)
    kept_job_ids: set[int] = set()

    for qt in list(task_queue.queue):
        job_id = qt.task.job.id
        if job_id in already_shadow_dropped or job_id in seen_here \
                or job_id in kept_job_ids:
            continue

        if should_shadow_drop_task(time, qt.task, worker_size, load_view,
                                   qt.priority):
            seen_here.add(job_id)
            newly_shadow_dropped.append((job_id, qt.task.task_id))
        elif per_job:
            kept_job_ids.add(job_id)

    return newly_shadow_dropped
