from queue import PriorityQueue

import core.configs.gen_config as gcfg

from core.task import Task
from schedulers.algo.boost_algo import _get_processing_time, get_min_exec_time


# NONE:      never drop
# LAZY:      drop once the job's deadline has already passed
# EARLY:     drop as soon as the job can no longer meet its deadline, were it the
#            only job in the cluster
# LOOKAHEAD: drop as soon as the job can no longer meet its deadline given the work
#            the cluster has already accepted, i.e. counting the time its remaining
#            stages will spend queued behind jobs that are already in the system
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


def get_remaining_processing_time(task: Task, worker_size: int | None = None) -> float:
    """Returns the minimum time still needed before [task]'s deadline can be met.

    Under job-level SLOs the deadline is the end of the pipeline, so what remains
    is the critical path through every incomplete task of the job, assuming batch
    size 1 and no queueing. Under per-stage (NEXUS) SLOs each stage carries its own
    deadline that already budgets for the stages behind it, so the only work left
    to fit is this stage's own execution.

    Args:
        task: Task whose remaining work to measure
        worker_size: Memory size (GB) of the worker holding the task, whose batch
        exec times to read. Callers that do not know where the task will run pass
        None, and the slowest worker size the model is profiled for is assumed.
    """
    if gcfg.SLO_TYPE == "NEXUS":
        exec_times = task.model_data.batch_exec_times
        if worker_size is not None and worker_size in exec_times:
            return exec_times[worker_size][1]

        return get_min_exec_time(task.model_data)

    complete_task_ids = {tid for tid, t in task.job._task_states.items() if t >= 0}
    return _get_processing_time(task.job, complete_task_ids)


def get_remaining_time_with_queueing(time: float, task: Task, load_view,
                                     worker_size: int | None = None) -> float:
    """Returns how long [task]'s job still needs before its deadline can be met,
    counting the time its remaining stages will spend waiting behind work the
    cluster has already accepted.

    This is the queue aware counterpart of [get_remaining_processing_time], and
    splits the same way. Under per-stage (NEXUS) SLOs the deadline being tested is
    this stage's, which already budgets for the stages behind it, so only this
    stage's wait and execution are counted. Under job-level SLOs the deadline is
    the end of the pipeline, so the whole critical path through the job's
    incomplete tasks is counted, each stage charged for the backlog standing in
    front of it.

    The walk mirrors [_get_processing_time]: stages that can run in parallel are
    charged the slowest of them, since a join waits for its last incoming branch.

    Args:
        time: Time at which the estimate is made
        task: Task whose job's remaining work to measure
        load_view: ClusterLoadView describing the work already in the system
        worker_size: Memory size (GB) of the worker holding the task, whose batch
        exec times to read for this stage. None assumes the slowest worker size
        the model is profiled for.

    Returns:
        remaining: Time still needed, including queueing (ms)
    """
    def _exec_time(model_data, this_stage: bool) -> float:
        if this_stage and worker_size is not None \
                and worker_size in model_data.batch_exec_times:
            return model_data.batch_exec_times[worker_size][1]

        return get_min_exec_time(model_data)

    if gcfg.SLO_TYPE == "NEXUS":
        return (load_view.get_queue_delay(task.model_data.id, time)
                + _exec_time(task.model_data, True))

    job = task.job
    complete_task_ids = {tid for tid, t in job._task_states.items() if t >= 0}

    dependencies: dict[int, set[int]] = {}
    dependents: dict[int, set[int]] = {}
    available = []
    for t in job.tasks:
        dependencies[t.task_id] = set(t.required_task_ids) - complete_task_ids
        dependents[t.task_id] = set(t.next_task_ids) - complete_task_ids

        if t.task_id not in complete_task_ids and not dependencies[t.task_id]:
            available.append(t)

    remaining = 0.0
    while available:
        # every stage in this wave waits out its own model's backlog, and the wave
        # is not done until its slowest stage is
        remaining += max(
            load_view.get_queue_delay(t.model_data.id, time)
            + _exec_time(t.model_data, t.task_id == task.task_id)
            for t in available)

        next_available = []
        for t in available:
            for dep in dependents[t.task_id]:
                dependencies[dep].discard(t.task_id)
                if not dependencies[dep]:
                    next_available.append(job.get_task_by_id(dep))

        available = next_available

    return remaining


def _drops_task(policy: str, time: float, task: Task,
                worker_size: int | None = None, load_view=None) -> bool:
    """Decides whether [policy] would drop the job owning [task] at [time].

    The applicable deadline comes from [task.get_task_deadline]: the job deadline
    under JOB_LEVEL SLOs, this stage's deadline under NEXUS SLOs. LAZY drops only
    once that deadline has already elapsed. EARLY drops as soon as it is
    guaranteed to be missed: when [time] plus the remaining processing time already
    exceeds it. Since remaining processing time is non-negative, EARLY drops
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
        remaining = get_remaining_time_with_queueing(time, task, load_view, worker_size)
        return (time + remaining) > deadline

    raise ValueError(f"Unrecognized drop policy {policy}")


def needs_load_view(policy: str) -> bool:
    """Whether [policy] can only decide with a ClusterLoadView."""
    return policy in _LOAD_AWARE_POLICIES


def should_drop_task(time: float, task: Task, worker_size: int | None = None,
                     load_view=None) -> bool:
    """Whether the job owning [task] should be dropped at [time] under DROP_POLICY,
    i.e. whether it is actually removed from the cluster. See [_drops_task].
    """
    return _drops_task(gcfg.DROP_POLICY, time, task, worker_size, load_view)


def should_shadow_drop_task(time: float, task: Task, worker_size: int | None = None,
                            load_view=None) -> bool:
    """Whether SHADOW_DROP_POLICY would have dropped the job owning [task] at
    [time]. The job keeps running either way; the answer is only recorded. See
    [_drops_task].
    """
    return _drops_task(gcfg.SHADOW_DROP_POLICY, time, task, worker_size, load_view)


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
    if load_view is not None and needs_load_view(gcfg.DROP_POLICY):
        # the loop below empties the queue as it decides, so fix the backlog now
        load_view.prime(time)

    kept = []
    newly_dropped: list[int] = []

    while task_queue.qsize() > 0:
        qt = task_queue.get()
        job = qt.task.job

        if job.id in already_dropped or job.id in newly_dropped:
            continue

        if should_drop_task(time, qt.task, worker_size, load_view):
            newly_dropped.append(job.id)
            continue

        kept.append(qt)

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

    seen_here: set[int] = set()
    newly_shadow_dropped: list[tuple[int, int]] = []

    for qt in list(task_queue.queue):
        job_id = qt.task.job.id
        if job_id in already_shadow_dropped or job_id in seen_here:
            continue

        if should_shadow_drop_task(time, qt.task, worker_size, load_view):
            seen_here.add(job_id)
            newly_shadow_dropped.append((job_id, qt.task.task_id))

    return newly_shadow_dropped
