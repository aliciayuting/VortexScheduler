from queue import PriorityQueue

import core.configs.gen_config as gcfg

from core.task import Task
from schedulers.algo.boost_algo import _get_processing_time, get_min_exec_time


# NONE:    never drop
# LAZY:    drop once the job's deadline has already passed
# EARLY:   drop as soon as the job can no longer meet its deadline
DROP_POLICIES = ["NONE", "LAZY", "EARLY"]

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


def _drops_task(policy: str, time: float, task: Task,
                worker_size: int | None = None) -> bool:
    """Decides whether [policy] would drop the job owning [task] at [time].

    The applicable deadline comes from [task.get_task_deadline]: the job deadline
    under JOB_LEVEL SLOs, this stage's deadline under NEXUS SLOs. LAZY drops only
    once that deadline has already elapsed. EARLY drops as soon as it is
    guaranteed to be missed: when [time] plus the remaining processing time already
    exceeds it. Since remaining processing time is non-negative, EARLY drops
    whatever LAZY would, and never later than LAZY does.

    Args:
        policy: One of DROP_POLICIES
        time: Time at which the drop decision is made
        task: Task whose job is considered for dropping
        worker_size: Memory size (GB) of the worker holding the task, if known (see
        [get_remaining_processing_time])

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

    raise ValueError(f"Unrecognized drop policy {policy}")


def should_drop_task(time: float, task: Task, worker_size: int | None = None) -> bool:
    """Whether the job owning [task] should be dropped at [time] under DROP_POLICY,
    i.e. whether it is actually removed from the cluster. See [_drops_task].
    """
    return _drops_task(gcfg.DROP_POLICY, time, task, worker_size)


def should_shadow_drop_task(time: float, task: Task,
                            worker_size: int | None = None) -> bool:
    """Whether SHADOW_DROP_POLICY would have dropped the job owning [task] at
    [time]. The job keeps running either way; the answer is only recorded. See
    [_drops_task].
    """
    return _drops_task(gcfg.SHADOW_DROP_POLICY, time, task, worker_size)


def drop_from_queue(time: float, task_queue: PriorityQueue,
                    already_dropped: set[int],
                    worker_size: int | None = None) -> list[int]:
    """Removes every queued task belonging to a job that should be dropped at
    [time], as well as tasks of jobs that were already dropped elsewhere.

    Args:
        time: Time at which the drop decision is made
        task_queue: Queue of QueuedTask to filter in place
        already_dropped: IDs of jobs known to be dropped already
        worker_size: Memory size (GB) of the worker holding the queue, if known (see
        [get_remaining_processing_time])

    Returns:
        newly_dropped: IDs of jobs dropped by this call, excluding jobs in
        [already_dropped]
    """
    kept = []
    newly_dropped: list[int] = []

    while task_queue.qsize() > 0:
        qt = task_queue.get()
        job = qt.task.job

        if job.id in already_dropped or job.id in newly_dropped:
            continue

        if should_drop_task(time, qt.task, worker_size):
            newly_dropped.append(job.id)
            continue

        kept.append(qt)

    for qt in kept:
        task_queue.put(qt)

    return newly_dropped


def shadow_drops_from_queue(time: float, task_queue: PriorityQueue,
                            already_shadow_dropped: set[int],
                            worker_size: int | None = None) -> list[tuple[int, int]]:
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

        if should_shadow_drop_task(time, qt.task, worker_size):
            seen_here.add(job_id)
            newly_shadow_dropped.append((job_id, qt.task.task_id))

    return newly_shadow_dropped
