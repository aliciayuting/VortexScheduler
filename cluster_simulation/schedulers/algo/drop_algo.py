from queue import PriorityQueue

import numpy as np

import core.configs.gen_config as gcfg

from core.job import Job
from core.task import Task
from schedulers.algo.boost_algo import _get_processing_time


# NONE:    never drop
# LAZY:    drop once the job's deadline has already passed
# EARLY:   drop as soon as the job can no longer meet its deadline
DROP_POLICIES = ["NONE", "LAZY", "EARLY"]

# NONE:          admit every job
# PROBABILISTIC: reject each arriving job independently with probability
#                ADMISSION_DROP_RATE
# ROUND_ROBIN:   reject one job out of every round(1 / ADMISSION_DROP_RATE)
ADMISSION_CONTROL_POLICIES = ["NONE", "PROBABILISTIC", "ROUND_ROBIN"]


class AdmissionController:
    """Flat rate load shedding, applied to jobs as they arrive at the scheduler.

    Independent of DROP_POLICY: admission control rejects a fixed fraction of
    arrivals at the front door without regard to deadlines, while the drop policy
    aborts jobs mid-flight once their deadline is known to be unreachable. Both
    may be enabled at once.
    """

    def __init__(self):
        # arrivals seen since the run began, for round robin rejection
        self.arrival_count = 0

    def should_reject(self, job: Job) -> bool:
        """Decides whether a newly arrived [job] should be rejected.

        Args:
            job: Job that just arrived at the scheduler

        Returns:
            should_reject: True if the job should be rejected on arrival
        """
        if gcfg.ADMISSION_CONTROL_POLICY == "NONE":
            return False

        rate = gcfg.ADMISSION_DROP_RATE
        assert(0 <= rate <= 1), f"ADMISSION_DROP_RATE must be in [0, 1], got {rate}"

        if rate == 0:
            return False

        self.arrival_count += 1

        if gcfg.ADMISSION_CONTROL_POLICY == "PROBABILISTIC":
            return np.random.random() < rate
        elif gcfg.ADMISSION_CONTROL_POLICY == "ROUND_ROBIN":
            # reject exactly one job out of every [period] arrivals
            period = max(1, round(1 / rate))
            return self.arrival_count % period == 0

        raise ValueError("Unrecognized admission control policy "
                         f"{gcfg.ADMISSION_CONTROL_POLICY}")

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


def get_remaining_processing_time(task: Task) -> float:
    """Returns the minimum time still needed before [task]'s deadline can be met.

    Under job-level SLOs the deadline is the end of the pipeline, so what remains
    is the critical path through every incomplete task of the job, assuming batch
    size 1 and no queueing. Under per-stage (NEXUS) SLOs each stage carries its own
    deadline that already budgets for the stages behind it, so the only work left
    to fit is this stage's own execution.
    """
    if gcfg.SLO_TYPE == "NEXUS":
        return task.model_data.batch_exec_times[24][1]

    complete_task_ids = {tid for tid, t in task.job._task_states.items() if t >= 0}
    return _get_processing_time(task.job, complete_task_ids)


def should_drop_task(time: float, task: Task) -> bool:
    """Decides whether the job owning [task] should be dropped at [time] under the
    configured drop policy.

    The applicable deadline comes from [task.get_task_deadline]: the job deadline
    under JOB_LEVEL SLOs, this stage's deadline under NEXUS SLOs. LAZY drops only
    once that deadline has already elapsed. EARLY drops as soon as it is
    guaranteed to be missed: when [time] plus the remaining processing time already
    exceeds it. Since remaining processing time is non-negative, EARLY drops
    whatever LAZY would, and never later than LAZY does.

    Args:
        time: Time at which the drop decision is made
        task: Task whose job is considered for dropping

    Returns:
        should_drop: True if the job should be dropped
    """
    if gcfg.DROP_POLICY == "NONE":
        return False

    deadline = task.get_task_deadline()

    if gcfg.DROP_POLICY == "LAZY":
        return time > deadline
    elif gcfg.DROP_POLICY == "EARLY":
        return (time + get_remaining_processing_time(task)) > deadline

    raise ValueError(f"Unrecognized drop policy {gcfg.DROP_POLICY}")


def drop_from_queue(time: float, task_queue: PriorityQueue,
                    already_dropped: set[int]) -> list[int]:
    """Removes every queued task belonging to a job that should be dropped at
    [time], as well as tasks of jobs that were already dropped elsewhere.

    Args:
        time: Time at which the drop decision is made
        task_queue: Queue of QueuedTask to filter in place
        already_dropped: IDs of jobs known to be dropped already

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

        if should_drop_task(time, qt.task):
            newly_dropped.append(job.id)
            continue

        kept.append(qt)

    for qt in kept:
        task_queue.put(qt)

    return newly_dropped
