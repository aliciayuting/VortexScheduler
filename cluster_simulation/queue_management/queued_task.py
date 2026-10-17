import heapq
from queue import PriorityQueue

import core.configs.gen_config as gcfg

from core.task import Task
from schedulers.algo.boost_algo import get_task_priority_by_boost, BoostPolicy, _get_processing_time


_BOOST_POLICY_MAP = {
    "JOB_SIZE":                    BoostPolicy.TOTAL_JOB_TIME,
    "REMAINING_EXEC_TIME":         BoostPolicy.REMAINING_EXEC_TIME,
    "REMAINING_TIME_TO_DEADLINE":  BoostPolicy.REMAINING_TIME_TO_DEADLINE,
    "LAXITY_BOOST":                BoostPolicy.LAXITY,
    "RELATIVE_LAXITY_BOOST":       BoostPolicy.RELATIVE_LAXITY,
}

# policies whose ordering changes as time passes, so a queue under one must be
# re-keyed at the time it is read (see [refresh_queue]) rather than trusted to
# keep the order its tasks were enqueued in
_TIME_VARYING_POLICIES = {"ACCUMULATING_PRIORITY"}


class QueuedTask:
    def __init__(self, task: Task, time: float = 0.0):
        self.task = task

        if gcfg.BOOST_POLICY == "FCFS":
            self.priority = task.job.create_time
        elif gcfg.BOOST_POLICY == "EDF":
            self.priority = task.get_task_deadline()
        elif gcfg.BOOST_POLICY == "LAXITY":
            complete = {tid for tid, t in task.job._task_states.items() if t != -1}
            remaining = _get_processing_time(task.job, complete)
            self.priority = task.get_task_deadline() - remaining - time
        elif gcfg.BOOST_POLICY == "RELATIVE_LAXITY":
            complete = {tid for tid, t in task.job._task_states.items() if t != -1}
            remaining = _get_processing_time(task.job, complete)
            self.priority = (task.get_task_deadline() - time - remaining) / remaining
        elif gcfg.BOOST_POLICY in _BOOST_POLICY_MAP:
            self.priority = get_task_priority_by_boost(
                time, task, _BOOST_POLICY_MAP[gcfg.BOOST_POLICY])
        elif gcfg.BOOST_POLICY == "ACCUMULATING_PRIORITY":
            # fixed for the job's lifetime, so computed once rather than per refresh
            self.job_size = _get_processing_time(task.job, set())
            assert(self.job_size > 0)
            self.refresh(time)
        else:
            raise ValueError(f"Unrecognized queue ordering policy {gcfg.BOOST_POLICY}")

    def refresh(self, time: float):
        """Recomputes the priority at [time] under a time varying policy; no-op
        under the others.

        ACCUMULATING_PRIORITY accrues priority at a rate inversely proportional to
        job size, so the job that has waited longest relative to its size, i.e. the
        one with the largest slowdown so far, is served first. Waiting is counted
        from job creation, so time spent in earlier stages counts toward it. Jobs of
        different sizes swap order as time passes, which is why this is recomputed
        on every read; jobs of equal size never do, so a single workflow queue
        orders exactly as FCFS.
        """
        if gcfg.BOOST_POLICY == "ACCUMULATING_PRIORITY":
            # lower is served first
            self.priority = -(time - self.task.job.create_time) / self.job_size

    def __lt__(self, other):
        return self.priority < other.priority
    
    def __str__(self):
        return f"[PRIORITY: {self.priority}] [TASK: {self.task}]"
    
    def __repr__(self):
        return self.__str__()


def refresh_queue(time: float, task_queue: PriorityQueue):
    """Re-keys every task in [task_queue] at [time] and restores heap order, so
    that a time varying ordering is read as it stands at [time]. No-op under the
    static policies, whose order is fixed at enqueue.

    Args:
        time: Time at which the queue is about to be read
        task_queue: Queue of QueuedTask to re-key in place
    """
    if gcfg.BOOST_POLICY not in _TIME_VARYING_POLICIES:
        return

    with task_queue.mutex:
        for qt in task_queue.queue:
            qt.refresh(time)
        heapq.heapify(task_queue.queue)
