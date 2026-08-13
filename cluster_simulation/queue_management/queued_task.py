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
        else:
            raise ValueError(f"Unrecognized queue ordering policy {gcfg.BOOST_POLICY}")

    def __lt__(self, other):
        return self.priority < other.priority
    
    def __str__(self):
        return f"[PRIORITY: {self.priority}] [TASK: {self.task}]"
    
    def __repr__(self):
        return self.__str__()