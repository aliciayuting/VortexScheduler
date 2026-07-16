import numpy as np
import core.configs.gen_config as gcfg

from core.task import Task
from core.data_models.task_data import TaskData
from schedulers.algo.boost_algo import get_task_priority_by_boost, BoostPolicy


def _remaining_rank(task_data: TaskData) -> float:
    """Critical path execution time from task_data to the end of its DAG,
    averaged over all partition sizes."""
    avg_exec = np.mean([
        task_data.model_data.batch_exec_times[s][1]
        for s in task_data.model_data.batch_exec_times
    ])
    if not task_data.next_tasks:
        return avg_exec
    return avg_exec + max(_remaining_rank(nt) for nt in task_data.next_tasks)


_BOOST_POLICY_MAP = {
    "JOB_SIZE":                    BoostPolicy.TOTAL_JOB_TIME,
    "REMAINING_JOB_TIME":          BoostPolicy.REMAINING_JOB_TIME,
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
            task_data = task.job.workflow.tasks[task.task_id]
            self.priority = task.get_task_deadline() - _remaining_rank(task_data) - time
        elif gcfg.BOOST_POLICY == "RELATIVE_LAXITY":
            task_data = task.job.workflow.tasks[task.task_id]
            remaining = _remaining_rank(task_data)
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