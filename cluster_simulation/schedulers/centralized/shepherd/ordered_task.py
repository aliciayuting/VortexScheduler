from core.task import Task
from core.config import *


class OrderedTask:
    """
        Task wrapper for PriorityQueue. Ordered by deadline.
    """

    def __init__(self, task: Task, current_time: float):
        self.task = task
        self.task_arrival_time = current_time

        if SLO_GRANULARITY == "TASK":
            self.deadline = task.job.create_time + task.slo
        else:
            self.deadline = task.job.create_time + task.job.slo * (1 + SLO_SLACK)

    def __lt__(self, other):
        return self.deadline < other.deadline
    
    def __str__(self):
        return f"[DEADLINE: {self.deadline}] {self.task}"
    
    def __repr__(self):
        return self.__str__()