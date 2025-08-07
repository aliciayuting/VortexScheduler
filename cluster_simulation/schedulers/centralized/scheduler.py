from core.job import Job
from core.task import Task
from core.events import *


class Scheduler:
    """
        Base class for a centralized scheduler.
    """

    def __init__(self, simulation, herd_assignment=None):
        self.simulation = simulation
        self.herd_assignment = herd_assignment

    def update_herd_assignment(self, herd_assignment):
        self.herd_assignment = herd_assignment
    
    def schedule_job_on_arrival(self, job: Job, current_time: float) -> list[EventOrders]:
        """
            Schedules all [job.tasks].
        """
        raise NotImplementedError()
    
    def schedule_tasks_on_arrival(self, tasks: list[Task], current_time: float) -> list[EventOrders]:
        """
            Schedules [tasks] which may or may not belong to the same jobs or
            be of the same type.
        """
        raise NotImplementedError()
    
    def schedule_tasks_on_queue(self, current_time: float) -> list[EventOrders]:
        """
            Schedules tasks that may be on queue without requiring
            newly arrived tasks.
        """
        raise NotImplementedError()