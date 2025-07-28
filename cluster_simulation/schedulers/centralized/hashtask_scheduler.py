from core.config import *
from core.events import *

from schedulers.centralized.scheduler import Scheduler


class HashTaskScheduler(Scheduler):
    """
        Scheduler class for a round robin centralized task scheduler.
    """

    def __init__(self, simulation, herd_assignment=None):
        super().__init__(simulation, herd_assignment)
        self.next_worker_id = { jt: [0 for _ in get_task_types([jt])] for jt in self.simulation.job_types_list }

    def schedule_job_on_arrival(self, job, current_time):
        self._assign_adfg(job.tasks, current_time)

        task_arrival_events = []

        initial_tasks = [task for task in job.tasks if len(task.required_task_ids) == 0]
        for task in initial_tasks:
            task_arrival_time = current_time + CPU_to_CPU_delay(task.input_size)
            worker_index = task.ADFG[task.task_id]
            task_arrival_events.append(EventOrders(
                task_arrival_time, TaskArrival(self.simulation.workers[worker_index], task, task.job.id)))

        return task_arrival_events

    def schedule_tasks_on_arrival(self, tasks, current_time):
        task_arrival_events = []

        self._assign_adfg(tasks, current_time)
        
        for task in tasks:
            task_arrival_time = current_time + CPU_to_CPU_delay(task.input_size)
            worker_index = task.ADFG[task.task_id]
            task_arrival_events.append(EventOrders(
                task_arrival_time, TaskArrival(self.simulation.workers[worker_index], task, task.job.id)))

        return task_arrival_events
    
    def _assign_adfg(self, tasks, current_time):
        for task in tasks:
            if ENABLE_DYNAMIC_MODEL_LOADING:
                if ALLOCATION_STRATEGY == "HERD":
                    # don't choose worker that is not in the correct group
                    while task.model and task.model.model_id not in self.herd_assignment.group_models[self.simulation.workers[self.next_worker_id[task.task_type[0]][task.task_id]].group_id] and \
                        self.simulation.workers[self.next_worker_id[task.task_type[0]][task.task_id]].total_memory * 10**6 < task.model.model_size:
                        self.next_worker_id[task.task_type[0]][task.task_id] = (self.next_worker_id[task.task_type[0]][task.task_id] + 1) % len(self.simulation.workers)
                else:
                    # don't choose partition that is too small
                    while task.model and self.simulation.workers[self.next_worker_id[task.task_type[0]][task.task_id]].total_memory * 10**6 < task.model.model_size:
                        self.next_worker_id[task.task_type[0]][task.task_id] = (self.next_worker_id[task.task_type[0]][task.task_id] + 1) % len(self.simulation.workers)
            else:
                # don't choose worker without the required model
                while task.model and all(m.model_id != task.model.model_id for m in self.simulation.workers[self.next_worker_id[task.task_type[0]][task.task_id]].GPU_state.placed_models(current_time)):
                    self.next_worker_id[task.task_type[0]][task.task_id] = (self.next_worker_id[task.task_type[0]][task.task_id] + 1) % len(self.simulation.workers)
            task.ADFG[task.task_id] = self.next_worker_id[task.task_type[0]][task.task_id]
            task.job.ADFG[task.task_id] = task.ADFG
            self.next_worker_id[task.task_type[0]][task.task_id] = (self.next_worker_id[task.task_type[0]][task.task_id] + 1) % len(self.simulation.workers)