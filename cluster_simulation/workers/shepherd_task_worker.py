from workers.worker import *
from workers.taskworker import *

from core.network import *
from core.events import *

import numpy as np


class ShepherdWorker(TaskWorker):
    
    def free_slot(self, current_time, batch: Batch, task_type):
        """ Attempts to launch another task. """
        events = super().free_slot(current_time, batch, task_type)
        events += self.simulation.scheduler.flex_schedule_on_batch_completion(
            self, batch, current_time)
        return events
    
    def maybe_start_batch(self, batch, current_time):
        events = super().maybe_start_batch(batch, current_time)
        if not events:
            return [EventOrders(
                current_time + CPU_to_CPU_delay(batch.size()*batch.tasks[0].input_size),
                BatchRejectionAtWorker(self.simulation, self, batch))]
        return events
    
    def preempt_batch(self, old_batch_id: int, new_batch: Batch, current_time: float):
        evicted_batch = self.evict_batch(old_batch_id, current_time)

        if self.can_run_task(current_time, new_batch.model) == self._CAN_RUN_ON_EVICT:
            current_time += self.evict_models_from_GPU_until(
                current_time, new_batch.model.model_size, self.FCFS_EVICTION)

        events, _ = self.batch_execute(new_batch, current_time)
        assert(events)
        events.append(EventOrders(
            current_time + CPU_to_CPU_delay(evicted_batch.tasks[0].input_size * evicted_batch.size()), 
            TasksArrivalAtScheduler(self.simulation, evicted_batch.tasks)))
        return events

    #  ---------------------------  Subsequent TASK Transfer   --------------------

    def send_results_to_next_workers(self, current_time: float, batch: Batch) -> list:
        """
        Send the result of a task to the next worker in the inference pipeline (it may be the same worker)
        """
        new_tasks = []
        for task in batch.tasks:
            new_tasks += task.job.newly_available_tasks(task)
        if len(new_tasks) > 0:
            return [EventOrders(
                current_time + CPU_to_CPU_delay(task.result_size), 
                TasksArrivalAtScheduler(self.simulation, new_tasks))]
        return []
    
    def get_wait_time(self, current_time: float, model_id: int) -> float:
        if model_id < 0:
            return 0 # GPU not needed
        
        if any(s.model.model_id == model_id 
               for s in self.GPU_state.placed_model_states(current_time)):
            if any(s.model.model_id == model_id and not s.reserved_batch
                   for s in self.GPU_state.placed_model_states(current_time)):
                return 0 # model available
            else:
                # duplicate model cannot be loaded, return time to available
                time_to_free = min(s.reserved_until for s in self.GPU_state.placed_model_states(current_time) 
                                    if s.model.model_id == model_id) - current_time
                return time_to_free # time remaining until model becomes available
        
        all_models = [m for wf in list(self.simulation.metadata_service.job_type_models.values()) for m in wf]
        model = [m for m in all_models if m.model_id == model_id][0]

        # model is 1) not placed already and 2) can be fetched with eviction if necessary
        if all(m.model_id != model_id for m in self.GPU_state.placed_models(current_time)) and \
            self.GPU_state.can_fetch_model_on_eviction(model, current_time):
            fetch_time = SameMachineCPUtoGPU_delay(model.model_size)
            return fetch_time
        
        # model cannot be fetched and is not loaded
        return np.inf