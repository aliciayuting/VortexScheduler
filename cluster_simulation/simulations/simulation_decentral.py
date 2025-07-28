from core.simulation import *
from core.config import *

from workers.taskworker import *


class Simulation_decentral(Simulation):
    def __init__(self, simulation_name="", job_split="", num_workers=1, job_types_list=[0], dynamic_adjust=True, consider_load=True, consider_cache=True, produce_breakdown=False):

        Simulation.__init__(self, simulation_name=simulation_name, job_split=job_split, \
                            centralized_scheduler=False, \
                            dynamic_adjust=dynamic_adjust, \
                            total_workers=num_workers, \
                            job_types_list=job_types_list,\
                            produce_breakdown=produce_breakdown)

        self.consider_load, self.consider_cache = consider_load, consider_cache
        self.initialize_workers()

    def add_job_completion_time(self, job_id, task_id, completion_time):
        job_is_completed = self.jobs[job_id].job_completed(
            completion_time, task_id)
        if job_is_completed:
            self.remaining_jobs -= 1

    def run(self):
        self.generate_all_jobs()

        last_time = 0
        while (self.remaining_jobs - len(self.task_drop_log)) > 0:
            cur_event = self.event_queue.get()

            print(cur_event.to_string())
            print(f"Jobs left: {self.remaining_jobs}")

            worker_id = -1
            if type(cur_event.event).is_worker_event():
                worker_id = cur_event.event.worker.worker_id
            self.event_log.loc[len(self.event_log)] = [cur_event.current_time, worker_id, cur_event.event.to_string()]

            assert cur_event.current_time >= last_time
            last_time = cur_event.current_time
            new_events = cur_event.event.run(cur_event.current_time)
            for new_event in new_events:
                last_time = cur_event.current_time
                self.event_queue.put(new_event)
        self.run_finish(last_time, by_job_type=True)