from core.simulation import *
from core.config import *

from workers.taskworker import *

from schedulers.algo.herd_algo import *
from schedulers.centralized.hashtask_scheduler import HashTaskScheduler
from schedulers.centralized.heft_scheduler import HeftScheduler
from schedulers.centralized.shepherd.shepherd_scheduler import ShepherdScheduler


class Simulation_central(Simulation):
    
    def __init__(self, simulation_name="", job_split="", num_workers=1, job_types_list=[0], produce_breakdown=False):

        Simulation.__init__(self, simulation_name=simulation_name, job_split=job_split,\
                            centralized_scheduler=True,\
                            total_workers=num_workers,\
                            job_types_list=job_types_list,\
                            produce_breakdown=produce_breakdown)
        
        self.herd_assignment = None
        self.initialize_workers()

        if self.simulation_name == "centralheft":
            self.scheduler = HeftScheduler(self, self.herd_assignment)
        elif self.simulation_name == "hashtask":
            self.scheduler = HashTaskScheduler(self, self.herd_assignment)
        elif self.simulation_name == "shepherd":
            self.scheduler = ShepherdScheduler(self, self.herd_assignment)

    def schedule_job_and_send_tasks(self, job, current_time):
        return self.scheduler.schedule_job_on_arrival(job, current_time)
        
    def schedule_tasks_on_arrival(self, tasks, current_time):
        return self.scheduler.schedule_tasks_on_arrival(tasks, current_time)
    
    def schedule_tasks_on_queue(self, current_time):
        return self.scheduler.schedule_tasks_on_queue(current_time)

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

            if cur_event.event.should_abandon_event(cur_event.current_time, {}):
                continue
            
            print(cur_event.to_string())
            print(f"Jobs left: {self.remaining_jobs}, dropped: {len(self.task_drop_log)}")

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