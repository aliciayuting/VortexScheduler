import core.configs.gen_config as gcfg
import numpy as np

from core.job import Job
from core.task import Task

from core.data_models.workflow import Workflow

from workers.worker import Worker

from schedulers.scheduler import Scheduler

from events.event_manager import EventManager
from events.event import *
from events.event_types import *


class CentralRoundRobinScheduler(Scheduler):

    def __init__(self, em: EventManager, workers: dict[UUID, Worker], workflows: list[Workflow], scheduler_worker_id: UUID):
        super().__init__(em)

        self.workers = workers
        self.workflows = workflows
        self.scheduler_worker_id = scheduler_worker_id

        # (job ID, task ID) -> worker ID on which task result/output is stored
        self.output_locs: dict[tuple[int, int], UUID] = {}

        # for round robin dispatch: model ID -> (worker ID, instance ID)
        self.last_sent_tasks_to: dict[int, tuple[UUID, UUID]] = {}


    def on_job_arrival(self, time: float, job: Job):
        return self.on_tasks_arrival(time, 
                                     [t for t in job.tasks if len(t.required_task_ids) == 0])
    

    def on_tasks_arrival(self, time: float, tasks: list[Task]):
        tasks_to_send: dict[UUID, list[Task]] = {}
        for task in tasks:
            relevant_instances = [(w.id, s.model.id) 
                                  for w in sorted(self.workers.values(), key=lambda w: (w.create_time, w.id))
                                  for s in sorted(w.GPU_state.state_at(time), key=lambda s: (s.model.created_at, s.model.id))
                                  if s.model.data.id == task.model_data.id]
            
            next_idx = 0
            if task.model_data.id not in self.last_sent_tasks_to:
                next_idx = np.random.randint(0, len(relevant_instances))
            else:
                last_idx = relevant_instances.index(self.last_sent_tasks_to[task.model_data.id])
                next_idx = (last_idx + 1) % len(relevant_instances)

            (worker_id, instance_id) = relevant_instances[next_idx]
            
            if worker_id not in tasks_to_send: tasks_to_send[worker_id] = []
            tasks_to_send[worker_id].append(task)

            # update round robin ptr
            self.last_sent_tasks_to[task.model_data.id] = (worker_id, instance_id)
        
        for worker_id, worker_tasks in tasks_to_send.items():
            self.em.add_event(
                Event(time,
                      EVENT_TYPES[EventIds.TASKS_ASSIGNED_TO_WORKER],
                      kwargs={"worker_id": worker_id,
                              "tasks": worker_tasks}), 
                self.emitter_id)
            
            # tell workers to send required outputs to chosen worker
            inputs_from_scheduler: list[Task] = []
            outputs_from_workers: dict[UUID, list[tuple[int, int]]] = {}
            for task in worker_tasks:
                if len(task.required_task_ids) == 0:
                    inputs_from_scheduler.append(task)

                else:
                    for rt in task.required_task_ids:
                        wid = self.output_locs[(task.job.id, rt)]
                        if wid not in outputs_from_workers:
                            outputs_from_workers[wid] = []
                        outputs_from_workers[wid].append((task.job.id, rt))

            if inputs_from_scheduler:
                self.em.add_event(
                    Event(time, 
                            EVENT_TYPES[EventIds.TASKS_INPUTS_SENT_TO_WORKER],
                            kwargs={"tasks": worker_tasks,
                                    "from_worker_id": self.scheduler_worker_id,
                                    "to_worker_id": worker_id,
                                    "ignore_transfer_time": not gcfg.ENABLE_NETWORKING_DELAYS}),
                    self.emitter_id)
            
            if outputs_from_workers:
                for (from_worker_id, job_task_ids) in outputs_from_workers.items():
                    self.em.add_event(
                        Event(time, 
                            EVENT_TYPES[EventIds.TASKS_OUTPUTS_ASSIGNED_TO_WORKER],
                            kwargs={"job_task_ids": job_task_ids,
                                    "from_worker_id": from_worker_id,
                                    "to_worker_id": worker_id}),
                        self.emitter_id)
    

    def on_jobs_dropped(self, time: float, job_ids: list[int]):
        pass
    

    def on_batch_start(self, time, batch, worker_id, instance_id):
        pass


    def on_batch_finish(self, time: float, batch: Batch, worker_id: UUID, instance_id: UUID):
        for task in batch.tasks:
            assert((task.job.id, task.task_id) not in self.output_locs)
            self.output_locs[(task.job.id, task.task_id)] = worker_id