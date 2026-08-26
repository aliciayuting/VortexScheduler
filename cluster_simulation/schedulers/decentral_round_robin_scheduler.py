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


class DecentralRoundRobinScheduler(Scheduler):

    def __init__(self, em: EventManager, workers: dict[UUID, Worker], workflows: dict[int, Workflow]):
        super().__init__(em, workers, workflows)

        # (job ID, task ID) -> worker ID on which task result/output is stored
        self.output_locs: dict[tuple[int, int], UUID] = {}

        # (job ID, task ID) -> worker ID to execute
        self.scheduled_task_to_worker: dict[tuple[int, int], UUID] = {}

        # for GLOBAL round robin dispatch: model ID -> (worker ID, instance ID)
        self.last_sent_tasks_to: dict[int, tuple[UUID, UUID]] = {}

        # for LOCAL round robin dispatch (per-worker): worker ID -> model ID -> (worker ID, instance ID)
        self.worker_last_sent_tasks_to: dict[UUID, dict[int, tuple[UUID, UUID]]] = {}


    def on_job_arrival(self, time: float, job: Job):
        return self.on_tasks_arrival(time, 
                                     [t for t in job.tasks if len(t.required_task_ids) == 0])
    

    def on_tasks_arrival(self, time: float, tasks: list[Task]):
        assert(all(len(t.required_task_ids) == 0 for t in tasks))

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
            
            self.em.add_event(
                Event(time, 
                        EVENT_TYPES[EventIds.TASKS_INPUTS_SENT_TO_WORKER],
                        kwargs={"tasks": worker_tasks,
                                "from_worker_id": None,
                                "to_worker_id": worker_id,
                                "ignore_transfer_time": True}), # always ignore sched -> worker for decentral
                self.emitter_id)
    

    def on_jobs_dropped(self, time: float, job_ids: list[int]):
        pass
    

    def on_batch_start(self, time, batch, worker_id, instance_id):
        pass


    def on_batch_finish(self, time: float, batch: Batch, worker_id: UUID, instance_id: UUID):
        if worker_id not in self.worker_last_sent_tasks_to:
            self.worker_last_sent_tasks_to[worker_id] = {}
        
        for task in batch.tasks:
            assert((task.job.id, task.task_id) not in self.output_locs)
            self.output_locs[(task.job.id, task.task_id)] = worker_id

            # workers drop for decentralized schedulers; do not dispatch further
            # stages of a job that was already dropped
            if task.job.id in self.dropped_job_ids:
                continue

            for next_task_id in task.next_task_ids:
                # if next task was already scheduled, send outputs to assigned worker
                if (task.job.id, next_task_id) in self.scheduled_task_to_worker:
                    self.em.add_event(
                        Event(time,
                              EVENT_TYPES[EventIds.TASKS_OUTPUTS_ASSIGNED_TO_WORKER],
                              kwargs={
                                  "job_task_ids": [(task.job.id, task.task_id)],
                                  "from_worker_id": worker_id,
                                  "to_worker_id": self.scheduled_task_to_worker[(task.job.id, next_task_id)]
                              }),
                        self.emitter_id)
                    
                else: # otherwise, round robin assign a new worker & assign outputs
                    next_task = task.job.workflow.tasks[next_task_id]

                    relevant_instances = [(w.id, s.model.id)
                                          for w in sorted(self.workers.values(), key=lambda w: (w.create_time, w.id))
                                          for s in sorted(w.GPU_state.state_at(time), key=lambda s: (s.model.created_at, s.model.id))
                                          if s.model.data.id == next_task.model_data.id]
                    
                    next_idx = 0
                    if next_task.model_data.id not in self.worker_last_sent_tasks_to[worker_id]:
                        next_idx = np.random.randint(0, len(relevant_instances))
                    else:
                        last_idx = relevant_instances.index(
                            self.worker_last_sent_tasks_to[worker_id][next_task.model_data.id])
                        next_idx = (last_idx + 1) % len(relevant_instances)

                    (next_worker_id, next_instance_id) = relevant_instances[next_idx]

                    self.em.add_event(
                        Event(time,
                            EVENT_TYPES[EventIds.TASKS_ASSIGNED_TO_WORKER],
                            kwargs={"worker_id": next_worker_id,
                                    "tasks": [task.job.get_task_by_id(next_task_id)]}), 
                        self.emitter_id)
                    
                    self.em.add_event(
                        Event(time,
                              EVENT_TYPES[EventIds.TASKS_OUTPUTS_ASSIGNED_TO_WORKER],
                              kwargs={
                                  "job_task_ids": [(task.job.id, task.task_id)],
                                  "from_worker_id": worker_id,
                                  "to_worker_id": next_worker_id
                              }),
                        self.emitter_id)

                    # update round robin pointer (per worker)
                    self.worker_last_sent_tasks_to[worker_id][next_task.model_data.id] = \
                        (next_worker_id, next_instance_id)
                    
                    # update scheduling decision
                    self.scheduled_task_to_worker[(task.job.id, next_task_id)] = next_worker_id