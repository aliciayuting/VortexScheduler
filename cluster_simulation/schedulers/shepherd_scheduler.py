import numpy as np

import core.configs.gen_config as gcfg

from queue import PriorityQueue

from core.job import Job
from core.task import Task
from core.model import Model
from core.data_models.workflow import Workflow

from workers.worker import Worker
from workers.gpu_state import ModelState

from schedulers.scheduler import Scheduler
from queue_management.queued_task import QueuedTask
from queue_management.batching import TaskBatcher

from events.event_manager import EventManager
from events.event import *
from events.event_types import *


class ShepherdScheduler(Scheduler):

    def __init__(self, em: EventManager, workers: dict[UUID, Worker], workflows: list[Workflow], scheduler_worker_id: UUID):
        super().__init__(em)

        self.workers = workers
        self.workflows = workflows
        self.scheduler_worker_id = scheduler_worker_id

        # (worker ID, instance ID) -> list[(job ID, task ID)] to record scheduling decisions
        self.scheduled_batch_to_instance: dict[tuple[UUID, UUID], list[tuple[int, int]]] = {}

        # job ID -> Job instance
        self.arrived_jobs: dict[int, Job] = {}

        # model ID -> model queue
        self.queues: dict[int, PriorityQueue] = {}

        # (job ID, task ID) -> worker ID on which task result/output is stored
        self.output_locs: dict[tuple[int, int], UUID] = {}

        # for round robin dispatch: model ID -> (worker ID, instance ID)
        self.last_sent_tasks_to: dict[int, tuple[UUID, UUID]] = {}


    def on_job_arrival(self, time: float, job: Job):
        self.arrived_jobs[job.id] = job
        return self.on_tasks_arrival(time, 
                                     [t for t in job.tasks if len(t.required_task_ids) == 0])
    

    def on_tasks_arrival(self, time: float, tasks: list[Task]):
        model_ids_to_check = set()
        for task in tasks:
            model_ids_to_check.add(task.model_data.id)
            if task.model_data.id not in self.queues:
                self.queues[task.model_data.id] = PriorityQueue()
            self.queues[task.model_data.id].put(QueuedTask(task, time))
        
        all_instance_states = [(w, s) for w in self.workers.values() for s in w.GPU_state.state_at(time)]
        for model_id in model_ids_to_check:
            # sort instances by availability, then create time, then ID
            relevant_instances: list[tuple[Worker, ModelState]] = sorted(
                [(w, s) for (w, s) in all_instance_states if s.model.data.id == model_id], 
                key=lambda k: (0 if (k[0].id, k[1].model.id) not in self.scheduled_batch_to_instance or \
                                    self.scheduled_batch_to_instance[(k[0].id, k[1].model.id)] == None else 1,
                               k[1].model.created_at,
                               k[1].model.id))
            
            for (worker, instance_state) in relevant_instances:
                self._schedule_instance_if_idle(time, worker.id, instance_state.model.id,
                                                not gcfg.ENABLE_NETWORKING_DELAYS)

                # TODO: preemption
                # curr_batch = None
                # if (worker.id, state.model.id) in self.scheduled_batch_to_instance:
                #     curr_batch = self.scheduled_batch_to_instance.inv[(worker.id, state.model.id)]

                # if not curr_batch:

                # elif gcfg.ENABLE_PREEMPTION and queued_batch.size() >= gcfg.FLEX_LAMBDA * curr_batch.size():
                #     self.scheduled_batch_to_instance[(worker.id, state.model.id)] = queued_batch
                #     for task in queued_batch.tasks:
                #         self.model_queues[state.model.data.id].remove(task)
                    
                #     events.append(EventOrders(
                #         current_time + CPU_to_CPU_delay(sum(t.input_size for t in queued_batch.tasks)), 
                #         BatchPreemptionScheduledAtWorker(self.simulation, worker, state.model.id, queued_batch, curr_batch.id)))
    

    def on_jobs_dropped(self, time: float, job_ids: list[int]):
        pass
    

    def on_batch_start(self, time, batch, worker_id, instance_id):
        pass


    def on_batch_finish(self, time: float, batch: Batch, worker_id: UUID, instance_id: UUID):
        # update scheduling state
        self.scheduled_batch_to_instance[(worker_id, instance_id)] = None
        for task in batch.tasks:
            assert((task.job.id, task.task_id) not in self.output_locs)
            self.output_locs[(task.job.id, task.task_id)] = worker_id

        # assign new batch if queue is not empty
        self._schedule_instance_if_idle(time, worker_id, instance_id, not gcfg.ENABLE_NETWORKING_DELAYS)
    

    def _schedule_instance_if_idle(self, time: float, worker_id: UUID, instance_id: UUID, ignore_transfer_time: bool):
        worker = self.workers[worker_id]
        instance_state = worker.GPU_state.get_instance_state(instance_id, time)

        # skip if queue is empty
        if self.queues[instance_state.model.data.id].qsize() == 0:
            return
    
        # if instance is idle, send a new batch
        if (worker.id, instance_state.model.id) not in self.scheduled_batch_to_instance or \
            self.scheduled_batch_to_instance[(worker.id, instance_state.model.id)] == None:
            
            queued_batch = TaskBatcher.get_batch(
                time, worker.total_memory_gb, self.queues[instance_state.model.data.id], True)

            # skip if cannot form batch
            if not queued_batch: return

            self.scheduled_batch_to_instance[(worker.id, instance_state.model.id)] = [
                (t.job.id, t.task_id) for t in queued_batch.tasks]

            # assign batch to worker
            self.em.add_event(
                Event(time,
                      EVENT_TYPES[EventIds.TASKS_ASSIGNED_TO_WORKER],
                      kwargs={"tasks": queued_batch.tasks,
                              "worker_id": worker.id,
                              "force_instance_id": instance_state.model.id}),
                self.emitter_id)
            
            # tell workers to send required outputs to chosen worker
            inputs_from_scheduler: list[Task] = []
            outputs_from_workers: dict[UUID, list[tuple[int, int]]] = {}
            for task in queued_batch.tasks:
                if len(task.required_task_ids) == 0:
                    inputs_from_scheduler.append(task)

                else:
                    for rt in task.required_task_ids:
                        worker_id = self.output_locs[(task.job.id, rt)]
                        if worker_id not in outputs_from_workers:
                            outputs_from_workers[worker_id] = []
                        outputs_from_workers[worker_id].append((task.job.id, rt))

            if inputs_from_scheduler:
                self.em.add_event(
                    Event(time, 
                            EVENT_TYPES[EventIds.TASKS_INPUTS_SENT_TO_WORKER],
                            kwargs={"tasks": queued_batch.tasks,
                                    "from_worker_id": self.scheduler_worker_id,
                                    "to_worker_id": worker.id,
                                    "force_instance_id": instance_id,
                                    "ignore_transfer_time": ignore_transfer_time}),
                    self.emitter_id)
            
            if outputs_from_workers:
                for (from_worker_id, job_task_ids) in outputs_from_workers.items():
                    self.em.add_event(
                        Event(time, 
                            EVENT_TYPES[EventIds.TASKS_OUTPUTS_ASSIGNED_TO_WORKER],
                            kwargs={"job_task_ids": job_task_ids,
                                    "from_worker_id": from_worker_id,
                                    "to_worker_id": worker.id}),
                        self.emitter_id)