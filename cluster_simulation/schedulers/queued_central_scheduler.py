import core.configs.gen_config as gcfg
import numpy as np

from core.job import Job
from core.task import Task
from core.batch import Batch
from core.data_models.workflow import Workflow

from queue import PriorityQueue
from queue_management.queued_task import QueuedTask

from workers.worker import Worker

from schedulers.scheduler import Scheduler

from events.event_manager import EventManager
from events.event import *
from events.event_types import *


class QueuedCentralScheduler(Scheduler):
    """
    Centralized scheduler with scheduler-side queue management and drop logic.

    Tasks are held in per-model PriorityQueues at the scheduler level, ordered
    by BOOST_POLICY. Before forwarding to a worker, expired tasks are removed
    according to DROP_POLICY (SLO = 0 is treated as no deadline). Tasks are
    then dispatched to worker queues round-robin; workers form batches
    themselves via CHECK_QUEUE_AT_WORKER, as in CentralRoundRobinScheduler.

    Drop policy (DROP_POLICY):
      NONE             — never drop
      LATEST_POSSIBLE  — drop any task whose deadline cannot be met even with
                         a solo batch (uses the minimum exec time across all
                         partition sizes as the best-case estimate)
    """

    def __init__(self, em: EventManager, workers: dict[UUID, Worker],
                 workflows: dict[int, Workflow], scheduler_worker_id: UUID):
        super().__init__(em)

        self.workers = workers
        self.workflows = workflows
        self.scheduler_worker_id = scheduler_worker_id

        # (job ID, task ID) -> worker ID holding that task's output
        self.output_locs: dict[tuple[int, int], UUID] = {}

        # per-model scheduler-side task queue ordered by BOOST_POLICY
        self.queues: dict[int, PriorityQueue] = {}

        # round-robin pointer: model ID -> (worker ID, instance ID)
        self.last_sent_tasks_to: dict[int, tuple[UUID, UUID]] = {}


    def on_job_arrival(self, time: float, job: Job):
        self.on_tasks_arrival(time, [t for t in job.tasks if not t.required_task_ids])


    def on_tasks_arrival(self, time: float, tasks: list[Task]):
        model_ids = set()
        for task in tasks:
            mid = task.model_data.id
            if mid not in self.queues:
                self.queues[mid] = PriorityQueue()
            self.queues[mid].put(QueuedTask(task, time))
            model_ids.add(mid)

        for mid in model_ids:
            self._dispatch_from_queue(time, mid)


    def _dispatch_from_queue(self, time: float, model_id: int):
        """Drop expired tasks then forward all remaining to worker queues round-robin."""
        if model_id not in self.queues or self.queues[model_id].qsize() == 0:
            return

        self._drop_expired(time, model_id)

        tasks_to_send: dict[UUID, list[Task]] = {}

        while self.queues[model_id].qsize() > 0:
            task = self.queues[model_id].get().task

            relevant_instances = [
                (w.id, s.model.id)
                for w in sorted(self.workers.values(), key=lambda w: (w.create_time, w.id))
                for s in sorted(w.GPU_state.state_at(time),
                                key=lambda s: (s.model.created_at, s.model.id))
                if s.model.data.id == model_id
            ]

            if model_id not in self.last_sent_tasks_to:
                next_idx = np.random.randint(0, len(relevant_instances))
            else:
                last_idx = relevant_instances.index(self.last_sent_tasks_to[model_id])
                next_idx = (last_idx + 1) % len(relevant_instances)

            worker_id, instance_id = relevant_instances[next_idx]
            self.last_sent_tasks_to[model_id] = (worker_id, instance_id)
            tasks_to_send.setdefault(worker_id, []).append(task)

        for worker_id, worker_tasks in tasks_to_send.items():
            self.em.add_event(
                Event(time,
                      EVENT_TYPES[EventIds.TASKS_ASSIGNED_TO_WORKER],
                      kwargs={"worker_id": worker_id, "tasks": worker_tasks}),
                self.emitter_id)

            inputs_from_scheduler = [t for t in worker_tasks if not t.required_task_ids]
            outputs_from_workers: dict[UUID, list[tuple[int, int]]] = {}
            for task in worker_tasks:
                for rt in task.required_task_ids:
                    wid = self.output_locs[(task.job.id, rt)]
                    outputs_from_workers.setdefault(wid, []).append((task.job.id, rt))

            if inputs_from_scheduler:
                self.em.add_event(
                    Event(time,
                          EVENT_TYPES[EventIds.TASKS_INPUTS_SENT_TO_WORKER],
                          kwargs={"tasks": worker_tasks,
                                  "from_worker_id": self.scheduler_worker_id,
                                  "to_worker_id": worker_id,
                                  "ignore_transfer_time": not gcfg.ENABLE_NETWORKING_DELAYS}),
                    self.emitter_id)

            for from_wid, job_task_ids in outputs_from_workers.items():
                self.em.add_event(
                    Event(time,
                          EVENT_TYPES[EventIds.TASKS_OUTPUTS_ASSIGNED_TO_WORKER],
                          kwargs={"job_task_ids": job_task_ids,
                                  "from_worker_id": from_wid,
                                  "to_worker_id": worker_id}),
                    self.emitter_id)


    def _drop_expired(self, time: float, model_id: int):
        """Remove tasks from the queue that cannot meet their deadline even in the
        best case (solo batch on the fastest partition). Emits JOBS_DROPPED."""
        if gcfg.DROP_POLICY == "NONE":
            return

        q = self.queues[model_id]
        qt_list = []
        while q.qsize() > 0:
            qt_list.append(q.get())

        dropped_job_ids = []
        for qt in qt_list:
            task = qt.task
            if task.job.slo == 0:
                q.put(qt)
                continue
            min_exec = min(task.model_data.batch_exec_times[s][1]
                           for s in task.model_data.batch_exec_times)
            deadline = task.job.create_time + task.job.slo * (1 + gcfg.SLO_SLACK)
            if time + min_exec > deadline:
                dropped_job_ids.append(task.job.id)
            else:
                q.put(qt)

        if dropped_job_ids:
            self.em.add_event(
                Event(time,
                      EVENT_TYPES[EventIds.JOBS_DROPPED],
                      kwargs={"job_ids": dropped_job_ids}),
                self.emitter_id)


    def on_jobs_dropped(self, time: float, job_ids: list[int]):
        job_id_set = set(job_ids)
        for q in self.queues.values():
            qt_list = []
            while q.qsize() > 0:
                qt_list.append(q.get())
            for qt in qt_list:
                if qt.task.job.id not in job_id_set:
                    q.put(qt)


    def on_batch_start(self, time: float, batch: Batch, worker_id: UUID, instance_id: UUID):
        pass


    def on_batch_finish(self, time: float, batch: Batch, worker_id: UUID, _instance_id: UUID):
        for task in batch.tasks:
            assert (task.job.id, task.task_id) not in self.output_locs
            self.output_locs[(task.job.id, task.task_id)] = worker_id

        for mid in list(self.queues):
            if self.queues[mid].qsize() > 0:
                self._dispatch_from_queue(time, mid)
