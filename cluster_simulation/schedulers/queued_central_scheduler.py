import core.configs.gen_config as gcfg

from core.job import Job
from core.task import Task
from core.batch import Batch
from core.data_models.workflow import Workflow

from queue import PriorityQueue
from queue_management.queued_task import QueuedTask
from queue_management.batching import TaskBatcher

from workers.worker import Worker

from schedulers.scheduler import Scheduler

from events.event_manager import EventManager
from events.event import *
from events.event_types import *


class QueuedCentralScheduler(Scheduler):
    """
    Centralized scheduler with scheduler-side queue management and drop logic.

    Unlike CentralRoundRobinScheduler (which dispatches tasks to worker queues
    immediately on arrival), this scheduler holds tasks in per-model queues and
    only dispatches when a worker instance is idle. Batches are force-assigned to
    specific instances, bypassing worker-side queue management entirely.

    Drop policies (DROP_POLICY):
      NONE             — never drop; tasks wait indefinitely
      LATEST_POSSIBLE  — before each dispatch, expire any task whose deadline
                         cannot be met even with a solo batch (batch size 1)
    """

    def __init__(self, em: EventManager, workers: dict[UUID, Worker],
                 workflows: dict[int, Workflow], scheduler_worker_id: UUID):
        super().__init__(em)

        self.workers = workers
        self.workflows = workflows
        self.scheduler_worker_id = scheduler_worker_id

        # (job ID, task ID) -> worker ID holding that task's output
        self.output_locs: dict[tuple[int, int], UUID] = {}

        # per-model scheduler-side task queue
        self.queues: dict[int, PriorityQueue] = {}

        # (worker ID, instance ID) -> scheduled job_task list, or None if idle
        self.scheduled_to_instance: dict[tuple[UUID, UUID], list[tuple[int, int]] | None] = {}

        # round-robin pointer: model ID -> (worker ID, instance ID)
        self.last_dispatched_to: dict[int, tuple[UUID, UUID]] = {}


    def on_job_arrival(self, time: float, job: Job):
        self.on_tasks_arrival(time, [t for t in job.tasks if not t.required_task_ids])


    def on_tasks_arrival(self, time: float, tasks: list[Task]):
        model_ids = set()
        for task in tasks:
            mid = task.model_data.id
            if mid not in self.queues:
                self.queues[mid] = PriorityQueue()
            self.queues[mid].put(QueuedTask(task))
            model_ids.add(mid)

        for mid in model_ids:
            self._try_dispatch(time, mid)


    def _try_dispatch(self, time: float, model_id: int):
        """Drain the queue for model_id into idle instances, round-robin across workers."""
        if model_id not in self.queues or self.queues[model_id].qsize() == 0:
            return

        idle_instances = self._idle_instances_rr(time, model_id)

        for worker, instance_id in idle_instances:
            if self.queues[model_id].qsize() == 0:
                break

            self._drop_expired(time, model_id, worker.total_memory_gb)
            if self.queues[model_id].qsize() == 0:
                break

            batch = TaskBatcher.get_batch(time, worker.total_memory_gb,
                                          self.queues[model_id], update_queue=True)
            if not batch:
                break

            self.scheduled_to_instance[(worker.id, instance_id)] = [
                (t.job.id, t.task_id) for t in batch.tasks]
            self.last_dispatched_to[model_id] = (worker.id, instance_id)

            self._dispatch(time, batch, worker, instance_id)


    def _idle_instances_rr(self, time: float, model_id: int) -> list[tuple[Worker, UUID]]:
        """Return idle instances for model_id in round-robin order from last dispatch."""
        all_instances = sorted(
            [(w, s.model.id)
             for w in self.workers.values()
             for s in w.GPU_state.state_at(time)
             if s.model.data.id == model_id],
            key=lambda x: (x[0].create_time, x[0].id, x[1]))

        idle_set = {(w.id, iid) for w, iid in all_instances
                    if self.scheduled_to_instance.get((w.id, iid)) is None}

        if not idle_set:
            return []

        if model_id in self.last_dispatched_to:
            last = self.last_dispatched_to[model_id]
            all_keys = [(w.id, iid) for w, iid in all_instances]
            if last in all_keys:
                pivot = (all_keys.index(last) + 1) % len(all_keys)
                all_instances = [all_instances[(pivot + i) % len(all_instances)]
                                 for i in range(len(all_instances))]

        return [(w, iid) for w, iid in all_instances if (w.id, iid) in idle_set]


    def _drop_expired(self, time: float, model_id: int, partition_size: int):
        """Remove tasks from the queue that can no longer meet their deadline
        even if executed alone (batch size 1). Emits a single JOBS_DROPPED event."""
        if gcfg.DROP_POLICY == "NONE":
            return

        q = self.queues[model_id]
        qt_list = []
        while q.qsize() > 0:
            qt_list.append(q.get())

        dropped_job_ids = []
        for qt in qt_list:
            task = qt.task
            min_exec = task.model_data.batch_exec_times[partition_size][1]
            deadline = task.job.create_time + task.job.slo * (1 + gcfg.SLO_SLACK)
            if task.job.slo > 0 and time + min_exec > deadline:
                dropped_job_ids.append(task.job.id)
            else:
                q.put(qt)

        if dropped_job_ids:
            self.em.add_event(
                Event(time,
                      EVENT_TYPES[EventIds.JOBS_DROPPED],
                      kwargs={"job_ids": dropped_job_ids}),
                self.emitter_id)


    def _dispatch(self, time: float, batch: Batch, worker: Worker, instance_id: UUID):
        self.em.add_event(
            Event(time,
                  EVENT_TYPES[EventIds.TASKS_ASSIGNED_TO_WORKER],
                  kwargs={"worker_id": worker.id,
                          "tasks": batch.tasks,
                          "force_instance_id": instance_id}),
            self.emitter_id)

        inputs_from_scheduler = [t for t in batch.tasks if not t.required_task_ids]
        outputs_from_workers: dict[UUID, list[tuple[int, int]]] = {}
        for task in batch.tasks:
            for rt in task.required_task_ids:
                wid = self.output_locs[(task.job.id, rt)]
                outputs_from_workers.setdefault(wid, []).append((task.job.id, rt))

        if inputs_from_scheduler:
            self.em.add_event(
                Event(time,
                      EVENT_TYPES[EventIds.TASKS_INPUTS_SENT_TO_WORKER],
                      kwargs={"tasks": batch.tasks,
                              "from_worker_id": self.scheduler_worker_id,
                              "to_worker_id": worker.id,
                              "force_instance_id": instance_id,
                              "ignore_transfer_time": not gcfg.ENABLE_NETWORKING_DELAYS}),
                self.emitter_id)

        for from_wid, job_task_ids in outputs_from_workers.items():
            self.em.add_event(
                Event(time,
                      EVENT_TYPES[EventIds.TASKS_OUTPUTS_ASSIGNED_TO_WORKER],
                      kwargs={"job_task_ids": job_task_ids,
                              "from_worker_id": from_wid,
                              "to_worker_id": worker.id}),
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


    def on_batch_finish(self, time: float, batch: Batch, worker_id: UUID, instance_id: UUID):
        self.scheduled_to_instance[(worker_id, instance_id)] = None

        for task in batch.tasks:
            assert (task.job.id, task.task_id) not in self.output_locs
            self.output_locs[(task.job.id, task.task_id)] = worker_id

        self._try_dispatch(time, batch.model_data.id)
