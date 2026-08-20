import core.configs.gen_config as gcfg

from queue import PriorityQueue

from core.job import Job
from core.task import Task

from schedulers.algo.drop_algo import (should_drop_task, drop_from_queue,
                                       scheduler_manages_queues, AdmissionController)

from events.event_manager import EventManager
from events.event import *
from events.event_types import *


class Scheduler(EventListener):

    def __init__(self, em: EventManager):
        super().__init__(Agent.SCHEDULER)

        self.em = em

        # drop where the queues are: a scheduler that queues and batches tasks
        # itself drops here, otherwise the workers own the queues and drop there
        self.drops_at_scheduler = scheduler_manages_queues()

        # jobs known to be dropped, by this scheduler or by a worker
        self.dropped_job_ids: set[int] = set()

        # flat rate load shedding at the front door, independent of DROP_POLICY.
        # Jobs arrive at the scheduler under every dispatch policy, so this runs
        # here regardless of where queues are managed.
        self.admission_controller = AdmissionController()

        self.em.register_listener(self, {
            EVENT_TYPES[EventIds.JOB_ARRIVAL_AT_SCHEDULER],
            EVENT_TYPES[EventIds.TASKS_ARRIVAL_AT_SCHEDULER],
            EVENT_TYPES[EventIds.JOBS_DROPPED],
            EVENT_TYPES[EventIds.BATCH_STARTED_AT_WORKER],
            EVENT_TYPES[EventIds.BATCH_FINISHED_AT_WORKER]
        })

        self.emitter_id = self.em.register_emitter(Agent.SCHEDULER, {
            EVENT_TYPES[EventIds.TASKS_ASSIGNED_TO_WORKER],
            EVENT_TYPES[EventIds.TASKS_INPUTS_SENT_TO_WORKER],
            EVENT_TYPES[EventIds.TASKS_OUTPUTS_ASSIGNED_TO_WORKER],
            EVENT_TYPES[EventIds.JOBS_DROPPED]
        })

    def on_event(self, event: Event):
        if event.type.id == EventIds.JOB_ARRIVAL_AT_SCHEDULER:
            job: Job = event.kwargs["job"]
            if self._reject_on_arrival(event.time, job):
                return

            source_tasks = [t for t in job.tasks if len(t.required_task_ids) == 0]
            if self._drop_jobs(event.time, source_tasks):
                return

            self.on_job_arrival(event.time, job)

        elif event.type.id == EventIds.TASKS_ARRIVAL_AT_SCHEDULER:
            # a task of an already dropped job can still arrive here, if a
            # predecessor was executing when the job was dropped
            tasks = [t for t in event.kwargs["tasks"]
                     if t.job.id not in self.dropped_job_ids]

            dropped = self._drop_jobs(event.time, tasks)
            tasks = [t for t in tasks if t.job.id not in dropped]

            if not tasks:
                return

            self.on_tasks_arrival(event.time, tasks)

        elif event.type.id == EventIds.JOBS_DROPPED:
            self.dropped_job_ids.update(event.kwargs["job_ids"])
            self.on_jobs_dropped(event.time, event.kwargs["job_ids"])
        elif event.type.id == EventIds.BATCH_STARTED_AT_WORKER:
            self.on_batch_start(event.time, event.kwargs["batch"], event.kwargs["worker_id"],
                                event.kwargs["model_instance_id"])
        elif event.type.id == EventIds.BATCH_FINISHED_AT_WORKER:
            self.on_batch_finish(event.time, event.kwargs["batch"], event.kwargs["worker_id"], 
                                 event.kwargs["model_instance_id"])
        else:
            raise ValueError(f"Scheduler received unregistered event: {event}")

    def _reject_on_arrival(self, time: float, job: Job) -> bool:
        """Applies flat rate admission control to a job that just arrived, emitting
        JOBS_DROPPED for it if the configured rate says to shed it.

        Args:
            time: Time the job arrived at the scheduler
            job: Job to admit or reject

        Returns:
            rejected: True if the job was rejected and should not be dispatched
        """
        if not self.admission_controller.should_reject(job):
            return False

        self._emit_drops(time, [job.id])
        return True

    def _drop_jobs(self, time: float, tasks: list[Task]) -> set[int]:
        """Applies the configured drop policy to the jobs owning [tasks] and emits
        a single JOBS_DROPPED event for those that fail it. No-op when the workers
        own the queues, in which case they make the drop decision.

        The decision is keyed on tasks rather than jobs because under per-stage
        (NEXUS) SLOs the deadline being tested belongs to the stage at hand.

        Args:
            time: Time at which the drop decision is made
            tasks: Tasks whose jobs to consider dropping

        Returns:
            dropped: IDs of the jobs dropped by this call
        """
        if not self.drops_at_scheduler or gcfg.DROP_POLICY == "NONE":
            return set()

        dropped = []
        for task in tasks:
            if task.job.id in self.dropped_job_ids or task.job.id in dropped:
                continue

            if should_drop_task(time, task):
                dropped.append(task.job.id)

        self._emit_drops(time, dropped)
        return set(dropped)

    def _drop_queued_jobs(self, time: float, task_queue: PriorityQueue) -> set[int]:
        """Applies the configured drop policy to a scheduler side queue, removing
        the tasks of any job that is dropped. No-op when the workers own the queues.

        Args:
            time: Time at which the drop decision is made
            task_queue: Queue of QueuedTask to filter in place

        Returns:
            dropped: IDs of the jobs dropped by this call
        """
        if not self.drops_at_scheduler or gcfg.DROP_POLICY == "NONE":
            return set()

        dropped = drop_from_queue(time, task_queue, self.dropped_job_ids)
        self._emit_drops(time, dropped)
        return set(dropped)

    def _emit_drops(self, time: float, job_ids: list[int]):
        if not job_ids:
            return

        self.dropped_job_ids.update(job_ids)
        self.em.add_event(
            Event(time,
                  EVENT_TYPES[EventIds.JOBS_DROPPED],
                  kwargs={"job_ids": job_ids}),
            self.emitter_id)

    def on_job_arrival(self, time: float, job: Job):
        raise NotImplementedError()
    
    def on_tasks_arrival(self, time: float, tasks: list[Task]):
        raise NotImplementedError()

    def on_jobs_dropped(self, time: float, job_ids: int):
        raise NotImplementedError()
    
    def on_batch_start(self, time: float, batch: Batch, worker_id: UUID, instance_id: UUID):
        raise NotImplementedError()
    
    def on_batch_finish(self, time: float, batch: Batch, worker_id: UUID, instance_id: UUID):
        raise NotImplementedError()
