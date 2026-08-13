from core.job import Job
from core.task import Task

from events.event_manager import EventManager
from events.event import *
from events.event_types import *


class Scheduler(EventListener):

    def __init__(self, em: EventManager):
        super().__init__(Agent.SCHEDULER)

        self.em = em

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
            self.on_job_arrival(event.time, event.kwargs["job"])
        elif event.type.id == EventIds.TASKS_ARRIVAL_AT_SCHEDULER:
            self.on_tasks_arrival(event.time, event.kwargs["tasks"])
        elif event.type.id == EventIds.JOBS_DROPPED:
            self.on_jobs_dropped(event.time, event.kwargs["job_ids"])
        elif event.type.id == EventIds.BATCH_STARTED_AT_WORKER:
            self.on_batch_start(event.time, event.kwargs["batch"], event.kwargs["worker_id"],
                                event.kwargs["model_instance_id"])
        elif event.type.id == EventIds.BATCH_FINISHED_AT_WORKER:
            self.on_batch_finish(event.time, event.kwargs["batch"], event.kwargs["worker_id"], 
                                 event.kwargs["model_instance_id"])
        else:
            raise ValueError(f"Scheduler received unregistered event: {event}")

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