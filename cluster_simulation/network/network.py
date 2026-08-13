from events.event_manager import EventManager
from events.event import *
from events.event_types import *

from uuid import UUID


class Network(EventListener):

    def __init__(self, em: EventManager, scheduler_worker_id: UUID | None):
        super().__init__(Agent.NETWORK)

        self.scheduler_worker_id = scheduler_worker_id
        self.em = em

        self.em.register_listener(self, {
            EVENT_TYPES[EventIds.JOB_SENT_TO_SCHEDULER],
            EVENT_TYPES[EventIds.TASKS_INPUTS_SENT_TO_WORKER],
            EVENT_TYPES[EventIds.TASKS_OUTPUTS_SENT_TO_WORKER],
            EVENT_TYPES[EventIds.RESPONSE_SENT_TO_CLIENT]
        })

        self.emitter_id = self.em.register_emitter(Agent.NETWORK, {
            EVENT_TYPES[EventIds.JOB_ARRIVAL_AT_SCHEDULER],
            EVENT_TYPES[EventIds.TASKS_INPUTS_ARRIVAL_AT_WORKER],
            EVENT_TYPES[EventIds.TASKS_OUTPUTS_ARRIVAL_AT_WORKER],
            EVENT_TYPES[EventIds.RESPONSE_RECEIVED_AT_CLIENT]
        })

    def on_event(self, event: Event):
        if event.type.id == EventIds.JOB_SENT_TO_SCHEDULER:
            transfer_time = 0
            if not event.kwargs["ignore_transfer_time"]:
                msg_size = sum(t.input_size for t in event.kwargs["job"].tasks
                               if len(t.required_task_ids) == 0)
                transfer_time = msg_size / 12500
            
            self.em.add_event(
                Event(event.time + transfer_time,
                      EVENT_TYPES[EventIds.JOB_ARRIVAL_AT_SCHEDULER],
                      kwargs={"job": event.kwargs["job"]}), 
                self.emitter_id)

        elif event.type.id == EventIds.TASKS_INPUTS_SENT_TO_WORKER:
            transfer_time = 0
            if not event.kwargs["ignore_transfer_time"] and \
                event.kwargs["from_worker_id"] != event.kwargs["to_worker_id"]:
                
                msg_size = sum(t.input_size for t in event.kwargs["tasks"])
                transfer_time = msg_size / 12500
            
            kwargs = {"tasks": event.kwargs["tasks"],
                      "from_worker_id": event.kwargs["from_worker_id"],
                      "to_worker_id": event.kwargs["to_worker_id"]}
            if "force_instance_id" in event.kwargs:
                kwargs["force_instance_id"] = event.kwargs["force_instance_id"]

            self.em.add_event(
                Event(event.time + transfer_time,
                      EVENT_TYPES[EventIds.TASKS_INPUTS_ARRIVAL_AT_WORKER],
                      kwargs=kwargs), 
                self.emitter_id)
            
        elif event.type.id == EventIds.TASKS_OUTPUTS_SENT_TO_WORKER:
            transfer_time = 0
            if not event.kwargs["ignore_transfer_time"] and \
                event.kwargs["from_worker_id"] != event.kwargs["to_worker_id"]:
                
                msg_size = sum(t.result_size for t in event.kwargs["tasks"])
                transfer_time = msg_size / 12500
            
            self.em.add_event(
                Event(event.time + transfer_time,
                      EVENT_TYPES[EventIds.TASKS_OUTPUTS_ARRIVAL_AT_WORKER],
                      kwargs={"tasks": event.kwargs["tasks"],
                              "from_worker_id": event.kwargs["from_worker_id"],
                              "to_worker_id": event.kwargs["to_worker_id"]}), 
                self.emitter_id)
        
        elif event.type.id == EventIds.RESPONSE_SENT_TO_CLIENT:
            transfer_time = 0
            if not event.kwargs["ignore_transfer_time"]:
                msg_size = sum(t.result_size for t in event.kwargs["job"].tasks
                               if len(t.next_task_ids) == 0)
                transfer_time = msg_size / 12500
            
            self.em.add_event(
                Event(event.time + transfer_time,
                      EVENT_TYPES[EventIds.RESPONSE_RECEIVED_AT_CLIENT],
                      kwargs={"job": event.kwargs["job"],
                              "client_id": event.kwargs["client_id"],
                              "worker_id": event.kwargs["worker_id"]}), 
                self.emitter_id)

        else:
            raise ValueError(f"Network received unregistered event: {event}")