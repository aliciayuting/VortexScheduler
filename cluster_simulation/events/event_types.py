from core.job import Job
from core.task import Task
from core.batch import Batch
from events.event import EventType, Agent
from uuid import UUID


class EventIds:
    """
    Events notifying scheduler of new request from client.
    """
    JOB_SENT_TO_SCHEDULER               = 0
    JOB_ARRIVAL_AT_SCHEDULER            = 1
    
    """
    Events notifying workers of assigned tasks and sending necessary
    task inputs/outputs.
    """
    TASKS_ASSIGNED_TO_WORKER            = 2
    TASKS_INPUTS_SENT_TO_WORKER         = 3
    TASKS_INPUTS_ARRIVAL_AT_WORKER      = 4
    TASKS_OUTPUTS_ASSIGNED_TO_WORKER    = 5
    TASKS_OUTPUTS_SENT_TO_WORKER        = 6
    TASKS_OUTPUTS_ARRIVAL_AT_WORKER     = 7

    """
    Events notifying scheduler of available tasks, do NOT carry input.
    """
    TASKS_ARRIVAL_AT_SCHEDULER          = 9

    """
    Execution markers.
    """
    JOBS_DROPPED                        = 10
    BATCH_FINISHED_AT_WORKER            = 11
    CHECK_QUEUE_AT_WORKER               = 12
    BATCH_STARTED_AT_WORKER             = 13

    """
    Events notifying client of job status.
    """
    RESPONSE_SENT_TO_CLIENT             = 14
    RESPONSE_RECEIVED_AT_CLIENT         = 15


EVENT_TYPES: dict[int, EventType] = {
    EventIds.JOB_SENT_TO_SCHEDULER: EventType(
        EventIds.JOB_SENT_TO_SCHEDULER, "Job Sent to Scheduler",
        kwargs={
            "job": True,                    # : Job
            "from_client_id": True,         # : UUID
            "ignore_transfer_time": True    # : bool
        },
        emitter_types=[Agent.CLIENT],
        listener_types=[Agent.NETWORK, Agent.LOGGER, Agent.VERIFIER]),
    
    EventIds.JOB_ARRIVAL_AT_SCHEDULER: EventType(
        EventIds.JOB_ARRIVAL_AT_SCHEDULER, "Job Arrival at Scheduler", 
        kwargs={"job": True},
        emitter_types=[Agent.NETWORK],
        listener_types=[Agent.SCHEDULER, Agent.LOGGER, Agent.VERIFIER]),


    EventIds.TASKS_ASSIGNED_TO_WORKER: EventType(
        EventIds.TASKS_ASSIGNED_TO_WORKER, "Tasks Assigned to Worker", 
        kwargs={"tasks": True,              # : list[Task]
                "force_instance_id": False, # : UUID (instance ID)
                "worker_id": True},         # : UUID
        emitter_types=[Agent.SCHEDULER],
        listener_types=[Agent.WORKER, Agent.LOGGER, Agent.VERIFIER]),

    EventIds.TASKS_INPUTS_SENT_TO_WORKER: EventType(
        EventIds.TASKS_INPUTS_SENT_TO_WORKER, "Tasks Inputs Sent to Worker", 
        kwargs={"tasks": True,                  # : list[Task]
                "from_worker_id": True,         # : UUID
                "to_worker_id": True,           # : UUID
                "ignore_transfer_time": True,   # : bool
                "force_instance_id": False},    # : UUID (instance ID)
        emitter_types=[Agent.SCHEDULER, Agent.WORKER],
        listener_types=[Agent.NETWORK, Agent.LOGGER, Agent.VERIFIER]),
    
    EventIds.TASKS_INPUTS_ARRIVAL_AT_WORKER: EventType(
        EventIds.TASKS_INPUTS_ARRIVAL_AT_WORKER, "Tasks Inputs Arrival at Worker", 
        kwargs={"tasks": True,                  # : list[Task]
                "from_worker_id": True,         # : UUID
                "to_worker_id": True,           # : UUID
                "force_instance_id": False},    # : UUID (instance ID)
        emitter_types=[Agent.NETWORK],
        listener_types=[Agent.WORKER, Agent.LOGGER, Agent.VERIFIER]),

    EventIds.TASKS_OUTPUTS_ASSIGNED_TO_WORKER: EventType(
        EventIds.TASKS_OUTPUTS_ASSIGNED_TO_WORKER, "Tasks Outputs Assigned to Worker", 
        kwargs={"job_task_ids": True,       # : list[tuple[int, int]]
                "from_worker_id": True,     # : UUID
                "to_worker_id": True},      # : UUID
        emitter_types=[Agent.SCHEDULER],
        listener_types=[Agent.WORKER, Agent.LOGGER, Agent.VERIFIER]),

    EventIds.TASKS_OUTPUTS_SENT_TO_WORKER: EventType(
        EventIds.TASKS_OUTPUTS_SENT_TO_WORKER, "Tasks Outputs Sent to Worker", 
        kwargs={"tasks": True,                  # : list[Task]
                "from_worker_id": True,         # : UUID
                "to_worker_id": True,           # : UUID
                "ignore_transfer_time": True},  # : bool     
        emitter_types=[Agent.WORKER],
        listener_types=[Agent.NETWORK, Agent.LOGGER, Agent.VERIFIER]),
    
    EventIds.TASKS_OUTPUTS_ARRIVAL_AT_WORKER: EventType(
        EventIds.TASKS_OUTPUTS_ARRIVAL_AT_WORKER, "Tasks Outputs Arrival at Worker", 
        kwargs={"tasks": True,              # : list[Task]
                "from_worker_id": True,     # : UUID
                "to_worker_id": True},      # : UUID
        emitter_types=[Agent.NETWORK],
        listener_types=[Agent.WORKER, Agent.LOGGER, Agent.VERIFIER]),

    
    EventIds.TASKS_ARRIVAL_AT_SCHEDULER: EventType(
        EventIds.TASKS_ARRIVAL_AT_SCHEDULER, "Tasks Arrival at Scheduler", 
        kwargs={"tasks": True},
        emitter_types=[Agent.SCHEDULER, Agent.WORKER],
        listener_types=[Agent.SCHEDULER, Agent.LOGGER, Agent.VERIFIER]),
    

    EventIds.JOBS_DROPPED: EventType(
        EventIds.JOBS_DROPPED, "Jobs Dropped",
        kwargs={"job_ids": True},
        emitter_types=[Agent.SCHEDULER, Agent.WORKER],
        listener_types=[Agent.SCHEDULER, Agent.WORKER, Agent.CLIENT, Agent.LOGGER, Agent.VERIFIER]),

    EventIds.CHECK_QUEUE_AT_WORKER: EventType(
        EventIds.CHECK_QUEUE_AT_WORKER, "Check Queue for Execution at Worker",
        kwargs={"model_id": True, "worker_id": True},
        emitter_types=[Agent.WORKER],
        listener_types=[Agent.WORKER, Agent.LOGGER, Agent.VERIFIER]),
    
    EventIds.BATCH_STARTED_AT_WORKER: EventType(
        EventIds.BATCH_STARTED_AT_WORKER, "Batch Started at Worker",
        kwargs={"batch": True, 
                "model_instance_id": True, 
                "worker_id": True},
        emitter_types=[Agent.WORKER],
        listener_types=[Agent.SCHEDULER, Agent.WORKER, Agent.LOGGER, Agent.VERIFIER]),
    
    EventIds.BATCH_FINISHED_AT_WORKER: EventType(
        EventIds.BATCH_FINISHED_AT_WORKER, "Batch Finished at Worker",
        kwargs={"batch": True, 
                "model_instance_id": True, 
                "worker_id": True},
        emitter_types=[Agent.WORKER],
        listener_types=[Agent.SCHEDULER, Agent.WORKER, Agent.LOGGER, Agent.VERIFIER]),


    EventIds.RESPONSE_SENT_TO_CLIENT: EventType(
        EventIds.RESPONSE_SENT_TO_CLIENT, "Response Sent to Client",
        kwargs={"job": True, 
                "client_id": True, 
                "worker_id": True,
                "ignore_transfer_time": True},
        emitter_types=[Agent.WORKER],
        listener_types=[Agent.NETWORK, Agent.LOGGER, Agent.VERIFIER]),

    EventIds.RESPONSE_RECEIVED_AT_CLIENT: EventType(
        EventIds.RESPONSE_RECEIVED_AT_CLIENT, "Response Received at Client",
        kwargs={"job": True, 
                "client_id": True, 
                "worker_id": True},
        emitter_types=[Agent.NETWORK],
        listener_types=[Agent.CLIENT, Agent.LOGGER, Agent.VERIFIER]),
}