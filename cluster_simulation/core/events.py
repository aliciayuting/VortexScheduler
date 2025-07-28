from core.job import *
from core.network import *
from core.config import *

from workers.worker import Worker


class Event(object):
    """ Abstract class representing events. """

    def __init__(self):
        raise NotImplementedError("Event is an abstract class and cannot be "
                                  "instantiated directly")

    def run(self, current_time):
        """ Returns any events that should be added to the queue. """
        raise NotImplementedError("The run() method must be implemented by "
                                  "each class subclassing Event")
    
    def should_abandon_event(self, current_time, kwargs: dict):
        return False

    def to_string(self, current_time):
        """ Returns the string describing the event """
        raise NotImplementedError("The to_string() method must be implemented by "
                                  "each class subclassing Event")
    
    def is_worker_event():
        raise NotImplementedError("is_worker_event : () -> bool not implemented")


class JobArrivalAtScheduler(Event):
    """
    Event signifying that a Job arrived to a Centralized scheduler.
    Only for Centralized Schedulers
    """

    def __init__(self, simulation, job):
        self.simulation = simulation
        self.job = job

    def run(self, current_time):
        # Schedule job
        if self.simulation.job_split == "PER_TASK":
            for task in self.job.tasks:
                task.log.task_arrival_at_scheduler_timestamp = current_time
            new_events = self.simulation.schedule_job_and_send_tasks(
                self.job, current_time)
        # elif self.simulation.job_split == "PER_JOB":
        #     new_events = self.simulation.schedule_job_and_send_job(
        #         self.job, current_time)
        return new_events
    
    def to_string(self):
        return "[Job Arrival at Scheduler (Job {})] ++".format(self.job.id)
    
    def is_worker_event():
        return False
    

class TasksArrivalAtScheduler(Event):
    """
    Event signifying that Task(s) arrived at a Centralized scheduler.
    Only for Centralized Schedulers
    """

    def __init__(self, simulation, tasks):
        assert(len(tasks) > 0)
        self.simulation = simulation
        self.tasks = tasks

    def run(self, current_time):
        # leave out dropped tasks
        self.tasks = [task for task in self.tasks 
                      if not (self.simulation.task_drop_log["job_id"] == task.job_id).any()]
        if not self.tasks:
            return []

        for task in self.tasks:
            # only set if not set already (avoid changing order for preempted tasks)
            if task.log.task_arrival_at_scheduler_timestamp == 0:
                task.log.task_arrival_at_scheduler_timestamp = current_time

        return self.simulation.schedule_tasks_on_arrival(self.tasks, current_time)
   
    def to_string(self):
        return f"[Tasks Arrival at Scheduler (Type: {self.tasks[0].task_type}, Job IDs: {list(map(lambda t: t.job_id, self.tasks))})] ++"
    
    def is_worker_event():
        return False


class BatchRejectionAtWorker(Event):
    """
    Event signifying that a worker was busy when a batch was sent for execution, and
    the batch has been sent back to the Centralized scheduler for rescheduling.
    """

    def __init__(self, simulation, worker, batch, current_worker_batch=None):
        self.simulation = simulation
        self.worker = worker
        self.batch = batch
        self.current_worker_batch = current_worker_batch

    def run(self, current_time):
        if self.simulation.simulation_name == "shepherd":
            self.simulation.scheduler.worker_rejected_batch(self.worker.worker_id, self.batch, self.current_worker_batch)
        if self.batch.size() == 0:
            return [] # possible if all tasks were dropped
        return [EventOrders(current_time, TasksArrivalAtScheduler(self.simulation, self.batch.tasks))] # reschedule batch

    def to_string(self):
        return f"[Batch {self.batch.id} Sent Back by Worker {self.worker.worker_id}]"
    
    def is_worker_event():
        return True


class BatchArrivalAtWorker(Event):
    """
    Event signifying that a batch of tasks arrived for execution at a worker.
    """

    def __init__(self, simulation, worker, batch):
        self.simulation = simulation
        self.worker = worker
        self.batch = batch

    def run(self, current_time):
        self.batch.tasks = [task for task in self.batch.tasks 
                            if not (self.simulation.task_drop_log["job_id"] == task.job_id).any()]
        if (not ENABLE_DYNAMIC_MODEL_LOADING and not self.worker.GPU_state.does_have_idle_copy(self.batch.model, current_time)) or \
            (not ENABLE_MULTITHREADING and any(s.reserved_batch for s in self.worker.GPU_state.state_at(current_time))) or \
                self.worker.did_abandon_batch(self.batch.id) or self.batch.size() == 0:
            current_batches = [s.reserved_batch for s in self.worker.GPU_state.state_at(current_time) if s.reserved_batch]
            transfer_delay = CPU_to_CPU_delay(self.batch.size()*self.batch.tasks[0].input_size) if self.batch.size() > 0 else 0
            return [EventOrders(current_time + transfer_delay, 
                                BatchRejectionAtWorker(self.simulation, self.worker, self.batch,
                                                       current_worker_batch=(current_batches[0] if current_batches else None)))]
        for task in self.batch.tasks:
            task.log.set_task_placed_on_worker_queue_timestamp(current_time)
        return self.worker.maybe_start_batch(self.batch, current_time)

    def to_string(self):
        return f"[Batch {self.batch.id} Arrival at Worker {self.worker.worker_id} (Type: {self.batch.tasks[0].task_type}, Job IDs: {self.batch.job_ids})] ++"
    
    def is_worker_event():
        return True


class BatchPreemptionAtWorker(Event):
    """
    Event signifying that a batch should be preempted at a worker.
    """

    def __init__(self, simulation, worker, batch, old_batch_id):
        self.simulation = simulation
        self.worker = worker
        self.batch = batch # replacement batch
        self.old_batch_id = old_batch_id # preempted batch

    def run(self, current_time):
        self.batch.tasks = [task for task in self.batch.tasks 
                            if not (self.simulation.task_drop_log["job_id"] == task.job_id).any()]

        # check if batch to be preempted still exists/is actively executing
        if len(self.batch.tasks) > 0 and any(s.reserved_batch and s.reserved_batch.id == self.old_batch_id 
               for s in self.worker.GPU_state.state_at(current_time)):
            for task in self.batch.tasks:
                task.log.set_task_placed_on_worker_queue_timestamp(current_time)
            return self.worker.preempt_batch(self.old_batch_id, self.batch, current_time)
        else:
            # NOTE: marks as abandoned anyway in case prior assigned batch has not yet arrived
            Worker._abandoned_batches.append(self.old_batch_id)
            # if outdated decision, or all batch tasks were dropped, send back tasks for rescheduling
            current_batches = [s.reserved_batch for s in self.worker.GPU_state.state_at(current_time) if s.reserved_batch]
            return [EventOrders(
                current_time + (0 if self.batch.size() == 0 else CPU_to_CPU_delay(self.batch.size()*self.batch.tasks[0].input_size)),
                BatchRejectionAtWorker(self.simulation, self.worker, self.batch, 
                                       current_worker_batch=(current_batches[0] if current_batches else None)))]
    
    def to_string(self):
        return f"[Batch Preemption at Worker {self.worker.worker_id} (Batch {self.old_batch_id} preempted)]"
    
    def is_worker_event():
        return True


class JobArrivalAtWorker(Event):
    """
    Event signifying that a Job arrived to a Cascade node (a Worker).
    Only for Decentralized Schedulers
    """

    def __init__(self, simulation, job, worker_id):
        self.simulation = simulation
        self.worker = self.simulation.workers[worker_id]
        self.worker_id = worker_id
        self.job = job

    def run(self, current_time):
        # Schedule job
        new_events = []
        if self.simulation.job_split == "PER_TASK":
            new_events = self.simulation.workers[self.worker_id].schedule_job_heft(
                current_time, self.job)
        # elif self.simulation.job_split == "PER_JOB":
        #     new_events = self.simulation.schedule_job_and_send_job(
        #         self.job, current_time)
        return new_events

    def to_string(self):
        return "[Job Arrival at Worker (Job {})] ++".format(self.job.id)
    
    def is_worker_event():
        return True


# for PER_TASK scheduler
class TaskArrival(Event):
    """ Event to signify a TASK arriving at a WORKER. """

    def __init__(self, worker, task, job_id):
        self.worker = worker
        self.task = task
        self.job_id = job_id

    def run(self, current_time):
        # log tracking for this task
        self.task.log.set_task_placed_on_worker_queue_timestamp(current_time)
        return self.worker.add_task(current_time, self.task)

    def to_string(self):
        return "[Task Arrival (Job {} - Task {}) at {}] ---".format(self.job_id, self.task.task_id, self.worker)
    
    def is_worker_event():
        return True
    

# for PER_TASK scheduler
class TasksArrival(Event):
    """ Event to signify TASKs arriving at a WORKER. """

    def __init__(self, worker, tasks):
        self.worker = worker
        self.tasks = tasks

    def run(self, current_time):
        # log tracking for this task
        drop_log = self.worker.simulation.task_drop_log
        relevant_tasks = [t for t in self.tasks if not (drop_log[current_time >= drop_log["drop_time"]]["job_id"] == t.job_id).any()]
        for task in relevant_tasks:
            task.log.set_task_placed_on_worker_queue_timestamp(current_time)
        return self.worker.add_tasks(current_time, self.tasks)

    def to_string(self):
        return f"[Tasks Arrival (Job {[t.job_id for t in self.tasks]} - Task Types {set([t.task_type for t in self.tasks])}) at {self.worker.worker_id}] ---"
    
    def is_worker_event():
        return True


class InterResultArrival(Event):
    """ Event to signify a TASK arriving at a WORKER. (?) """

    def __init__(self, worker, prev_task, cur_task):
        self.worker = worker
        self.prev_task = prev_task
        self.cur_task = cur_task

    def run(self, current_time):
        self.cur_task.log.set_task_arrival_at_worker_buffer_timestamp(
            current_time)
        return self.worker.receive_intermediate_result(current_time, self.prev_task, self.cur_task)

    def to_string(self):
        return "[Intermediate Results Arrival]: worker:" + str(self.worker.worker_id) + ", prev_task_id:" + str(self.prev_task.task_id) + ", cur_task_id:" + str(self.cur_task.task_id)
    
    def is_worker_event():
        return True


class BatchStartEvent(Event):
    """ Event to signify that a BATCH has been started by the WORKER. """

    def __init__(self, worker, batch_id=-1, job_ids=[], task_type=(-1, -1)):
        self.worker = worker
        self.batch_id = batch_id
        self.job_ids = job_ids    # integers representing the job_ids
        self.task_type = task_type # (workflow_id, task_id)

    def run(self, current_time):
        return []
    
    def should_abandon_event(self, current_time, kwargs: dict):
        return self.worker.did_abandon_batch(self.batch_id)

    def to_string(self):
        jobs = ",".join([str(id) for id in self.job_ids])
        return f"[Batch {self.batch_id} Start (Task {self.task_type}, Jobs {jobs}) at Worker {self.worker.worker_id}]"
    
    def is_worker_event():
        return True


class BatchEndEvent(Event):
    """ Event to signify that a BATCH has been performed by the WORKER. """

    def __init__(self, worker, batch, job_ids=[], task_type=(-1, -1)):
        self.worker = worker
        self.batch = batch
        self.job_ids = job_ids    # integers representing the job_ids
        self.task_type = task_type # (workflow_id, task_id)

    def run(self, current_time):
        if self.worker.did_abandon_batch(self.batch.id):
            return []
        return self.worker.free_slot(current_time, self.batch, self.task_type)

    def should_abandon_event(self, current_time, kwargs: dict):
        return self.worker.did_abandon_batch(self.batch.id)

    def to_string(self):
        jobs = ",".join([str(id) for id in self.job_ids])
        return f"[Batch {self.batch.id} End (Task {self.task_type}, Jobs {jobs}) at Worker {self.worker.worker_id}]"
    
    def is_worker_event():
        return True


# for PER_JOB scheduler
# class JobAssignEvent(Event):
#     """
#     Used in PER_JOB scheduler.
#     Event to signify that a JOB has been assigned to a worker for execution.
#     JobArrivalAtScheduler delay generate one another in a chain reaction.
#     """

#     def __init__(self, worker, job):
#         self.worker = worker
#         self.job = job

#     def run(self, current_time):
#         return self.worker.add_job(current_time, self.job)

#     def to_string(self):
#         return "[Job Assign] ---"


# class JobEndEvent(Event):
#     """ Event to signify that a JOB has been executed by the NODE.
#     JobArrivalAtScheduler delay generate one another in a chain reaction."""

#     def __init__(self, worker, job):
#         self.worker = worker
#         self.job = job

#     def run(self, current_time):
#         return self.worker.free_slot(current_time, self.job)

#     def to_string(self):
#         return "[Job End] ==="


class AbortAllJobsEvent(Event):
    """
        Event signifying that all workers should immediately abort all
        currently executing batches and send them back to the Centralized
        scheduler for rescheduling.
    """

    def __init__(self, simulation, run_herd_sched=False):
        self.simulation = simulation
        self.run_herd_sched = run_herd_sched

    def run(self, current_time):
        # NOTE: assumes abort is costless

        events = []
        for worker in self.simulation.workers:
            curr_batch_ids = [s.reserved_batch.id for s in worker.GPU_state.state_at(current_time) if s.reserved_batch]
            for batch_id in curr_batch_ids:
                evicted_batch = worker.evict_batch(batch_id, current_time)
                if self.simulation.centralized_scheduler:
                    events.append(EventOrders(
                        current_time + CPU_to_CPU_delay(evicted_batch.tasks[0].input_size * evicted_batch.size()), 
                        TasksArrivalAtScheduler(self.simulation, evicted_batch.tasks)))
                else:
                    # TODO: decentral + HERD case
                    raise NotImplementedError("Job abort for decentralized scheduler")
            
            if self.simulation.simulation_name == "shepherd":
                assigned_batch = self.simulation.scheduler.worker_states[worker.worker_id]
                if assigned_batch and not worker.did_abandon_batch(assigned_batch.id):
                    Worker._abandoned_batches.append(assigned_batch.id)

        assert(all(all(not s.reserved_batch for s in w.GPU_state.state_at(current_time)) 
                    for w in self.simulation.workers))
        
        if self.run_herd_sched:
            events.append(EventOrders(current_time, RerunHerdScheduler(self.simulation)))

        return events

    def to_string(self):
        return "[Abort All Jobs]"
    
    def is_worker_event():
        return False


class StartHerdSchedulerRerun(Event):
    """
        Event signifying that the HERD scheduler should be run again to reallocate GPUs.
        Triggers job abortion and scheduler rerun.
    """

    def __init__(self, simulation):
        self.simulation = simulation

    def run(self, current_time):
        if self.simulation.remaining_jobs:
            return [EventOrders(current_time, 
                                AbortAllJobsEvent(self.simulation, run_herd_sched=True))]
        return []

    def to_string(self):
        return "[HERD Scheduler Rerun Queued]"
    
    def is_worker_event():
        return False


class RerunHerdScheduler(Event):
    """
        Event signifying that the HERD scheduler will be run again to
        reallocate GPUs.
    """

    def __init__(self, simulation):
        self.simulation = simulation

    def run(self, current_time):
        self.simulation.run_herd_scheduler(current_time)
        self.simulation.scheduler.update_herd_assignment(self.simulation.herd_assignment)
        events = []
        if self.simulation.centralized_scheduler:
            events = self.simulation.scheduler.schedule_tasks_on_queue(current_time)

        if HERD_PERIODICITY == np.inf:
            return events
        return events + [EventOrders(current_time + HERD_PERIODICITY,
                                     StartHerdSchedulerRerun(self.simulation))]
    
    def to_string(self):
        return "[HERD Scheduler Rerun]"
    
    def is_worker_event():
        return False


class EventOrders:
    """
    Used so that the Simulation keeps track of the priority queue order
    """

    def __init__(self, current_time, event):
        self.priority = current_time
        self.current_time = current_time
        self.event = event

    def __lt__(self, other):
        return self.priority < other.priority

    def to_string(self):
        return ""+str(self.current_time) + " " + self.event.to_string()
