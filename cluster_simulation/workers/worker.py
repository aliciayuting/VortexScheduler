import core.configs.gen_config as gcfg

from queue import PriorityQueue
from queue_management.queued_task import QueuedTask
from queue_management.batching import TaskBatcher

from events.event_manager import EventManager
from events.event import *
from events.event_types import *

from core.network import *
from core.data_models.model_data import ModelData
from core.batch import Batch

from workers.gpu_state import ModelState, GPUState

from uuid import UUID


class Worker(EventListener):

    _abandoned_batches = []

    def __init__(self, id: UUID, em: EventManager, total_memory_gb: int, create_time: float,
                 is_centralized: bool):
        super().__init__(Agent.WORKER)

        self.id = id
        self.em = em
        self.is_centralized = is_centralized
        self.total_memory_gb = total_memory_gb
        self.create_time = create_time
        self.GPU_state = GPUState(total_memory_gb * (10**6))

        self.em.register_listener(self, {
            EVENT_TYPES[EventIds.TASKS_ASSIGNED_TO_WORKER],
            EVENT_TYPES[EventIds.TASKS_INPUTS_ARRIVAL_AT_WORKER],
            EVENT_TYPES[EventIds.TASKS_OUTPUTS_ASSIGNED_TO_WORKER],
            EVENT_TYPES[EventIds.TASKS_OUTPUTS_ARRIVAL_AT_WORKER],
            EVENT_TYPES[EventIds.CHECK_QUEUE_AT_WORKER],
            EVENT_TYPES[EventIds.BATCH_STARTED_AT_WORKER],
            EVENT_TYPES[EventIds.BATCH_FINISHED_AT_WORKER],
            EVENT_TYPES[EventIds.JOBS_DROPPED],
        })

        self.emitter_id = self.em.register_emitter(Agent.WORKER, {
            EVENT_TYPES[EventIds.TASKS_OUTPUTS_SENT_TO_WORKER],
            EVENT_TYPES[EventIds.TASKS_ARRIVAL_AT_SCHEDULER],
            EVENT_TYPES[EventIds.CHECK_QUEUE_AT_WORKER],
            EVENT_TYPES[EventIds.BATCH_STARTED_AT_WORKER],
            EVENT_TYPES[EventIds.BATCH_FINISHED_AT_WORKER],
            EVENT_TYPES[EventIds.JOBS_DROPPED],
            EVENT_TYPES[EventIds.RESPONSE_SENT_TO_CLIENT]
        })

        self.queues: dict[int, PriorityQueue] = {}
        self.completed_tasks: dict[tuple[int, int], Task] = {}
        self.scheduled_task_to_worker: dict[tuple[int, int], UUID] = {} # (job ID, task ID) -> worker ID

        self.awaiting_task_to_deps: dict[Task, list[int]] = {}      # list of dep task IDs remaining
        self.awaiting_dep_to_task: dict[tuple[int, int], Task] = {} # dep task ID being waited on -> waiting task
        self.awaiting_batch_to_tasks: dict[UUID, list[Task]] = {}   # instance ID -> assigned batch
        self.awaiting_task_to_batch: dict[Task, UUID] = {}          # waiting task -> assigned instance ID


    def on_event(self, event: Event):
        if event.type.id == EventIds.TASKS_ASSIGNED_TO_WORKER:
            if event.kwargs["worker_id"] != self.id:
                return
            
            self.on_tasks_assigned(event.kwargs["tasks"],
                                   event.kwargs["force_instance_id"] 
                                   if "force_instance_id" in event.kwargs else None)
        
        elif event.type.id == EventIds.TASKS_INPUTS_ARRIVAL_AT_WORKER:
            if event.kwargs["to_worker_id"] != self.id:
                return
            
            if "force_instance_id" in event.kwargs:
                self.on_batch_ready(event.time, event.kwargs["tasks"], 
                                    event.kwargs["force_instance_id"])
            else:
                self.on_tasks_ready(event.time, event.kwargs["tasks"])
        
        elif event.type.id == EventIds.TASKS_OUTPUTS_ASSIGNED_TO_WORKER:
            if event.kwargs["from_worker_id"] != self.id:
                return
            
            self.on_outputs_assigned(event.time, event.kwargs["job_task_ids"],
                                     event.kwargs["to_worker_id"])

        elif event.type.id == EventIds.TASKS_OUTPUTS_ARRIVAL_AT_WORKER:
            if event.kwargs["to_worker_id"] != self.id:
                return
            
            self.on_outputs_arrival(event.time, event.kwargs["tasks"])
        
        elif event.type.id == EventIds.JOBS_DROPPED:
            self._drop_tasks(event.kwargs["job_ids"])

        elif event.type.id == EventIds.CHECK_QUEUE_AT_WORKER:
            if event.kwargs["worker_id"] != self.id:
                return
            
            self.on_check_queue(event.time, event.kwargs["model_id"])

        elif event.type.id == EventIds.BATCH_STARTED_AT_WORKER:
            if event.kwargs["worker_id"] != self.id:
                return
            
            self.exec_batch(event.time, event.kwargs["batch"], event.kwargs["model_instance_id"])

        elif event.type.id == EventIds.BATCH_FINISHED_AT_WORKER:
            if event.kwargs["worker_id"] != self.id:
                return
            
            self.on_batch_finish(event.time, event.kwargs["batch"], event.kwargs["model_instance_id"])

        else:
            raise ValueError(f"Worker received unregistered event: {event}")
        

    def on_tasks_assigned(self, tasks: list[Task], forced_instance_id: UUID | None):
        for task in tasks:
            self.awaiting_task_to_deps[task] = task.required_task_ids.copy()
            for rt in task.required_task_ids:
                self.awaiting_dep_to_task[(task.job.id, rt)] = task
            
        if forced_instance_id:
            self.awaiting_batch_to_tasks[forced_instance_id] = tasks
            for task in tasks:
                self.awaiting_task_to_batch[task] = forced_instance_id


    def on_batch_ready(self, time: float, tasks: list[Task], instance_id: UUID):
        """If scheduler assigned a pre-formed batch to this worker, and all the 
        dependency results for the batched tasks have been received, start batch
        execution.
        """
        state = self.GPU_state.get_instance_state(instance_id, time)
        assert(state.reserved_batch == None)

        self.em.add_event(
            Event(time, 
                  EVENT_TYPES[EventIds.BATCH_STARTED_AT_WORKER],
                  kwargs={"batch": Batch(tasks), 
                          "model_instance_id": instance_id, 
                          "worker_id": self.id}),
            self.emitter_id)


    def on_tasks_ready(self, time: float, tasks: list[Task]):
        """If scheduler has NOT assigned a pre-formed batch and a set of tasks has
        arrived on the worker, add the tasks to a queue and attempt to execute 
        batch(es) if there are idle model instances.
        """
        for task in tasks:
            if task.model_data.id not in self.queues:
                self.queues[task.model_data.id] = PriorityQueue()
            self.queues[task.model_data.id].put(QueuedTask(task))

        for model_id in set(t.model_data.id for t in tasks):
            self.em.add_event(
                Event(time, 
                      EVENT_TYPES[EventIds.CHECK_QUEUE_AT_WORKER],
                      kwargs={"model_id": model_id,
                              "worker_id": self.id}),
                self.emitter_id)


    def on_outputs_assigned(self, time: float, job_task_ids: list[tuple[int, int]], to_worker_id: UUID):
        """When scheduler assigns a task's output to another worker, if the output
        exists already, send the output to the worker. Otherwise store the decision
        to send when the output is ready.
        """
        tasks_ready = []
        for (jid, tid) in job_task_ids:
            if (jid, tid) in self.completed_tasks:
                tasks_ready.append(self.completed_tasks[(jid, tid)])
            else:
                self.scheduled_task_to_worker[(jid, tid)] = to_worker_id
        
        if tasks_ready:
            self.em.add_event(
                Event(time,
                      EVENT_TYPES[EventIds.TASKS_OUTPUTS_SENT_TO_WORKER],
                      kwargs={"tasks": tasks_ready,
                              "from_worker_id": self.id,
                              "to_worker_id": to_worker_id,
                              "ignore_transfer_time": not gcfg.ENABLE_NETWORKING_DELAYS}),
                self.emitter_id)
            
    
    def on_outputs_arrival(self, time: float, tasks: list[Task]):
        """When outputs for waiting tasks arrive, attempt to add tasks to worker queue
        or start an assigned batch if all required deps are satisfied.
        """
        ready_tasks_for_worker_queue = []
        for task in tasks:
            assert((task.job.id, task.task_id) in self.awaiting_dep_to_task)

            waiting_task = self.awaiting_dep_to_task[(task.job.id, task.task_id)]
            self.awaiting_task_to_deps[waiting_task].remove(task.task_id)
            self.awaiting_dep_to_task.pop((task.job.id, task.task_id))

            # if task has no more deps to wait for
            if len(self.awaiting_task_to_deps[waiting_task]) == 0:
                self.awaiting_task_to_deps.pop(waiting_task)

                # if task was assigned to a batch
                if waiting_task in self.awaiting_task_to_batch:
                    waiting_batch = self.awaiting_task_to_batch[waiting_task]

                    # if batch has no more tasks to wait for
                    if all(t not in self.awaiting_task_to_deps 
                           for t in self.awaiting_batch_to_tasks[waiting_batch]):
                        
                        self.on_batch_ready(time, 
                                            self.awaiting_batch_to_tasks[waiting_batch],
                                            waiting_batch)
                else: # if task was not assigned (can join queue on worker instead)
                    ready_tasks_for_worker_queue.append(waiting_task)
        
        if ready_tasks_for_worker_queue:
            self.on_tasks_ready(time, ready_tasks_for_worker_queue)


    def _drop_tasks(self, job_ids: list[int]):
        """Remove all tasks associated with given jobs from queues. Does not
        affect executing batches.
        """
        for q in self.queues.values():
            filtered = []
            while q.qsize() > 0:
                task: Task = q.get()
                if task.job.id not in job_ids: 
                    filtered.append(task)
            
            for t in filtered: q.put(t)


    def on_check_queue(self, time: float, model_id: int):
        # if no tasks queued, do nothing
        if model_id not in self.queues or self.queues[model_id].qsize() == 0:
            return

        idle_instances = [s.model.id for s in self.GPU_state.state_at(time)
                          if (not s.reserved_batch) and (s.model.data.id == model_id)]
        
        while self.queues[model_id].qsize() > 0 and len(idle_instances) > 0:
            batch = TaskBatcher.get_batch(time, 
                                            self.total_memory_gb, 
                                            self.queues[model_id],
                                            True)
            
            if not batch: break

            self.em.add_event(
                Event(time, 
                        EVENT_TYPES[EventIds.BATCH_STARTED_AT_WORKER],
                        kwargs={"batch": batch, 
                                "model_instance_id": idle_instances.pop(0), 
                                "worker_id": self.id}),
                self.emitter_id)
    

    def on_batch_finish(self, time: float, batch: Batch, instance_id):
        assert(not self.did_abandon_batch(batch.id))

        # release model
        if batch.model_data != None:
            self.GPU_state.release_busy_model(batch.id, time)

        tasks_to_send: dict[UUID, list[Task]] = {}
        for task in batch.tasks:
            task.job.set_completion_time(time, task.task_id)
            self.completed_tasks[(task.job.id, task.task_id)] = task

            # if central scheduler directed worker to send results, add to send batch
            if (task.job.id, task.task_id) in self.scheduled_task_to_worker:
                next_worker_id = self.scheduled_task_to_worker[(task.job.id, task.task_id)]
                if next_worker_id not in tasks_to_send: tasks_to_send[next_worker_id] = []
                tasks_to_send[next_worker_id].append(task)

            # if any jobs are complete, send response to client
            if task.job.is_complete():
                self.em.add_event(
                    Event(time,
                          EVENT_TYPES[EventIds.RESPONSE_SENT_TO_CLIENT],
                          kwargs={"job": task.job,
                                  "ignore_transfer_time": not gcfg.ENABLE_NETWORKING_DELAYS,
                                  "client_id": task.job.client_id,
                                  "worker_id": self.id}),
                    self.emitter_id)
        
        # send outputs if required
        for worker_id, send_batch in tasks_to_send.items():
            self.em.add_event(
                Event(time,
                    EVENT_TYPES[EventIds.TASKS_OUTPUTS_SENT_TO_WORKER],
                    kwargs={"tasks": send_batch,
                            "from_worker_id": self.id,
                            "to_worker_id": worker_id,
                            "ignore_transfer_time": not gcfg.ENABLE_NETWORKING_DELAYS}),
                self.emitter_id)

        # notify centralized scheduler of newly available tasks, but keep outputs on current worker
        if self.is_centralized:
            all_available = [nt for task in batch.tasks for nt in task.job.newly_available_tasks(task)]
            if all_available:
                self.em.add_event(
                    Event(time, 
                        EVENT_TYPES[EventIds.TASKS_ARRIVAL_AT_SCHEDULER],
                        kwargs={"tasks": all_available}), 
                    self.emitter_id)
        
        assert(self.GPU_state.does_have_idle_copy(batch.model_data.id, time))

        self.em.add_event(
            Event(time, 
                    EVENT_TYPES[EventIds.CHECK_QUEUE_AT_WORKER],
                    kwargs={"model_id": batch.model_data.id,
                            "worker_id": self.id}),
            self.emitter_id)


    def exec_batch(self, time: float, batch: Batch, instance_id=None):
        """Reserves an idle copy of any required GPU models and executes the given batch.

        Args:
            time: Time to start batch execution
            batch: Batch to execute
            instance_id: Optional ID of model instance to execute
        """

        batch_exec_time = batch.model_data.get_randomized_exec_time(
            batch.size(), self.total_memory_gb)
        reserved_instance_id = None

        if batch.model_data != None:
            assert(self.GPU_state.does_have_idle_copy(batch.model_data.id, time))

            if instance_id:
                reserved_instance_id = instance_id
                self.GPU_state.reserve_instance(instance_id, time, batch, batch_exec_time)
            else:
                reserved_instance_id = self.GPU_state.reserve_idle_copy(
                    batch.model_data, time, batch, batch_exec_time)

            # verify reserved instance
            reserved_state = [s for s in self.GPU_state.state_at(time)
                                if s.model.id == reserved_instance_id][0]
            assert(reserved_state.reserved_until >= (time + batch_exec_time))
            assert(reserved_state.reserved_batch == batch)
            assert(reserved_state.model.data.id == batch.model_data.id)
            assert(reserved_state.state == ModelState.PLACED)

            batch_exec_time = reserved_state.reserved_until - time

        task_end_time = time + batch_exec_time + \
            SameMachineCPUtoGPU_delay(sum(t.input_size for t in batch.tasks)) + \
            SameMachineGPUtoCPU_delay(sum(t.result_size for t in batch.tasks))
        
        self.em.add_event(
            Event(task_end_time, 
                  EVENT_TYPES[EventIds.BATCH_FINISHED_AT_WORKER],
                  kwargs={"batch": batch, 
                          "model_instance_id": reserved_instance_id, 
                          "worker_id": self.id}),
            self.emitter_id)
        
    def did_abandon_batch(self, batch_id: int):
        return batch_id in Worker._abandoned_batches
    
    def get_qlen(self, model_id: int):
        return self.queues[model_id].qsize() if model_id in self.queues else 0

    def __hash__(self):
        return hash(self.id)

    def __str__(self):
        return f"[WORKER {self.id}] [STATE: {self.GPU_state}]"
    
    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        return isinstance(other, Worker) and self.id == other.id