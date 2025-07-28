from queue import PriorityQueue

from core.config import *
from core.task import Task
from core.batch import Batch
from core.events import *
from core.network import *
from core.workflow import *

from workers.worker import Worker

from schedulers.centralized.scheduler import Scheduler
from schedulers.centralized.shepherd.ordered_task import OrderedTask


class ShepherdScheduler(Scheduler):
    """
        Scheduler class for a Flex scheduler.
    """

    def __init__(self, simulation, herd_assignment=None):
        super().__init__(simulation, herd_assignment)
        
        self.model_queues = { mid: PriorityQueue() for mid in set(get_model_id_for_task_type(tt) 
                                                                  for tt in get_task_types(self.simulation.job_types_list)) }

        # init currently executing batch ids
        self.worker_states = {}
        for group in self.herd_assignment.worker_groups:
            for worker in group:
                self.worker_states[worker.worker_id] = None

        # for round-robin worker ordering
        self.next_worker_idxs = [0 for _ in self.herd_assignment.worker_groups]

    def update_herd_assignment(self, herd_assignment):
        super().update_herd_assignment(herd_assignment)

        # currently executing batch ids
        self.worker_states = {}
        for group in self.herd_assignment.worker_groups:
            for worker in group:
                self.worker_states[worker.worker_id] = None

        # for round-robin worker ordering
        self.next_worker_idxs = [0 for _ in self.herd_assignment.worker_groups]

    def schedule_job_on_arrival(self, job, current_time):
        arrived_groups = set()
        for task in job.tasks:
            if len(task.required_task_ids) == 0:
                self.model_queues[task.model.model_id].put(OrderedTask(task, current_time))
                arrived_groups.add(self.herd_assignment.task_type_to_group[task.task_type])

        events = []
        for group in arrived_groups:
            events += self._flex_schedule_tasks_on_arrival(group, current_time)
        return events
    
    def schedule_tasks_on_arrival(self, tasks, current_time):
        """
            While there are unchecked workers and queued tasks, creates the largest batch
            possible across all models and attempts to assign a worker to the batch in round 
            robin order.
        """
        arrived_groups = set()
        for task in tasks:
            if task.model.model_id not in self.model_queues:
                self.model_queues[task.model.model_id] = PriorityQueue()
            self.model_queues[task.model.model_id].put(OrderedTask(task, current_time))
            arrived_groups.add(self.herd_assignment.task_type_to_group[task.task_type])
        events = []
        for group in arrived_groups:
            events += self._flex_schedule_tasks_on_arrival(group, current_time)
        return events

    def schedule_tasks_on_queue(self, current_time):
        events = []
        for group in range(len(self.herd_assignment.worker_groups)):
            events += self._flex_schedule_tasks_on_arrival(group, current_time)
        return events
    
    # STATE MANAGEMENT --------------------------------------------------------------------
    
    def worker_completed_batch(self, worker_id: int, batch: Batch):
        assert(self.worker_states[worker_id] == batch)
        self.worker_states[worker_id] = None

    def worker_rejected_batch(self, worker_id: int, batch: Batch, current_worker_batch: Batch):
        # if alr. scheduled a different batch, do NOT update state
        if self.worker_states[worker_id] != batch:
            return
        self.worker_states[worker_id] = current_worker_batch

    def preempt_batch_on_worker(self, worker_id: int, new_batch: Batch):
        assert(self.worker_states[worker_id] is not None)
        self.worker_states[worker_id] = new_batch

    def assign_batch_to_worker(self, worker_id: int, batch: Batch):
        assert(self.worker_states[worker_id] is None)
        self.worker_states[worker_id] = batch
        
    # FLEX ALGO ----------------------------------------------------------------------------

    def _flex_schedule_tasks_on_arrival(self, group, current_time):
        self._drop_bad_tasks(current_time)

        events = []
        
        unassigned_workers = self.herd_assignment.worker_groups[group].copy()
        unassigned_workers = unassigned_workers[self.next_worker_idxs[group]:] + unassigned_workers[:self.next_worker_idxs[group]]

        worker_idx = 0

        largest_batch_model_id, largest_batch_size = self._flex_get_largest_candidate_batch(group, current_time)
        while worker_idx < len(unassigned_workers) and largest_batch_size > 0:
            next_worker = unassigned_workers[worker_idx]
            
            if not ENABLE_DYNAMIC_MODEL_LOADING:
                if all(m.model_id != largest_batch_model_id for m in next_worker.GPU_state.placed_models(current_time)):
                    worker_idx += 1
                    continue
            
            # when it is impossible for worker to load model for some reason
            if next_worker.get_wait_time(current_time, largest_batch_model_id) == np.inf:
                worker_idx += 1
                continue

            # TODO: allow shepherd workers to run concurrent models
            # NOTE: workers are assumed to run only 1 batch at a time
            curr_batch = self.worker_states[next_worker.worker_id]
            curr_batch_size = curr_batch.size() if not curr_batch is None else 0

            if curr_batch_size == 0:
                # assign batch to best worker
                batch = self._flex_form_largest_batch(largest_batch_model_id, current_time)
                self.assign_batch_to_worker(next_worker.worker_id, batch)
                events.append(EventOrders(
                    current_time + CPU_to_CPU_delay(batch.size()*batch.tasks[0].input_size), 
                    BatchArrivalAtWorker(self.simulation, next_worker, batch)))
                # update candidate batch
                largest_batch_model_id, largest_batch_size = self._flex_get_largest_candidate_batch(group, current_time)
            elif largest_batch_size >= FLEX_LAMBDA * curr_batch_size:
                # assign batch to best worker
                batch = self._flex_form_largest_batch(largest_batch_model_id, current_time)
                old_batch_id = self.worker_states[next_worker.worker_id].id
                self.preempt_batch_on_worker(next_worker.worker_id, batch)
                events.append(EventOrders(
                    current_time + CPU_to_CPU_delay(batch.size()*batch.tasks[0].input_size), 
                    BatchPreemptionAtWorker(self.simulation, next_worker, batch, old_batch_id)))
                # update candidate batch
                largest_batch_model_id, largest_batch_size = self._flex_get_largest_candidate_batch(group, current_time)
            
            # remove worker from consideration
            worker_idx += 1

        self.next_worker_idxs[group] = (self.next_worker_idxs[group] + worker_idx) % len(self.next_worker_idxs)
        
        return events
    
    def _drop_bad_tasks(self, time: float):
        """
            For each model queue in [self.model_queues], for each task in queue, if 1) [time] >= task arrival 
            at scheduler and 2) [time] + exec time of the smallest batch for this model > task deadline + 
            grace period, then drop [task.job].
        """
        for model_queue in self.model_queues.values():
            skipped_tasks = []
            while model_queue.qsize() > 0:
                ot = model_queue.get()
                if time < ot.task.log.task_arrival_at_scheduler_timestamp:
                    skipped_tasks.append(ot)
                    continue
                # drop tasks whose SLOs can't be satisfied within a grace period
                # earliest task end time >= deadline + grace period
                if (time + ot.task.model.batch_exec_times[24][0]) > ot.deadline * (1 + SLO_SLACK):
                    self.simulation.task_drop_log.loc[len(self.simulation.task_drop_log)] = {
                        "job_id": ot.task.job_id,
                        "workflow_id": ot.task.task_type[0],
                        "task_id": ot.task.task_type[1],
                        "drop_time": time,
                        "arrival_time": ot.task_arrival_time,
                        "slo": ot.task.slo,
                        "deadline": ot.deadline
                    }
                else:
                    skipped_tasks.append(ot)
            
            for ot in skipped_tasks:
                model_queue.put(ot)

    def _flex_form_largest_batch(self, model_id: int, time: float) -> Batch:
        """
            Form the largest batch available in [self.model_queues[model_id]].
        """
        tasks = []
        skipped_tasks = []
        model_queue = self.model_queues[model_id]
        while model_queue.qsize() > 0:
            ot = model_queue.get()
            if time < ot.task.log.task_placed_on_worker_queue_timestamp:
                skipped_tasks.append(ot)
                continue
            tasks.append(ot.task)
            if len(tasks) == tasks[0].max_batch_size:
                break
        for ot in skipped_tasks:
            model_queue.put(ot)
        return Batch(tasks)

    def _flex_get_largest_candidate_batch(self, group: int, current_time: float):
        """
            Returns (model_id, batch_size) of the largest batch that can be formed from currently 
            queued tasks across all of [model_queues] where model_id is required by some task in
            [task_types].
        """
        task_types = self.herd_assignment.group_task_types[group]
        largest_batch_model_id = -1
        largest_batch_size = 0
        for task_type in task_types:
            mid = get_model_id_for_task_type(task_type)
            if mid not in self.model_queues or self.model_queues[mid].qsize() == 0:
                continue # no tasks queued
            candidate_batch_size = min(len([t for t in self.model_queues[mid].queue if current_time >= t.task.log.task_placed_on_worker_queue_timestamp]),
                                       self.model_queues[mid].queue[0].task.max_batch_size)
            if candidate_batch_size > largest_batch_size:
                largest_batch_size = candidate_batch_size
                largest_batch_model_id = mid
        return (largest_batch_model_id, largest_batch_size)

    def flex_schedule_on_batch_completion(self, worker: Worker, completed_batch: Batch, current_time: float):
        self._drop_bad_tasks(current_time)
        
        # if alr. assigned to a new batch do nothing
        if self.worker_states[worker.worker_id].id != completed_batch.id:
            return []

        self.worker_completed_batch(worker.worker_id, completed_batch)

        all_task_types = []
        for m in worker.GPU_state.placed_models(current_time):
            all_task_types += get_task_types_for_model(m.model_id)
        
        largest_batch_model_id, largest_batch_size = self._flex_get_largest_candidate_batch(
            self.herd_assignment.task_type_to_group[completed_batch.tasks[0].task_type], 
            current_time)

        if largest_batch_size > 0:
            batch = self._flex_form_largest_batch(largest_batch_model_id, current_time)
            self.assign_batch_to_worker(worker.worker_id, batch)
            return [EventOrders(
                current_time + CPU_to_CPU_delay(batch.size()*batch.tasks[0].input_size), 
                BatchArrivalAtWorker(self.simulation, worker, batch))]
        return []