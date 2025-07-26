import queue
from collections import defaultdict

from workers.taskworker import *
from workers.worker import *

from core.network import *
from core.events import *
from core.batch import Batch

from schedulers.centralized.heft_scheduler import HeftScheduler


class HeftTaskWorker(TaskWorker):
    def __init__(self, simulation, worker_id, total_memory, group_id=-1):
        super().__init__(simulation, worker_id, total_memory, group_id=group_id)
        # {task_obj1:[(preq_task_id0,arrival_time0), (preq_taks_id0, arrival_time1), ...], task2:[( ...],}
        self.waiting_tasks_buffer = defaultdict(lambda: [])
        # keep track of the queue information at time:  [ (time1,[task0,task1,]), (time2,[task1,...]),...]
        self.queue_history = {}
        self.involved = False
        self.max_wait_times = {}

        self.next_task_type_idx = 0
        self.next_worker_id = { mid: 0 for mid in set([get_model_id_for_task_type(tt) for tt in get_task_types(self.simulation.job_types_list) ]) }

    def add_task(self, current_time, task):
        """
        Add task into the local task queue
        """
        if not ENABLE_DYNAMIC_MODEL_LOADING and task.model != None and task.model not in self.GPU_state.placed_models(current_time):
            print("Static allocation received task that cannot be executed")
            print(f"Allocated: {self.GPU_state.state_at(current_time)}, Requested ID: {task.model.model_id}")
            assert(False)

        # Update when the task is sent to the worker
        assert (task.log.task_placed_on_worker_queue_timestamp <= current_time)
        self.add_task_to_queue_history(task, current_time) # Update when the task is sent to the worker

        # Initialize max wait time
        if task.task_type not in self.max_wait_times or self.max_wait_times[task.task_type] < 0:
            self.max_wait_times[task.task_type] = current_time + task.max_wait_time

        if (not ENABLE_MULTITHREADING) and any(s.reserved_batch for s in self.GPU_state.state_at(current_time)):
            return []

        events = self.check_task_queue(task.task_type, current_time)
        return events
    
    def add_tasks(self, current_time, tasks):
        """
        Add tasks into the local task queue
        """
        if len(tasks) == 0:
            return []

        if not ENABLE_DYNAMIC_MODEL_LOADING and tasks[0].model != None and tasks[0].model not in self.GPU_state.placed_models(current_time):
            print("Static allocation received task that cannot be executed")
            print(f"Allocated: {self.GPU_state.state_at(current_time)}, Requested ID: {tasks[0].model.model_id}")
            assert(False)

        # Update when the task is sent to the worker
        assert (tasks[0].log.task_placed_on_worker_queue_timestamp <= current_time)
        for task in tasks:
            self.add_task_to_queue_history(task, current_time) # Update when the task is sent to the worker

        # Initialize max wait time
        if tasks[0].task_type not in self.max_wait_times or self.max_wait_times[tasks[0].task_type] < 0:
            self.max_wait_times[tasks[0].task_type] = current_time + task.max_wait_time

        if (not ENABLE_MULTITHREADING) and any(s.reserved_batch for s in self.GPU_state.state_at(current_time)):
            return []

        events = self.check_task_queue(tasks[0].task_type, current_time)
        return events

    def free_slot(self, current_time, batch: Batch, task_type):
        """ Attempts to launch another task. """
        events = super().free_slot(current_time, batch, task_type)

        # update queue history
        for task in batch.tasks: # rm all tasks in batch
            self.rm_task_in_queue_history(task, current_time)

        # update max wait time for executed batch task type
        queued_tasks = queue.Queue()
        [queued_tasks.put(task) for task in self.get_queue_history(current_time, task_type, info_staleness=0)]
        if not queued_tasks.empty():
            earliest_remaining_arrival = -1
            while not queued_tasks.empty():
                task = queued_tasks.get()
                if earliest_remaining_arrival < 0 or \
                    task.log.task_placed_on_worker_queue_timestamp < earliest_remaining_arrival:
                    earliest_remaining_arrival = task.log.task_placed_on_worker_queue_timestamp
            self.max_wait_times[task_type] = earliest_remaining_arrival + batch.tasks[0].max_wait_time
        else:
            self.max_wait_times[task_type] = -1

        # start next batch
        if ENABLE_MULTITHREADING:
            for task_type in self.queue_history.keys():
                batch_end_events = self.check_task_queue(task_type, current_time)
                events += batch_end_events
        else:
            all_task_types = get_task_types(self.simulation.job_types_list)
            self.next_task_type_idx = all_task_types.index(task_type)
            task_types_left = len(all_task_types)
            batch_end_events = []
            while not batch_end_events and task_types_left:
                if all_task_types[self.next_task_type_idx] in self.queue_history:
                    batch_end_events = self.check_task_queue(all_task_types[self.next_task_type_idx], current_time)
                    events += batch_end_events
                self.next_task_type_idx = (self.next_task_type_idx + 1) % len(all_task_types)
                task_types_left -= 1
        return events

    #  --------------------------- DECENTRALIZED WORKER SCHEDULING  ----------------------
    
    def schedule_job_heft(self, current_time, job):
        """ HEFT scheduler to schedule Tasks and send the initial task to worker 
        This implementation so far, assume scheduling thread is different from the execution thread,
            where even schedule the initial task on this same worker, 
            this task needs to be waiting from the end of queue after scheduling
        """
        task_arrival_events = []  # List to store the TaskArrivalEvent to the receiving Workers
        # 1. compute scheduling decisions based on decentralized HEFT
        # {task_id0->worker_id0, ...}
        activation_graph = HeftScheduler.nav_heft_job_plan(job, \
                                             self.simulation.workers, \
                                             current_time, \
                                             initial_worker_id=self.worker_id, \
                                             consider_load=self.simulation.consider_load, \
                                             consider_cache=self.simulation.consider_cache)

        # 2. assign the planned ADFG to job object
        job.assign_ADFG(activation_graph)

        # 3. send the first task to allocated worker
        initial_tasks = [task for task in job.tasks if len(task.required_task_ids) == 0]
        for initial_task in initial_tasks:
            worker_index = activation_graph[initial_task.task_id]
            task_arrival_time = current_time
            if(worker_index != self.worker_id):
                task_arrival_time = current_time + \
                    CPU_to_CPU_delay(initial_task.input_size)
            task_arrival_events.append(EventOrders(
                task_arrival_time, TaskArrival(self.simulation.workers[worker_index], initial_task, job.id)))
        
        return task_arrival_events
    
    def check_task_queue(self, task_type, current_time):
        task_queue = self.get_queue_history(current_time, task_type, info_staleness=0)
        tasks = []
        for task in task_queue:
            if current_time >= task.log.task_placed_on_worker_queue_timestamp:
                # skip dropped tasks
                if (self.simulation.task_drop_log["job_id"]==task.job_id).any():
                    continue

                # get correct task deadline
                if self.simulation.centralized_scheduler:
                    if SLO_GRANULARITY == "TASK":
                        task_deadline = task.log.task_arrival_at_scheduler_timestamp + task.slo
                        task_arrival = task.log.task_arrival_at_scheduler_timestamp 
                    else:
                        task_deadline = task.log.job_creation_timestamp + task.job.slo
                        task_arrival = task.log.job_creation_timestamp
                else:
                    if SLO_GRANULARITY == "TASK":
                        task_deadline = task.log.task_placed_on_worker_queue_timestamp + task.slo
                        task_arrival = task.log.task_placed_on_worker_queue_timestamp
                    else:
                        task_deadline = task.log.job_creation_timestamp + task.job.slo
                        task_arrival = task.log.job_creation_timestamp


                # drop tasks whose SLO can't be met
                if (current_time + task.model.batch_exec_times[24][0]) >= task_deadline * (1 + SLO_SLACK):
                    for job_task in task.job.tasks:
                        self.rm_task_in_queue_history(job_task, current_time)
                    self.simulation.task_drop_log.loc[len(self.simulation.task_drop_log)] = {
                        "job_id": task.job_id, "workflow_id": task.task_type[0], "task_id": task.task_type[1],
                        "drop_time": current_time, 
                        "arrival_time": task_arrival,
                        "slo": task.slo if SLO_GRANULARITY == "TASK" else task.job.slo, 
                        "deadline": task_deadline
                    }
                    continue
                
                tasks.append(task)

        if len(tasks) == 0:
            return []
        else:
            batch = Batch(tasks[:tasks[0].max_batch_size])
            return self.maybe_start_batch(batch, current_time)

    #  ---------------------------  Subsequent TASK Transfer   --------------------

    def send_results_to_next_workers(self, current_time, batch) -> list:
        """
        Send the result of a task to the next worker in the inference pipeline (it may be the same worker)
        """
        events = []

        if self.simulation.simulation_name == "hashtask":
            ready_tasks = []
            for task in batch.tasks:
                ready_tasks += task.job.newly_available_tasks(task)

            if len(ready_tasks) == 0:
                return []
            
            prev_curr = current_time
            for i in range(0, len(ready_tasks), 4):
                curr_send_batch = ready_tasks[i:(i+4)]

                transfer_delay = 0
                if ENABLE_DYNAMIC_MODEL_LOADING:
                    if ALLOCATION_STRATEGY == "HERD":
                        # don't choose worker that is not in the correct group
                        while curr_send_batch[0].model and curr_send_batch[0].model.model_id not in self.simulation.herd_assignment.group_models[self.simulation.workers[self.next_worker_id[curr_send_batch[0].model.model_id]].group_id] and \
                            self.simulation.workers[self.next_worker_id[curr_send_batch[0].model.model_id]].total_memory * 10**6 < curr_send_batch[0].model.model_size:
                            self.next_worker_id[curr_send_batch[0].model.model_id] = (self.next_worker_id[curr_send_batch[0].model.model_id] + 1) % len(self.simulation.workers)
                    else:
                        while curr_send_batch[0].model and self.simulation.workers[self.next_worker_id[curr_send_batch[0].model.model_id]].total_memory * 10**6 < curr_send_batch[0].model.model_size:
                            self.next_worker_id[curr_send_batch[0].model.model_id] = (self.next_worker_id[curr_send_batch[0].model.model_id] + 1) % len(self.simulation.workers)
                else:
                    while curr_send_batch[0].model and all(m.model_id != curr_send_batch[0].model.model_id for m in self.simulation.workers[self.next_worker_id[curr_send_batch[0].model.model_id]].GPU_state.placed_models(current_time)):
                        self.next_worker_id[curr_send_batch[0].model.model_id] = (self.next_worker_id[curr_send_batch[0].model.model_id] + 1) % len(self.simulation.workers)
                
                if self.next_worker_id[curr_send_batch[0].model.model_id] != self.worker_id:  # The next worker on the pipeline is NOT the same node
                    transfer_delay = CPU_to_CPU_delay(task.result_size * len(curr_send_batch))

                events.append(EventOrders(prev_curr + transfer_delay, TasksArrival(
                    self.simulation.workers[self.next_worker_id[curr_send_batch[0].model.model_id]], curr_send_batch)))
                
                prev_curr = prev_curr + transfer_delay

                self.next_worker_id[curr_send_batch[0].model.model_id] = (self.next_worker_id[curr_send_batch[0].model.model_id] + 1) % len(self.simulation.workers)
        else:
            for task in batch.tasks:
                cur_job = self.simulation.jobs[task.job_id]
                for cur_task_id in task.next_task_ids:
                    cur_task = cur_job.tasks[cur_task_id]
                    assigned_worker_id = task.ADFG[cur_task.task_id]
                    if self.simulation.simulation_name != "hashtask" and self.simulation.dynamic_adjust:
                        assigned_worker_id = HeftScheduler.nav_heft_task_adjustment(cur_job, cur_task_id, \
                                                                    self.simulation.workers, \
                                                                    current_time, \
                                                                    self.worker_id, \
                                                                    assigned_worker_id)
                    next_worker = self.simulation.workers[assigned_worker_id]
                    transfer_delay = 0
                    if assigned_worker_id != self.worker_id:  # The next worker on the pipeline is NOT the same node
                        transfer_delay = CPU_to_CPU_delay(task.result_size)
                    events.append(EventOrders(current_time + transfer_delay, InterResultArrival(
                        worker=next_worker, prev_task=task, cur_task=cur_task)))
        return events

    def receive_intermediate_result(self, current_time, prev_task, cur_task) -> list:
        """
        event handling when taskworker receives the result of prev_task and put it to the waiting_buffer of cur_task
        @param: current_time: the time when the prev_task result arrives at this worker
        @param: prev_task: the task that has been executed and sent the result to this worker
        @param: cur_task: the task that is waiting for the result of prev_task, to be put to the waiting_buffer
        """
        events = []
        # 0. add this arrived prev_task result to the buffer of cur_task
        if prev_task != None:
            self.waiting_tasks_buffer[cur_task].append(
                [prev_task.task_id, current_time])
        # check if we have collected all the preq_tasks for current_task to be put on the task queue
        prev_arrived_list = self.waiting_tasks_buffer[cur_task]
        if(len(prev_arrived_list) != len(cur_task.required_task_ids)):
            # the cur_task hasn't received all the prerequisite task it needs to execute. SKIP this round
            return events
        # time when all the pre-requisite tasks have arrived
        receive_time = current_time
        for element in prev_arrived_list:
            receive_time = max(receive_time, element[1])
        events.append(EventOrders(
            receive_time, TaskArrival(self, cur_task, cur_task.job_id)))
        return events
    
    # ------------------------- queue history update helper functions ---------------

    def add_task_to_queue_history(self, task, current_time):
        # 0. Base case (first entry)
        if task.task_type not in self.queue_history:
            self.queue_history[task.task_type] = [(current_time, [task])]
            return

        # 1. Find the time_stamp place to add this queue information
        last_index = len(self.queue_history[task.task_type]) - 1
        while last_index >= 0:
            if self.queue_history[task.task_type][last_index][0] == current_time:
                if task not in self.queue_history[task.task_type][last_index][1]:
                    self.queue_history[task.task_type][last_index][1].append(task)
                break
            if self.queue_history[task.task_type][last_index][0] < current_time:
                # print("2")
                if task not in self.queue_history[task.task_type][last_index][1]:
                    next_queue = self.queue_history[task.task_type][last_index][1].copy()
                    next_queue.append(task)
                    last_index += 1
                    self.queue_history[task.task_type].insert(
                        last_index, (current_time, next_queue)
                    )
                break
            # check the previous entry
            last_index -= 1

        # 2. added the task to all the subsequent timestamp tuples
        while last_index < len(self.queue_history[task.task_type]):
            if task not in self.queue_history[task.task_type][last_index][1]:
                self.queue_history[task.task_type][last_index][1].append(task)
            last_index += 1

    def rm_task_in_queue_history(self, task, current_time):
        # 0. base case: shouldn't happen
        if task.task_type not in self.queue_history:
            AssertionError("rm model cached location to an empty list")
            return

        last_index = len(self.queue_history[task.task_type]) - 1
        
        # 1. find the place to add this remove_event to the tuple list
        while last_index >= 0:
            if self.queue_history[task.task_type][last_index][0] == current_time:
                if task in self.queue_history[task.task_type][last_index][1]:
                    self.queue_history[task.task_type][last_index][1].remove(task)
                break
            if self.queue_history[task.task_type][last_index][0] < current_time:
                if task in self.queue_history[task.task_type][last_index][1]:
                    next_tasks_in_queue = self.queue_history[task.task_type][last_index][1].copy()
                    next_tasks_in_queue.remove(task)
                    last_index = last_index + 1
                    self.queue_history[task.task_type].insert(
                        last_index, (current_time, next_tasks_in_queue)
                    )
                break
            last_index -= 1  # go to prev time
        # 2. remove the task from all the subsequent tuple
        while last_index < len(self.queue_history[task.task_type]):
            if task in self.queue_history[task.task_type][last_index]:
                self.queue_history[task.task_type][last_index][1].remove(task)
            last_index += 1  # do this for the remaining element after

    def get_queue_history(self, current_time, task_type, info_staleness=0) -> list:
        return self.get_history(self.queue_history[task_type], current_time, info_staleness)

    def get_task_queue_waittime(self, current_time, task_type, info_staleness=0, requiring_worker_id=None):
        if requiring_worker_id != None and requiring_worker_id != self.worker_id:
            info_staleness = 0

        task_model_id = WORKFLOW_LIST[task_type[0]]["TASKS"][task_type[1]]["MODEL_ID"]
        if task_model_id < 0: # CPU only
            return 0
        
        task_model_states = list(filter(lambda s: s.model.model_id == task_model_id, 
                                        self.GPU_state.placed_model_states(current_time)))    
        if len(task_model_states) == 0: # model not currently on worker
            if not ENABLE_DYNAMIC_MODEL_LOADING: # static alloc
                return np.inf
            else:
                task_model = self.simulation.get_model_from_id(task_model_id)
                fetch_time = SameMachineCPUtoGPU_delay(task_model.model_size)
                if self.GPU_state.can_fetch_model_on_eviction(task_model, current_time):
                    # evictions are free
                    return fetch_time
                elif self.GPU_state._total_memory < task_model.model_size:
                    return np.inf # partition too small
                elif ALLOCATION_STRATEGY == "HERD" and task_model.model_id not in self.simulation.herd_assignment.group_models[self.group_id]:
                    return np.inf # model ID not in worker's HERD-assigned group
                else: # not enough space to load right away
                    latest_avail = 0
                    placed_model_states = [s for s in self.GPU_state.state_at(current_time) 
                                           if s.state in [ModelState.IN_FETCH, ModelState.PLACED]]
                    total_avail_mem = self.GPU_state.available_memory(current_time)
                    i = 0
                    while total_avail_mem < task_model.model_size:
                        total_avail_mem += placed_model_states[i].model.model_size
                        avail_at = 0
                        if placed_model_states[i].state == ModelState.IN_FETCH:
                            avail_at = current_time + self.GPU_state.shortest_time_to_fetch_end(
                                placed_model_states[i].model.model_id, current_time)
                        else:
                            avail_at = placed_model_states[i].reserved_until
                        latest_avail = max(latest_avail, avail_at)
                        i += 1
                    return latest_avail - current_time + fetch_time

        if self.GPU_state.does_have_idle_copy(task_model_states[0].model, current_time):
            return 0
        
        return min(s.reserved_until for s in task_model_states) - current_time