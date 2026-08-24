import numpy as np

from core.data_models.workflow import Workflow


class NexusSLOSplitter:
    """SLO split algorithm adapted from pg. 330 (sec 6.2): 
    https://homes.cs.washington.edu/~arvind/papers/nexus.pdf
    """

    @classmethod
    def get_task_arrival_rates(cls, workflow: Workflow, job_arrival_rate: float,
                               workers: dict, time: float = 0) -> dict[int, float]:
        """Estimates the request rate seen by a single instance of each task's model.

        Every job runs each task exactly once, so a task's aggregate arrival rate is
        the job arrival rate, spread over the workers hosting that task's model.

        Args:
            workflow: Workflow whose tasks to estimate rates for
            job_arrival_rate: Job arrival rate for this workflow (qps)
            workers: Map of worker ID -> worker object
            time: Time at which to read model placements

        Returns:
            task_arrival_rates: Task ID -> per worker arrival rate (qps)
        """
        task_arrival_rates = {}
        for task in workflow.tasks.values():
            num_model_workers = len([
                w for w in workers.values()
                if any(s.model.data.id == task.model_data.id for s in w.GPU_state.state_at(time))])

            assert(num_model_workers > 0)
            task_arrival_rates[task.id] = job_arrival_rate / num_model_workers

        return task_arrival_rates


    @classmethod
    def get_task_worker_sizes(cls, workflow: Workflow, workers: dict,
                              time: float = 0) -> dict[int, int]:
        """Returns the worker memory size (GB) each task's batch exec times should
        be read from.

        A task can be dispatched to any worker hosting its model, and the same model
        runs at different speeds on different MIG partition sizes, so the split is
        planned against the slowest partition the model is actually placed on.

        Args:
            workflow: Workflow whose tasks to find worker sizes for
            workers: Map of worker ID -> worker object
            time: Time at which to read model placements

        Returns:
            task_worker_sizes: Task ID -> worker memory size to profile against (GB)
        """
        task_worker_sizes = {}
        for task in workflow.tasks.values():
            placed_sizes = {
                w.total_memory_gb for w in workers.values()
                if any(s.model.data.id == task.model_data.id
                       for s in w.GPU_state.state_at(time))}
            profiled_sizes = placed_sizes & set(task.model_data.batch_exec_times.keys())

            assert(profiled_sizes), \
                (f"Model {task.model_data.id} has no batch exec times profiled for "
                 f"any worker size it is placed on ({sorted(placed_sizes)})")

            task_worker_sizes[task.id] = max(
                profiled_sizes, key=lambda ws: task.model_data.batch_exec_times[ws][1])

        return task_worker_sizes


    @classmethod
    def _max_feasible_batch_size(cls, model_data, worker_size: int, budget: float) -> int:
        """Returns the largest batch of [model_data] that runs within [budget] on a
        worker of [worker_size], falling back to 1 when even that does not fit.
        """
        feasible = [b for b in range(1, model_data.max_batch_size + 1)
                    if model_data.batch_exec_times[worker_size][b] <= budget]

        return max(feasible) if feasible else 1


    @classmethod
    def generate_task_slos(cls, workflow: Workflow, slo: float,
                           task_arrival_rates: dict[int, float],
                           task_worker_sizes: dict[int, int]) -> dict[int, tuple[float, int]]:
        """Split job SLO over workflow tasks.

        Args:
            workflow: Workflow to split SLO for
            slo: Job-level SLO to split across tasks (ms)
            task_arrival_rates: Task ID -> per worker arrival rate (qps), as
            produced by [get_task_arrival_rates]
            task_worker_sizes: Task ID -> worker memory size whose batch exec times
            to plan against (GB), as produced by [get_task_worker_sizes]

        Returns:
            task_slos: (SLO, max batch size) for each task ID in workflow
        """

        # granularity of SLOs in ms
        TIME_STEP = 5

        slo = int(slo)
        assert(slo >= TIME_STEP)

        # task id -> task SLO -> (min # gpus, max batch size, (SLO for curr task, SLO for subtree))
        min_gpus = {task.id: {} for task in workflow.tasks.values()}
        
        # base case: exit points / leaf nodes
        final_tasks = [t for t in workflow.tasks.values() if len(t.next_tasks) == 0]
        assert(len(final_tasks) == 1) # NOTE: algorithm is for fork-join graphs
        final_task = final_tasks[0]

        def _min_gpu_single(model, req_rate, k, worker_size):
            exec_times = model.batch_exec_times[worker_size]
            bsizes = [b for b in range(1, model.max_batch_size + 1) if exec_times[b] <= k]
            if not bsizes:
                return np.inf
            return min([req_rate * exec_times[bsize] / bsize / 1000 for bsize in bsizes])

        for t in range(TIME_STEP, slo + 1, TIME_STEP):
            min_gpus[final_task.id][t] = min(
                [(k, _min_gpu_single(final_task.model_data, task_arrival_rates[final_task.id], k,
                                     task_worker_sizes[final_task.id]))
                 for k in range(TIME_STEP, t + 1, TIME_STEP)],
                key=lambda x: x[1])
        
        # reverse traverse tree to find remaining SLO splits
        computed_task_ids = set([final_task.id])
        rem_tasks = [t for t in final_task.prev_tasks if all(nt.id in computed_task_ids for nt in t.next_tasks)]
        while rem_tasks:
            for task in rem_tasks:
                for t in range(TIME_STEP, slo + 1, TIME_STEP):
                    min_gpus[task.id][t] = min(
                        [(k, 
                          _min_gpu_single(task.model_data, task_arrival_rates[task.id], k,
                                          task_worker_sizes[task.id]) + \
                          (np.inf if t - k < TIME_STEP else
                           min(sum(min_gpus[v.id][t_prime][1] for v in task.next_tasks)
                               for t_prime in range(TIME_STEP, t - k + 1, TIME_STEP))))
                         for k in list(range(TIME_STEP, t + 1, TIME_STEP))],
                        key=lambda x: x[1])
                computed_task_ids.add(task.id)
            rem_tasks = set([pt for t in rem_tasks for pt in t.prev_tasks 
                             if pt.id not in computed_task_ids and all(pt_nt.id in computed_task_ids for pt_nt in pt.next_tasks)])

        slos = {}

        # walk the DAG in topological order, reading off the stage budget the DP
        # chose for whatever job SLO is left after the stage's predecessors. A
        # join cannot start until every incoming branch is done, so the budget
        # consumed ahead of it is the MAX over its predecessors, not the budget of
        # whichever branch happens to be visited last.
        consumed = {}
        ready = list(workflow.initial_tasks)
        while ready:
            next_ready = []
            for task in ready:
                if task.id in slos:
                    continue

                spent = max([consumed[pt.id] for pt in task.prev_tasks], default=0)
                subtree_slo = (slo - spent) // TIME_STEP * TIME_STEP

                if subtree_slo < TIME_STEP:
                    raise ValueError(
                        f"Job SLO {slo}ms is too small to split across workflow "
                        f"{workflow.id}: no budget left for task {task.id}")

                task_slo = min_gpus[task.id][subtree_slo][0]
                slos[task.id] = (task_slo,
                                 cls._max_feasible_batch_size(task.model_data,
                                                              task_worker_sizes[task.id],
                                                              task_slo))
                consumed[task.id] = spent + task_slo

                next_ready.extend([t for t in task.next_tasks
                                   if t.id not in slos and all(pt.id in slos for pt in t.prev_tasks)])

            ready = next_ready

        assert(len(slos) == len(workflow.tasks))

        # The DP stops buying budget as soon as a stage's GPU cost bottoms out, which
        # can leave most of the job SLO handed to no stage at all: in Nexus the
        # leftover is reclaimed downstream, where the bin packer spends a stage's
        # budget on the largest batch that fits it, but nothing here reclaims it.
        # Scale every stage by the same factor so the longest root to leaf path lands
        # on the job SLO, keeping the proportions the DP chose but spending all of it.
        critical_path = max(consumed.values())
        assert(critical_path <= slo)

        if critical_path > 0 and critical_path < slo:
            scale = slo / critical_path
            for tid, (task_slo, _) in slos.items():
                # keep stage budgets on the DP's TIME_STEP grid, rounding down so the
                # scaled path cannot overshoot the job SLO
                scaled_slo = int(task_slo * scale) // TIME_STEP * TIME_STEP
                slos[tid] = (scaled_slo,
                             cls._max_feasible_batch_size(workflow.tasks[tid].model_data,
                                                          task_worker_sizes[tid],
                                                          scaled_slo))

        return slos


    @classmethod
    def redistribute_task_slos(cls, time: float, simulation, workflow: Workflow, slos: dict[int, int], realloc_amt_ms: float):
        last_slo_update_time = simulation.scheduler.task_slo_log[simulation.scheduler.task_slo_log["workflow_id"]==workflow.id]["time"].max()
        if time - last_slo_update_time < 1000:
            return # at least 1s must pass from last update

        drop_df = simulation.task_drop_log[(simulation.task_drop_log["workflow_id"]==workflow.id) & \
                                                (simulation.task_drop_log["drop_time"] <= time) & \
                                                (simulation.task_drop_log["drop_time"] > last_slo_update_time)]
        
        task_drop_rates = {}
        for task in workflow.tasks.values():
            drop_rate = (drop_df["task_id"]==task.id).sum() / (time - last_slo_update_time) * 1000
            task_drop_rates[task.id] = drop_rate

        print("WF DROP RATES: ", task_drop_rates)

        MAX_ALLOWABLE_DROP_RATE = 2

        # TODO
        if workflow.id == 1:
            unsat_tasks = sorted([task for task in workflow.tasks.values() if task_drop_rates[task.id] > MAX_ALLOWABLE_DROP_RATE],
                             key=lambda task: task_drop_rates[task.id],
                             reverse=True)
            
            sat_tasks = sorted([task for task in workflow.tasks.values() if task not in unsat_tasks],
                    key=lambda task: slos[task.id][0],
                    reverse=True)
        elif workflow.id == 4:
            unsat_tasks = sorted([task for task in workflow.tasks.values() if task.id not in [3,4] and task_drop_rates[task.id] > MAX_ALLOWABLE_DROP_RATE],
                             key=lambda task: task_drop_rates[task.id] if task.id != 2 else max(task_drop_rates[3] + task_drop_rates[4], task_drop_rates[2]),
                             reverse=True)
            
            sat_tasks = sorted([task for task in workflow.tasks.values() if task not in unsat_tasks and task.id not in [3,4]],
                    key=lambda task: slos[task.id][0],
                    reverse=True)
        elif workflow.id == 5 or workflow.id == 0:
            unsat_tasks = sorted([task for task in workflow.tasks.values() if task.id != 0 and task_drop_rates[task.id] > MAX_ALLOWABLE_DROP_RATE],
                                key=lambda task: task_drop_rates[task.id] if task.id != 1 else max(task_drop_rates[0], task_drop_rates[1]),
                                reverse=True)
            
            # tasks with drop rate < threshold in desc magnitude of SLO
            sat_tasks = sorted([task for task in workflow.tasks.values() if task not in unsat_tasks and task.id != 0],
                            key=lambda task: slos[task.id][0],
                            reverse=True)
        else:
            raise NotImplementedError()

        if not unsat_tasks and workflow.id != 4:
            return
        
        if not sat_tasks and workflow.id != 4:
            return
        
        if workflow.id == 4:
            if not sat_tasks and task_drop_rates[3] > MAX_ALLOWABLE_DROP_RATE and \
                task_drop_rates[4] > MAX_ALLOWABLE_DROP_RATE:
                return
            
            if not unsat_tasks and task_drop_rates[3] <= MAX_ALLOWABLE_DROP_RATE and \
                task_drop_rates[4] <= MAX_ALLOWABLE_DROP_RATE:
                return
        
        for sat_task in sat_tasks:
            if slos[sat_task.id][0] < realloc_amt_ms:
                continue
            
            new_sat_task_slo = slos[sat_task.id][0] - realloc_amt_ms
            if sat_task.model_data.batch_exec_times[24][1] > new_sat_task_slo:
                continue

            slos[sat_task.id] = (new_sat_task_slo,
                 max([bsize for bsize in range(1,sat_task.model_data.max_batch_size+1) 
                      if sat_task.model_data.batch_exec_times[24][bsize] <= new_sat_task_slo]))
            
            unsat_task = unsat_tasks.pop(0)

            new_unsat_task_slo = slos[unsat_task.id][0] + realloc_amt_ms
            slos[unsat_task.id] = \
                (new_unsat_task_slo,
                 max([bsize for bsize in range(1,unsat_task.model_data.max_batch_size+1) 
                      if unsat_task.model_data.batch_exec_times[24][bsize] <= new_unsat_task_slo]))
            
            # TODO
            if workflow.id in [0, 5]:
                if sat_task.id == 1:
                    task_0 = [t for t in workflow.tasks.values() if t.id ==0][0]
                    slos[0] = \
                        (new_sat_task_slo,
                        max([bsize for bsize in range(1,task_0.model_data.max_batch_size+1) 
                            if task_0.model_data.batch_exec_times[24][bsize] <= new_sat_task_slo]))
                
                if unsat_task.id == 1:
                    task_0 = [t for t in workflow.tasks.values() if t.id ==0][0]
                    slos[0] = \
                        (new_unsat_task_slo,
                        max([bsize for bsize in range(1,task_0.model_data.max_batch_size+1) 
                            if task_0.model_data.batch_exec_times[24][bsize] <= new_unsat_task_slo]))
                    
            elif workflow.id == 4:
                if sat_task.id == 2:
                    task_3 = [t for t in workflow.tasks.values() if t.id ==3][0]
                    slo3 = slos[3][0] - 2.5
                    slos[3] = \
                        (slo3,
                        max([bsize for bsize in range(1,task_3.model_data.max_batch_size+1) 
                            if task_3.model_data.batch_exec_times[24][bsize] <= slo3]))
                    
                    task_4 = [t for t in workflow.tasks.values() if t.id ==4][0]
                    slo4 = slos[4][0] - 2.5
                    slos[4] = \
                        (slo4,
                        max([bsize for bsize in range(1,task_4.model_data.max_batch_size+1) 
                            if task_4.model_data.batch_exec_times[24][bsize] <= slo4]))

                if unsat_task.id == 2:
                    task_3 = [t for t in workflow.tasks.values() if t.id ==3][0]
                    slo3 = slos[3][0] + 2.5
                    slos[3] = \
                        (slo3,
                        max([bsize for bsize in range(1,task_3.model_data.max_batch_size+1) 
                            if task_3.model_data.batch_exec_times[24][bsize] <= slo3]))
                    
                    task_4 = [t for t in workflow.tasks.values() if t.id ==4][0]
                    slo4 = slos[4][0] + 2.5
                    slos[4] = \
                        (slo4,
                        max([bsize for bsize in range(1,task_4.model_data.max_batch_size+1) 
                            if task_4.model_data.batch_exec_times[24][bsize] <= slo4]))
        
            if not unsat_tasks:
                break

        if workflow.id == 4:
            if task_drop_rates[3] <= MAX_ALLOWABLE_DROP_RATE and \
                task_drop_rates[4] <= MAX_ALLOWABLE_DROP_RATE:
                return

            if task_drop_rates[3] > MAX_ALLOWABLE_DROP_RATE and \
                task_drop_rates[4] > MAX_ALLOWABLE_DROP_RATE:
                return
            
            if task_drop_rates[3] > MAX_ALLOWABLE_DROP_RATE and \
                task_drop_rates[4] <= MAX_ALLOWABLE_DROP_RATE:

                slo3 = slos[3][0] + realloc_amt_ms
                slo4 = slos[4][0] - realloc_amt_ms

            if task_drop_rates[3] <= MAX_ALLOWABLE_DROP_RATE and \
                task_drop_rates[4] > MAX_ALLOWABLE_DROP_RATE:

                slo3 = slos[3][0] - realloc_amt_ms
                slo4 = slos[4][0] + realloc_amt_ms

            task_3 = [t for t in workflow.tasks.values() if t.id ==3][0]
            task_4 = [t for t in workflow.tasks.values() if t.id ==4][0]
            
            slos[3] = \
                (slo3,
                max([bsize for bsize in range(1, task_3.model_data.max_batch_size+1) 
                    if task_3.model_data.batch_exec_times[24][bsize] <= slo3]))
            slos[4] = \
                (slo4,
                max([bsize for bsize in range(1, task_4.model_data.max_batch_size+1) 
                    if task_4.model_data.batch_exec_times[24][bsize] <= slo4]))