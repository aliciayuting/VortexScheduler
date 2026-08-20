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
    def generate_task_slos(cls, workflow: Workflow, slo: float,
                           task_arrival_rates: dict[int, float]) -> dict[int, tuple[float, int]]:
        """Split job SLO over workflow tasks.

        Args:
            workflow: Workflow to split SLO for
            slo: Job-level SLO to split across tasks (ms)
            task_arrival_rates: Task ID -> per worker arrival rate (qps), as
            produced by [get_task_arrival_rates]

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

        def _min_gpu_single(model, req_rate, k):
            bsizes = [b for b in range(1, model.max_batch_size + 1) if model.batch_exec_times[24][b] <= k]
            if not bsizes:
                return np.inf
            return min([req_rate * model.batch_exec_times[24][bsize] / bsize / 1000 for bsize in bsizes])

        for t in range(TIME_STEP, slo + 1, TIME_STEP):
            min_gpus[final_task.id][t] = min(
                [(k, _min_gpu_single(final_task.model_data, task_arrival_rates[final_task.id], k)) 
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
                          _min_gpu_single(task.model_data, task_arrival_rates[task.id], k) + \
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
                feasible_bsizes = [b for b in range(1, task.model_data.max_batch_size + 1)
                                   if task.model_data.batch_exec_times[24][b] <= task_slo]

                slos[task.id] = (task_slo, max(feasible_bsizes) if feasible_bsizes else 1)
                consumed[task.id] = spent + task_slo

                next_ready.extend([t for t in task.next_tasks
                                   if t.id not in slos and all(pt.id in slos for pt in t.prev_tasks)])

            ready = next_ready

        assert(len(slos) == len(workflow.tasks))

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