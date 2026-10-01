import numpy as np

import core.configs.gen_config as gcfg

from core.data_models.workflow import Workflow


class NexusSLOSplitter:
    """SLO split algorithm adapted from pg. 330 (sec 6.2): 
    https://homes.cs.washington.edu/~arvind/papers/nexus.pdf
    """

    # granularity of SLOs in ms
    TIME_STEP = 5

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
    def _region_topo_order(cls, region: set[int], workflow: Workflow) -> list[int]:
        """Returns the task IDs of [region] in topological order, following only the
        edges that stay inside the region.
        """
        indeg = {tid: len([pt for pt in workflow.tasks[tid].prev_tasks if pt.id in region])
                 for tid in region}

        order = []
        ready = sorted([tid for tid, d in indeg.items() if d == 0])
        while ready:
            tid = ready.pop(0)
            order.append(tid)
            for nt in workflow.tasks[tid].next_tasks:
                if nt.id not in region:
                    continue
                indeg[nt.id] -= 1
                if indeg[nt.id] == 0:
                    ready.append(nt.id)

        assert(len(order) == len(region))

        return order


    @classmethod
    def _region_duration(cls, region: set[int], workflow: Workflow,
                         budgets: dict[int, float]) -> float:
        """Returns the largest total budget along any path through [region], i.e.
        how long the region takes when every stage spends its whole budget.
        """
        longest = {}
        for tid in cls._region_topo_order(region, workflow):
            spent = max([longest[pt.id] for pt in workflow.tasks[tid].prev_tasks
                         if pt.id in region], default=0)
            longest[tid] = spent + budgets[tid]

        return max(longest.values(), default=0)


    @classmethod
    def _region_branches(cls, region: set[int], workflow: Workflow) -> list[set[int]]:
        """Splits [region] into the independent branches it is made of: its weakly
        connected components, which for a fork-join graph are exactly the parallel
        paths between the fork and the join that bound the region.
        """
        branches = []
        remaining = set(region)
        while remaining:
            branch = {min(remaining)}
            frontier = list(branch)
            while frontier:
                task = workflow.tasks[frontier.pop()]
                for nb in task.prev_tasks + task.next_tasks:
                    if nb.id in region and nb.id not in branch:
                        branch.add(nb.id)
                        frontier.append(nb.id)

            branches.append(branch)
            remaining -= branch

        return branches


    @classmethod
    def _series_elements(cls, region: set[int],
                         workflow: Workflow) -> list[tuple[str, object]]:
        """Decomposes [region] into the sequence of elements that every path through
        it crosses, in order.

        An element is either ("task", task_id) for a stage that lies on every path
        through the region, or ("parallel", {task ids}) for the group of stages
        sitting between two such stages, which different paths cross differently.
        Since every path crosses every element, the region's duration is the SUM over
        elements, while a parallel element's own duration is the MAX over its
        branches: the two rules a proportional stretch has to respect.

        Found by sweeping the region in topological order and tracking how many edges
        cross the cut just behind the sweep. A stage lies on every path exactly when
        the only edges crossing the cut in front of it are its own incoming ones.
        """
        indeg, outdeg = {}, {}
        for tid in region:
            task = workflow.tasks[tid]
            # a region source is entered once, from the fork bounding the region;
            # a region sink likewise leaves once, into the join
            indeg[tid] = max(1, len([pt for pt in task.prev_tasks if pt.id in region]))
            outdeg[tid] = max(1, len([nt for nt in task.next_tasks if nt.id in region]))

        elements = []
        group = []
        cut = len([tid for tid in region
                   if not any(pt.id in region for pt in workflow.tasks[tid].prev_tasks)])

        for tid in cls._region_topo_order(region, workflow):
            if cut == indeg[tid]:
                if group:
                    elements.append(("parallel", set(group)))
                    group = []
                elements.append(("task", tid))
            else:
                group.append(tid)

            cut += outdeg[tid] - indeg[tid]

        if group:
            elements.append(("parallel", set(group)))

        return elements


    @classmethod
    def _stretch_region(cls, region: set[int], workflow: Workflow,
                        budgets: dict[int, float], target: float):
        """Scales the budgets of [region] in place so the region's longest path takes
        [target] ms, keeping the proportions the split chose.

        Recurses on the region's series-parallel structure: the elements of a series
        share [target] in proportion to what they already take, and each branch of a
        parallel element is then stretched to that element's whole share, which is
        what makes parallel branches come out equal.
        """
        elements = cls._series_elements(region, workflow)
        durations = [budgets[payload] if kind == "task"
                     else cls._region_duration(payload, workflow, budgets)
                     for kind, payload in elements]

        total = sum(durations)
        if total <= 0:
            return

        # the region's longest path is the sum over its series elements, so a region
        # is never asked to fit in less than it already takes
        assert(target >= total)
        scale = target / total

        for (kind, payload), duration in zip(elements, durations):
            # keep budgets on the DP's grid, rounding down so the stretched path
            # cannot overshoot [target]
            share = int(duration * scale) // cls.TIME_STEP * cls.TIME_STEP

            if kind == "task":
                budgets[payload] = share
            else:
                for branch in cls._region_branches(payload, workflow):
                    cls._stretch_region(branch, workflow, budgets, share)


    @classmethod
    def equalize_parallel_stage_slos(cls, workflow: Workflow,
                                     slos: dict[int, tuple[float, int]],
                                     task_worker_sizes: dict[int, int]) -> dict[int, tuple[float, int]]:
        """Spreads the slack the split leaves on parallel branches over the stages of
        those branches, so every branch of a fan-in takes as long as the longest one.

        The split sizes each stage on its own GPU cost in isolation, so on a fan-in
        the cheap branch is handed a much smaller budget than the branch beside it.
        The join waits for both branches regardless, so the difference is slack that
        no stage is allowed to spend: the cheap branch's stage deadlines fire early
        for no reason, and the batch caps they imply are smaller than the job SLO
        actually requires.

        The slack cannot simply be handed to each stage that has some, because stages
        on nested branches draw on the same room as the fork above them and would
        spend it twice. It is instead pushed down the workflow's series-parallel
        structure (see [_stretch_region]), which gives each branch of a fan-in the
        same total while dividing that total among the branch's own stages in the
        proportions the split chose for them. The longest path is left exactly as it
        was, so no deadline moves outward and the job SLO still bounds the split.

        Args:
            workflow: Workflow whose split to stretch
            slos: (stage SLO, max batch size) per task ID, as produced by
            [generate_task_slos]
            task_worker_sizes: Task ID -> worker memory size whose batch exec times
            to plan against (GB), as produced by [get_task_worker_sizes]

        Returns:
            slos: (stage SLO, max batch size) per task ID, stretched in place
        """
        region = set(workflow.tasks.keys())
        budgets = {tid: slo for tid, (slo, _) in slos.items()}

        critical_path = cls._region_duration(region, workflow, budgets)
        cls._stretch_region(region, workflow, budgets, critical_path)

        # stretching only ever spends slack that was already inside the workflow
        assert(cls._region_duration(region, workflow, budgets) <= critical_path)

        for tid, budget in budgets.items():
            assert(budget >= slos[tid][0])
            slos[tid] = (budget,
                         cls._max_feasible_batch_size(workflow.tasks[tid].model_data,
                                                      task_worker_sizes[tid],
                                                      budget))

        return slos


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

        TIME_STEP = cls.TIME_STEP

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

        if gcfg.EQUALIZE_PARALLEL_STAGE_SLOS:
            slos = cls.equalize_parallel_stage_slos(workflow, slos, task_worker_sizes)

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