import core.configs.gen_config as gcfg

from core.data_models.model_data import ModelData
from core.data_models.workflow import Workflow
from core.allocation import ModelAllocation
from schedulers.algo.capacity_algo import get_instance_throughput

import uuid
from collections import Counter

try:
    import gurobipy as gp
    from gurobipy import GRB
except ImportError:
    print("Gurobipy not found")

class VortexPlanner:

    # total node count -> allocation
    nodes_to_worker_cfgs: dict[int, ModelAllocation] = {}

    @classmethod
    def _get_allocation(cls, simulation, node_count: int, workflow: Workflow, slo: float) -> ModelAllocation:
        assert(node_count >= gcfg.MIN_NUM_NODES and node_count <= gcfg.MAX_NUM_NODES)

        # a model with no memory footprint (e.g. an aggregation step) needs no
        # partition of its own, so it is left out of the placement and co-located
        # on the first worker instead
        colocated = sorted(m.id for m in workflow.get_models() if m.size == 0)
        models = [m.id for m in workflow.get_models() if m.size > 0]
        configs = [sz // 10**6 for sz in gcfg.VALID_WORKER_SIZES]

        # a job issues one task per stage, and a workflow may run several stages on
        # one model (e.g. fanning a detection out over many crops), so a model's
        # throughput in jobs is its task throughput divided by its tasks per job
        tasks_per_job = Counter(t.model_data.id for t in workflow.tasks.values()
                                if t.model_data)

        # choose max bsize s.t. SLO respected if bsize=1 for rem. tasks
        max_batch_sizes = {}
        for mem_size in configs:
            max_batch_sizes[mem_size] = {}
            for model in workflow.get_models():
                if mem_size not in model.batch_exec_times:
                    continue

                min_proc_without_model = workflow.get_min_processing_time() - model.batch_exec_times[mem_size][1]
                max_batch_sizes[mem_size][model.id] = max(
                    i for i in range(1, model.max_batch_size+1, 1)
                    if model.batch_exec_times[mem_size][i] < slo - min_proc_without_model)
        
        # jobs/s one instance sustains on each partition size, at whichever batch
        # size up to the SLO-feasible maximum serves the most tasks per ms
        throughput = {
            m.id: {
                mem_size: get_instance_throughput(m, mem_size, max_batch_sizes[mem_size][m.id]) * \
                   1000 / tasks_per_job[m.id]
                   for mem_size in m.batch_exec_times.keys()}
            for m in workflow.get_models() if m.id in models
        }

        valid_layouts = [[24], [12, 12], [6, 6, 6, 6], [12, 6, 6]]
        
        
        nodes = [f"GPU{i}" for i in range(node_count)]

        def solve_leximin(locked_lower_bounds):
            m = gp.Model("Leximin_Level")
            x, y = {}, {}
            T_m = {}
            Z = m.addVar(lb=0, vtype=GRB.CONTINUOUS, name="Z")

            for model in models:
                for node in nodes:
                    for c in configs:
                        if c in throughput[model]:
                            x[model, node, c] = m.addVar(vtype=GRB.INTEGER, name=f"x_{model}_{node}_{c}")

            for node in nodes:
                for lid, layout in enumerate(valid_layouts):
                    y[node, lid] = m.addVar(vtype=GRB.BINARY, name=f"y_{node}_{lid}")

            for model in models:
                T_m[model] = m.addVar(lb=0, vtype=GRB.CONTINUOUS, name=f"T_{model}")
                m.addConstr(
                    T_m[model] == gp.quicksum(
                        x[model, node, c] * throughput[model][c]
                        for node in nodes for c in configs if (model, node, c) in x
                    )
                )

                if model in locked_lower_bounds:
                    m.addConstr(T_m[model] >= locked_lower_bounds[model])
                else:
                    m.addConstr(Z <= T_m[model])

                m.addConstr(gp.quicksum(
                    x[model, node, c] for node in nodes for c in configs if (model, node, c) in x
                ) >= 1)

            for node in nodes:
                m.addConstr(gp.quicksum(y[node, lid] for lid in range(len(valid_layouts))) == 1)
                for c in configs:
                    m.addConstr(
                        gp.quicksum(
                            x[model, node, c] for model in models if (model, node, c) in x
                        ) <= gp.quicksum(
                            y[node, lid] * layout.count(c)
                            for lid, layout in enumerate(valid_layouts)
                        )
                    )

            m.setObjective(Z, GRB.MAXIMIZE)
            m.setParam("OutputFlag", 0)
            m.optimize()

            status = m.Status

            if m.Status != GRB.OPTIMAL:
                print(f"{node_count} nodes infeasible (status code: {m.Status}), skipping...")
                return None, None

            throughput_vals = {model: T_m[model].X for model in models}
            assignment = {
                (model, node, c): int(x[model, node, c].X)
                for model in models for node in nodes for c in configs
                if (model, node, c) in x and x[model, node, c].X > 0.5
            }
            return throughput_vals, assignment

        # Leximin loop
        locked = {}
        for _ in range(len(models)):
            T_vals, assignment = solve_leximin(locked)
            if assignment == None:
                return None # not feasible

            unlocked = [m for m in models if m not in locked]
            if not unlocked:
                break
            min_model = min(unlocked, key=lambda m: T_vals[m])
            locked[min_model] = T_vals[min_model]

        if not locked:
            return None

        # Print results
        # print(f"\nPipeline Throughput: {min(locked.values()):.2f}")
        # print(" Leximin Model Throughputs:")
        # for model in models:
        #     print(f" {model}: {locked[model]:.2f}")

        # print("\nAssignments:")
        # for (model, node, c), count in assignment.items():
        #     print(f" - Model {model} assigned {count}x to {node} with MIG {c}GB")
        
        # x counts partitions of size c on a node, each hosting one instance, so
        # every unit of it is a worker of its own
        worker_cfgs = {}
        for (model, node, c), count in sorted(assignment.items(), key=lambda kv: -kv[0][2]):
            for _ in range(count):
                worker_cfgs[uuid.uuid4()] = (c, [model])

        if colocated:
            first = next(iter(worker_cfgs))
            size, mids = worker_cfgs[first]
            worker_cfgs[first] = (size, mids + colocated)

        allocation = ModelAllocation(simulation, worker_cfgs, False)
        for model in workflow.get_models():
            # TODO: max batch size per partition size
            allocation.models[model.id].max_batch_size = min(
                max_bsizes[model.id] for max_bsizes in max_batch_sizes.values()
                if model.id in max_bsizes)
            
        return allocation
    
    @classmethod
    def get_allocation(cls, simulation, node_count: int, workflow: Workflow, slo: float) -> ModelAllocation:
        if node_count not in cls.nodes_to_worker_cfgs:
            allocation = cls._get_allocation(simulation, node_count, workflow, slo)
            cls.nodes_to_worker_cfgs[node_count] = allocation
            return allocation
        return cls.nodes_to_worker_cfgs[node_count]
    
    @classmethod
    def does_alloc_sat(cls, simulation, workflow: Workflow, slo: float, node_count: int, arrival_rate: float) -> bool:
        if node_count not in cls.nodes_to_worker_cfgs:
            allocation = cls._get_allocation(simulation, node_count, workflow, slo)
            cls.nodes_to_worker_cfgs[node_count] = allocation

        if cls.nodes_to_worker_cfgs[node_count] == None:
            return False

        max_throughput = 0
        allocation = cls.nodes_to_worker_cfgs[node_count]

        throughputs = {} # task ID -> max throughput
        completed_tasks = set()
        available_tasks = [t for t in workflow.initial_tasks]
        remaining_tasks = [t for t in workflow.tasks.values()]

        while remaining_tasks:
            for task in available_tasks:
                model_count = allocation.count(task.model_data.id)
                
                total_input_rate = min(throughputs[t.id] for t in task.prev_tasks) \
                    if task.prev_tasks else arrival_rate
                input_rate = total_input_rate / model_count
                
                expected_bsize = max(b for b in range(1, allocation.models[task.model_data.id].max_batch_size+1, 1) 
                                        if b == 1 or input_rate >= b / task.model_data.batch_exec_times[24][b] * 1000)
                throughputs[task.id] = min(input_rate, expected_bsize / task.model_data.batch_exec_times[24][expected_bsize] * 1000) * model_count
                
                print("IN: ", input_rate, " OUT: ", throughputs[task.id], " COUNT: ", model_count)

                remaining_tasks.remove(task)
                completed_tasks.add(task.id)

            max_throughput = min(throughputs[t.id] for t in available_tasks)

            available_tasks = []
            for task in remaining_tasks:
                if all(t.id in completed_tasks for t in task.prev_tasks):
                    available_tasks.append(task)  
        
        return max_throughput >= arrival_rate
    
    @classmethod
    def get_min_node_alloc(cls, simulation, workflow: Workflow, slo: float, arrival_rate: float) -> ModelAllocation:
        # init allocations
        for count in range(gcfg.MIN_NUM_NODES, gcfg.MAX_NUM_NODES+1, 1):
            if count not in cls.nodes_to_worker_cfgs:
                allocation = cls._get_allocation(simulation, count, workflow, slo)
                cls.nodes_to_worker_cfgs[count] = allocation

        for count in sorted(cls.nodes_to_worker_cfgs.keys()):
            if cls.does_alloc_sat(simulation, workflow, slo, count, arrival_rate):
                return cls.nodes_to_worker_cfgs[count]
        
        print("No satisfying alloc exists, use largest...")
        return cls.nodes_to_worker_cfgs[count]
    
    @classmethod
    def check_scale(cls, time: float, simulation, workflow: Workflow, slo: float, curr_allocation: ModelAllocation, arrival_rate: float):
        best_alloc = cls.get_min_node_alloc(simulation, workflow, slo, arrival_rate)
        if best_alloc.model_ids == curr_allocation.model_ids:
            return []
        
        best_alloc_wids = sorted(best_alloc.worker_cfgs.keys(),
                                 key=lambda wid: (best_alloc.worker_cfgs[wid][0], 
                                                  sorted(best_alloc.worker_cfgs[wid][1])))
        curr_alloc_wids = sorted(curr_allocation.worker_cfgs.keys(),
                                 key=lambda wid: (curr_allocation.worker_cfgs[wid][0], 
                                                  sorted(curr_allocation.worker_cfgs[wid][1])))

        delta_cfgs: list[tuple[uuid.UUID, tuple[int, list[int]]]] = []
        for i, wid in enumerate(best_alloc_wids):
            if i >= len(curr_alloc_wids):
                delta_cfgs.append((wid, best_alloc.worker_cfgs[wid]))

            elif best_alloc.worker_cfgs[wid] != curr_allocation.worker_cfgs[curr_alloc_wids[i]]:
                delta_cfgs.append((curr_alloc_wids[i], best_alloc.worker_cfgs[wid]))

        if len(curr_alloc_wids) > len(best_alloc_wids):
            delta_cfgs += [(wid, None) for wid in curr_alloc_wids[len(best_alloc_wids):]]

        curr_allocation.apply_delta(time, delta_cfgs)

        return delta_cfgs
