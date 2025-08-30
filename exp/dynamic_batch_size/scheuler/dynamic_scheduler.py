from collections import deque
# from main import Request
import math
import csv
import os
import gurobipy as gp
from gurobipy import GRB
import logging
from utils import SortedQueue


class DynamicScheduler:
    max_batch_size: int
    batch_runtimes: dict
    slo: float
    scheduled_batches: dict[int, list[int]]

    def __init__(self, max_batch_size: int, batch_runtimes: dict, logger=None):
        # Use the logger passed from main, or create one for this module
        if logger is not None:
            self.logger = logger
        else:
            # Use the same logger name as main.py to inherit the configuration
            self.logger = logging.getLogger('__main__')

        # self.slo = slo
        self.max_batch_size = max_batch_size
        self.current_solution = None
        self.batch_runtimes = batch_runtimes
        self.scheduled_batches = []

    def get_batch_duration(self, s_k):
        """Function f that maps batch size to runtime"""
        return self.batch_runtimes.get(s_k, 0)

    def preempt(self, current_batch: SortedQueue, queue: SortedQueue, current_time: float, batch_finish_time: float) -> bool:
        N = len(queue) + len(current_batch)
        assert len(queue) > 0 and len(current_batch) > 0, f"Queue and current batch size is not greater than 0: {len(queue)} and {len(current_batch)}"
        assert batch_finish_time != math.inf, f"Batch finish time is not math.inf: {batch_finish_time}"
        
        # check the performance of finishing the current batch and schedule the remaining requests
        self.logger.info(f"Checking the performance of finishing the current batch and schedule the remaining requests")
        queue_copy = queue.copy()
        while len(queue_copy) > 0 and queue_copy[0].deadline < batch_finish_time + self.get_batch_duration(1):
            queue_copy.pop()
        if len(queue_copy) == 0:
            future_num_satisfied = len(current_batch)
            future_finish_time = batch_finish_time
        else:
            _, future_solution = self.schedule(SortedQueue(), queue_copy, batch_finish_time)
            future_obj = future_solution["obj"]
            future_max_completion_time = future_solution["max_completion_time"]
            future_num_satisfied = len(current_batch) + future_obj
            assert future_num_satisfied >= 0, f"{len(current_batch)} + {future_obj} = {future_num_satisfied}"
            future_finish_time = future_max_completion_time + batch_finish_time

        # check the performance of preempting the current batch and schedule the remaining requests
        self.logger.info(f"Checking the performance of preempting the current batch")
        preempt_batch = SortedQueue()
        queue_copy = queue.copy()
        queue_copy.extend(current_batch)        
        _, preempt_solution = self.schedule(preempt_batch, queue_copy, current_time)
        preempt_num_satisfied = preempt_solution["obj"]
        assert preempt_num_satisfied >= 0, f"{preempt_num_satisfied}"

        preempt_finish_time = current_time + preempt_solution["max_completion_time"]

        self.logger.info(f"Preempting compare: current plan ({future_num_satisfied}, {future_finish_time}), preempt plan ({preempt_num_satisfied}, {preempt_finish_time})")


        if preempt_num_satisfied > future_num_satisfied or (preempt_num_satisfied == future_num_satisfied and preempt_finish_time < future_finish_time):
            while len(current_batch) > 0:
                req = current_batch.pop()
                req.preempt()
                queue.append(req)

            for req in preempt_batch:
                queue.remove(req)
                req.schedule(current_time, len(preempt_batch), self.get_batch_duration(len(preempt_batch)))
                current_batch.append(req)

            assert len(current_batch) + len(queue) == N, f"Current batch and queue size is not equal to the original size: {len(current_batch)} + {len(queue)} and {N}"
            assert len(current_batch) == len(preempt_batch), f"Current batch size is not equal to the preempted batch size: {len(current_batch)} and {len(preempt_batch)}"
            return True

        return False


    def schedule(self, current_batch: SortedQueue, queue: SortedQueue, current_time: float) -> float:
        N = len(queue)
        B = self.max_batch_size
        req_deadlines = {req.id: req.deadline - current_time for req in queue}
        d = list(req_deadlines.values())
        self.logger.info(f"deadline: {req_deadlines}")
        self.logger.info(f"base latency: {self.batch_runtimes}")
        
        assert (len(current_batch) == 0, f"Without preemption, schedule should only happen when the current batch is finished (current_batch_size: {len(current_batch)})")

        if N == 0:
            return math.inf, None

        result = solve_ilp(N, B, d, self.batch_runtimes)

        if result is None:
            assert False, "No solution found"
    
        # Print assignments for each j
        self.logger.info(f"satisfied: {result['obj']} / {N}, max completion time: {result['max_completion_time']}")
        cumulated_latency = 0
        for j in range(N):
            batsh_size = result['s'][j]
            batch_latency = self.batch_runtimes.get(batsh_size, 0)
            self.logger.info(f"Position j={j} (batch size s[{j}]={result['s'][j]}), cumulated latency = {cumulated_latency}, batch_finish_time = {batch_latency}")
            batch = []
            for i in range(N):
                if result["x"][(i, j)] == 1:
                    self.logger.info(f"\tRequest {queue[i].id}: time remaining: {queue[i].deadline - current_time - cumulated_latency}")
                    batch.append(queue[i].id)
            cumulated_latency += batch_latency
            self.scheduled_batches.append(batch)

        req_ids = set([queue[i].id for i in range(N) if result["x"][(i,0)] == 1])

        for id in req_ids:
            req = queue.get_by_id(id)
            current_batch.append(req)
            queue.remove(req)
        
        assert len(current_batch) + len(queue) == N, f"Current batch and queue size is not equal to the original size: {len(current_batch)} + {len(queue)} and {N}"                
        assert len(current_batch) == result["s"][0], f"Current batch size is not equal to the scheduled batch size: {len(current_batch)} and {result['s'][0]}"
        assert len(current_batch) <= self.max_batch_size, f"Current batch size is greater than the max batch size: {len(current_batch)} and {self.max_batch_size}"
        assert len(current_batch) > 0, "Current batch size is 0"
        

        return math.inf, result

    def offline_schedule(self, current_batch: SortedQueue, queue: SortedQueue, current_time: float, finished_reqs: list) -> float:
        current_time = 0
        queue_copy = queue.copy()
        self.schedule(current_batch, queue_copy, current_time)

        num_batch = 0
        for num_batch, batch in enumerate(self.scheduled_batches):
            # self.logger.info(f"####### Batch: {num_batch} Time: {currnet_time} #######")
            # deadline = {req.id: req.deadline - currnet_time for req in queue}
            # self.logger.info(f"deadline: {deadline}")

            batch_size = len(batch)
            if batch_size > 0:
                batch_time = self.batch_runtimes[batch_size]
                # self.logger.info(f"current batch (size: {batch_size}, time: {batch_time}) {batch}")

                for id in batch:
                    req = queue.get_by_id(id)
                    req.schedule(current_time, batch_size, self.batch_runtimes[batch_size])
                    finished_reqs.append(req)
                    queue.remove(req)
                    # self.logger.info(f"\tRequest {req.id}: time remaining: {req.deadline - currnet_time}")


                current_time += batch_time
                num_batch += 1

        for req in queue:
            req.get_dropped(current_time)
            finished_reqs.append(req)


            self.logger.info(f"--------------------------------")

        return current_time



def solve_ilp(N, B, d, batch_runtimes):
    """
    Solve the ILP:
    min sum_i Ind{d_i < t_i}
    s.t. sum_j x_{i,j} = 1, forall i
        sum_i x_{i,j} = s_j, forall j  
        s_j <= B, forall j
        e_j = sum_{k=1}^j f(s_k), forall j
        t_i = sum_j x_{i,j} e_j, forall i
    """
    model = gp.Model("IndicatorMinimization")

    for i in range(N):
        assert d[i] >= 0, f"Deadline is negative: {d[i]}"

    # Suppress Gurobi output
    model.setParam('OutputFlag', 0)
    # Set number of threads (cores) to use
    model.setParam('Threads', 4)  # Adjust this number as needed

    # Variables
    x = model.addVars(N, N, vtype=GRB.BINARY, name="x")  # x_{i,j}
    s = model.addVars(N, vtype=GRB.INTEGER, lb=0, ub=B, name="s")  # s_j
    e = model.addVars(N, vtype=GRB.CONTINUOUS, name="e")  # e_j
    t = model.addVars(N, vtype=GRB.CONTINUOUS, name="t")  # t_i
    z = model.addVars(N, vtype=GRB.BINARY, name="z")  # Indicator: z_i = 1 if d_i < t_i
    # position_used = model.addVars(N, vtype=GRB.BINARY, name="position_used")  # position_used[j] = 1 if s[j] > 0

    # Big-M value for indicator constraints
    M = max(d) + sum(batch_runtimes.get(s_val, 0) for s_val in range(B + 1)) * N

    # Constraint 1: sum_j x_{i,j} <= 1, forall i (allow not scheduling)
    for i in range(N):
        model.addConstr(gp.quicksum(x[i, j] for j in range(N)) <= 1, name=f"assign_row_{i}")

    # Constraint 2: sum_i x_{i,j} = s_j, forall j
    for j in range(N):
        model.addConstr(gp.quicksum(x[i, j] for i in range(N)) == s[j], name=f"assign_col_{j}")

    # Constraint 3: s_j <= B (handled by variable bounds)

    # Constraint 3.5: Sequential batch positions - if s[j] = 0, then s[j'] = 0 for all j' > j
    for j in range(N-1):
        for j_prime in range(j+1, N):
            # If s[j] = 0, then s[j_prime] = 0
            # This is equivalent to: s[j_prime] <= M * s[j] where M is a large constant
            model.addConstr(s[j_prime] <= B * s[j], name=f"sequential_{j}_{j_prime}")

    # Model constraint 4 with fitted linear model.
    for j in range(N):
        if j == 0:
            model.addConstr(
                e[j] == 21.576762387982544 * s[j] + 14.724000781249913
            )
        else:
            model.addConstr(
                e[j]
                == e[j - 1]
                + (21.576762387982544 * s[j] + 14.724000781249913),
                name=f"e_{j}",
            )

    # Constraint 5: t_i = sum_j x_{i,j} e_j, forall i (if scheduled), otherwise t_i = large_value
    for i in range(N):
        # If request i is scheduled, t_i = sum_j x[i,j] * e[j]
        # If request i is not scheduled, t_i = M (large value to ensure z[i] = 1)
        model.addConstr(
            t[i] == gp.quicksum(x[i, j] * e[j] for j in range(N)) + M * (1 - gp.quicksum(x[i, j] for j in range(N))), 
            name=f"t_{i}"
        )

    # Indicator constraint: z_i = 1 if d_i < t_i
    eps = 1e-6  # Small epsilon for strict inequality
    for i in range(N):
        # If t_i > d_i + eps, then z_i = 1
        model.addConstr(t[i] - d[i] + M * (1 - z[i]) >= eps, name=f"ind_lb_{i}")
        # If t_i <= d_i, then z_i = 0
        model.addConstr(t[i] - d[i] <= M * z[i], name=f"ind_ub_{i}")
    
    # Additional constraint: if z[i] = 1 (deadline cannot be met), then don't schedule request i
    for i in range(N):
        # If z[i] = 1, then sum_j x[i,j] = 0 (don't schedule)
        # If z[i] = 0, then sum_j x[i,j] = 1 (must schedule)
        model.addConstr(gp.quicksum(x[i, j] for j in range(N)) <= 1 - z[i], name=f"schedule_if_meet_deadline_{i}")


    # Two-stage optimization: first minimize deadline violations, then minimize positions used
    # Stage 1: Minimize deadline violations
    model.setObjective(gp.quicksum(z[i] for i in range(N)), GRB.MINIMIZE)
    
    # Solve first stage
    model.optimize()

    optimal_deadline_violations = model.ObjVal
    
    # if model.status == GRB.OPTIMAL:
    #     

    #     # Stage 2: Add constraint for optimal deadline violations and minimize completion time
    #     model.addConstr(gp.quicksum(z[i] for i in range(N)) == optimal_deadline_violations, name="optimal_deadline_constraint")
        
    #     # Change objective to minimize the completion time of all scheduled requests
    #     # The completion time is the end time of the last batch position that has scheduled requests
    #     # We can use e[j] which represents the cumulative time up to position j
    #     # We need to find the maximum e[j] where s[j] > 0 (i.e., where there are scheduled requests)
        
    #     # Add a variable to represent the maximum completion time
    #     max_completion_time = model.addVar(vtype=GRB.CONTINUOUS, name="max_completion_time")
        
    #     # Constraint: max_completion_time >= e[j] for all j where s[j] > 0
    #     for j in range(N):
    #         # If s[j] > 0, then max_completion_time >= e[j]
    #         # Since s[j] is an integer >= 0, we need a binary indicator to check if s[j] > 0
    #         indicator = model.addVar(vtype=GRB.BINARY, name=f"indicator_{j}")
    #         # indicator = 1 if s[j] >= 1, 0 otherwise
    #         model.addConstr(indicator <= s[j], name=f"indicator_lb_{j}")
    #         model.addConstr(s[j] <= M * indicator, name=f"indicator_ub_{j}")
    #         # If indicator = 1, then max_completion_time >= e[j]
    #         model.addConstr(max_completion_time >= e[j] - M * (1 - indicator), name=f"max_completion_time_{j}")
        
    #     # Set objective to minimize the maximum completion time
    #     model.setObjective(max_completion_time, GRB.MINIMIZE)
        
    #     # Solve second stage
    #     model.optimize()

    if model.status == GRB.OPTIMAL:
        solution = {
            "x": {(i, j): round(x[i, j].X) for i in range(N) for j in range(N)},
            "s": [round(s[j].X) for j in range(N)],
            "e": [e[j].X for j in range(N)],
            "t": [t[i].X for i in range(N)],
            "z": [round(z[i].X) for i in range(N)],
            "obj": N - round(optimal_deadline_violations),
            # "max_completion_time": max_completion_time.X
             "max_completion_time": 0
        }
        return solution
    else:
        print(f"Model status: {model.status}")
        return None

