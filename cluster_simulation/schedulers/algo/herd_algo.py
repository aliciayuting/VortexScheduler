from core.config import *
from core.model import Model
from core.workflow import *

from scipy import stats

import gurobipy as gp
from gurobipy import GRB


def _model_affinity(m1: Model, m2: Model) -> float:
    alpha_1, beta_1, r1, p1, err1 = stats.linregress(m1.batch_sizes, m1.batch_exec_times[24])
    alpha_2, beta_2, r2, p2, err2 = stats.linregress(m2.batch_sizes, m2.batch_exec_times[24])

    assert(alpha_1 > 0 and alpha_2 > 0)

    if alpha_2 + beta_2 - beta_1 <= 0:
        return (alpha_1 + beta_1) / alpha_2
    else:
        return min((alpha_1 + beta_1) / alpha_2,
                   max(alpha_1 / (alpha_2 + beta_2 - beta_1), 
                       alpha_1 / alpha_2))


def _get_affinity_sets(all_models: list[Model]) -> list[list[Model]]:
    affinity_sets = []
    while all_models:
        curr_model = all_models.pop()
        affinity_set = [curr_model]
        for model in all_models[::-1]:
            if all(_model_affinity(model, m) <= HERD_K for m in affinity_set):
                affinity_set.append(model)
                all_models.remove(model)
        affinity_sets.append(affinity_set)
    return affinity_sets
    

def get_herd_assignment(task_types: list[tuple[int,int]], all_models: list[Model], 
                        task_tputs: dict[tuple[int,int], float], send_rates: dict[int,float]) -> tuple[list[int], list[tuple[int,int]]]:
    """
        Solves HERD ILP from p. 792 of SHEPHERD paper.
        Returns tuple[list[int], list[tuple[int,int]]] with the no. of GPUs
        in each group and the assignment of streams to groups given as a
        list of (task type index, group index).
    """
    
    affinity_sets = _get_affinity_sets(all_models)

    I = len(task_types)         # req. stream count = GPU needing task type count
    J = len(affinity_sets)      # no. serving groups; >= len(aff sets)
    K = len(all_models)
    C = len(affinity_sets)

    model = gp.Model("HERD")

    x = model.addVars(I, J, vtype=GRB.BINARY, name="x") # x[i][j] == 1 iff stream i -> group j
    z = model.addVars(K, J, vtype=GRB.BINARY, name="z") # z[k][j] == 1 iff model k -> group j
    y = model.addVars(C, J, vtype=GRB.BINARY, name="y") # y[c][j] == 1 iff affinity set c -> group j
    size = model.addVars(J, vtype=GRB.INTEGER, lb=1, name="size") # size[j] is no. GPUs in group j
    z_ij = model.addVars(I, J, vtype=GRB.CONTINUOUS, name="z_ij") # x[i][j] * size[j]
    B = model.addVar(vtype=GRB.CONTINUOUS, name="B") # min. burst tolerance over i

    N = TOTAL_NUM_OF_NODES      # no. GPUs
    G = 12                       # max GPUs per group, 12 in paper
    mem = GPU_MEMORY_SIZE * G   # cluster total memory

    n = [send_rates[wid] / task_tputs[(wid,tid)] for wid,tid in task_types]  # avg load rate / max goodput over GPUs
    m_size = [m.model_size for m in all_models]

    # h[i][k] == 1 iff stream i uses model k
    h = [[(1 if m.model_id == get_model_id_for_task_type(task_type) else 0) for m in all_models] 
         for task_type in task_types]

    # q[c][k] = 1 if model k is in affinity-set c
    q = [[(1 if m in c else 0) for m in all_models] for c in affinity_sets]

    # (a) Cluster size constraint
    model.addConstr(gp.quicksum(size[j] for j in range(J)) <= N)

    for j in range(J):
        # (b) Group size constraint
        model.addConstr(size[j] <= G)
        # (c) Group memory constraint
        model.addConstr(gp.quicksum(z[k,j] * m_size[k] for k in range(K)) <= mem)

    for i in range(I):
        # (d) Group surjectivity
        model.addConstr(gp.quicksum(x[i,j] for j in range(J)) == 1)

    for i in range(I):
        for j in range(J):
            for k in range(K):
                # (d) Group surjectivity
                model.addConstr(h[k][i] * x[i,j] <= z[k,j])

    for j in range(J):
        # (e) Affinity set surjectivity
        model.addConstr(gp.quicksum(y[c,j] for c in range(C)) == 1)

    for j in range(J):
        for k in range(K):
            for c in range(C):
                # (e) Affinity set surjectivity
                model.addConstr(q[c][k] * z[k,j] <= y[c,j])

    # Linearization to ensure z_ij[i,j] = x[i,j] * size[j]
    for i in range(I):
        for j in range(J):
            model.addConstr(z_ij[i,j] <= size[j])
            model.addConstr(z_ij[i,j] <= G * x[i,j])
            model.addConstr(z_ij[i,j] >= size[j] - G * (1 - x[i,j]))
            model.addConstr(z_ij[i,j] >= 0)

    for i in range(I):
        # B = minimum burst tolerance
        model.addConstr((gp.quicksum(z_ij[i,j] for j in range(J))) / n[i] >= B)

    model.setObjective(B, GRB.MAXIMIZE)
    model.optimize()

    for j in range(J):
        print(f"Group {j} has {size[j].X} GPUs")

    for i in range(I):
        for j in range(J):
            if x[i,j].X > 0.5:
                print(f"Stream {i} assigned to group {j}")

    for k in range(K):
        for j in range(J):
            if z[k,j].X > 0.5:
                print(f"Model {k} assigned to group {j}")
                
    for c in range(C):
        for j in range(J):
            if y[c,j].X > 0.5:
                print(f"Aff set {c} assigned to group {j}")
    
    return ([size[j].X for j in range(J)], # no. GPUs per group
            [(i,j) for i in range(I) for j in range(J) if x[i,j].X > 0.5]) # (stream id, group id)