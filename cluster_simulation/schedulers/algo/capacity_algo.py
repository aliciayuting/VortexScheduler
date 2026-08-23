"""Statically estimates the sustainable throughput of the cluster under the current
model allocation, in jobs per second per workflow.
"""

import core.configs.gen_config as gcfg

from core.data_models.model_data import ModelData
from core.data_models.workflow import Workflow


def get_instance_throughput(model_data: ModelData, worker_size_gb: int,
                            max_batch_size: int = None) -> float:
    """Returns the highest sustained throughput (jobs/ms) one instance of
    [model_data] can deliver on a worker of [worker_size_gb].

    A batch of size b occupies the instance for exec_time(b) and retires b jobs, so
    it serves b / exec_time(b) jobs per ms. Larger batches amortize better, but the
    batch size may be capped below the model's maximum, e.g. by a per-stage SLO,
    so this takes the best rate over the batch sizes actually allowed.

    Args:
        model_data: Model to estimate throughput for
        worker_size_gb: Memory size of the worker hosting the instance (GB)
        max_batch_size: Largest batch size allowed, defaulting to the model's own

    Returns:
        throughput: Sustainable throughput of one instance (jobs/ms)
    """
    assert(worker_size_gb in model_data.batch_exec_times)

    cap = model_data.max_batch_size
    if max_batch_size is not None:
        cap = min(cap, max_batch_size)
    if gcfg.DISABLE_BATCHING:
        cap = 1

    assert(cap >= 1)

    exec_times = model_data.batch_exec_times[worker_size_gb]
    return max(bsize / exec_times[bsize] for bsize in range(1, cap + 1))


def get_model_capacities(workers: dict, time: float = 0,
                         max_batch_sizes: dict[int, int] = None) -> dict[int, float]:
    """Returns the aggregate sustainable throughput (jobs/ms) of every model in the
    allocation, summed over its instances.

    Args:
        workers: Map of worker ID -> worker object
        time: Time at which to read model placements
        max_batch_sizes: Model ID -> largest batch size allowed for that model

    Returns:
        model_capacities: Model ID -> aggregate throughput (jobs/ms)
    """
    max_batch_sizes = max_batch_sizes or {}

    capacities: dict[int, float] = {}
    for worker in workers.values():
        for state in worker.GPU_state.state_at(time):
            model_data = state.model.data
            capacities[model_data.id] = capacities.get(model_data.id, 0) + \
                get_instance_throughput(model_data, worker.total_memory_gb,
                                        max_batch_sizes.get(model_data.id))

    return capacities


def get_workflow_capacities(workflows: dict[int, Workflow], workers: dict,
                            demand_rates: dict[int, float],
                            time: float = 0) -> dict[int, float]:
    """Estimates the job rate (qps) each workflow can sustain on the current
    allocation.

    A workflow runs every one of its stages once per job, so its throughput is set
    by its bottleneck stage: the model whose share of capacity, divided by how many
    of the workflow's tasks use it, is smallest.

    Models shared by several workflows have to be divided between them, and the
    split determines how much of the cluster each tenant is allowed to claim. Each
    model's capacity is shared in proportion to the demand placed on it, so a
    workflow that asks for twice the rate is allotted twice the share, and a
    workflow that shares nothing keeps its models entirely. Splitting by demand
    keeps a single overloaded tenant from consuming a shared model and starving the
    others, at the cost of not handing an idle tenant's share to a busy one.

    Args:
        workflows: Map of workflow ID -> workflow
        workers: Map of worker ID -> worker object
        demand_rates: Workflow ID -> the send rate (qps) it is expected to offer,
        used to weight the split of shared models
        time: Time at which to read model placements

    Returns:
        workflow_capacities: Workflow ID -> sustainable job rate (qps)
    """
    # per-stage SLOs cap how large a batch each stage may form, which lowers the
    # throughput its model can reach
    max_batch_sizes: dict[int, int] = {}
    if gcfg.SLO_TYPE == "NEXUS":
        for workflow in workflows.values():
            for task_id, bsize in workflow.task_max_batch_sizes.items():
                model_id = workflow.tasks[task_id].model_data.id
                max_batch_sizes[model_id] = min(
                    max_batch_sizes.get(model_id, bsize), bsize)

    model_capacities = get_model_capacities(workers, time, max_batch_sizes)

    weights = {wid: demand_rates.get(wid, 0) for wid in workflows.keys()}
    if sum(weights.values()) <= 0:
        # nothing to go on: divide every shared model evenly
        weights = {wid: 1 for wid in workflows.keys()}

    # total weighted demand on each model, counting a workflow once per task that
    # uses the model
    model_demand: dict[int, float] = {}
    for workflow in workflows.values():
        for task in workflow.tasks.values():
            model_demand[task.model_data.id] = \
                model_demand.get(task.model_data.id, 0) + weights[workflow.id]

    capacities: dict[int, float] = {}
    for workflow in workflows.values():
        stage_capacities = []
        for task in workflow.tasks.values():
            model_id = task.model_data.id
            assert(model_id in model_capacities), \
                f"Model {model_id} of workflow {workflow.id} is not in the allocation"

            demand = model_demand[model_id]
            share = weights[workflow.id] / demand if demand > 0 else 0

            # this workflow's slice of the model, spread over the tasks of a
            # single job that need it
            stage_capacities.append(model_capacities[model_id] * share * 1000)

        capacities[workflow.id] = min(stage_capacities)

    return capacities
