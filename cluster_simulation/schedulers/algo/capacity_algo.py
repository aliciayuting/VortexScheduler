"""Statically estimates the sustainable throughput of the cluster under the current
model allocation, in jobs per second per workflow.
"""

import math

import core.configs.gen_config as gcfg

from core.data_models.model_data import ModelData
from core.data_models.workflow import Workflow
from schedulers.algo.nexus_algo import NexusSLOSplitter


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


def get_workflow_stage_capacities(workflows: dict[int, Workflow], workers: dict,
                                  demand_rates: dict[int, float],
                                  time: float = 0) -> dict[int, dict[int, float]]:
    """Estimates the job rate (qps) each individual stage of each workflow can
    sustain on the current allocation.

    A stage's rate is its workflow's share of the model it runs on. Models shared
    by several workflows have to be divided between them, and the split determines
    how much of the cluster each tenant is allowed to claim. Each model's capacity
    is shared in proportion to the demand placed on it, so a workflow that asks for
    twice the rate is allotted twice the share, and a workflow that shares nothing
    keeps its models entirely. Splitting by demand keeps a single overloaded tenant
    from consuming a shared model and starving the others, at the cost of not
    handing an idle tenant's share to a busy one. A workflow that runs two of its
    own stages on one model is counted twice against it, so the two stages split
    that model between them as well.

    Args:
        workflows: Map of workflow ID -> workflow
        workers: Map of worker ID -> worker object
        demand_rates: Workflow ID -> the send rate (qps) it is expected to offer,
        used to weight the split of shared models
        time: Time at which to read model placements

    Returns:
        stage_capacities: Workflow ID -> task ID -> sustainable job rate (qps)
    """
    # under NEXUS a stage may not form a batch larger than its own SLO budget
    # affords, which lowers the throughput its model can reach. NEXUS_SPLIT takes
    # only the stage deadlines from the split and leaves batching uncapped
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

    stage_capacities: dict[int, dict[int, float]] = {}
    for workflow in workflows.values():
        stage_capacities[workflow.id] = {}
        for task in workflow.tasks.values():
            model_id = task.model_data.id
            assert(model_id in model_capacities), \
                f"Model {model_id} of workflow {workflow.id} is not in the allocation"

            demand = model_demand[model_id]
            share = weights[workflow.id] / demand if demand > 0 else 0

            # this workflow's slice of the model, spread over the tasks of a
            # single job that need it
            stage_capacities[workflow.id][task.id] = \
                model_capacities[model_id] * share * 1000

    return stage_capacities


def get_workflow_capacities(workflows: dict[int, Workflow], workers: dict,
                            demand_rates: dict[int, float],
                            time: float = 0) -> dict[int, float]:
    """Estimates the job rate (qps) each workflow can sustain on the current
    allocation.

    A workflow runs every one of its stages once per job, so its throughput is set
    by its bottleneck stage: the slowest of the per-stage rates computed by
    [get_workflow_stage_capacities].

    Args:
        workflows: Map of workflow ID -> workflow
        workers: Map of worker ID -> worker object
        demand_rates: Workflow ID -> the send rate (qps) it is expected to offer,
        used to weight the split of shared models
        time: Time at which to read model placements

    Returns:
        workflow_capacities: Workflow ID -> sustainable job rate (qps)
    """
    stage_capacities = get_workflow_stage_capacities(
        workflows, workers, demand_rates, time)

    return {wid: min(stages.values()) for wid, stages in stage_capacities.items()}


def get_task_exec_times(workflow: Workflow, workers: dict,
                        time: float = 0) -> dict[int, float]:
    """Returns how long each task of [workflow] takes to run on its own, at batch
    size 1 on the slowest partition size its model is placed on, the same
    partition the Nexus split plans each stage against.

    Args:
        workflow: Workflow whose tasks to time
        workers: Map of worker ID -> worker object
        time: Time at which to read model placements

    Returns:
        exec_times: Task ID -> exec time (ms)
    """
    worker_sizes = NexusSLOSplitter.get_task_worker_sizes(workflow, workers, time)

    return {task_id: workflow.tasks[task_id].model_data.batch_exec_times[size][1]
            for task_id, size in worker_sizes.items()}


def get_job_runtime(workflow: Workflow, workers: dict, time: float = 0) -> float:
    """Returns how long one job of [workflow] takes with the cluster to itself on
    the current allocation, i.e. the job size its SLO is set as a multiple of.

    This is the longest path through the DAG with each task charged its
    [get_task_exec_times] time, times the rounds its own job forces its stage
    into: a stage fanned out into more tasks than its model has instances cannot
    run them at once, and a batch takes at most one task per job, so they queue
    behind each other even on an idle cluster. Tasks only contend for a model's
    instances with tasks at the same DAG depth released at the same time, and a
    source task cannot start before its input arrives. Mirrors job_runtime in
    experiments/run_core_sweep.py, which computes the same quantity from the
    config alone.

    Args:
        workflow: Workflow to size
        workers: Map of worker ID -> worker object
        time: Time at which to read model placements

    Returns:
        runtime: Time one job needs on an idle cluster (ms)
    """
    exec_times = get_task_exec_times(workflow, workers, time)

    instances: dict[int, int] = {}
    for worker in workers.values():
        for state in worker.GPU_state.state_at(time):
            model_id = state.model.data.id
            instances[model_id] = instances.get(model_id, 0) + 1

    depths: dict[int, int] = {}

    def depth(task_id: int) -> int:
        if task_id not in depths:
            depths[task_id] = max([depth(t.id) + 1
                                   for t in workflow.tasks[task_id].prev_tasks],
                                  default=0)
        return depths[task_id]

    def stage(task_id: int) -> tuple[int, int, float]:
        task = workflow.tasks[task_id]
        return (task.model_data.id, depth(task_id), task.arrival_offset)

    task_counts: dict[tuple[int, int, float], int] = {}
    for task_id in workflow.tasks:
        task_counts[stage(task_id)] = task_counts.get(stage(task_id), 0) + 1

    finish: dict[int, float] = {}

    def finish_by(task_id: int) -> float:
        if task_id not in finish:
            task = workflow.tasks[task_id]
            rounds = math.ceil(task_counts[stage(task_id)]
                               / max(1, instances.get(task.model_data.id, 1)))
            start = max([finish_by(t.id) for t in task.prev_tasks],
                        default=task.arrival_offset)
            finish[task_id] = start + exec_times[task_id] * rounds
        return finish[task_id]

    return max(finish_by(task_id) for task_id in workflow.tasks)
