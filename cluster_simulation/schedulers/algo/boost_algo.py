import numpy as np

import core.configs.gen_config as gcfg
from core.task import Task
from core.job import Job


class BoostPolicy:
    TOTAL_JOB_TIME = 0
    REMAINING_JOB_TIME = 1
    REMAINING_TIME_TO_DEADLINE = 2


def _get_processing_time(job: Job, complete_task_ids: set[int]) -> float:
    dependencies: dict[int, set[int]] = {}
    dependents: dict[int, set[int]] = {}
    available_tasks: list[Task] = []
    for task in job.tasks:
        dependencies[task.task_id] = set(task.required_task_ids) - complete_task_ids
        dependents[task.task_id] = set(task.next_task_ids) - complete_task_ids
        
        if task.task_id not in complete_task_ids and len(dependencies[task.task_id]) == 0:
            available_tasks.append(task)
    
    max_cum_processing_time = 0
    while available_tasks:
        next_available_tasks = []
        # TODO: Figure out the right way to compute "processing time" here.
        # Do we include things like GPU_to_GPU_delay? Does it make sense to
        # use execution_time from a previous run to start?
        max_cum_processing_time += max([
            task.model_data.batch_exec_times[24][1] for task in available_tasks
        ])
        for task in available_tasks:
            for dep in dependents[task.task_id]:
                dependencies[dep].remove(task.task_id)
                if len(dependencies[dep]) == 0:
                    next_available_tasks.append(job.get_task_by_id(dep))

        available_tasks = next_available_tasks
    return max_cum_processing_time


def get_job_boost_size(time: float, job: Job, boost_policy: int) -> float:
    """
    Compute a boost for the job based on the [boost_policy].
    """
    assert(boost_policy in [BoostPolicy.TOTAL_JOB_TIME, 
                            BoostPolicy.REMAINING_JOB_TIME,
                            BoostPolicy.REMAINING_TIME_TO_DEADLINE])

    if boost_policy == BoostPolicy.TOTAL_JOB_TIME:
        return _get_processing_time(job, set())
    elif boost_policy == BoostPolicy.REMAINING_JOB_TIME:
        return _get_processing_time(job, {tid for tid, t in job._task_states.items() if t != -1})
    elif boost_policy == BoostPolicy.REMAINING_TIME_TO_DEADLINE:
        assert(gcfg.SLO_TYPE == "JOB_LEVEL")
        return (job.create_time + (1 + gcfg.SLO_SLACK) * job.slo) - time


def get_task_priority_by_boost(time, task: Task, boost_policy: int, boost_parameter=gcfg.BOOST_PARAMETER) -> float:
    boost_size = get_job_boost_size(time, task.job, boost_policy)
    # print(f"Task {task}, boost {boost_size}")

    # past deadline tasks
    if boost_size <= 0:
        return np.inf
    return task.get_task_deadline() - 1 / boost_parameter * np.log(
        1 / (1 - np.exp(-boost_parameter * boost_size)))
