from core.data_models.model_data import ModelData
from core.data_models.task_data import TaskData
from core.allocation import ModelAllocation

from bidict import bidict


class Workflow:
    """
    Represents a DAG pipeline.
    """

    def __init__(self, workflow_cfg, models, slo_type):
        self.id = workflow_cfg["JOB_TYPE"]
        
        self.initial_tasks: list[TaskData] = []
        self.tasks = {}
        for cfg in workflow_cfg["TASKS"]:
            task = TaskData(
                cfg["TASK_INDEX"],
                models[cfg["MODEL_ID"]],
                cfg["INPUT_SIZE"],
                cfg["OUTPUT_SIZE"],
                cfg["SLO"] if slo_type == "NEXUS" else None)
            
            self.tasks[task.id] = task

            if len(cfg["PREV_TASK_INDEX"]) == 0:
                self.initial_tasks.append(task)

        assert(len(self.initial_tasks) > 0)
        
        for cfg in workflow_cfg["TASKS"]:
            task = self.tasks[cfg["TASK_INDEX"]]
            task.prev_tasks = [self.tasks[id] 
                               for id in cfg["PREV_TASK_INDEX"]]
            task.next_tasks = [self.tasks[id]
                               for id in cfg["NEXT_TASK_INDEX"]]
            
    def get_models(self) -> list[ModelData]:
        """
        Returns all models used by any task in workflow.
        """
        return list(set([task.model_data for task in self.tasks.values() if task.model_data]))

    def get_min_processing_time(self) -> float:
        return self.get_processing_time(
            lambda t: t.model_data.batch_exec_times[24][1])

    def get_processing_time(self, get_exec_time) -> float:
        dependencies: dict[int, set[int]] = {}
        dependents: dict[int, set[int]] = {}
        available_tasks: list[TaskData] = []
        for task in self.tasks.values():
            dependencies[task.id] = set([t.id for t in task.prev_tasks])
            dependents[task.id] = set([t.id for t in task.next_tasks])
            
            if len(dependencies[task.id]) == 0:
                available_tasks.append(task)
        
        max_cum_processing_time = 0
        while available_tasks:
            next_available_tasks = []
            max_cum_processing_time += max([
                get_exec_time(task) for task in available_tasks
            ])
            for task in available_tasks:
                for dep in dependents[task.id]:
                    dependencies[dep].remove(task.id)
                    if len(dependencies[dep]) == 0:
                        next_available_tasks.append(self.tasks[dep])

            available_tasks = next_available_tasks
        return max_cum_processing_time