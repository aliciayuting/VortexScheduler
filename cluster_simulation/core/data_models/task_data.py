from core.data_models.model_data import ModelData
from core.task import Task

import core.configs.gen_config as gcfg


class TaskData:
    """
    Represents one node in a DAG pipeline.
    """

    def __init__(self, id: int, model_data: ModelData | None, input_size: float, output_size: float,
                 slo: float | None):

        self.id = id
        self.model_data = model_data
        self.input_size = input_size
        self.output_size = output_size
        self.slo = slo

        self.prev_tasks: list[TaskData] = []
        self.next_tasks: list[TaskData] = []

    def __eq__(self, value):
        if type(value) != TaskData:
            return False
        
        if value.id != self.id:
            return False
        
        if value.model_data.id != self.model_data.id:
            return False
        
        return True
    
    def __hash__(self):
        return hash((self.id, self.model_data.id))
    
    def __str__(self):
        return f"[TaskData ID: {self.id}, Model ID: {self.model_data.id}]"

    def __repr__(self):
        return self.__str__()
    
    def create_task(self, job) -> Task:
        task = Task(job,
                    self.id,
                    self.model_data,
                    self.input_size,
                    self.output_size,
                    self.slo if gcfg.SLO_TYPE == "NEXUS" else 0)
        
        for prev_task in self.prev_tasks:
            task.required_task_ids.append(prev_task.id)
        
        for next_task in self.next_tasks:
            task.next_task_ids.append(next_task.id)

        return task