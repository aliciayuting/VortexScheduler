import uuid

from core.task import Task


class Batch:
    """
        Represents a batch of Tasks executing on the same model.
    """
    def __init__(self, tasks: list[Task]):
        assert(len(tasks) > 0)

        self.id = uuid.uuid4()
        self.tasks = tasks
        self.model = tasks[0].model
        self.job_ids = list(map(lambda t: t.job_id, tasks))

    def size(self) -> int:
        return len(self.tasks)
    
    def __hash__(self):
        return hash(self.id)
    
    def __eq__(self, value):
        return value is not None and type(value) == Batch and value.id == self.id
    
    def __str__(self):
        return f"[Batch {self.id} | Type {self.tasks[0].task_type}] <Jobs {self.job_ids}>"
    
    def __repr__(self):
        return self.__str__()