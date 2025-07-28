from random import randint
from core.workflow import *
from core.config import *


class Model:
    """
    Description of Model
    """

    def __init__(self, job_type_id: int, model_id: int, model_size: float, 
                 batch_sizes: list[int], batch_exec_times: list[float], exec_time_cv: float):
        self.job_type_id = job_type_id
        self.model_id = model_id
        self.model_size = model_size
        self.batch_sizes = batch_sizes
        self.batch_exec_times = batch_exec_times
        self.exec_time_cv = exec_time_cv

    def __hash__(self):
        return hash(self.model_id)

    def __eq__(self, other):
        if isinstance(other, Model):
            return self.model_id == other.model_id and self.job_type_id == other.job_type_id
        return False

    def __ne__(self, other):
        return not (self == other)

    def __str__(self):
        return ("Model belongs to Job "
                + str(self.job_type_id)
                + ", Model ID: "
                + str(self.model_id)
                + " (size:"
                + str(self.model_size)
                + "KB)")

    def to_string(self):
        return (
            "Model: job_id"
            + str(self.job_id)
            + " , model_id:"
            + str(self.model_id)
            + " (size:"
            + str(self.model_size)
            + ")"
        )
    
    def get_exec_time(self, batch_size: int, partition_size: int) -> float:
        """
            Returns the execution time of a batch of [batch_size] on a partition of 
            [partition_size] sampled from a Normal distribution with a CV of [self.cv].
        """
        exact_exec_time = self.batch_exec_times[partition_size][self.batch_sizes.index(batch_size)]
        stddev = self.exec_time_cv * exact_exec_time
        
        randomized_time = np.random.normal(loc=exact_exec_time, scale=stddev, size=1)
        while randomized_time <= 0:
            randomized_time = np.random.normal(loc=exact_exec_time, scale=stddev, size=1)

        return randomized_time[0]

def parse_models_from_workflows() -> dict:
    """
    Helper function for Simulation: upon Resource Module initialization read "workflow.py" and extract how many models there are
    Returns: {job_type_id1:[model1, model2 ...], job_type_id2:[]} a dict of model for different jobs
    """

    job_models_dict = dict()
    for i, job in enumerate(WORKFLOW_LIST):
        # print(job)
        models = []
        for model in WORKFLOW_LIST[i]["TASKS"]:
            if model["MODEL_ID"] != -1:
                models.append(Model(
                    job_type_id=job["JOB_TYPE"], 
                    model_id=model["MODEL_ID"], 
                    model_size=model["MODEL_SIZE"],
                    batch_sizes=model["BATCH_SIZES"],
                    batch_exec_times=model["MIG_BATCH_EXEC_TIMES"],
                    exec_time_cv=model["EXEC_TIME_CV"]))
        job_models_dict[job["JOB_TYPE"]] = models

    # print(job_models_dict)
    return job_models_dict
