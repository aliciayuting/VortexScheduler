from core.data_models.model_data import ModelData

import core.configs.gen_config as gcfg


class Task(object):
    def __init__(self, job, task_id: int, model_data: ModelData | None,
                 input_size: float, result_size: float,
                 slo: float | None=None):

        self.job = job
        self.task_id = task_id
        self.model_data = model_data
        self.input_size = input_size
        self.result_size = result_size
        self.slo = slo
        
        # list of Tasks (inputs) that this task requires ( list will be appended as the job generated)
        self.required_task_ids = []                        # list of task ids
        self.next_task_ids = []                            # list of task ids
        self.assigned_worker_id = None
        self.executing_worker_id = -1
        self.ADFG = {}                                  # ADFG assigned to the job that this task belongs to

    def get_task_deadline(self):
        """Returns the time by which this task must finish. Under job-level SLOs
        that is the job's own deadline; under NEXUS SLOs it is the deadline of
        this task's pipeline stage.
        """
        if gcfg.SLO_TYPE == "NEXUS":
            # read through the workflow rather than a value copied at task creation,
            # since the split is computed after model placement, i.e. after jobs exist
            offset = self.job.workflow.get_task_deadline_offset(self.task_id)
            return self.job.create_time + offset * (1 + gcfg.SLO_SLACK)
        else:
            return self.job.create_time + self.job.slo * (1 + gcfg.SLO_SLACK)

    def __hash__(self):
        return hash((self.task_id, self.job.id))

    def __eq__(self, other):
        if (isinstance(other, Task)):
            return self.task_id == other.task_id and self.job.id == other.job.id
        return False

    def __ne__(self, other):
        return not (self.__eq__(other))

    def __str__(self):
        return f"[Job ID {self.job.id}, Task ID {self.task_id}]"

    def __repr__(self):
        return self.__str__()

    def print_task_log(self):
        print(self.log.toString())
