from core.workflow import *
from core.model import *
from core.task import *
from core.config import *


class Job(object):

    def __init__(self, create_time, job_type_id, job_id, client_id, slo):
        """
        A job is a unique object across the simulation execution that has a specific graph of task dependencies (job_type_id)
        """

        self.client_id = client_id
        self.id = job_id  # unique ID for each job
        self.job_type_id = job_type_id
        self.job_name, self.tasks = None, []  # List of Task objects
        self.tasks = []
        # TODO: this is called everytime now. OPTIMIZE BY CALLING IT ONLY ONCE
        self.job_generate_from_workflow()
        self.ADFG = {}     # Activated Dataflow Graph scheduled by scheduler. map: task_id->worker_id
        # List containing which tasks tha constitute the job have been completed : [(task,timestamp),...]
        self.completed_tasks = []
        self.create_time = create_time  
        self.end_time = create_time
        self.slo = 0 if SLO_GRANULARITY == "TASK" else slo


    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        if isinstance(other, Job):
            return self.id == other.id
        return False

    def __ne__(self, other):
        return not (self == other)

    def __str__(self):
        return "JobID: {}".format(self.id)

    def get_task_by_id(self, task_id) -> Task:
        for task in self.tasks:
            if task.task_id == task_id:
                return task
        return None

    def assign_ADFG(self, ADFG):
        """
        Function to assign the ADFG to the job and tasks within the job
        :param ADFG: ADFG to be assigned to the job
        """
        self.ADFG = ADFG
        for task in self.tasks:
            task.ADFG = ADFG

    def job_completed(self, completion_time, task_id) -> bool:
        """ 
        Check if the all tasks in the job have completed
        Returns True if the job has completed, and False otherwise. 
        """
        if task_id not in self.completed_tasks:
            self.completed_tasks.append(task_id)
        self.end_time = max(completion_time, self.end_time)
        assert len(self.completed_tasks) <= len(self.tasks)
        return len(self.completed_tasks) == len(self.tasks)

    def job_generate_from_workflow(self):
        """
        Access WORKFLOW_LIST in workflow.py and fill up the self members based on the job_type_id
        """
        job_cfg = WORKFLOW_LIST[self.job_type_id]
        self.job_name = job_cfg["JOB_NAME"]
        
        for task_cfg in job_cfg["TASKS"]:
            required_model_for_task = None
            if task_cfg["MODEL_ID"] > -1:
                required_model_for_task = Model(job_type_id=job_cfg["JOB_TYPE"],
                                                model_id=task_cfg["MODEL_ID"],
                                                model_size=task_cfg["MODEL_SIZE"],
                                                batch_sizes=task_cfg["BATCH_SIZES"],
                                                batch_exec_times=task_cfg["MIG_BATCH_EXEC_TIMES"],
                                                exec_time_cv=task_cfg["EXEC_TIME_CV"])

            current_task = Task(self,
                                self.id,  # ID of the associated unique Job
                                task_cfg["TASK_INDEX"],  # taskID
                                (self.job_type_id, task_cfg["TASK_INDEX"]), # task type
                                task_cfg["EXECUTION_TIME"], 
                                required_model_for_task, 
                                task_cfg["INPUT_SIZE"],
                                task_cfg["OUTPUT_SIZE"],
                                task_cfg["MAX_BATCH_SIZE"],
                                task_cfg["MAX_WAIT_TIME"],
                                task_cfg["SLO"] if SLO_GRANULARITY == "TASK" else 0,
                                task_cfg["MAX_EMIT_BATCH_SIZE"])

            self.tasks.append(current_task)

        # Assign dependencies among Tasks
        for current_task_index in range(len(job_cfg["TASKS"])):
            for prev_idx in job_cfg["TASKS"][current_task_index]["PREV_TASK_INDEX"]:
                self.tasks[current_task_index].required_task_ids.append(prev_idx)
            for next_idx in job_cfg["TASKS"][current_task_index]["NEXT_TASK_INDEX"]:
                self.tasks[current_task_index].next_task_ids.append(next_idx)

    def finished_task(self, task: Task) -> bool:
        for f_task in self.completed_tasks:
            if task.job_id == self.id and task.task_id == f_task[0].task_id:
                return True
        return False
    
    def remaining_tasks(self) -> list[Task]:
        return list(filter(lambda t: t.task_id not in self.completed_tasks, self.tasks))
    
    def newly_available_tasks(self, newly_completed: Task) -> list[Task]:
        """
            Returns a list of incomplete tasks that are now ready for
            execution given the completion of [newly_completed] task.
        """
        ready_tasks = []
        for task in self.remaining_tasks():
            if newly_completed.task_id in task.required_task_ids and \
                all(t in self.completed_tasks for t in task.required_task_ids):
                ready_tasks.append(task)
        return ready_tasks

    def print_job_info(self):
        for task in self.tasks:
            print("task{}, duration{}, required_task_ids{}".format(task.task_id, task.task_exec_duration,
                                                                task.required_task_ids))