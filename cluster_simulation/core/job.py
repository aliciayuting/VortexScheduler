from core.configs.workflow_config import *
from core.task import Task
from core.data_models.workflow import Workflow


class Job(object):
    def __init__(self, workflow: Workflow, job_id: int, client_id: int, slo: float, created_at: float):

        self.client_id: int = client_id
        self.id: int = job_id
        self.job_type_id: int = workflow.id

        self.workflow: Workflow = workflow

        self.tasks: list[Task] = []
        for _, at in sorted(self.workflow.tasks.items(), key=lambda item: item[0]):
            self.tasks.append(at.create_task(self))

        # task ID -> completion time, -1 for not complete
        self._task_states: dict[Task, float] = {tid: -1 for tid in workflow.tasks.keys()}
        
        self.create_time = created_at
        self.slo: float = slo

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        return isinstance(other, Job) and self.id == other.id

    def __str__(self):
        return f"[Job {self.id}, Workflow {self.job_type_id}]"
    
    def __repr__(self):
        return self.__str__()

    def get_min_remaining_processing_time(self, init_proc_times={}, batch_sizes={}) -> float:
        """
            Returns the minimum remaining time to finish the job
            without batching, given [init_proc_times: task_id -> time] where
            each [task_id: int] is processed in [time: float] amount of time.
            
            By default, assumes all completed tasks take 0 extra time to
            complete, and calculates tasks not in [init_proc_times] with best
            execution time.
        """
        def _get_min_proc_time(target_task, proc_times):
            if target_task.task_id in proc_times:
                return proc_times[target_task.task_id]
            else:
                bsize = batch_sizes[target_task.task_id] if target_task.task_id in batch_sizes else 1
                proc_time = target_task.model_data.batch_exec_times[24][bsize]
                if target_task.required_task_ids:
                    proc_time += max(_get_min_proc_time([t for t in self.tasks if t.task_id == tid][0], proc_times) 
                                    for tid in target_task.required_task_ids)
                proc_times[target_task.task_id] = proc_time
                return proc_time
        
        initial_proc_times = {tid: 0 for tid in self.completed_tasks}
        for k,v in init_proc_times.items(): initial_proc_times[k] = v

        return min(_get_min_proc_time(t, initial_proc_times) for t in self.tasks if not t.next_task_ids)

    def get_task_by_id(self, task_id) -> Task:
        for task in self.tasks:
            if task.task_id == task_id:
                return task
        return None

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

    def finished_task(self, task: Task) -> bool:
        for f_task in self.completed_tasks:
            if task.job.id == self.id and task.task_id == f_task[0].task_id:
                return True
        return False
    
    def is_complete(self) -> bool:
        return all(t >= 0 for t in self._task_states.values())
    
    def set_completion_time(self, time: float, task_id: int):
        assert(self._task_states[task_id] < 0)
        self._task_states[task_id] = time
    
    def newly_available_tasks(self, dependency: Task) -> list[Task]:
        """Fetches a list of incomplete dependents of a given task for which the
        given task is the LAST dependency completed.

        Args:
            dependency: Filter incomplete tasks by those which require this
            dependency as a direct predecessor.

        Returns:
            ready_tasks: List of dependents that are ready for execution after
            given dependency is complete.
        """
        ready_tasks = []
        for task in self.tasks:
            if self._task_states[task.task_id] < 0 and \
                dependency.task_id in task.required_task_ids and \
                all(self._task_states[rt_id] >= 0 for rt_id in task.required_task_ids) and \
                self._task_states[dependency.task_id] == max([self._task_states[rt_id] for rt_id in task.required_task_ids]):
                
                ready_tasks.append(task)
        return ready_tasks