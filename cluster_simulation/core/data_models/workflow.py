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

        # per-stage SLO split, populated by assign_task_slos() when SLO_TYPE is NEXUS
        self.job_slo: float | None = None
        self.task_slos: dict[int, float] = {}
        self.task_max_batch_sizes: dict[int, int] = {}
        self.task_deadline_offsets: dict[int, float] = {}

    def assign_task_slos(self, task_slos: dict[int, tuple[float, int]], job_slo: float):
        """Records a per-stage SLO split (see NexusSLOSplitter) and derives the
        deadline offset of each task from job create time.

        A stage cannot start until every incoming branch has finished, so its
        deadline offset is its own budget plus the LARGEST offset among its
        predecessors.

        Args:
            task_slos: (stage SLO, max batch size) for each task ID, as produced
            by NexusSLOSplitter.generate_task_slos
            job_slo: Job-level SLO the split was derived from (ms)
        """
        assert(set(task_slos.keys()) == set(self.tasks.keys()))

        self.job_slo = job_slo
        self.task_slos = {tid: slo for tid, (slo, _) in task_slos.items()}
        self.task_max_batch_sizes = {tid: bsize for tid, (_, bsize) in task_slos.items()}

        self.task_deadline_offsets = {}
        ready = list(self.initial_tasks)
        while ready:
            next_ready = []
            for task in ready:
                if task.id in self.task_deadline_offsets:
                    continue

                spent = max([self.task_deadline_offsets[pt.id] for pt in task.prev_tasks],
                            default=0)
                self.task_deadline_offsets[task.id] = spent + self.task_slos[task.id]

                next_ready.extend([
                    t for t in task.next_tasks
                    if t.id not in self.task_deadline_offsets
                    and all(pt.id in self.task_deadline_offsets for pt in t.prev_tasks)])

            ready = next_ready

        assert(len(self.task_deadline_offsets) == len(self.tasks))

        # meeting every stage deadline must imply meeting the job deadline
        assert(max(self.task_deadline_offsets.values()) <= job_slo)

    def get_task_deadline_offset(self, task_id: int) -> float:
        """Returns the time from job create time by which [task_id] must finish
        under per-stage (NEXUS) SLOs.
        """
        assert(self.task_deadline_offsets), \
            f"No per-stage SLO split assigned for workflow {self.id}"

        return self.task_deadline_offsets[task_id]

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