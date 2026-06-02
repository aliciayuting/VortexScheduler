import core.configs.gen_config as gcfg
import numpy as np

from core.job import Job
from core.task import Task
from core.batch import Batch
from core.data_models.workflow import Workflow

from workers.worker import Worker
from schedulers.scheduler import Scheduler

from events.event_manager import EventManager
from events.event import *
from events.event_types import *


class HEFTScheduler(Scheduler):

    def __init__(self, em: EventManager, workers: dict[UUID, Worker],
                 workflows: dict[int, Workflow], scheduler_worker_id: UUID):
        super().__init__(em)

        self.workers = workers
        self.workflows = workflows
        self.scheduler_worker_id = scheduler_worker_id

        # (job ID, task ID) -> worker ID on which task result/output is stored
        self.output_locs: dict[tuple[int, int], UUID] = {}

        # precompute upward ranks (critical path to end of DAG) per workflow
        self._ranks: dict[int, dict[int, float]] = {
            wid: self._compute_ranks(wf) for wid, wf in workflows.items()
        }

    def _compute_ranks(self, workflow: Workflow) -> dict[int, float]:
        """Upward rank for each task: average exec time + max(comm + rank(successor)).

        Averaged over all worker memory sizes to stay machine-agnostic.
        """
        ranks = {}

        def rank(task_data):
            if task_data.id in ranks:
                return ranks[task_data.id]
            avg_exec = np.mean([
                task_data.model_data.batch_exec_times[mem][1]
                for mem in task_data.model_data.batch_exec_times
            ])
            if not task_data.next_tasks:
                ranks[task_data.id] = avg_exec
            else:
                comm_cost = task_data.output_size / 12500 if gcfg.ENABLE_NETWORKING_DELAYS else 0
                ranks[task_data.id] = avg_exec + max(comm_cost + rank(succ)
                                                     for succ in task_data.next_tasks)
            return ranks[task_data.id]

        for td in workflow.tasks.values():
            rank(td)
        return ranks

    def _laxity(self, time: float, task: Task) -> float:
        """Time remaining until deadline minus critical path length from this task."""
        deadline = task.job.create_time + task.job.slo
        return deadline - time - self._ranks[task.job.job_type_id][task.task_id]

    def _compute_eft(self, time: float, task: Task, worker: Worker,
                     extra_queued: int = 0) -> float:
        """Estimated finish time for task on worker.

        extra_queued: tasks of the same model already dispatched to this worker
        in the current scheduling round, used to project a more accurate batch size.
        """
        model_id = task.model_data.id
        instances = [s for s in worker.GPU_state.state_at(time)
                     if s.model.data.id == model_id]
        if not instances:
            return float('inf')

        # earliest any model instance on this worker will be free
        worker_ready = min(
            s.reserved_until if s.reserved_batch else time
            for s in instances
        )

        # when all inputs arrive at this worker
        inputs_arrive = time
        if gcfg.ENABLE_NETWORKING_DELAYS:
            if not task.required_task_ids:
                inputs_arrive = time + task.input_size / 12500
            else:
                for rt_id in task.required_task_ids:
                    from_wid = self.output_locs.get((task.job.id, rt_id))
                    if from_wid and from_wid != worker.id:
                        pred_task = task.job.get_task_by_id(rt_id)
                        inputs_arrive = max(inputs_arrive,
                                            time + pred_task.result_size / 12500)

        est = max(worker_ready, inputs_arrive)

        # project batch size: existing queue + tasks dispatched this round + this task
        proj_batch = min(
            worker.get_qlen(model_id) + extra_queued + 1,
            task.model_data.max_batch_size
        )
        exec_time = task.model_data.batch_exec_times[worker.total_memory_gb][proj_batch]

        return est + exec_time

    def _pick_worker(self, time: float, task: Task,
                     extra_queued: dict[tuple[UUID, int], int]) -> UUID | None:
        """Return the worker ID with the lowest EFT for this task."""
        best_worker_id = None
        best_eft = float('inf')

        for w in sorted(self.workers.values(), key=lambda w: (w.create_time, w.id)):
            extra = extra_queued.get((w.id, task.model_data.id), 0)
            eft = self._compute_eft(time, task, w, extra)
            if eft < best_eft:
                best_eft = eft
                best_worker_id = w.id

        return best_worker_id

    def on_job_arrival(self, time: float, job: Job):
        self.on_tasks_arrival(time, [t for t in job.tasks if not t.required_task_ids])

    def on_tasks_arrival(self, time: float, tasks: list[Task]):
        # most urgent (lowest laxity) first
        sorted_tasks = sorted(tasks, key=lambda t: self._laxity(time, t))

        tasks_to_send: dict[UUID, list[Task]] = {}
        # track tasks already dispatched to each (worker, model) pair this round
        extra_queued: dict[tuple[UUID, int], int] = {}

        for task in sorted_tasks:
            worker_id = self._pick_worker(time, task, extra_queued)
            if worker_id is None:
                continue
            tasks_to_send.setdefault(worker_id, []).append(task)
            key = (worker_id, task.model_data.id)
            extra_queued[key] = extra_queued.get(key, 0) + 1

        for worker_id, worker_tasks in tasks_to_send.items():
            self.em.add_event(
                Event(time,
                      EVENT_TYPES[EventIds.TASKS_ASSIGNED_TO_WORKER],
                      kwargs={"worker_id": worker_id, "tasks": worker_tasks}),
                self.emitter_id)

            inputs_from_scheduler = [t for t in worker_tasks if not t.required_task_ids]
            outputs_from_workers: dict[UUID, list[tuple[int, int]]] = {}

            for task in worker_tasks:
                for rt in task.required_task_ids:
                    wid = self.output_locs[(task.job.id, rt)]
                    outputs_from_workers.setdefault(wid, []).append((task.job.id, rt))

            if inputs_from_scheduler:
                self.em.add_event(
                    Event(time,
                          EVENT_TYPES[EventIds.TASKS_INPUTS_SENT_TO_WORKER],
                          kwargs={"tasks": worker_tasks,
                                  "from_worker_id": self.scheduler_worker_id,
                                  "to_worker_id": worker_id,
                                  "ignore_transfer_time": not gcfg.ENABLE_NETWORKING_DELAYS}),
                    self.emitter_id)

            for from_wid, job_task_ids in outputs_from_workers.items():
                self.em.add_event(
                    Event(time,
                          EVENT_TYPES[EventIds.TASKS_OUTPUTS_ASSIGNED_TO_WORKER],
                          kwargs={"job_task_ids": job_task_ids,
                                  "from_worker_id": from_wid,
                                  "to_worker_id": worker_id}),
                    self.emitter_id)

    def on_jobs_dropped(self, time: float, job_ids: list[int]):
        pass

    def on_batch_start(self, time, batch, worker_id, instance_id):
        pass

    def on_batch_finish(self, time: float, batch: Batch, worker_id: UUID, instance_id: UUID):
        for task in batch.tasks:
            assert((task.job.id, task.task_id) not in self.output_locs)
            self.output_locs[(task.job.id, task.task_id)] = worker_id
