from events.event_manager import EventManager
from events.event import *
from events.event_types import *

from workers.worker import Worker

import pandas as pd
import numpy as np


_TASK_LOG_COLUMNS = [
    "job_id", "task_id", "client_id", "workflow_id", "model_id", "executing_worker_id",
    "arrival_at_scheduler_timestamp", "last_dep_dispatch_timestamp", "arrival_at_worker_timestamp",
    "execution_start_timestamp", "execution_end_timestamp", "dropped_timestamp",
    "curr_unfinished_jobs", "curr_idle_instances", "executing_worker_qlen_at_arrival",
]

_WORKER_LOG_COLUMNS = [
    "worker_id", "instance_id", "model_id", "batch_id", "batched_job_task_ids",
    "batch_size", "execution_start_timestamp", "execution_end_timestamp", "preempted_timestamp",
]


class Logger(EventListener):

    def __init__(self, em: EventManager, workers: dict[UUID, Worker]):
        super().__init__(Agent.LOGGER)

        self.em = em
        self.workers = workers

        self.em.register_listener(self, {
            EVENT_TYPES[EventIds.JOB_SENT_TO_SCHEDULER],
            EVENT_TYPES[EventIds.JOB_ARRIVAL_AT_SCHEDULER],
            EVENT_TYPES[EventIds.TASKS_ARRIVAL_AT_SCHEDULER],

            EVENT_TYPES[EventIds.TASKS_ASSIGNED_TO_WORKER],
            EVENT_TYPES[EventIds.TASKS_INPUTS_SENT_TO_WORKER],
            EVENT_TYPES[EventIds.TASKS_INPUTS_ARRIVAL_AT_WORKER],
            EVENT_TYPES[EventIds.TASKS_OUTPUTS_ASSIGNED_TO_WORKER],
            EVENT_TYPES[EventIds.TASKS_OUTPUTS_SENT_TO_WORKER],
            EVENT_TYPES[EventIds.TASKS_OUTPUTS_ARRIVAL_AT_WORKER],

            EVENT_TYPES[EventIds.JOBS_DROPPED],
            EVENT_TYPES[EventIds.BATCH_STARTED_AT_WORKER],
            EVENT_TYPES[EventIds.BATCH_FINISHED_AT_WORKER],
            EVENT_TYPES[EventIds.RESPONSE_SENT_TO_CLIENT],
            EVENT_TYPES[EventIds.RESPONSE_RECEIVED_AT_CLIENT]
        })

        # Row buffers — converted to DataFrames by finalize() after simulation ends.
        self._task_log_rows: list[dict] = []
        self._task_idx: dict[tuple[int, int], int] = {}  # (job_id, task_id) -> row index
        self._job_task_keys: dict[int, list] = {}        # job_id -> [(job_id, task_id), ...]

        self._worker_log_rows: list[dict] = []
        self._batch_idx: dict[int, int] = {}             # batch_id -> row index

        self.unfinished_jobs: set[int] = set()
        self.deps_to_task = {}

    def finalize(self):
        """Build DataFrames from row buffers. Must be called once after the event loop ends."""
        self.task_log = pd.DataFrame(self._task_log_rows, columns=_TASK_LOG_COLUMNS)
        self.worker_log = pd.DataFrame(self._worker_log_rows, columns=_WORKER_LOG_COLUMNS)

    def _get_curr_idle_instances(self, time: float):
        curr_idle_instances = 0
        for w in self.workers.values():
            for s in w.GPU_state.state_at(time):
                if not s.reserved_batch:
                    curr_idle_instances += 1
        return curr_idle_instances

    def _add_task_row(self, row: dict):
        key = (row["job_id"], row["task_id"])
        self._task_idx[key] = len(self._task_log_rows)
        self._job_task_keys.setdefault(row["job_id"], []).append(key)
        self._task_log_rows.append(row)

    def on_event(self, event: Event):
        if event.type.id == EventIds.JOB_ARRIVAL_AT_SCHEDULER:
            job: Job = event.kwargs["job"]
            self.unfinished_jobs.add(job.id)
            idle = self._get_curr_idle_instances(event.time)
            for task in job.tasks:
                if len(task.required_task_ids) == 0:
                    self._add_task_row({
                        "job_id": job.id, "task_id": task.task_id, "client_id": job.client_id,
                        "model_id": task.model_data.id, "workflow_id": job.job_type_id,
                        "executing_worker_id": "N/A",
                        "arrival_at_scheduler_timestamp": event.time,
                        "last_dep_dispatch_timestamp": np.nan,
                        "arrival_at_worker_timestamp": np.nan,
                        "execution_start_timestamp": np.nan,
                        "execution_end_timestamp": np.nan,
                        "dropped_timestamp": np.nan,
                        "curr_unfinished_jobs": len(self.unfinished_jobs),
                        "curr_idle_instances": idle,
                        "executing_worker_qlen_at_arrival": np.nan,
                    })

        elif event.type.id == EventIds.TASKS_ARRIVAL_AT_SCHEDULER:
            tasks: list[Task] = event.kwargs["tasks"]
            idle = self._get_curr_idle_instances(event.time)
            for task in tasks:
                self._add_task_row({
                    "job_id": task.job.id, "task_id": task.task_id, "client_id": task.job.client_id,
                    "model_id": task.model_data.id, "workflow_id": task.job.job_type_id,
                    "executing_worker_id": "N/A",
                    "arrival_at_scheduler_timestamp": event.time,
                    "last_dep_dispatch_timestamp": np.nan,
                    "arrival_at_worker_timestamp": np.nan,
                    "execution_start_timestamp": np.nan,
                    "execution_end_timestamp": np.nan,
                    "dropped_timestamp": np.nan,
                    "curr_unfinished_jobs": len(self.unfinished_jobs),
                    "curr_idle_instances": idle,
                    "executing_worker_qlen_at_arrival": np.nan,
                })

        elif event.type.id == EventIds.TASKS_INPUTS_SENT_TO_WORKER:
            for task in event.kwargs["tasks"]:
                key = (task.job.id, task.task_id)
                self._task_log_rows[self._task_idx[key]]["last_dep_dispatch_timestamp"] = event.time

        elif event.type.id == EventIds.TASKS_INPUTS_ARRIVAL_AT_WORKER:
            to_worker_id = event.kwargs["to_worker_id"]
            for task in event.kwargs["tasks"]:
                key = (task.job.id, task.task_id)
                assert key in self._task_idx
                row = self._task_log_rows[self._task_idx[key]]
                qlen = self.workers[to_worker_id].get_qlen(task.model_data.id)
                row["executing_worker_id"] = to_worker_id
                row["arrival_at_worker_timestamp"] = event.time
                row["executing_worker_qlen_at_arrival"] = qlen

                if "force_instance_id" not in event.kwargs:
                    assert qlen > 0

        elif event.type.id == EventIds.TASKS_ASSIGNED_TO_WORKER:
            tasks: list[Task] = event.kwargs["tasks"]
            for task in tasks:
                key = (task.job.id, task.task_id)
                if key not in self._task_idx:
                    self._add_task_row({
                        "job_id": task.job.id, "task_id": task.task_id, "client_id": task.job.client_id,
                        "model_id": task.model_data.id, "workflow_id": task.job.job_type_id,
                        "executing_worker_id": event.kwargs["worker_id"],
                        "arrival_at_scheduler_timestamp": np.nan,
                        "last_dep_dispatch_timestamp": np.nan,
                        "arrival_at_worker_timestamp": event.time,
                        "execution_start_timestamp": np.nan,
                        "execution_end_timestamp": np.nan,
                        "dropped_timestamp": np.nan,
                        "curr_unfinished_jobs": len(self.unfinished_jobs),
                        "curr_idle_instances": self._get_curr_idle_instances(event.time),
                        "executing_worker_qlen_at_arrival": np.nan,
                    })
                else:
                    self._task_log_rows[self._task_idx[key]]["executing_worker_id"] = event.kwargs["worker_id"]

                for rt in task.required_task_ids:
                    self.deps_to_task.setdefault((task.job.id, rt), []).append(task)

        elif event.type.id == EventIds.TASKS_OUTPUTS_SENT_TO_WORKER:
            to_worker_id = event.kwargs["to_worker_id"]
            for task in event.kwargs["tasks"]:
                for succ in self.deps_to_task[(task.job.id, task.task_id)]:
                    key = (task.job.id, succ.task_id)
                    if key not in self._task_idx:
                        continue
                    row = self._task_log_rows[self._task_idx[key]]
                    if row["executing_worker_id"] != to_worker_id:
                        continue
                    row["last_dep_dispatch_timestamp"] = event.time

        elif event.type.id == EventIds.TASKS_OUTPUTS_ARRIVAL_AT_WORKER:
            to_worker_id = event.kwargs["to_worker_id"]
            for task in event.kwargs["tasks"]:
                for succ in self.deps_to_task[(task.job.id, task.task_id)]:
                    key = (task.job.id, succ.task_id)
                    if key not in self._task_idx:
                        continue
                    row = self._task_log_rows[self._task_idx[key]]
                    if row["executing_worker_id"] != to_worker_id:
                        continue
                    row["executing_worker_id"] = to_worker_id
                    row["arrival_at_worker_timestamp"] = event.time
                    row["executing_worker_qlen_at_arrival"] = \
                        self.workers[to_worker_id].get_qlen(succ.model_data.id)

        elif event.type.id == EventIds.BATCH_STARTED_AT_WORKER:
            batch: Batch = event.kwargs["batch"]
            for task in batch.tasks:
                key = (task.job.id, task.task_id)
                self._task_log_rows[self._task_idx[key]]["execution_start_timestamp"] = event.time

            self._batch_idx[batch.id] = len(self._worker_log_rows)
            self._worker_log_rows.append({
                "worker_id": event.kwargs["worker_id"],
                "instance_id": event.kwargs["model_instance_id"],
                "model_id": batch.model_data.id,
                "batch_id": batch.id,
                "batched_job_task_ids": [(t.job.id, t.task_id) for t in batch.tasks],
                "batch_size": batch.size(),
                "execution_start_timestamp": event.time,
                "execution_end_timestamp": np.nan,
                "preempted_timestamp": np.nan,
            })

        elif event.type.id == EventIds.BATCH_FINISHED_AT_WORKER:
            batch: Batch = event.kwargs["batch"]
            for task in batch.tasks:
                key = (task.job.id, task.task_id)
                self._task_log_rows[self._task_idx[key]]["execution_end_timestamp"] = event.time
            self._worker_log_rows[self._batch_idx[batch.id]]["execution_end_timestamp"] = event.time

        elif event.type.id == EventIds.JOBS_DROPPED:
            for job_id in event.kwargs["job_ids"]:
                self.unfinished_jobs.remove(job_id)
                for key in self._job_task_keys.get(job_id, []):
                    self._task_log_rows[self._task_idx[key]]["dropped_timestamp"] = event.time

        elif event.type.id == EventIds.RESPONSE_SENT_TO_CLIENT:
            self.unfinished_jobs.remove(event.kwargs["job"].id)
