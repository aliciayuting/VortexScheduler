import sys
import os
import ast
import importlib.util

import numpy as np
import pandas as pd

from core.network import *


class LogVerifier:

    def __init__(self, job_log: pd.DataFrame, task_log: pd.DataFrame, batch_log: pd.DataFrame,
                 worker_log: pd.DataFrame, centralized: bool, gcfg, mcfg, wcfg):
        
        self.job_log = job_log
        self.task_log = task_log
        self.batch_log = batch_log
        self.worker_log = worker_log

        self.is_centralized = centralized

        self.gcfg = gcfg
        self.mcfg = mcfg
        self.wcfg = wcfg


    def run(self):
        self.trace_task_arrivals()


    def trace_task_arrivals(self):
        # trace state
        worker_model_qs = {} # worker ID -> model ID -> awaiting tasks [(job ID, task ID)]
        worker_instances = {} # worker ID -> model ID -> [instance ID]

        # parse configs
        sched_worker_id = self.worker_log["worker_id"][0]
        wf_cfgs = {}
        for cfg in self.wcfg.WORKFLOW_LIST:
            wf_cfgs[cfg["JOB_TYPE"]] = {tcfg["TASK_INDEX"]: tcfg 
                                        for tcfg in cfg["TASKS"]}
            
        for _, wrow in self.worker_log.iterrows():
            if wrow["worker_id"] not in worker_instances:
                worker_instances[wrow["worker_id"]] = {}

            if wrow["model_id"] not in worker_instances[wrow["worker_id"]]:
                worker_instances[wrow["worker_id"]][wrow["model_id"]] = []

            worker_instances[wrow["worker_id"]][wrow["model_id"]].append(
                wrow["instance_id"])
            
        total_rows = len(self.task_log)

        prev_time = {(wid, mid): 0 for wid in worker_instances.keys()
                     for mid in set(self.task_log["model_id"])}
        
        for i, row in self.task_log.sort_values(["arrival_at_worker_timestamp", 
                                                 "executing_worker_qlen_at_arrival"]).iterrows():
            print(f"{i} / {total_rows} = {i / total_rows * 100:.1f}% done...")
            print(row["job_id"], row["task_id"])
            print()

            tcfg = wf_cfgs[row["workflow_id"]][row["task_id"]]

            if len(tcfg["PREV_TASK_INDEX"]) == 0:
                # if initial task, check arrival at scheduler timestamp
                if (not self.is_centralized) or (not self.gcfg.ENABLE_NETWORKING_DELAYS) or \
                    (sched_worker_id == row["executing_worker_id"]):
                    assert(row["arrival_at_scheduler_timestamp"] == row["arrival_at_worker_timestamp"])
                else:
                    assert(row["arrival_at_worker_timestamp"] ==
                           (row["arrival_at_scheduler_timestamp"] + CPU_to_CPU_delay(tcfg["INPUT_SIZE"])))

            if not self.gcfg.ENABLE_NETWORKING_DELAYS:
                assert(row["last_dep_dispatch_timestamp"]==row["arrival_at_worker_timestamp"])

            if row["executing_worker_id"] not in worker_model_qs:
                worker_model_qs[row["executing_worker_id"]] = {}
            
            if row["model_id"] not in worker_model_qs[row["executing_worker_id"]]:
                worker_model_qs[row["executing_worker_id"]][row["model_id"]] = []

            # should not duplicate tasks
            assert((row["job_id"], row["task_id"]) not in 
                   worker_model_qs[row["executing_worker_id"]][row["model_id"]])

            worker_model_qs[row["executing_worker_id"]][row["model_id"]].append(
                (row["job_id"], row["task_id"]))
            
            # update q based on batch exec history
            started_batches = self.batch_log[(self.batch_log["worker_id"] == row["executing_worker_id"]) &
                                             (self.batch_log["model_id"] == row["model_id"]) &
                                             (self.batch_log["execution_start_timestamp"] < row["arrival_at_worker_timestamp"])]
            for _, brow in started_batches.iterrows():
                for (j, t) in ast.literal_eval(brow["batched_job_task_ids"]):
                    if (j, t) in worker_model_qs[row["executing_worker_id"]][row["model_id"]]:
                        worker_model_qs[row["executing_worker_id"]][row["model_id"]].remove((j, t))
            
            prev_time[(row["executing_worker_id"], row["model_id"])] = row["arrival_at_worker_timestamp"]

            print(row["executing_worker_qlen_at_arrival"])
            print(worker_model_qs[row["executing_worker_id"]][row["model_id"]])

            # check qlen logging
            assert(row["executing_worker_qlen_at_arrival"] == 
                   len(worker_model_qs[row["executing_worker_id"]][row["model_id"]]))

            # verify exec against batch log
            mask = ((self.batch_log["model_id"]==row["model_id"]) & 
                    (self.batch_log["execution_start_timestamp"] == row["execution_start_timestamp"]) &
                    (self.batch_log["execution_end_timestamp"] == row["execution_end_timestamp"]))

            assert(not self.batch_log.loc[mask].empty)

            s = f"({row['job_id']}, {row['task_id']})"
            assert(any(s in r["batched_job_task_ids"] for  _, r in self.batch_log.loc[mask].iterrows()))

            # if qlen <= max bsize && idle instance exists, should start task right away
            should_start_exec_immediately = False
            if len(worker_model_qs[row["executing_worker_id"]][row["model_id"]]) <= \
                self.mcfg.MODELS[row["model_id"]]["MAX_BATCH_SIZE"]:

                for instance_id in worker_instances[row["executing_worker_id"]][row["model_id"]]:
                    mask = ((self.batch_log["instance_id"]==instance_id) & 
                            (self.batch_log["execution_start_timestamp"] < row["arrival_at_worker_timestamp"]) &
                            (self.batch_log["execution_end_timestamp"] > row["arrival_at_worker_timestamp"]))
                    
                    # if no batch is being executed currently
                    if self.batch_log.loc[mask].empty:
                        should_start_exec_immediately = True

            if should_start_exec_immediately:
                assert(row["arrival_at_worker_timestamp"] == row["execution_start_timestamp"])


if __name__ == "__main__":
    results_dir = sys.argv[1]
    is_centralized = bool(sys.argv[2])

    gcfg_path = os.path.join(results_dir, "configs/gen_config.py")
    mcfg_path = os.path.join(results_dir, "configs/model_config.py")
    wcfg_path = os.path.join(results_dir, "configs/workflow_config.py")

    modules = {}

    for path in [gcfg_path, mcfg_path, wcfg_path]:
        spec = importlib.util.spec_from_file_location(f"results_{path.replace('/', '_')}", path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load spec from {path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        modules[path] = module

    exec_verifier = LogVerifier(
        pd.read_csv(os.path.join(results_dir, "sim_logs/job_log.csv")),
        pd.read_csv(os.path.join(results_dir, "sim_logs/task_log.csv")),
        pd.read_csv(os.path.join(results_dir, "sim_logs/worker_batch_log.csv")),
        pd.read_csv(os.path.join(results_dir, "sim_logs/worker_config_log.csv")),
        is_centralized,
        modules[gcfg_path],
        modules[mcfg_path],
        modules[wcfg_path]
    )

    exec_verifier.run()