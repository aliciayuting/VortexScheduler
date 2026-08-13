#!/usr/bin/env python3
"""
Standalone correctness checks on simulation log CSVs.

Usage:
    python smoke_test.py <results_dir>

Checks:
  1. Each (job_id, task_id) appears exactly once; completed jobs have all tasks.
  2. DAG order: every task starts only after all predecessors have finished.
  3. Model assignment: each task runs on a worker that has the required model loaded.
  4. Memory allocation: each worker's loaded model memory fits within a valid partition size.
"""

import sys
import os
import importlib.util

import pandas as pd


# ── helpers ──────────────────────────────────────────────────────────────────

def load_module(path: str):
    spec = importlib.util.spec_from_file_location(path.replace("/", "_"), path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def build_wf_cfgs(wcfg) -> dict:
    """Returns {workflow_id: {task_index: task_cfg}}"""
    return {cfg["JOB_TYPE"]: {t["TASK_INDEX"]: t for t in cfg["TASKS"]}
            for cfg in wcfg.WORKFLOW_LIST}


def report(name: str, failures: list[str]) -> int:
    if failures:
        print(f"  FAIL  {name}")
        for f in failures[:10]:
            print(f"        {f}")
        if len(failures) > 10:
            print(f"        ... and {len(failures) - 10} more")
    else:
        print(f"  PASS  {name}")
    return len(failures)


# ── checks ───────────────────────────────────────────────────────────────────

def check_each_task_executes_once(task_log, job_log, wcfg) -> int:
    failures = []

    dupes = task_log.groupby(["job_id", "task_id"]).size()
    dupes = dupes[dupes > 1]
    if not dupes.empty:
        for (jid, tid), count in dupes.items():
            failures.append(f"(job={jid}, task={tid}) appears {count} times")
        return report("each task executes once", failures)

    wf_tasks = {cfg["JOB_TYPE"]: cfg["TASKS"] for cfg in wcfg.WORKFLOW_LIST}
    executed = set(zip(task_log["job_id"].astype(int), task_log["task_id"].astype(int)))
    for _, jrow in job_log[job_log["was_completed"]].iterrows():
        for t in wf_tasks[int(jrow["workflow_id"])]:
            if (int(jrow["job_id"]), t["TASK_INDEX"]) not in executed:
                failures.append(
                    f"job={int(jrow['job_id'])} workflow={int(jrow['workflow_id'])}: "
                    f"task {t['TASK_INDEX']} missing")

    return report("each task executes once", failures)


def check_dag_order(task_log, wcfg) -> int:
    wf_cfgs = build_wf_cfgs(wcfg)
    task_end = {
        (int(r["job_id"]), int(r["task_id"])): r["execution_end_timestamp"]
        for _, r in task_log.dropna(subset=["execution_end_timestamp"]).iterrows()
    }
    failures = []
    for _, row in task_log.dropna(subset=["execution_start_timestamp"]).iterrows():
        tcfg = wf_cfgs[int(row["workflow_id"])][int(row["task_id"])]
        for prev_tid in tcfg["PREV_TASK_INDEX"]:
            key = (int(row["job_id"]), prev_tid)
            if key not in task_end:
                failures.append(
                    f"(job={int(row['job_id'])}, task={int(row['task_id'])}): "
                    f"predecessor task {prev_tid} never completed")
            elif task_end[key] > row["execution_start_timestamp"]:
                failures.append(
                    f"(job={int(row['job_id'])}, task={int(row['task_id'])}): "
                    f"started at {row['execution_start_timestamp']:.2f} but "
                    f"predecessor {prev_tid} ended at {task_end[key]:.2f}")
    return report("DAG execution order", failures)


def check_task_model_assignment(task_log, job_log, worker_log, wcfg) -> int:
    wf_cfgs = build_wf_cfgs(wcfg)
    worker_models: dict[str, set] = (
        worker_log.groupby("worker_id")["model_id"].apply(set).to_dict())
    dropped_jobs = set(job_log.loc[~job_log["was_completed"], "job_id"])
    failures = []
    for _, row in task_log.iterrows():
        if int(row["job_id"]) in dropped_jobs and pd.isna(row["execution_start_timestamp"]):
            continue
        expected = wf_cfgs[int(row["workflow_id"])][int(row["task_id"])]["MODEL_ID"]
        if int(row["model_id"]) != expected:
            failures.append(
                f"(job={int(row['job_id'])}, task={int(row['task_id'])}): "
                f"logged model_id={int(row['model_id'])} but workflow requires {expected}")
        wid = row["executing_worker_id"]
        if wid not in worker_models or expected not in worker_models[wid]:
            failures.append(
                f"(job={int(row['job_id'])}, task={int(row['task_id'])}): "
                f"worker {wid} has no instance of model {expected}")
    return report("task model assignment", failures)


def check_memory_allocation(worker_log, mcfg, gcfg) -> int:
    model_sizes = {i: m["MODEL_SIZE"] for i, m in enumerate(mcfg.MODELS)}
    valid_sizes = gcfg.VALID_WORKER_SIZES
    failures = []
    for worker_id, group in worker_log.groupby("worker_id"):
        total = sum(model_sizes[int(mid)] for mid in group["model_id"])
        if not any(total <= v for v in valid_sizes):
            failures.append(
                f"worker {worker_id}: {total:,} kB loaded, "
                f"exceeds all valid sizes {[f'{v:,}' for v in valid_sizes]}")
    return report("memory allocation", failures)


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <results_dir>")
        sys.exit(1)

    results_dir = sys.argv[1]
    logs_dir = os.path.join(results_dir, "sim_logs")
    cfg_dir = os.path.join(results_dir, "configs")

    gcfg = load_module(os.path.join(cfg_dir, "gen_config.py"))
    mcfg = load_module(os.path.join(cfg_dir, "model_config.py"))
    wcfg = load_module(os.path.join(cfg_dir, "workflow_config.py"))

    task_log   = pd.read_csv(os.path.join(logs_dir, "task_log.csv"))
    job_log    = pd.read_csv(os.path.join(logs_dir, "job_log.csv"))
    worker_log = pd.read_csv(os.path.join(logs_dir, "worker_config_log.csv"))

    dropped = job_log[~job_log["was_completed"]]

    print(f"\nVerifying: {results_dir}")
    print(f"  {len(task_log)} tasks, {len(job_log)} jobs, {len(worker_log)} worker instances")
    print(f"  {len(dropped)} job(s) dropped" + (f": {sorted(dropped['job_id'].tolist())}" if len(dropped) else ""))
    print()

    total_failures = 0
    total_failures += check_each_task_executes_once(task_log, job_log, wcfg)
    total_failures += check_dag_order(task_log, wcfg)
    total_failures += check_task_model_assignment(task_log, job_log, worker_log, wcfg)
    total_failures += check_memory_allocation(worker_log, mcfg, gcfg)

    print()
    if total_failures:
        print(f"FAILED — {total_failures} violation(s)")
        sys.exit(1)
    else:
        print("All checks passed")


if __name__ == "__main__":
    main()
