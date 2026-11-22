"""Run WF15 line-count workloads across the standard scheduler policies.

The workload uses the three-peak staircase from ``core.configs.wf15_workload``
scaled to each requested peak line count. The allocation is the canonical WF15
custom placement produced by the Vortex allocation DP and used by the existing
core sweep. Each output run keeps its job log and config snapshot for
``postprocess/plot_tail.py``; large per-task logs are removed after summary
metrics have been collected.

Example (Python 3.10+):
    python3 experiments/run_wf15_line_policy_sweep.py --parallel 2
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

SIM_DIR = Path(__file__).resolve().parents[1]
EXP_DIR = SIM_DIR / "experiments"
sys.path.insert(0, str(SIM_DIR))
os.environ.setdefault("SIMULATION_DIR", str(SIM_DIR))

from core.configs.wf15_workload import (
    DEFAULT_HORIZON_MS,
    DEFAULT_LINE_COUNT_STEPS,
    DEFAULT_OBJECT_INTERVAL_MS,
    LineCountStep,
    generate_object_arrivals,
    write_arrival_trace,
)
from experiments import run_alitrace_pipeline_sweep as policy_source
from experiments import run_core_sweep


WORKFLOW_ID = 15
SLO_MS = 2_000
DEFAULT_PEAKS = (100, 120, 140, 160)
TOKEN_BURST_SIZES = (1, 5, 10, 20)


def policy_matrix() -> dict[str, dict]:
    """Return requested options, including auto and tuned token buckets."""
    policies = {
        "no_drop": policy_source.POLICIES["nodrop"],
        "early_drop": policy_source.POLICIES["early"],
        "lookahead_drop": policy_source.POLICIES["lookahead"],
        "nexus": policy_source.POLICIES["nexus"],
        "flat_admission_10pct": policy_source.POLICIES["flat10"],
        "token_bucket_auto": policy_source.POLICIES["token_auto"],
    }
    for size in TOKEN_BURST_SIZES:
        policies[f"token_bucket_burst_{size}"] = policy_source.POLICIES[
            f"token_burst{size}"
        ]
    return policies


def scaled_profile(peak_lines: int) -> tuple[LineCountStep, ...]:
    base_peak = max(step.active_lines for step in DEFAULT_LINE_COUNT_STEPS)
    return tuple(
        LineCountStep(step.start_ms, round(step.active_lines * peak_lines / base_peak))
        for step in DEFAULT_LINE_COUNT_STEPS
    )


def _trace_path(root: Path, peak_lines: int) -> Path:
    return root / "arrival_traces" / f"wf15_peak_{peak_lines}.csv"


def prepare_traces(root: Path, peak_lines: list[int]) -> None:
    for peak in peak_lines:
        trace = _trace_path(root, peak)
        arrivals = generate_object_arrivals(
            horizon_ms=DEFAULT_HORIZON_MS,
            object_interval_ms=DEFAULT_OBJECT_INTERVAL_MS,
            line_count_steps=scaled_profile(peak),
            align_cycles=False,
        )
        write_arrival_trace(arrivals, trace)
        print(f"[trace] peak={peak} lines; jobs={len(arrivals)}; {trace}", flush=True)


def child_environment() -> dict[str, str]:
    env = os.environ.copy()
    env["SIMULATION_DIR"] = str(SIM_DIR)
    env["PYTHONPATH"] = os.pathsep.join(
        path for path in (str(SIM_DIR), env.get("PYTHONPATH", "")) if path
    )
    return env


def run_one(policy: str, peak_lines: int, output: Path, root: Path) -> None:
    import core.configs.gen_config as gcfg
    from run_experiment import run_experiment

    policies = policy_matrix()
    if policy not in policies:
        raise ValueError(f"unknown policy {policy!r}")

    trace_path = _trace_path(root, peak_lines).resolve()
    profile = scaled_profile(peak_lines)
    arrivals = generate_object_arrivals(
        horizon_ms=DEFAULT_HORIZON_MS,
        object_interval_ms=DEFAULT_OBJECT_INTERVAL_MS,
        line_count_steps=profile,
        align_cycles=False,
    )
    allocation = run_core_sweep.ALLOCATIONS["wf15"]
    overrides = {
        "CLIENT_CONFIGS": [{WORKFLOW_ID: {
            "WORKLOAD": {
                "ARRIVAL_PROCESS": "ALITRACE",
                "TRACE_FILE_PATH": str(trace_path),
            },
            "SLO": SLO_MS,
        }}],
        "ALLOCATION_STRATEGY": "CUSTOM",
        "CUSTOM_ALLOCATION": allocation,
        "DISPATCH_POLICY": "ROUND_ROBIN",
        "PRODUCE_EVENT_LOG": False,
        "PRODUCE_QUEUE_LOG": True,
        "EXEC_TIME_DISTRIBUTION": "NORMAL",
        "BOOST_POLICY": "FCFS",
    }
    overrides.update(policies[policy])
    for option, value in overrides.items():
        setattr(gcfg, option, value)

    output.mkdir(parents=True, exist_ok=True)
    (output / "sim_logs").mkdir(exist_ok=True)
    run_experiment(True, str(output))

    run_metadata = {
        "config": f"wf15_peak_{peak_lines}_{policy}",
        "workflow_id": WORKFLOW_ID,
        "peak_lines": peak_lines,
        "arrival_jobs": len(arrivals),
        "line_count_steps": [
            {"start_ms": step.start_ms, "active_lines": step.active_lines}
            for step in profile
        ],
        "cycle_alignment": "staggered",
        "object_interval_ms": DEFAULT_OBJECT_INTERVAL_MS,
        "horizon_ms": DEFAULT_HORIZON_MS,
        "slo_ms": SLO_MS,
        "allocation_family": "wf15",
        "allocation_source": "run_core_sweep.ALLOCATIONS[\"wf15\"] (Vortex DP custom allocation)",
        "overrides": overrides,
    }
    with (output / "overrides.json").open("w") as f:
        json.dump(run_metadata, f, indent=2)

    with (output / "configs" / "gen_config.py").open("a") as f:
        f.write(f"\n\n# Overrides applied by run_wf15_line_policy_sweep.py ({policy}, peak {peak_lines})\n")
        for key, value in overrides.items():
            f.write(f"{key} = {value!r}\n")


def summarize_run(run_path: Path, peak_lines: int, policy: str) -> dict:
    logs_dir = run_path / "sim_logs"
    jobs = pd.read_csv(logs_dir / "job_log.csv")
    completed = jobs["was_completed"].astype(bool)
    in_slo = completed & (
        jobs["response_time"] <= jobs["deadline"] - jobs["create_time"]
    )
    completion_times = jobs.loc[completed, "response_time"]
    queue_path = logs_dir / "queue_length_log.csv"
    queues = pd.read_csv(queue_path) if queue_path.exists() else pd.DataFrame()
    result = {
        "peak_lines": peak_lines,
        "policy": policy,
        "jobs": len(jobs),
        "completed_fraction": float(completed.mean()) if len(jobs) else 0.0,
        "within_slo_fraction": float(in_slo.mean()) if len(jobs) else 0.0,
        "p50_completed_response_ms": float(completion_times.quantile(0.50)) if len(completion_times) else np.nan,
        "p95_completed_response_ms": float(completion_times.quantile(0.95)) if len(completion_times) else np.nan,
        "max_recorded_queue": int(queues["queue_length"].max()) if not queues.empty else 0,
    }

    # Tail plotting only needs job_log.csv and the config snapshot. Keep those,
    # plus worker_config_log.csv to preserve the actual model placement; discard
    # bulky diagnostic logs generated for simulation internals.
    for path in logs_dir.iterdir():
        if path.is_file() and path.name != "job_log.csv":
            path.unlink()
    return result


def write_summary(root: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path = root / "summary.csv"
    rows = sorted(rows, key=lambda row: (row["peak_lines"], row["policy"]))
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_matrix(root: Path, peaks: list[int], policies: list[str], parallel: int,
               python: str) -> list[str]:
    all_policies = policy_matrix()
    pending = [(peak, policy) for peak in peaks for policy in policies]
    running = []
    failed = []
    rows = []
    summary_path = root / "summary.csv"
    if summary_path.exists():
        rows = pd.read_csv(summary_path).to_dict("records")
    completed_keys = {(int(row["peak_lines"]), row["policy"]) for row in rows}
    pending = [pair for pair in pending if pair not in completed_keys]

    while pending or running:
        while pending and len(running) < parallel:
            peak, policy = pending.pop(0)
            name = f"peak_{peak:03d}_{policy}"
            run_path = root / name
            run_path.mkdir(parents=True, exist_ok=True)
            log = (run_path / "run.log").open("w")
            command = [
                python, "-u", str(Path(__file__).resolve()),
                "--run-one", policy,
                "--peak-lines-one", str(peak),
                "--out", str(run_path),
                "--root", str(root),
            ]
            process = subprocess.Popen(
                command, cwd=SIM_DIR, env=child_environment(),
                stdout=log, stderr=subprocess.STDOUT,
            )
            running.append((peak, policy, run_path, process, log))
            print(f"[start] peak={peak} policy={policy}", flush=True)

        for item in list(running):
            peak, policy, run_path, process, log = item
            if process.poll() is None:
                continue
            log.close()
            running.remove(item)
            if process.returncode:
                failed.append(f"peak_{peak:03d}_{policy}")
                print(f"[FAILED] peak={peak} policy={policy}; see {run_path / 'run.log'}", flush=True)
                continue
            row = summarize_run(run_path, peak, policy)
            rows = [r for r in rows if not (
                int(r["peak_lines"]) == peak and r["policy"] == policy
            )]
            rows.append(row)
            write_summary(root, rows)
            print(
                f"[done] peak={peak} policy={policy} jobs={row['jobs']} "
                f"within-SLO={row['within_slo_fraction']:.3%} "
                f"p95={row['p95_completed_response_ms']:.1f}ms "
                f"max-queue={row['max_recorded_queue']}", flush=True
            )

        if running:
            time.sleep(1)
    return failed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path,
                        default=SIM_DIR / "results" / "wf15_line_policy_sweep")
    parser.add_argument("--peak-lines", nargs="+", type=int, default=list(DEFAULT_PEAKS))
    parser.add_argument("--policies", nargs="+", choices=sorted(policy_matrix()),
                        default=list(policy_matrix()))
    parser.add_argument("--parallel", type=int, default=2)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--run-one", choices=sorted(policy_matrix()), help=argparse.SUPPRESS)
    parser.add_argument("--peak-lines-one", dest="peak_one", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--root", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.run_one:
        if args.peak_one is None or args.out is None or args.root is None:
            parser.error("internal single-run arguments are incomplete")
        run_one(args.run_one, args.peak_one, args.out, args.root)
        return
    if args.parallel < 1:
        parser.error("--parallel must be at least 1")
    if any(peak < 1 for peak in args.peak_lines):
        parser.error("peak line counts must be positive")
    args.out.mkdir(parents=True, exist_ok=True)

    prepare_traces(args.out, args.peak_lines)
    failures = run_matrix(args.out, args.peak_lines, args.policies,
                          args.parallel, args.python)
    if failures:
        raise SystemExit(f"{len(failures)} run(s) failed: {', '.join(failures)}")
    print(f"Summary: {args.out / 'summary.csv'}")
    print("Plot commands:")
    for peak in args.peak_lines:
        run_paths = [str(args.out / f"peak_{peak:03d}_{policy}" / "sim_logs")
                     for policy in args.policies]
        labels = list(args.policies)
        print(
            f"  python3 experiments/postprocess/plot_tail.py --srcs "
            f"{' '.join(run_paths)} --labels {' '.join(labels)} --pdf --per-slo "
            f"--out {args.out / f'tail_by_slo_peak_{peak}.pdf'}"
        )


if __name__ == "__main__":
    main()
