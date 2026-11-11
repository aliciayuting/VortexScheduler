"""Estimate WF15 assembly-line capacity from the configured simulator models.

Run from any directory with the repository Python environment, for example:
    python experiments/run_wf15_capacity_sweep.py

Each trial scales the default three-peak WF15 line-count profile so its peak is
the requested number of lines, then simulates staggered 3-second line cycles.
The summary reports the fraction of jobs completed within WF15's 2-second SLO.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

SIM_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SIM_DIR))
os.environ.setdefault("SIMULATION_DIR", str(SIM_DIR))

from core.configs import gen_config as gcfg
from core.configs import model_config as mcfg
from core.configs.wf15_workload import (
    DEFAULT_HORIZON_MS,
    DEFAULT_LINE_COUNT_STEPS,
    DEFAULT_OBJECT_INTERVAL_MS,
    LineCountStep,
    generate_object_arrivals,
)
from simulations.simulation import Simulation


SLO_MS = 2_000
DETECTOR_MODEL_IDS = (17, 18)
TASKS_PER_DETECTOR_PER_OBJECT = 24


def wf15_allocation() -> list[tuple[int, list[int]]]:
    """Use WF15's 16-partition, 8-per-detector placement from core_sweep."""
    return (
        [(6, [17])] * 7
        + [(6, [17, 19])]
        + [(6, [18])] * 8
    )


def theoretical_capacity() -> dict[str, float]:
    """Ideal throughput ceiling from the fastest legal batch per detector."""
    allocation = wf15_allocation()
    capacity_by_model = {}
    for model_id in DETECTOR_MODEL_IDS:
        instances = sum(model_id in model_ids for _, model_ids in allocation)
        model_cfg = mcfg.MODELS[model_id]
        times_by_partition = model_cfg["MIG_BATCH_EXEC_TIMES"]
        legal_batch = model_cfg["MAX_BATCH_SIZE"]
        best_tasks_per_second_per_instance = max(
            batch_size / exec_ms * 1000
            for partition_times in times_by_partition.values()
            for batch_size, exec_ms in partition_times.items()
            if batch_size <= legal_batch
        )
        capacity_by_model[model_id] = instances * best_tasks_per_second_per_instance

    bottleneck_tasks_per_second = min(capacity_by_model.values())
    objects_per_second = bottleneck_tasks_per_second / TASKS_PER_DETECTOR_PER_OBJECT
    lines = objects_per_second * DEFAULT_OBJECT_INTERVAL_MS / 1000
    return {
        "objects_per_second": objects_per_second,
        "ideal_continuous_lines": lines,
        "detector_17_tasks_per_second": capacity_by_model[17],
        "detector_18_tasks_per_second": capacity_by_model[18],
    }


def scale_profile(peak_lines: int) -> tuple[LineCountStep, ...]:
    base_peak = max(step.active_lines for step in DEFAULT_LINE_COUNT_STEPS)
    return tuple(
        LineCountStep(step.start_ms, round(step.active_lines * peak_lines / base_peak))
        for step in DEFAULT_LINE_COUNT_STEPS
    )


def run_trial(peak_lines: int, output_dir: Path) -> dict:
    if peak_lines < 1:
        raise ValueError("peak line count must be positive")

    profile = scale_profile(peak_lines)
    arrivals = generate_object_arrivals(
        horizon_ms=DEFAULT_HORIZON_MS,
        object_interval_ms=DEFAULT_OBJECT_INTERVAL_MS,
        line_count_steps=profile,
        align_cycles=False,
    )
    trace_path = output_dir / f"wf15_peak_{peak_lines}.csv"
    with trace_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["arrival_time"])
        writer.writerows((arrival,) for arrival in arrivals)

    gcfg.CLIENT_CONFIGS = [{15: {
        "WORKLOAD": {
            "ARRIVAL_PROCESS": "ALITRACE",
            "TRACE_FILE_PATH": str(trace_path),
        },
        "SLO": SLO_MS,
    }}]
    gcfg.ALLOCATION_STRATEGY = "CUSTOM"
    gcfg.CUSTOM_ALLOCATION = wf15_allocation()
    gcfg.SLO_TYPE = "JOB_LEVEL"
    gcfg.DISPATCH_POLICY = "ROUND_ROBIN"
    gcfg.DROP_POLICY = "NONE"
    gcfg.SHADOW_DROP_POLICY = "NONE"
    gcfg.ADMISSION_CONTROL_POLICY = "NONE"
    gcfg.PRODUCE_EVENT_LOG = False
    gcfg.PRODUCE_QUEUE_LOG = True
    gcfg.ENABLE_VERIFICATION = False
    gcfg.ENABLE_LIVE_VERIFICATION = False
    np.random.seed(42)

    trial_dir = output_dir / f"peak_{peak_lines}"
    trial_dir.mkdir(parents=True, exist_ok=True)
    sim = Simulation(True, str(trial_dir))
    sim.run()

    jobs = pd.read_csv(trial_dir / "job_log.csv")
    completed = jobs["was_completed"].astype(bool)
    within_slo = completed & (
        jobs["response_time"] <= jobs["deadline"] - jobs["create_time"]
    )
    completed_times = jobs.loc[completed, "response_time"]

    queue_path = trial_dir / "queue_length_log.csv"
    queues = pd.read_csv(queue_path) if queue_path.exists() else pd.DataFrame()
    return {
        "peak_lines": peak_lines,
        "arrival_jobs": len(jobs),
        "completed_fraction": float(completed.mean()) if len(jobs) else 0.0,
        "within_slo_fraction": float(within_slo.mean()) if len(jobs) else 0.0,
        "p50_response_ms": float(completed_times.quantile(0.50)) if len(completed_times) else math.nan,
        "p95_response_ms": float(completed_times.quantile(0.95)) if len(completed_times) else math.nan,
        "max_recorded_queue": int(queues["queue_length"].max()) if not queues.empty else 0,
        "average_arrival_rate_per_second": len(arrivals) / (DEFAULT_HORIZON_MS / 1000),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--peak-lines", nargs="+", type=int, default=[64, 128, 192, 256, 320],
        help="peak line counts to simulate (default: 64 128 192 256 320)",
    )
    parser.add_argument(
        "--out", type=Path,
        default=SIM_DIR / "results" / "wf15_capacity_estimate",
        help="directory for the compact capacity summary CSV",
    )
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    estimate = theoretical_capacity()
    print(
        "Ideal detector-only ceiling: "
        f"{estimate['objects_per_second']:.2f} objects/s, "
        f"{estimate['ideal_continuous_lines']:.1f} continuous lines "
        f"({estimate['detector_17_tasks_per_second']:.0f}/"
        f"{estimate['detector_18_tasks_per_second']:.0f} tasks/s on detectors 17/18)"
    )

    summary_path = args.out / "summary.csv"
    if summary_path.exists():
        prior = pd.read_csv(summary_path).to_dict("records")
        rerun_peaks = set(args.peak_lines)
        summaries = [row for row in prior if int(row["peak_lines"]) not in rerun_peaks]
    else:
        summaries = []
    with tempfile.TemporaryDirectory(prefix="wf15_capacity_") as scratch:
        for peak in args.peak_lines:
            print(f"Running WF15 trial at peak {peak} active lines...", flush=True)
            result = run_trial(peak, Path(scratch))
            summaries.append(result)
            print(
                f"  jobs={result['arrival_jobs']}, "
                f"within-SLO={result['within_slo_fraction']:.3%}, "
                f"completed={result['completed_fraction']:.3%}, "
                f"p95={result['p95_response_ms']:.1f} ms, "
                f"max queue={result['max_recorded_queue']}"
            )
            # Persist after every trial so an interrupted longer sweep retains
            # all completed measurements.
            with summary_path.open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(summaries[0].keys()))
                writer.writeheader()
                writer.writerows(summaries)
    print(f"Summary written to {summary_path}")


if __name__ == "__main__":
    main()
