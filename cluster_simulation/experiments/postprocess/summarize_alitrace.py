"""Summarize and plot AliTrace policy runs for any workflow.

Usage from ``cluster_simulation/``:
    python experiments/postprocess/summarize_alitrace.py \
        --src results/alitrace_wf6
"""

import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def summarize(src: str) -> pd.DataFrame:
    rows = []
    for run_name in sorted(os.listdir(src)):
        run_dir = os.path.join(src, run_name)
        jobs_path = os.path.join(run_dir, "sim_logs", "job_log.csv")
        if not os.path.isfile(jobs_path):
            continue

        jobs = pd.read_csv(jobs_path)
        with open(os.path.join(run_dir, "overrides.json")) as f:
            metadata = json.load(f)
        completed = jobs.loc[jobs["was_completed"].astype(bool)]
        if jobs.empty:
            continue
        slo = float(jobs["deadline"].iloc[0] - jobs["create_time"].iloc[0])
        within_slo = completed["response_time"] <= slo
        rows.append({
            "run": run_name,
            "workflow_id": metadata.get("workflow_id"),
            "allocation_family": metadata.get("allocation_family"),
            "trace": metadata.get("trace"),
            "jobs_offered": len(jobs),
            "completed": len(completed),
            "incomplete": len(jobs) - len(completed),
            "completion_pct": 100 * len(completed) / len(jobs),
            "within_slo_pct_of_offered": 100 * int(within_slo.sum()) / len(jobs),
            "median_completed_latency_ms": completed["response_time"].median(),
            "p95_completed_latency_ms": completed["response_time"].quantile(0.95),
            "mean_completed_latency_ms": completed["response_time"].mean(),
            "slo_ms": slo,
            "admission_policy": metadata["overrides"]["ADMISSION_CONTROL_POLICY"],
            "drop_policy": metadata["overrides"]["DROP_POLICY"],
            "slo_type": metadata["overrides"]["SLO_TYPE"],
            "admission_burst_size": metadata["overrides"].get("ADMISSION_BURST_SIZE"),
            "admission_drop_rate": metadata["overrides"].get("ADMISSION_DROP_RATE"),
        })
    return pd.DataFrame(rows)


def plot(summary: pd.DataFrame, out_path: str) -> None:
    labels = summary["run"].str.replace(r"^wf\d+_", "", regex=True).tolist()
    x = np.arange(len(summary))
    fig, axes = plt.subplots(2, 1, figsize=(max(12, len(summary) * 1.2), 8),
                             sharex=True)
    axes[0].bar(x, summary["within_slo_pct_of_offered"], color="#277da1")
    axes[0].set_ylabel("Offered jobs completed within SLO (%)")
    axes[0].set_ylim(0, 100)
    axes[0].grid(axis="y", alpha=0.25)

    axes[1].bar(x, summary["p95_completed_latency_ms"], color="#f9844a",
                label="Completed jobs, p95")
    axes[1].plot(x, summary["slo_ms"], color="#b23a48", linestyle="--",
                 marker="o", label="Job SLO")
    axes[1].set_ylabel("Latency (ms)")
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].legend()
    axes[1].set_xticks(x, labels, rotation=30, ha="right")
    workflow_ids = sorted(summary["workflow_id"].dropna().unique())
    workflow_label = ", ".join(f"WF{int(wid)}" for wid in workflow_ids) or "Workflow"
    fig.suptitle(f"{workflow_label} AliTrace policy comparison")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True, help="AliTrace run output directory")
    args = parser.parse_args()
    summary = summarize(args.src)
    if summary.empty:
        raise SystemExit(f"No completed run logs found under {args.src}")
    workflow_ids = sorted(summary["workflow_id"].dropna().unique())
    stem = "wf" + "_".join(str(int(wid)) for wid in workflow_ids) if workflow_ids \
        else "workflow"
    summary.to_csv(os.path.join(args.src, f"{stem}_policy_summary.csv"), index=False)
    plot(summary, os.path.join(args.src, f"{stem}_policy_comparison.png"))
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
