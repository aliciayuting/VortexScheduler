"""Draws WF14 (time-series defect inspection) as one job runs on an idle cluster
under the core sweep's WF14 allocation: when each round of images arrives, which
partition each detection runs on and for how long, the aggregation that waits on
all 48 detections, the job's SLO deadline, and each stage's runtime and peak
throughput.

Stage runtimes are at batch size 1. Throughput is at the batch size that maximizes
images per second for one instance, as the admission controller's capacity
estimate takes it (capacity_algo.get_instance_throughput).

Writes [out] (default experiments/results/wf14_core/figures/wf14_diagram.png).

Usage (from cluster_simulation/, after `source set_env.sh`):
    python experiments/postprocess/plot_wf14_diagram.py
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

SIM_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(SIM_DIR, "experiments"))

import run_core_sweep
import core.configs.model_config as mcfg
import core.configs.workflow_config as wcfg

from plot_bucket_size_sweep import INK, INK_SECONDARY, GRID, SURFACE, SERIES

FAMILY, WORKFLOW_ID = "wf14", 14
DETECTOR_COLORS = {"crack": SERIES[0][2], "hole": SERIES[1][2]}
AGG_COLOR = INK


def _best_batch(model: dict, size_gb: int) -> tuple[int, float]:
    """Returns the batch size that maximizes one instance's throughput, and that
    batch's runtime (ms)."""
    times = model["MIG_BATCH_EXEC_TIMES"][size_gb]
    batch = max(range(1, model["MAX_BATCH_SIZE"] + 1), key=lambda b: b / times[b])
    return batch, times[batch]


def draw(out_path: str):
    tasks = next(c for c in wcfg.WORKFLOW_LIST if c["JOB_TYPE"] == WORKFLOW_ID)["TASKS"]
    detections = [t for t in tasks if t["PREV_TASK_INDEX"] == []]
    agg = next(t for t in tasks if t["PREV_TASK_INDEX"])
    alloc = run_core_sweep.ALLOCATIONS[FAMILY]
    runtime = run_core_sweep.job_runtime(FAMILY, WORKFLOW_ID)
    slo = run_core_sweep.job_slo(FAMILY, WORKFLOW_ID)
    src_family, src_wid = run_core_sweep.SLO_FROM[FAMILY]
    src_runtime = run_core_sweep.job_runtime(src_family, src_wid)

    # one lane per partition; each detector's partitions in allocation order
    lanes = {"crack": [], "hole": []}
    model_of = {"crack": detections[0]["MODEL_ID"], "hole": detections[1]["MODEL_ID"]}
    for worker, (size, models) in enumerate(alloc):
        for detector, mid in model_of.items():
            if mid in models:
                lanes[detector].append((worker, size, agg["MODEL_ID"] in models))
    lane_order = [("crack", i) for i in range(len(lanes["crack"]))] + \
                 [("hole", i) for i in range(len(lanes["hole"]))]
    lane_y = {key: len(lane_order) - 1 - i for i, key in enumerate(lane_order)}

    def exec_time(mid: int, worker: int, batch: int = 1) -> float:
        return mcfg.MODELS[mid]["MIG_BATCH_EXEC_TIMES"][alloc[worker][0]][batch]

    fig = plt.figure(figsize=(13, 9.2))
    fig.patch.set_facecolor(SURFACE)
    ax = fig.add_axes([0.13, 0.35, 0.84, 0.53])
    ax.set_facecolor(SURFACE)

    # --- detections: round r's images go one per partition, as on an idle cluster
    rounds = sorted({t["ARRIVAL_OFFSET"] for t in detections})
    agg_ready = 0
    for r, offset in enumerate(rounds, start=1):
        seen = {"crack": 0, "hole": 0}
        for t in (t for t in detections if t["ARRIVAL_OFFSET"] == offset):
            detector = t["DETECTOR"]
            worker = lanes[detector][seen[detector]][0]
            y = lane_y[(detector, seen[detector])]
            seen[detector] += 1
            dur = exec_time(t["MODEL_ID"], worker)
            agg_ready = max(agg_ready, offset + dur)
            ax.add_patch(Rectangle((offset, y - 0.36), dur, 0.72, facecolor=DETECTOR_COLORS[detector],
                                   edgecolor=SURFACE, linewidth=1, zorder=3))
            ax.text(offset + dur / 2, y, f"cam {t['CAMERA_ID']}", ha="center", va="center",
                    color=SURFACE, fontsize=6.5, zorder=4)

        # arrival marker and round label above the lanes
        top = len(lane_order) - 0.3
        ax.annotate("", xy=(offset, top), xytext=(offset, top + 1.5),
                    arrowprops=dict(arrowstyle="-|>", color=INK, linewidth=1.2), zorder=5)
        ax.text(offset + 1.5, top + 1.2, f"Round {r} arrives, +{offset:g} ms\n"
                f"8 images → 16 detections",
                ha="left", va="center", color=INK, fontsize=8.5)
        ax.axvline(offset, color=GRID, linewidth=0.8, zorder=1)

    # --- aggregation: on the partition that also hosts it, once all 48 are done
    agg_worker, agg_lane = next((w, i) for i, (w, _, has_agg) in enumerate(lanes["crack"]) if has_agg)
    agg_y = lane_y[("crack", agg_lane)]
    agg_dur = exec_time(agg["MODEL_ID"], agg_worker)
    assert(abs(agg_ready + agg_dur - runtime) < 1e-6), (agg_ready + agg_dur, runtime)

    # every detection feeds it: a collector across all lanes
    ax.plot([agg_ready, agg_ready], [-0.4, len(lane_order) - 0.6], color=INK_SECONDARY,
            linewidth=1.2, linestyle=(0, (2, 2)), zorder=2)
    ax.add_patch(Rectangle((agg_ready, agg_y - 0.36), agg_dur, 0.72, facecolor=AGG_COLOR,
                           edgecolor=AGG_COLOR, linewidth=1.5, zorder=4))
    ax.annotate(f"aggregate all 48 results\n({agg_dur:.1f} ms, on crack partition {agg_lane + 1})",
                xy=(agg_ready + agg_dur, agg_y), xytext=(agg_ready + 8, agg_y + 2.6),
                color=INK, fontsize=8.5, ha="left", va="center",
                arrowprops=dict(arrowstyle="-|>", color=INK, linewidth=1))
    ax.text(agg_ready - 1.5, -0.95, "waits for all\n48 detections", ha="right", va="top",
            color=INK_SECONDARY, fontsize=8)

    # --- job size, slack and deadline
    y0 = -2.3
    ax.annotate("", xy=(0, y0), xytext=(runtime, y0),
                arrowprops=dict(arrowstyle="<->", color=INK, linewidth=1.2, shrinkA=0, shrinkB=0))
    ax.text(runtime / 2, y0 - 0.35, f"job size {runtime:.1f} ms (idle cluster, batch 1)",
            ha="center", va="top", color=INK, fontsize=9)
    ax.axvspan(runtime, slo, color=GRID, alpha=0.45, zorder=0, linewidth=0)
    ax.text((runtime + slo) / 2, len(lane_order) / 2 - 0.5,
            f"slack\n{slo - runtime:.1f} ms", ha="center", va="center",
            color=INK_SECONDARY, fontsize=9)
    ax.axvline(slo, color=INK, linewidth=1.6, zorder=2)
    ax.text(slo + 1.5, len(lane_order) + 0.9,
            f"SLO deadline\n+{slo} ms\n= WF12's SLO\n({run_core_sweep.SLO_MULTIPLES[src_family]} × "
            f"{src_runtime:.1f} ms)", ha="left", va="center", color=INK, fontsize=8.5)

    ax.set_xlim(-4, slo + 30)
    ax.set_ylim(y0 - 1.2, len(lane_order) + 1.9)
    ax.set_yticks([lane_y[k] for k in lane_order])
    ax.set_yticklabels([f"{d} {i + 1}  ({lanes[d][i][1]} GB)" for d, i in lane_order])
    for label, (d, _) in zip(ax.get_yticklabels(), lane_order):
        label.set_color(INK_SECONDARY)
    ax.set_xticks(range(0, slo + 1, 25))
    ax.tick_params(colors=INK_SECONDARY, labelsize=8.5, length=0)
    ax.set_xlabel("Time since the job was created (ms)", color=INK_SECONDARY, fontsize=10)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.spines["bottom"].set_position(("data", y0 - 1.2))
    fig.text(0.015, 0.63, "GPU partition\n(detector instance)", rotation=90, ha="center",
             va="center", color=INK_SECONDARY, fontsize=10)

    n_gpus = sum(size for size, _ in alloc) // 24
    fig.text(0.02, 0.975, "WF14: time-series defect inspection", color=INK, fontsize=15,
             ha="left", va="top")
    fig.text(0.02, 0.945, f"3 rounds of 8 camera images arriving {rounds[1]:g} ms apart. Each "
             f"image is checked by a crack and a hole YOLOv5 detector, with no dependence between "
             f"rounds,\nand one aggregation waits on all 48 results. Shown: one job on an idle "
             f"cluster ({n_gpus} GPUs as {len(alloc)} × {alloc[0][0]} GB partitions)", color=INK_SECONDARY, fontsize=9.5, ha="left", va="top")

    # --- stage table
    rows = []
    for detector in ("crack", "hole"):
        model = mcfg.MODELS[model_of[detector]]
        n_inst = len(lanes[detector])
        size = lanes[detector][0][1]
        batch, batch_ms = _best_batch(model, size)
        per_inst = 1000 * batch / batch_ms
        per_job = sum(1 for t in detections if t["DETECTOR"] == detector)
        rows.append((f"{detector} detection", f"{model['MODEL_NAME']} ({model_of[detector]})",
                     f"{per_job}  (8 per round)", f"{n_inst} × {size} GB",
                     f"{exec_time(model_of[detector], lanes[detector][0][0]):.1f} ms",
                     f"{batch}  ({batch_ms:.1f} ms)",
                     f"{per_inst * n_inst:.0f} img/s  ({n_inst} × {per_inst:.1f})",
                     f"{per_inst * n_inst / per_job:.1f}", DETECTOR_COLORS[detector]))
    model = mcfg.MODELS[agg["MODEL_ID"]]
    batch, batch_ms = _best_batch(model, alloc[agg_worker][0])
    rows.append(("aggregation", f"{model['MODEL_NAME']} ({agg['MODEL_ID']})", "1",
                 f"1 × {alloc[agg_worker][0]} GB (shared)", f"{agg_dur:.1f} ms",
                 f"{batch}  ({batch_ms:.1f} ms, max)", f"{1000 * batch / batch_ms:.0f} tasks/s",
                 f"{1000 * batch / batch_ms:.0f}", AGG_COLOR))
    bottleneck = min(float(r[7]) for r in rows)

    header = ("Stage", "Model", "Tasks per job", "Instances", "Batch-1 runtime",
              "Best batch (runtime)", "Throughput at best batch", "Jobs/s")
    xs = [0.035, 0.14, 0.335, 0.435, 0.545, 0.645, 0.775, 0.955]
    y = 0.245
    for x, h in zip(xs, header):
        fig.text(x, y, h, color=INK_SECONDARY, fontsize=9, ha="right" if x == xs[-1] else "left",
                 va="center", weight="bold")
    fig.add_artist(plt.Line2D([0.02, 0.965], [y - 0.017] * 2, color=GRID, linewidth=1))
    for row in rows:
        y -= 0.037
        fig.add_artist(Rectangle((0.02, y - 0.008), 0.009, 0.016, color=row[-1],
                                 transform=fig.transFigure))
        for x, cell in zip(xs, row[:-1]):
            fig.text(x, y, cell, color=INK, fontsize=9, ha="right" if x == xs[-1] else "left",
                     va="center", weight="bold" if x == xs[-1] and float(cell) == bottleneck else None)

    wl = run_core_sweep.WORKLOADS[(FAMILY, "const")][WORKFLOW_ID]["PHASES"][0]
    burst = run_core_sweep.WORKLOADS[(FAMILY, "burst")][WORKFLOW_ID]["PHASES"][0]
    fig.add_artist(plt.Line2D([0.02, 0.965], [y - 0.025] * 2, color=GRID, linewidth=1))
    fig.text(0.02, y - 0.05,
             f"Bottleneck: detection, {bottleneck:.1f} jobs/s (each job needs 24 images through "
             f"each detector). Offered load: constant {wl['RATE']} jobs/s "
             f"({100 * wl['RATE'] / bottleneck:.0f}% of capacity); bursty {burst['RATE']} jobs/s with "
             f"{burst['BURST_DURATION'] / 1000:g} s spikes to {burst['BURST_RATE']} jobs/s "
             f"({100 * burst['BURST_RATE'] / bottleneck:.0f}%) every "
             f"{(burst['QUIET_DURATION'] + burst['BURST_DURATION']) / 1000:g} s.",
             color=INK_SECONDARY, fontsize=9, ha="left", va="top", wrap=True)

    fig.savefig(out_path, dpi=200, facecolor=SURFACE)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=str, default=os.path.join(
        SIM_DIR, "experiments", "results", "wf14_core", "figures", "wf14_diagram.png"))
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    draw(args.out)
    print(f"wrote {args.out}")
