"""Plot WF15's staggered assembly-line staircase profiles."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt

SIM_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SIM_DIR))

from core.configs.wf15_workload import (  # noqa: E402
    DEFAULT_HORIZON_MS,
    DEFAULT_LINE_COUNT_STEPS,
    DEFAULT_OBJECT_INTERVAL_MS,
    LineCountStep,
    generate_object_arrivals,
)


def scaled_profile(peak_lines: int) -> tuple[LineCountStep, ...]:
    base_peak = max(step.active_lines for step in DEFAULT_LINE_COUNT_STEPS)
    return tuple(
        LineCountStep(step.start_ms, round(step.active_lines * peak_lines / base_peak))
        for step in DEFAULT_LINE_COUNT_STEPS
    )


def plot_patterns(peaks: list[int], output: Path) -> None:
    fig, axes = plt.subplots(len(peaks), 1, figsize=(13, 2.5 * len(peaks)),
                             sharex=True, constrained_layout=True)
    if len(peaks) == 1:
        axes = [axes]

    colors = plt.cm.viridis([0.12 + 0.76 * i / max(len(peaks) - 1, 1)
                             for i in range(len(peaks))])
    horizon_s = DEFAULT_HORIZON_MS / 1000
    for ax, peak, color in zip(axes, peaks, colors):
        profile = scaled_profile(peak)
        times_s = [step.start_ms / 1000 for step in profile]
        counts = [step.active_lines for step in profile]
        ax.stairs(counts, times_s + [horizon_s], color=color,
                  linewidth=2.2, fill=True, alpha=0.13)
        ax.stairs(counts, times_s + [horizon_s], color=color,
                  linewidth=2.2)
        arrivals = generate_object_arrivals(
            horizon_ms=DEFAULT_HORIZON_MS,
            object_interval_ms=DEFAULT_OBJECT_INTERVAL_MS,
            line_count_steps=profile,
            align_cycles=False,
        )
        ax.set_ylabel(f"{peak} peak\nlines")
        ax.set_ylim(0, peak * 1.28)
        ax.set_yticks([0, peak // 2, peak])
        ax.grid(axis="y", alpha=0.25)
        ax.text(1.005, 0.5, f"{len(arrivals):,} objects",
                transform=ax.transAxes, rotation=270, va="center", ha="left")

    axes[-1].set_xlabel("Simulation time (seconds)")
    axes[-1].set_xlim(0, horizon_s)
    fig.suptitle("WF15 assembly-line workload patterns — staggered 3-second cycles")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--peaks", nargs="+", type=int,
                        default=[100, 120, 140, 160, 200, 240])
    parser.add_argument("--out", type=Path,
                        default=SIM_DIR / "results" / "wf15_line_policy_sweep"
                                / "wf15_workload_patterns.png")
    args = parser.parse_args()
    if not args.peaks or any(peak < 1 for peak in args.peaks):
        parser.error("peak counts must be positive")
    plot_patterns(args.peaks, args.out)


if __name__ == "__main__":
    main()
