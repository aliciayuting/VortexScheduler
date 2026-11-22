"""WF15 assembly-line workload configuration and arrival trace generator.

WF15's workflow definition supplies one object's 3-round image inspection DAG
(three image rounds, 600 ms apart). This module defines when objects enter the
cluster. Each active assembly line sends one object every ``OBJECT_INTERVAL_MS``.

The number of active lines is a piecewise-constant, stair-step schedule. It can
be configured from an externally chosen capacity profile; this module does not
infer cluster capacity from GPU timings. To use the generated object arrivals
with :class:`core.workload.Workload`, save them as a CSV containing
``arrival_time`` and configure WF15 with ``ARRIVAL_PROCESS="ALITRACE"`` and
that file as ``TRACE_FILE_PATH``.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


WF15_JOB_TYPE = 15
DEFAULT_OBJECT_INTERVAL_MS = 3_000.0
DEFAULT_HORIZON_MS = 300_000.0


@dataclass(frozen=True)
class LineCountStep:
    """Active assembly-line count from ``start_ms`` up to the next step."""

    start_ms: float
    active_lines: int


# Example stair profile: three separate peaks at eight concurrent lines, with
# lower-load ramps and cooldowns between them. Tune these boundaries and counts
# to the desired offered load and the capacity measured for the target cluster.
DEFAULT_LINE_COUNT_STEPS = (
    LineCountStep(0, 2),
    LineCountStep(15_000, 3),
    LineCountStep(30_000, 5),
    LineCountStep(45_000, 8),
    LineCountStep(65_000, 6),
    LineCountStep(80_000, 3),
    LineCountStep(100_000, 4),
    LineCountStep(115_000, 7),
    LineCountStep(130_000, 8),
    LineCountStep(150_000, 5),
    LineCountStep(170_000, 2),
    LineCountStep(190_000, 4),
    LineCountStep(205_000, 6),
    LineCountStep(220_000, 8),
    LineCountStep(245_000, 5),
    LineCountStep(270_000, 3),
)


def generate_object_arrivals(
    *,
    horizon_ms: float = DEFAULT_HORIZON_MS,
    object_interval_ms: float = DEFAULT_OBJECT_INTERVAL_MS,
    line_count_steps: Sequence[LineCountStep] = DEFAULT_LINE_COUNT_STEPS,
    align_cycles: bool = True,
    line_start_offsets_ms: Sequence[float] | None = None,
) -> list[float]:
    """Return sorted object job arrival times in milliseconds from time zero.

    ``line_count_steps`` is a staircase of active line counts. A line added at a
    step starts producing objects at that step's boundary; lines removed at a
    step stop producing from that boundary onward. With ``align_cycles=True``,
    each active line emits on the shared 3-second grid (cycles match). With it
    false, line offsets are staggered to spread arrivals across each cycle.
    Explicit ``line_start_offsets_ms`` can be supplied for repeatable custom
    phase alignment; the sequence is indexed by zero-based line ID.

    Since line counts may rise and fall, line IDs are assigned to the currently
    active set in ascending order. A reactivated line keeps its original phase
    offset but begins emitting again at the new step boundary.
    """
    if horizon_ms <= 0:
        raise ValueError("horizon_ms must be positive")
    if object_interval_ms <= 0:
        raise ValueError("object_interval_ms must be positive")
    if not line_count_steps:
        raise ValueError("line_count_steps must contain at least one step")

    steps = tuple(line_count_steps)
    if steps[0].start_ms != 0:
        raise ValueError("the first line-count step must start at 0 ms")
    if any(step.start_ms < 0 or step.active_lines < 0 for step in steps):
        raise ValueError("step times and active line counts must be nonnegative")
    if any(a.start_ms >= b.start_ms for a, b in zip(steps, steps[1:])):
        raise ValueError("line-count step times must be strictly increasing")

    max_lines = max(step.active_lines for step in steps)
    if line_start_offsets_ms is not None:
        offsets = tuple(float(value) for value in line_start_offsets_ms)
        if len(offsets) < max_lines:
            raise ValueError("need one start offset per possible assembly line")
        if any(value < 0 or value >= object_interval_ms for value in offsets):
            raise ValueError("line start offsets must be in [0, object_interval_ms)")
    elif align_cycles:
        offsets = (0.0,) * max_lines
    else:
        offsets = tuple(line_id * object_interval_ms / max(max_lines, 1)
                        for line_id in range(max_lines))

    arrivals: list[float] = []
    for index, step in enumerate(steps):
        segment_end = (steps[index + 1].start_ms
                       if index + 1 < len(steps) else horizon_ms)
        segment_end = min(segment_end, horizon_ms)
        if step.start_ms >= horizon_ms:
            break
        for line_id in range(step.active_lines):
            phase = offsets[line_id]
            first = step.start_ms + ((phase - step.start_ms) % object_interval_ms)
            # Preserve each line's phase on the global cycle grid and never
            # backfill work for time when the line was inactive.
            while first < segment_end:
                arrivals.append(first)
                first += object_interval_ms

    return sorted(arrivals)


def write_arrival_trace(arrivals_ms: Iterable[float], output_path: str | Path) -> Path:
    """Write arrivals as a CSV trace accepted by ``Workload(ALITRACE)``."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=["arrival_time"])
        writer.writeheader()
        writer.writerows({"arrival_time": float(time_ms)} for time_ms in arrivals_ms)
    return path


def wf15_client_config(trace_file_path: str | Path, *, slo_ms: float = 2_000) -> dict:
    """Return the standard client-config entry for a generated WF15 trace."""
    return {
        WF15_JOB_TYPE: {
            "WORKLOAD": {
                "ARRIVAL_PROCESS": "ALITRACE",
                "TRACE_FILE_PATH": str(trace_file_path),
            },
            "SLO": slo_ms,
        }
    }


if __name__ == "__main__":
    # Generate a ready-to-use default trace next to this module.
    trace_path = Path(__file__).with_name("wf15_object_arrivals.csv")
    arrivals = generate_object_arrivals()
    write_arrival_trace(arrivals, trace_path)
    print(f"Wrote {len(arrivals)} WF15 object arrivals to {trace_path}")
