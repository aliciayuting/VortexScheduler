"""Runs the core drop / admission control sweep over the WF12 and WF13 workloads.

The policies are the subset of run_admission_sweep.py's table that the comparison
usually turns on: nothing shedding work, the two SLO-aware drop policies, the queue
aware one, and token bucket admission on its own and alongside the Nexus split.
Taking them from that table rather than restating them keeps the two sweeps from
drifting apart.

What this adds over run_admission_sweep.py is the workload: that script takes the
workflow, allocation and send pattern from gen_config.py as currently written,
which means one sweep per workload and a config edit in between. Here each of the
four workloads is carried in the sweep itself and crossed with every policy, so a
single invocation covers all of them.

Both workflows fan a job out into many tasks on one model -- 24 detections per
model for WF12, four per classifier for WF13 -- so their send rates sit at about
0.82 of what their allocations sustain, and their SLOs are set where roughly 85%
of jobs make the deadline with nothing dropped. That leaves the policies something
to improve on without the workload being hopeless.

Each configuration runs in its own process so that the RNG seeding in
run_experiment.py is identical across configurations, i.e. every policy for a
given workload sees the same arrival trace. Results land in [out]/[config name]/.

Usage (from cluster_simulation/, after `source set_env.sh`):
    python experiments/run_core_sweep.py --name core_sweep
"""

import argparse
import itertools
import json
import math
import os
import subprocess
import sys

SIM_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXP_DIR = os.path.join(SIM_DIR, "experiments")

for _path in (SIM_DIR, EXP_DIR):
    if _path not in sys.path:
        sys.path.insert(0, _path)
os.environ.setdefault("SIMULATION_DIR", SIM_DIR)

from run_admission_sweep import SWEEP_CONFIGS

# the policies this sweep compares, named as they are in run_admission_sweep.py
CORE_POLICIES = {
    "nodrop": "central_rr_nodrop",
    "early": "central_rr_early",
    "nexus": "central_rr_nexus",
    "lookahead": "central_rr_lookahead",
    "token": "central_rr_token",
    "5token": "central_rr_5token",
    "tokennexus": "central_rr_token_nexus",
}

ALLOCATIONS = {
    # WF12: the two YOLOv5 detectors get equal shares, since a job issues 24 of
    # each; the 0GB aggregation model rides along on the first worker
    "wf12": [(6, [17, 19]), (6, [17]), (6, [17]), (6, [17]),
             (6, [18]), (6, [18]), (6, [18]), (6, [18])],
    # WF13: a 12GB worker for each of the four parallel tasks per DINO classifier,
    # with OCR, segmentation and aggregation sharing the 24GB worker
    "wf13": [(24, [20, 21, 24]),
             (12, [22]), (12, [22]), (12, [22]), (12, [22]),
             (12, [23]), (12, [23]), (12, [23]), (12, [23])],
    # the textvision placement with only WF6's tail model placed
    "wf6": [(24, [1]), (24, [1]), (24, [1]), (6, [3]), (6, [3]), (6, [3]),
            (6, [0, 2]), (6, [14]), (6, []), (6, []), (6, [])],
    # the same placement with all three variant models, 14, 15 and 16
    "wf6variants": [(24, [1]), (24, [1]), (24, [1]), (6, [3]), (6, [3]), (6, [3]),
                    (6, [0, 2]), (6, [14]), (6, [15]), (6, [16]), (6, [])],
    # pipeline 2 on its own, 8 nodes
    "ppl2": [(12, [4]), (12, [5, 6]),
             (12, [7]), (12, [7]), (12, [7]), (12, [7]), (12, [7]), (12, [7])],
    # 12 node multitenant pipeline 2, placing models 4-10 and 13
    "ppl2mt": [(12, [4]), (12, [6]), (12, [7]), (12, [7]), (12, [7]), (12, [7]),
               (12, [8]), (6, [5]), (6, [13]),
               (24, [9]), (24, [9]), (24, [9]), (24, [9]), (24, [9]),
               (24, [10]), (24, [10]), (24, [10])],
}

# the workflows sharing the cluster in each family
FAMILIES = {"wf12": [12], "wf13": [13], "wf6": [6], "wf6variants": [6, 7, 8],
            "ppl2": [1], "ppl2mt": [1, 4, 5]}

# Each tenant's job SLO is this many times its own job runtime. WF12 and WF13 are
# newer and were sized at 8x; the textvision and pipeline 2 workloads carry the 5x
# the rest of the repo has always used for them, so their runs stay comparable with
# the results already on disk
SLO_MULTIPLES = {"wf12": 8, "wf13": 8,
                 "wf6": 5, "wf6variants": 5, "ppl2": 5, "ppl2mt": 5}


def _const(rate: int, num_jobs: int) -> dict:
    return {"ARRIVAL_PROCESS": "POISSON",
            "PHASES": [{"KIND": "CONSTANT", "RATE": rate, "NUM_JOBS": num_jobs}]}


def _burst(rate: int, burst_rate: int, quiet: int, burst: int,
           burst_phase: int, tail: int) -> dict:
    """A workload that spikes for [burst] ms out of every [quiet] + [burst] ms for
    [burst_phase] ms, then holds [rate] for [tail] ms. A family with several tenants
    staggers their spikes so they do not all land at once.
    """
    return {"ARRIVAL_PROCESS": "POISSON",
            "PHASES": [{"KIND": "BURST", "RATE": rate, "BURST_RATE": burst_rate,
                        "QUIET_DURATION": quiet, "BURST_DURATION": burst,
                        "STOCHASTIC": False, "DURATION": burst_phase},
                       {"KIND": "CONSTANT", "RATE": rate, "DURATION": tail}]}


# workflow ID -> workload, per family and workload kind. The WF12 and WF13 bursts
# spike to 1.5x capacity for 10s out of every 40s and offer the same average rate
# as their constant workloads; the others are the workloads gen_config.py carries
WORKLOADS = {
    ("wf12", "const"): {12: _const(22, 10000)},
    ("wf12", "burst"): {12: _burst(16, 40, 30000, 10000, 320000, 185000)},
    ("wf13", "const"): {13: _const(18, 10000)},
    ("wf13", "burst"): {13: _burst(13, 33, 30000, 10000, 360000, 270000)},

    ("wf6", "const"): {6: _const(120, 20000)},
    ("wf6", "burst"): {6: _burst(100, 175, 15000, 3000, 162000, 17750)},

    ("wf6variants", "const"): {6: _const(40, 5000), 7: _const(40, 5000),
                               8: _const(40, 5000)},
    ("wf6variants", "burst"): {6: _burst(35, 120, 30000, 10000, 40000, 80000),
                               7: _burst(35, 120, 60000, 10000, 70000, 50000),
                               8: _burst(35, 120, 90000, 10000, 100000, 20000)},

    ("ppl2", "const"): {1: _const(40, 20000)},
    ("ppl2", "burst"): {1: _burst(30, 80, 30000, 5000, 490000, 60000)},

    ("ppl2mt", "const"): {1: _const(9, 5000), 4: _const(7, 5000), 5: _const(8, 5000)},
    ("ppl2mt", "burst"): {1: _burst(7, 24, 160000, 30000, 190000, 450000),
                          4: _burst(7, 24, 320000, 30000, 350000, 290000),
                          5: _burst(7, 24, 480000, 30000, 510000, 130000)},
}


def job_runtime(family: str, workflow_id: int) -> float:
    """Returns how long one job of [workflow_id] takes with the cluster to itself,
    under the allocation [family] places.

    This is the longest path through the DAG at batch size 1, with each stage
    charged for the rounds its own job forces it into: a stage fanned out into more
    tasks than its model has instances cannot run them at once, and a batch takes
    at most one task per job, so they queue behind each other even on an idle
    cluster. It is what the job SLOs here are a multiple of, and the same quantity
    plot_tail.get_workflow_job_sizes normalizes response times by.

    The critical path alone would be the wrong base: WF12's is 15.7 ms, but a job
    issues 24 detections per model over four instances, so it needs six rounds of
    its 14.7 ms detector and cannot finish faster than about 89 ms. Where a job has
    no more tasks on a model than there are instances, which is every stage of the
    textvision and pipeline 2 workflows, this is exactly the critical path.

    Args:
        family: Key into FAMILIES and ALLOCATIONS
        workflow_id: Workflow to size, one of FAMILIES[family]

    Returns:
        runtime: Time one job needs on an idle cluster (ms)
    """
    import core.configs.model_config as mcfg
    import core.configs.workflow_config as wcfg

    cfg = next(c for c in wcfg.WORKFLOW_LIST if c["JOB_TYPE"] == workflow_id)
    tasks = {t["TASK_INDEX"]: t for t in cfg["TASKS"]}

    instances: dict[int, int] = {}
    for _, model_ids in ALLOCATIONS[family]:
        for model_id in model_ids:
            instances[model_id] = instances.get(model_id, 0) + 1

    task_counts: dict[int, int] = {}
    for task in tasks.values():
        task_counts[task["MODEL_ID"]] = task_counts.get(task["MODEL_ID"], 0) + 1

    def exec_time(model_id: int) -> float:
        if model_id < 0:
            return 0.0
        times = mcfg.MODELS[model_id]["MIG_BATCH_EXEC_TIMES"]
        rounds = math.ceil(task_counts[model_id] / max(1, instances.get(model_id, 1)))
        return times[max(times)][1] * rounds

    finish: dict[int, float] = {}

    def finish_by(task_id: int) -> float:
        if task_id not in finish:
            task = tasks[task_id]
            start = max([finish_by(p) for p in task["PREV_TASK_INDEX"]], default=0)
            finish[task_id] = start + exec_time(task["MODEL_ID"])
        return finish[task_id]

    return max(finish_by(task_id) for task_id in tasks)


def config_names() -> list[str]:
    return [f"{family}_{workload}_{policy}" for family, workload, policy
            in itertools.product(FAMILIES, ["const", "burst"], CORE_POLICIES)]


def _parse(config_name: str) -> tuple[str, str, str]:
    family, workload, policy = config_name.split("_")
    return family, workload, policy


def _overrides(config_name: str) -> dict:
    family, workload, policy = _parse(config_name)
    multiple = SLO_MULTIPLES[family]

    overrides = {"CLIENT_CONFIGS": [
                     {wid: {"WORKLOAD": wl,
                            "SLO": int(job_runtime(family, wid) * multiple)}}
                     for wid, wl in WORKLOADS[(family, workload)].items()],
                 "ALLOCATION_STRATEGY": "CUSTOM",
                 "CUSTOM_ALLOCATION": ALLOCATIONS[family],
                 "DISPATCH_POLICY": "ROUND_ROBIN",
                 # both workflows run several tasks of one job on the same model,
                 # which batching handles by taking at most one of them per batch
                 "DISABLE_BATCHING": False,
                 # only job_log.csv and task_log.csv are read back, and the event
                 # log is by far the largest thing a run writes
                 "PRODUCE_EVENT_LOG": False}

    # the policy decides DROP_POLICY, SLO_TYPE and the admission options
    overrides.update(SWEEP_CONFIGS[CORE_POLICIES[policy]])

    return overrides


def _child_env() -> dict:
    env = os.environ.copy()
    env["SIMULATION_DIR"] = SIM_DIR
    env["PYTHONPATH"] = os.pathsep.join(
        p for p in [SIM_DIR, env.get("PYTHONPATH", "")] if p)
    return env


def run_one(config_name: str, out_path: str):
    """Runs a single sweep configuration into [out_path]."""
    os.environ.setdefault("SIMULATION_DIR", SIM_DIR)
    for path in (SIM_DIR, EXP_DIR):
        if path not in sys.path:
            sys.path.insert(0, path)

    import core.configs.gen_config as gcfg
    from run_experiment import run_experiment  # seeds the RNG on import

    overrides = _overrides(config_name)
    for option, value in overrides.items():
        setattr(gcfg, option, value)

    os.makedirs(os.path.join(out_path, "sim_logs"), exist_ok=True)
    run_experiment(True, out_path)

    with open(os.path.join(out_path, "overrides.json"), "w") as f:
        json.dump({"config": config_name, "scheduler_type": "central",
                   "overrides": overrides}, f, indent=2)

    # the config copy is written by run_experiment before the run, so append the
    # overrides (later assignments win) to keep it a faithful record. plot_tail
    # reads CUSTOM_ALLOCATION back out of it to size this workflow's jobs
    with open(os.path.join(out_path, "configs", "gen_config.py"), "a") as f:
        f.write(f"\n\n\"\"\" ----- overrides applied by run_core_sweep.py "
                f"({config_name}) ----- \"\"\"\n\n")
        for option, value in overrides.items():
            literal = f'"{value}"' if isinstance(value, str) else repr(value)
            f.write(f"{option} = {literal}\n")


def run_sweep(config_names: list[str], out_dir: str, max_parallel: int):
    """Runs each named configuration in its own process, up to [max_parallel] at once."""
    running: list[tuple[str, subprocess.Popen, object]] = []
    failed: list[str] = []
    pending = list(config_names)

    def reap(block: bool):
        for entry in list(running):
            name, proc, log = entry
            if not block and proc.poll() is None:
                continue
            code = proc.wait()
            log.close()
            running.remove(entry)
            if code == 0:
                print(f"[done] {name}")
            else:
                failed.append(name)
                print(f"[FAILED] {name} (exit {code}), see "
                      f"{os.path.join(out_dir, name, 'run.log')}")

    while pending or running:
        while pending and len(running) < max_parallel:
            name = pending.pop(0)
            run_path = os.path.join(out_dir, name)
            os.makedirs(run_path, exist_ok=True)
            log = open(os.path.join(run_path, "run.log"), "w")
            proc = subprocess.Popen(
                [sys.executable, "-u", os.path.abspath(__file__),
                 "--run-one", name, "--out", run_path],
                cwd=SIM_DIR, env=_child_env(), stdout=log, stderr=subprocess.STDOUT)
            running.append((name, proc, log))
            print(f"[start] {name} -> {run_path}")

        reap(block=len(running) >= max_parallel or not pending)

    return failed


if __name__ == "__main__":
    all_configs = config_names()

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-n", "--name", type=str,
                        help="Name of the sweep, used as the results directory name")
    parser.add_argument("-o", "--out", type=str,
                        help="Results directory (default: experiments/results/[name])")
    parser.add_argument("-c", "--configs", nargs="+", choices=all_configs,
                        default=all_configs,
                        help="Subset of configurations to run (default: all)")
    parser.add_argument("-j", "--jobs", type=int, default=4,
                        help="Number of configurations to run in parallel")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing results instead of erroring out")
    parser.add_argument("--run-one", type=str, choices=all_configs,
                        help=argparse.SUPPRESS)  # internal: run a single configuration

    args = parser.parse_args()

    if args.run_one:
        run_one(args.run_one, args.out)
        sys.exit(0)

    if not args.name and not args.out:
        parser.error("one of --name or --out is required")

    out_dir = os.path.abspath(args.out or os.path.join(EXP_DIR, "results", args.name))

    existing = [name for name in args.configs
                if os.path.isdir(os.path.join(out_dir, name))
                and os.listdir(os.path.join(out_dir, name))]
    if existing and not args.overwrite:
        parser.error(f"results already exist in {out_dir} for {len(existing)} configs. "
                     "Pass --overwrite to replace them or pick another --name.")

    os.makedirs(out_dir, exist_ok=True)
    failed = run_sweep(args.configs, out_dir, max(1, args.jobs))

    print(f"\nResults in {out_dir}")
    if failed:
        print(f"{len(failed)} configuration(s) failed: {', '.join(failed)}")
        sys.exit(1)
