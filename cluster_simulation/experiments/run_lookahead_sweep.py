"""Runs the LOOKAHEAD queue delay sweep.

LOOKAHEAD drops a job once the work the cluster has already accepted means it
cannot finish in time. Two things vary here:

  accounting  how much of a model's backlog a queued task is charged for: only
              the tasks ahead of it in queue order, or the whole backlog
              including the tasks queued behind it (LOOKAHEAD_WHOLE_QUEUE_DELAY,
              the superseded accounting)
  ordering    the queue order those positions are read from: FCFS, EDF, or
              accumulating priority (AP), which serves the job that has waited
              longest relative to its job size first

Every run uses job-level SLOs, so a job carries one deadline for the whole
pipeline and LOOKAHEAD weighs its entire remaining critical path against it. The
per-stage split is a separate mechanism and is not used here.

The workloads are multi-tenant on purpose. Under job-level SLOs a job's EDF key
is its create time plus its SLO, so with a single workflow EDF and FCFS order the
queue identically and only differ once tenants with different SLOs share the
cluster.

Each tenancy is run under a constant and a bursty workload, across job SLOs from
3x to 10x each workflow's job size (its critical path at batch size 1), so every
tenant's SLO scales together and their ratios, which is what EDF sorts on, stay
fixed.

Each configuration runs in its own process so that the RNG seeding in
run_experiment.py is identical across configurations, i.e. every configuration of
a (tenancy, workload) pair sees the same arrival trace. Results land in
[out]/[config name]/.

Usage (from cluster_simulation/, after `source set_env.sh`):
    python experiments/run_lookahead_sweep.py --name lookahead_multitenant
"""

import argparse
import itertools
import json
import os
import subprocess
import sys

SIM_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXP_DIR = os.path.join(SIM_DIR, "experiments")

# job size (ms) per workflow: its critical path at batch size 1, which each
# tenant's job SLO is set as a multiple of
JOB_SIZES = {1: 256.1, 4: 787.0, 5: 388.7,
             6: 62.48, 7: 70.48, 8: 80.48, 10: 70.48, 11: 80.48}

SLO_MULTIPLES = [3, 4, 5, 6, 7, 8, 9, 10]

# the workflows sharing the cluster in each tenancy. The two textvision tenancies
# run the same three pipelines with the variant model at opposite ends: in
# textvision its stage is last, in textvisionswapped first
TENANCIES = {"ppl2": [1, 4, 5],
             "textvision": [6, 7, 8],
             "textvisionswapped": [6, 10, 11]}

# When set, every tenant of a tenancy is given the same job SLO, that of its
# slowest pipeline, instead of each being given a multiple of its own job size.
# The tenants then differ only in the work they do and not in what they are
# promised, so a job's deadline no longer says which tenant it belongs to: EDF and
# FCFS order the shared queue identically, and LOOKAHEAD weighs every queued job
# against one common budget. The short pipelines are the ones this loosens, since
# they keep their own execution time but inherit the slowest tenant's deadline
EQUALIZE_SLOS = False

ALLOCATIONS = {
    # 12 node multitenant ppl 2, placing models 4-10 and 13
    "ppl2": [(12, [4]), (12, [6]), (12, [7]), (12, [7]), (12, [7]), (12, [7]),
             (12, [8]), (6, [5]), (6, [13]),
             (24, [9]), (24, [9]), (24, [9]), (24, [9]), (24, [9]),
             (24, [10]), (24, [10]), (24, [10])],
    # the textvision placement, whose three tails are models 14, 15 and 16. Both
    # textvision tenancies run on it: swapping which end of the pipeline the
    # variant model sits at does not change the set of models to place
    "textvision": [(24, [1]), (24, [1]), (24, [1]), (6, [3]), (6, [3]), (6, [3]),
                   (6, [0, 2]), (6, [14]), (6, [15]), (6, [16]), (6, [])],
}
ALLOCATIONS["textvisionswapped"] = ALLOCATIONS["textvision"]

def _const(rate: int, num_jobs: int) -> dict:
    return {"ARRIVAL_PROCESS": "POISSON",
            "PHASES": [{"KIND": "CONSTANT", "RATE": rate, "NUM_JOBS": num_jobs}]}


def _burst(rate: int, burst_rate: int, quiet: int, burst: int,
           burst_phase: int, tail: int) -> dict:
    """A workload that spikes for [burst] ms out of every [quiet] + [burst] ms for
    [burst_phase] ms, then holds [rate] for [tail] ms. Each tenancy staggers its
    tenants' spikes so they do not all land at once.
    """
    return {"ARRIVAL_PROCESS": "POISSON",
            "PHASES": [{"KIND": "BURST", "RATE": rate, "BURST_RATE": burst_rate,
                        "QUIET_DURATION": quiet, "BURST_DURATION": burst,
                        "STOCHASTIC": False, "DURATION": burst_phase},
                       {"KIND": "CONSTANT", "RATE": rate, "DURATION": tail}]}


# workflow ID -> workload, per tenancy and workload kind
WORKLOADS = {
    # matches the core sweep's ppl2mt const: the tenants share model 7, whose
    # 26.79 qps is the cluster's bottleneck, and these put it at about 0.95 of it,
    # with WF4's own model 9 at about 0.96
    ("ppl2", "const"): {1: _const(8.75, 5000), 4: _const(8.75, 5000), 5: _const(8, 5000)},
    ("ppl2", "burst"): {1: _burst(7, 24, 160000, 30000, 190000, 450000),
                        4: _burst(7, 24, 320000, 30000, 350000, 290000),
                        5: _burst(7, 24, 480000, 30000, 510000, 130000)},
    ("textvision", "const"): {6: _const(40, 5000), 7: _const(40, 5000),
                              8: _const(40, 5000)},
    ("textvision", "burst"): {6: _burst(35, 120, 30000, 10000, 40000, 80000),
                              7: _burst(35, 120, 60000, 10000, 70000, 50000),
                              8: _burst(35, 120, 90000, 10000, 100000, 20000)},
    ("textvisionswapped", "const"): {6: _const(40, 5000), 10: _const(40, 5000),
                                     11: _const(40, 5000)},
    ("textvisionswapped", "burst"): {6: _burst(35, 120, 30000, 10000, 40000, 80000),
                                     10: _burst(35, 120, 60000, 10000, 70000, 50000),
                                     11: _burst(35, 120, 90000, 10000, 100000, 20000)},
}

ORDERINGS = {"fcfs": "FCFS", "edf": "EDF", "ap": "ACCUMULATING_PRIORITY"}

# the fixed accounting against the one it replaced
ACCOUNTINGS = {"ahead": False, "whole": True}


def config_names() -> list[str]:
    return [f"{tenancy}_{workload}_{mult}x_{ordering}_{accounting}"
            for tenancy, workload, mult, ordering, accounting
            in itertools.product(TENANCIES, ["const", "burst"], SLO_MULTIPLES,
                                 ORDERINGS, ACCOUNTINGS)]


def _parse(config_name: str) -> tuple[str, str, int, str, str]:
    tenancy, workload, mult, ordering, accounting = config_name.split("_")
    return tenancy, workload, int(mult[:-1]), ordering, accounting


def _job_slos(tenancy: str, workload: str, mult: int) -> dict[int, int]:
    """Returns the job SLO (ms) each tenant of [tenancy] is given.

    Normally that is [mult] times the tenant's own job size, so every tenant is
    promised the same headroom over its own critical path. Under EQUALIZE_SLOS it
    is instead [mult] times the largest job size in the tenancy, so all of them
    carry the slowest pipeline's deadline.
    """
    workflow_ids = list(WORKLOADS[(tenancy, workload)])

    if EQUALIZE_SLOS:
        slo = int(max(JOB_SIZES[wid] for wid in workflow_ids) * mult)
        return {wid: slo for wid in workflow_ids}

    return {wid: int(JOB_SIZES[wid] * mult) for wid in workflow_ids}


def _overrides(config_name: str) -> dict:
    tenancy, workload, mult, ordering, accounting = _parse(config_name)
    slos = _job_slos(tenancy, workload, mult)

    return {"CLIENT_CONFIGS": [{wid: {"WORKLOAD": wl, "SLO": slos[wid]}}
                               for wid, wl in WORKLOADS[(tenancy, workload)].items()],
            "ALLOCATION_STRATEGY": "CUSTOM",
            "CUSTOM_ALLOCATION": ALLOCATIONS[tenancy],
            "DISPATCH_POLICY": "ROUND_ROBIN",
            "BOOST_POLICY": ORDERINGS[ordering],
            "SLO_TYPE": "JOB_LEVEL",
            "DROP_POLICY": "LOOKAHEAD",
            "LOOKAHEAD_WHOLE_QUEUE_DELAY": ACCOUNTINGS[accounting],
            "ADMISSION_CONTROL_POLICY": "NONE",
            "SHADOW_DROP_POLICY": "NONE",
            # only job_log.csv and task_log.csv are read back, and the event log is
            # by far the largest thing a run writes
            "PRODUCE_EVENT_LOG": False}


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
                 "--run-one", name, "--out", run_path]
                + (["--equalize-slos"] if EQUALIZE_SLOS else []),
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
    parser.add_argument("-j", "--jobs", type=int, default=8,
                        help="Number of configurations to run in parallel")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing results instead of erroring out")
    parser.add_argument("--equalize-slos", action="store_true",
                        help="Give every tenant of a tenancy the same job SLO, "
                             "that of its slowest pipeline, instead of a multiple "
                             "of each tenant's own job size")
    parser.add_argument("--run-one", type=str, choices=all_configs,
                        help=argparse.SUPPRESS)  # internal: run a single configuration

    args = parser.parse_args()

    EQUALIZE_SLOS = args.equalize_slos

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
