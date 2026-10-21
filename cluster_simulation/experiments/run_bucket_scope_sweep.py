"""Runs the token bucket scope sweep: per-workflow buckets against one shared bucket.

TOKEN_BUCKET admission control sizes its buckets from the throughput the cluster
can sustain, and ADMISSION_BUCKET_SCOPE decides who spends them:

  PER_WORKFLOW  every workflow meters its own work against its own share of the
                models it uses, so tenants are isolated and an idle tenant's
                unused share is never lent to a busy one
  AGGREGATE     the workflows share one bucket, refilling at the total of the
                rates they would have held separately, so the cluster admits the
                same rate overall but whoever asks first gets the tokens

Both runs of a pair therefore admit the same long run rate and differ only in how
that rate is divided, which is only visible on a multi-tenant workload. Every
tenancy here is three workflows sharing a cluster, run under a constant and a
bursty workload; the bursty ones stagger their tenants' spikes, so one tenant is
over its own share while the others are under theirs, which is exactly where the
two scopes part ways.

Each tenant's job SLO is SLO_MULTIPLE times its job size (its critical path at
batch size 1), matching the workloads as gen_config.py carries them.

Each configuration runs in its own process so that the RNG seeding in
run_experiment.py is identical across configurations, i.e. the two scopes of a
(tenancy, workload) pair see the same arrival trace. Results land in
[out]/[config name]/.

Usage (from cluster_simulation/, after `source set_env.sh`):
    python experiments/run_bucket_scope_sweep.py --name bucket_scope

The two runs of a pair are then read against each other with plot_tail.py, e.g.

    python experiments/postprocess/plot_tail.py --normalize --pdf \
        --out results/bucket_scope/figures/ppl2_const_tail.png \
        --srcs results/bucket_scope/ppl2_const_{perworkflow,aggregate}/sim_logs \
        --labels "per-workflow buckets" "aggregate bucket"

Normalizing is what makes the pair comparable: the tenants of a tenancy have
different pipeline lengths, so only in multiples of job size do their response
times share an axis, and each tenant's SLO then sits at the same SLO_MULTIPLE on
it. Adding --split gives one subplot per tenant, which is where the scopes differ
most: the aggregate bucket moves work between tenants, not just in total.
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
JOB_SIZES = {1: 186.3, 4: 725.2, 5: 326.7,
             6: 62.48, 7: 70.48, 8: 80.48, 10: 70.48, 11: 80.48}

SLO_MULTIPLE = 5

# the workflows sharing the cluster in each tenancy. The two textvision tenancies
# run the same three pipelines with the variant model at opposite ends: in
# textvision its stage is last, in textvision_swapped first, so the swapped one
# spends its variant capacity before the models all three tenants share rather
# than after
TENANCIES = {"ppl2": [1, 4, 5],
             "textvision": [6, 7, 8],
             "textvisionswapped": [6, 10, 11]}

ALLOCATIONS = {
    # 12 node multitenant ppl 2, placing models 4-10 and 13
    "ppl2": [(12, [4]), (12, [6]), (12, [7]), (12, [7]), (12, [7]), (12, [7]),
             (12, [8]), (6, [5]), (6, [13]),
             (24, [9]), (24, [9]), (24, [9]), (24, [9]), (24, [9]),
             (24, [10]), (24, [10]), (24, [10])],
    # the textvision placement, whose three variant models are 14, 15 and 16.
    # Both textvision tenancies run on it: swapping which end of the pipeline the
    # variant sits at does not change the set of models to place
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

# the comparison this sweep exists for
SCOPES = {"perworkflow": "PER_WORKFLOW", "aggregate": "AGGREGATE"}

# ADMISSION_BURST_SIZE per scope, i.e. the depth of one bucket, set from the
# command line and passed down to each child process. None derives the depth from
# the SLO, which pools correctly on its own: the aggregate bucket is built as the
# sum of the per-workflow budgets it replaces. A depth given here is instead taken
# literally as that one bucket's depth, so to keep a pair comparable the aggregate
# depth should be set to the per-workflow depth times the number of tenants
BURST_SIZES = {"perworkflow": None, "aggregate": None}


def config_names() -> list[str]:
    return [f"{tenancy}_{workload}_{scope}" for tenancy, workload, scope
            in itertools.product(TENANCIES, ["const", "burst"], SCOPES)]


def _parse(config_name: str) -> tuple[str, str, str]:
    tenancy, workload, scope = config_name.split("_")
    return tenancy, workload, scope


def _overrides(config_name: str) -> dict:
    tenancy, workload, scope = _parse(config_name)

    return {"CLIENT_CONFIGS": [{wid: {"WORKLOAD": wl,
                                      "SLO": int(JOB_SIZES[wid] * SLO_MULTIPLE)}}
                               for wid, wl in WORKLOADS[(tenancy, workload)].items()],
            "ALLOCATION_STRATEGY": "CUSTOM",
            "CUSTOM_ALLOCATION": ALLOCATIONS[tenancy],
            "DISPATCH_POLICY": "ROUND_ROBIN",
            "SLO_TYPE": "JOB_LEVEL",
            # admission control is the only thing shedding work here: a drop
            # policy would shed jobs the buckets had already admitted, which is a
            # separate mechanism and would blur what the scopes do differently
            "DROP_POLICY": "NONE",
            "SHADOW_DROP_POLICY": "NONE",
            "ADMISSION_CONTROL_POLICY": "TOKEN_BUCKET",
            "ADMISSION_GRANULARITY": "JOB",
            "ADMISSION_BUCKET_SCOPE": SCOPES[scope],
            "ADMISSION_BURST_SIZE": BURST_SIZES[scope],
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


def _burst_args() -> list[str]:
    """Returns the bucket depth options as command line arguments, so that a child
    process running one configuration is given the same depths as the sweep.
    """
    args = []
    for scope, depth in BURST_SIZES.items():
        if depth is not None:
            args += [f"--burst-{scope}", str(depth)]

    return args


def run_dir(out_dir: str, config_name: str, suffix: str = "") -> str:
    """Returns the directory a configuration's results go in.

    [suffix] distinguishes runs of the same configuration that differ in something
    the name does not carry, such as a bucket depth, so that several of them can
    share one results directory and be read against each other.
    """
    return os.path.join(out_dir, f"{config_name}_{suffix}" if suffix else config_name)


def run_sweep(config_names: list[str], out_dir: str, max_parallel: int,
              suffix: str = ""):
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
                      f"{os.path.join(run_dir(out_dir, name, suffix), 'run.log')}")

    while pending or running:
        while pending and len(running) < max_parallel:
            name = pending.pop(0)
            run_path = run_dir(out_dir, name, suffix)
            os.makedirs(run_path, exist_ok=True)
            log = open(os.path.join(run_path, "run.log"), "w")
            proc = subprocess.Popen(
                [sys.executable, "-u", os.path.abspath(__file__),
                 "--run-one", name, "--out", run_path] + _burst_args(),
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
    parser.add_argument("--suffix", type=str, default="",
                        help="Appended to each run directory name, so that runs "
                             "differing only in bucket depth can share a results "
                             "directory (e.g. --burst-aggregate 10 --suffix depth10)")
    parser.add_argument("--burst-perworkflow", type=float,
                        help="Depth of each per-workflow bucket, in jobs "
                             "(default: derive it from the SLO)")
    parser.add_argument("--burst-aggregate", type=float,
                        help="Depth of the shared bucket, in jobs. Set it to the "
                             "per-workflow depth times the number of tenants to "
                             "keep the pair comparable (default: derive it)")
    parser.add_argument("--run-one", type=str, choices=all_configs,
                        help=argparse.SUPPRESS)  # internal: run a single configuration

    args = parser.parse_args()

    BURST_SIZES["perworkflow"] = args.burst_perworkflow
    BURST_SIZES["aggregate"] = args.burst_aggregate

    if args.run_one:
        run_one(args.run_one, args.out)
        sys.exit(0)

    if not args.name and not args.out:
        parser.error("one of --name or --out is required")

    out_dir = os.path.abspath(args.out or os.path.join(EXP_DIR, "results", args.name))

    existing = [name for name in args.configs
                if os.path.isdir(run_dir(out_dir, name, args.suffix))
                and os.listdir(run_dir(out_dir, name, args.suffix))]
    if existing and not args.overwrite:
        parser.error(f"results already exist in {out_dir} for {len(existing)} configs. "
                     "Pass --overwrite to replace them or pick another --name.")

    os.makedirs(out_dir, exist_ok=True)
    failed = run_sweep(args.configs, out_dir, max(1, args.jobs), args.suffix)

    print(f"\nResults in {out_dir}")
    if failed:
        print(f"{len(failed)} configuration(s) failed: {', '.join(failed)}")
        sys.exit(1)
