"""Admission control: the decision about whether the cluster should accept a
piece of work at all, taken before that work consumes any GPU time.

Independent of DROP_POLICY, which aborts jobs already in flight once their
deadline is known to be unreachable. Admission control instead refuses work up
front, so that what is admitted is what the cluster can still serve on time. Both
may be enabled at once.

Under ADMISSION_GRANULARITY = "JOB" the decision is taken once, at the front door,
as a job arrives. Under "TASK" it is taken again at every pipeline stage, as each
task becomes schedulable.
"""

import numpy as np

import core.configs.gen_config as gcfg
import core.workload as workload

from core.job import Job
from core.task import Task
from core.data_models.workflow import Workflow
from schedulers.algo.capacity_algo import (get_workflow_capacities,
                                           get_workflow_stage_capacities)


# NONE:          admit every job
# PROBABILISTIC: reject each arriving job independently with probability
#                ADMISSION_DROP_RATE
# ROUND_ROBIN:   reject one job out of every round(1 / ADMISSION_DROP_RATE)
# TOKEN_BUCKET:  admit up to the rate the cluster can actually serve, allowing
#                bursts of the size it can still drain within the SLO
ADMISSION_CONTROL_POLICIES = ["NONE", "PROBABILISTIC", "ROUND_ROBIN", "TOKEN_BUCKET"]

# What a single token buys, i.e. the unit admission control decides on. Applies to
# TOKEN_BUCKET only.
#
# JOB:  one bucket per workflow, rated at the workflow's bottleneck stage. A job
#       spends a token on arrival and, once admitted, is never shed again by
#       admission control, so it either runs end to end or never starts.
# TASK: one bucket per pipeline stage, each rated at that stage's own capacity. A
#       task spends a token of its stage's bucket when it becomes schedulable, and
#       is discarded before it runs if the bucket is empty. Discarding a task
#       necessarily kills its job, since the stages behind it can never run, so
#       the work already spent on the job upstream is wasted. In exchange the
#       decision is made per stage and against fresh information: a stage that is
#       congested right now sheds, while an uncongested one keeps admitting, and
#       a non-bottleneck stage is no longer throttled at the bottleneck's rate.
ADMISSION_GRANULARITIES = ["JOB", "TASK"]


class TokenBucket:
    """Standard token bucket rate limiter.

    Tokens accumulate continuously at [rate] and are capped at [depth]; each
    admitted job spends one. The bucket therefore enforces the average rate over
    any long interval, while still letting through a burst of up to [depth] jobs
    that arrive back to back after a quiet period. Depth is what separates this
    from flat rate shedding: a burst that the cluster can absorb is admitted whole
    rather than thinned, and sustained overload is throttled to exactly the rate
    the cluster can serve.
    """

    def __init__(self, rate: float, depth: float, start_time: float = 0):
        """
        Args:
            rate: Token refill rate (tokens/ms)
            depth: Bucket size, i.e. largest burst admitted at once (tokens)
            start_time: Time the bucket starts filling (ms)
        """
        assert(rate >= 0 and depth >= 0)

        self.rate = rate
        self.depth = depth

        # start full: before any load has arrived the cluster is idle, so it can
        # absorb a full burst
        self.tokens = depth
        self.last_refill_time = start_time

    def _refill(self, time: float):
        elapsed = max(0.0, time - self.last_refill_time)
        self.tokens = min(self.depth, self.tokens + elapsed * self.rate)
        self.last_refill_time = time

    def try_consume(self, time: float, cost: float = 1) -> bool:
        """Spends [cost] tokens if the bucket holds that many at [time].

        Returns:
            consumed: True if the tokens were available and have been spent
        """
        self._refill(time)

        if self.tokens < cost:
            return False

        self.tokens -= cost
        return True


class AdmissionController:
    """Applies the configured admission control policy to arriving work.

    Under TOKEN_BUCKET the buckets are sized from the throughput the cluster can
    sustain (see capacity_algo), and how they are keyed depends on
    ADMISSION_GRANULARITY:

    JOB  gives each workflow one bucket, rated at its bottleneck stage.
    TASK gives each pipeline stage its own bucket, rated at that stage's capacity.

    Either way the buckets are per workflow, which isolates tenants: one
    workflow's burst is throttled against its own share of the models it uses, and
    cannot push another workflow's work out.
    """

    def __init__(self, workflows: dict[int, Workflow] = None, workers: dict = None,
                 start_time: float = 0):
        """
        Args:
            workflows: Map of workflow ID -> workflow, required for TOKEN_BUCKET
            workers: Map of worker ID -> worker object, required for TOKEN_BUCKET
            start_time: Time the buckets start filling (ms)
        """
        assert(gcfg.ADMISSION_CONTROL_POLICY in ADMISSION_CONTROL_POLICIES), \
            f"Unrecognized admission control policy {gcfg.ADMISSION_CONTROL_POLICY}"
        assert(gcfg.ADMISSION_GRANULARITY in ADMISSION_GRANULARITIES), \
            f"Unrecognized admission granularity {gcfg.ADMISSION_GRANULARITY}"
        # the other policies shed whole jobs by construction, so silently serving
        # them job-level admission would hide a misconfigured sweep
        assert(gcfg.ADMISSION_GRANULARITY == "JOB"
               or gcfg.ADMISSION_CONTROL_POLICY in ("NONE", "TOKEN_BUCKET")), \
            f"ADMISSION_GRANULARITY = {gcfg.ADMISSION_GRANULARITY} is only " \
            f"supported for TOKEN_BUCKET, not {gcfg.ADMISSION_CONTROL_POLICY}"

        # arrivals seen since the run began, for round robin rejection
        self.arrival_count = 0

        self.per_task = (gcfg.ADMISSION_CONTROL_POLICY == "TOKEN_BUCKET"
                         and gcfg.ADMISSION_GRANULARITY == "TASK")

        # bucket key -> [admitted, rejected]. The key is the workflow ID under JOB
        # granularity and (workflow ID, task ID) under TASK.
        self.admitted: dict = {}
        self.rejected: dict = {}

        # bucket key -> bucket, TOKEN_BUCKET only
        self.buckets: dict = {}
        # bucket key -> estimated sustainable rate (qps), for logging
        self.capacities: dict = {}

        if gcfg.ADMISSION_CONTROL_POLICY == "TOKEN_BUCKET":
            assert(workflows is not None and workers is not None), \
                "TOKEN_BUCKET admission control needs the workflows and workers to " \
                "size its buckets"
            if self.per_task:
                self._build_task_buckets(workflows, workers, start_time)
            else:
                self._build_buckets(workflows, workers, start_time)

    def _check_bucket_config(self):
        assert(0 < gcfg.ADMISSION_TARGET_UTILIZATION <= 1), \
            "ADMISSION_TARGET_UTILIZATION must be in (0, 1], got " \
            f"{gcfg.ADMISSION_TARGET_UTILIZATION}"
        assert(gcfg.ADMISSION_BURST_SIZE is None or gcfg.ADMISSION_BURST_SIZE >= 1), \
            f"ADMISSION_BURST_SIZE must admit at least one job, got {gcfg.ADMISSION_BURST_SIZE}"

    def _build_buckets(self, workflows: dict[int, Workflow], workers: dict,
                       start_time: float):
        """Creates one token bucket per workflow, rated at the throughput the
        cluster can sustain for it.

        The refill rate is the workflow's share of cluster capacity, scaled by
        ADMISSION_TARGET_UTILIZATION to leave headroom: the capacity estimate
        assumes every batch runs at the most efficient size, which a real queue
        only approaches under steady load.

        The depth is how large a burst may be let through at once. A burst of B
        jobs admitted on top of a saturated cluster takes B / capacity ms to drain,
        and the jobs at the back of it still have to meet their deadline, so the
        largest burst worth admitting is the one that drains within the slack left
        by the SLO after the job's own execution. Configuring ADMISSION_BURST_SIZE
        overrides that derivation with a fixed number of jobs.
        """
        self._check_bucket_config()

        self.capacities = get_workflow_capacities(
            workflows, workers, workload.get_workflow_mean_rates(), start_time)

        slos = workload.get_workflow_slos()

        for wid, capacity in self.capacities.items():
            rate = capacity * gcfg.ADMISSION_TARGET_UTILIZATION / 1000  # jobs/ms

            if gcfg.ADMISSION_BURST_SIZE is not None:
                depth = gcfg.ADMISSION_BURST_SIZE
            else:
                # time a job can afford to spend queued and still finish in time
                slack = max(0.0, slos[wid] - workflows[wid].get_min_processing_time())
                depth = rate * slack

            # a bucket that can never hold a single token would reject everything
            self.buckets[wid] = TokenBucket(rate, max(1.0, depth), start_time)

    def _build_task_buckets(self, workflows: dict[int, Workflow], workers: dict,
                            start_time: float):
        """Creates one token bucket per pipeline stage, rated at the throughput
        the cluster can sustain for that stage alone.

        Sizing mirrors [_build_buckets], with the workflow's bottleneck rate
        replaced by the stage's own: a stage running on a model the workflow barely
        shares is allowed to admit at that model's rate, rather than being held
        down to the slowest stage in the pipeline.

        The derived depth uses the slack of the stage rather than of the job. Under
        per-stage (NEXUS) SLOs that is the stage's own budget less the time it
        needs to execute, which is exactly the queueing the stage can absorb and
        still hit its deadline. Under job-level SLOs no such per-stage budget
        exists, so every stage falls back to the job's slack.
        """
        self._check_bucket_config()

        self.capacities = {}
        stage_capacities = get_workflow_stage_capacities(
            workflows, workers, workload.get_workflow_mean_rates(), start_time)

        slos = workload.get_workflow_slos()

        for wid, stages in stage_capacities.items():
            workflow = workflows[wid]
            job_slack = max(0.0, slos[wid] - workflow.get_min_processing_time())

            for task_id, capacity in stages.items():
                key = (wid, task_id)
                self.capacities[key] = capacity
                rate = capacity * gcfg.ADMISSION_TARGET_UTILIZATION / 1000  # jobs/ms

                if gcfg.ADMISSION_BURST_SIZE is not None:
                    depth = gcfg.ADMISSION_BURST_SIZE
                else:
                    depth = rate * self._get_stage_slack(workflow, task_id, job_slack)

                self.buckets[key] = TokenBucket(rate, max(1.0, depth), start_time)

    @staticmethod
    def _get_stage_slack(workflow: Workflow, task_id: int, job_slack: float) -> float:
        """Returns the time a task of [task_id] can spend queued and still meet the
        deadline of its stage, falling back to [job_slack] when the run has no
        per-stage SLO split to read.
        """
        if gcfg.SLO_TYPE != "NEXUS" or not workflow.task_slos:
            return job_slack

        task = workflow.tasks[task_id]
        # fastest this stage can run, matching the convention in
        # Workflow.get_min_processing_time
        min_exec_time = task.model_data.batch_exec_times[24][1]

        return max(0.0, workflow.task_slos[task_id] - min_exec_time)

    def should_reject(self, time: float, job: Job) -> bool:
        """Decides whether [job], which just arrived at the scheduler, should be
        rejected rather than admitted.

        Under TASK granularity a job carries no bucket of its own, so the caller
        admits it here and puts each of its tasks through [should_reject_task]
        instead, starting with the source tasks.

        Args:
            time: Time the job arrived at the scheduler (ms)
            job: Job to admit or reject

        Returns:
            should_reject: True if the job should be rejected on arrival
        """
        if self.per_task:
            return False

        reject = self._should_reject(time, job)
        self._count(job.job_type_id, reject)

        return reject

    def should_reject_task(self, time: float, task: Task) -> bool:
        """Decides whether [task], which has just become schedulable, should be
        discarded before it runs.

        Only TASK granularity rejects anything here; under JOB granularity the
        job was already admitted as a whole and every one of its tasks is allowed
        through.

        Rejecting a task ends its job: the stages behind it depend on its output
        and can never run. The caller is responsible for dropping the job.

        Args:
            time: Time the task became schedulable (ms)
            task: Task to admit or discard

        Returns:
            should_reject: True if the task should be discarded before running
        """
        if not self.per_task:
            return False

        key = (task.job.job_type_id, task.task_id)
        assert(key in self.buckets), \
            f"No admission bucket for stage {task.task_id} of workflow " \
            f"{task.job.job_type_id}"

        reject = not self.buckets[key].try_consume(time)
        self._count(key, reject)

        return reject

    def _count(self, key, reject: bool):
        counter = self.rejected if reject else self.admitted
        counter[key] = counter.get(key, 0) + 1

    def _should_reject(self, time: float, job: Job) -> bool:
        if gcfg.ADMISSION_CONTROL_POLICY == "NONE":
            return False

        if gcfg.ADMISSION_CONTROL_POLICY == "TOKEN_BUCKET":
            assert(job.job_type_id in self.buckets), \
                f"No admission bucket for workflow {job.job_type_id}"
            return not self.buckets[job.job_type_id].try_consume(time)

        rate = gcfg.ADMISSION_DROP_RATE
        assert(0 <= rate <= 1), f"ADMISSION_DROP_RATE must be in [0, 1], got {rate}"

        if rate == 0:
            return False

        self.arrival_count += 1

        if gcfg.ADMISSION_CONTROL_POLICY == "PROBABILISTIC":
            return np.random.random() < rate
        elif gcfg.ADMISSION_CONTROL_POLICY == "ROUND_ROBIN":
            # reject exactly one job out of every [period] arrivals
            period = max(1, round(1 / rate))
            return self.arrival_count % period == 0

        raise ValueError("Unrecognized admission control policy "
                         f"{gcfg.ADMISSION_CONTROL_POLICY}")

    def get_stats(self) -> list[dict]:
        """Returns what admission control did, for the run's
        admission_control.csv: one row per workflow under JOB granularity, one row
        per pipeline stage under TASK.
        """
        stats = []
        for key in sorted(set(self.admitted) | set(self.rejected) | set(self.buckets)):
            admitted = self.admitted.get(key, 0)
            rejected = self.rejected.get(key, 0)
            bucket = self.buckets.get(key)

            wid, task_id = key if self.per_task else (key, np.nan)

            stats.append({
                "workflow_id": wid,
                "task_id": task_id,
                "policy": gcfg.ADMISSION_CONTROL_POLICY,
                "granularity": gcfg.ADMISSION_GRANULARITY,
                "estimated_capacity_qps": self.capacities.get(key, np.nan),
                "admit_rate_qps": bucket.rate * 1000 if bucket else np.nan,
                "burst_size_jobs": bucket.depth if bucket else np.nan,
                "num_admitted": admitted,
                "num_rejected": rejected,
                "reject_fraction": rejected / (admitted + rejected) if admitted + rejected else 0,
            })

        return stats
