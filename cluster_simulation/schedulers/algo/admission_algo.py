"""Admission control: the decision, taken at the front door as a job arrives,
about whether the cluster should accept it at all.

Independent of DROP_POLICY, which aborts jobs already in flight once their
deadline is known to be unreachable. Admission control instead refuses work before
it consumes any GPU time, so that the jobs that are admitted are ones the cluster
can still serve on time. Both may be enabled at once.
"""

import numpy as np

import core.configs.gen_config as gcfg
import core.workload as workload

from core.job import Job
from core.data_models.workflow import Workflow
from schedulers.algo.capacity_algo import get_workflow_capacities


# NONE:          admit every job
# PROBABILISTIC: reject each arriving job independently with probability
#                ADMISSION_DROP_RATE
# ROUND_ROBIN:   reject one job out of every round(1 / ADMISSION_DROP_RATE)
# TOKEN_BUCKET:  admit up to the rate the cluster can actually serve, allowing
#                bursts of the size it can still drain within the SLO
ADMISSION_CONTROL_POLICIES = ["NONE", "PROBABILISTIC", "ROUND_ROBIN", "TOKEN_BUCKET"]


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
    """Applies the configured admission control policy to arriving jobs.

    Under TOKEN_BUCKET each workflow gets its own bucket, sized from the throughput
    the cluster can sustain for that workflow (see capacity_algo). Per workflow
    buckets isolate tenants: one workflow's burst is throttled against its own
    share of the models it uses, and cannot push another workflow's jobs out.
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

        # arrivals seen since the run began, for round robin rejection
        self.arrival_count = 0

        # workflow ID -> [admitted, rejected]
        self.admitted: dict[int, int] = {}
        self.rejected: dict[int, int] = {}

        # workflow ID -> bucket, TOKEN_BUCKET only
        self.buckets: dict[int, TokenBucket] = {}
        # workflow ID -> estimated sustainable rate (qps), for logging
        self.capacities: dict[int, float] = {}

        if gcfg.ADMISSION_CONTROL_POLICY == "TOKEN_BUCKET":
            assert(workflows is not None and workers is not None), \
                "TOKEN_BUCKET admission control needs the workflows and workers to " \
                "size its buckets"
            self._build_buckets(workflows, workers, start_time)

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
        assert(0 < gcfg.ADMISSION_TARGET_UTILIZATION <= 1), \
            "ADMISSION_TARGET_UTILIZATION must be in (0, 1], got " \
            f"{gcfg.ADMISSION_TARGET_UTILIZATION}"
        assert(gcfg.ADMISSION_BURST_SIZE is None or gcfg.ADMISSION_BURST_SIZE >= 1), \
            f"ADMISSION_BURST_SIZE must admit at least one job, got {gcfg.ADMISSION_BURST_SIZE}"

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

    def should_reject(self, time: float, job: Job) -> bool:
        """Decides whether [job], which just arrived at the scheduler, should be
        rejected rather than admitted.

        Args:
            time: Time the job arrived at the scheduler (ms)
            job: Job to admit or reject

        Returns:
            should_reject: True if the job should be rejected on arrival
        """
        reject = self._should_reject(time, job)

        counter = self.rejected if reject else self.admitted
        counter[job.job_type_id] = counter.get(job.job_type_id, 0) + 1

        return reject

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
        """Returns one row per workflow describing what admission control did,
        for the run's admission_control.csv.
        """
        stats = []
        for wid in sorted(set(self.admitted) | set(self.rejected) | set(self.buckets)):
            admitted = self.admitted.get(wid, 0)
            rejected = self.rejected.get(wid, 0)
            bucket = self.buckets.get(wid)

            stats.append({
                "workflow_id": wid,
                "policy": gcfg.ADMISSION_CONTROL_POLICY,
                "estimated_capacity_qps": self.capacities.get(wid, np.nan),
                "admit_rate_qps": bucket.rate * 1000 if bucket else np.nan,
                "burst_size_jobs": bucket.depth if bucket else np.nan,
                "num_admitted": admitted,
                "num_rejected": rejected,
                "reject_fraction": rejected / (admitted + rejected) if admitted + rejected else 0,
            })

        return stats
