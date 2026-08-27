"""Client workload specification: how a client's send rate behaves over time, and
how that rate is turned into individual job arrivals.

A workload is a sequence of PHASES, each describing the send rate over one stretch
of simulated time (steady or bursty), plus an ARRIVAL PROCESS that turns the
instantaneous rate into arrival times. The two are orthogonal: the phases say what
the average load looks like at each moment, the arrival process says how much
randomness there is around it.
"""

import os

import numpy as np
import pandas as pd

import core.configs.gen_config as gcfg


ARRIVAL_PROCESSES = ["CONSTANT", "POISSON", "GAMMA", "ALITRACE"]

# CONSTANT: fixed RATE
# BURST:    alternates between RATE and BURST_RATE, spending BURST_DURATION ms in
#           a burst and QUIET_DURATION ms between bursts. With STOCHASTIC set,
#           those two are means of exponential draws rather than exact lengths
PHASE_KINDS = ["CONSTANT", "BURST"]

# how far to jump ahead when the instantaneous rate is zero, and so no
# inter-arrival time can be drawn from it (ms)
_IDLE_STEP_MS = 1.0


class WorkloadPhase:
    """One stretch of a workload, over which the send rate follows a single shape.

    A phase ends either after DURATION ms or after NUM_JOBS jobs have been
    generated; exactly one of the two must be configured.
    """

    def __init__(self, cfg: dict):
        self.kind = cfg.get("KIND", "CONSTANT")
        assert(self.kind in PHASE_KINDS), f"Unknown workload phase kind {self.kind}"

        self.duration = cfg.get("DURATION")
        self.num_jobs = cfg.get("NUM_JOBS")
        assert((self.duration is None) != (self.num_jobs is None)), \
            f"Workload phase {cfg} must set exactly one of DURATION (ms) and NUM_JOBS"
        assert(self.duration is None or self.duration > 0)
        assert(self.num_jobs is None or self.num_jobs > 0)

        if self.kind == "CONSTANT":
            self.rate = cfg["RATE"]
            assert(self.rate >= 0)

        elif self.kind == "BURST":
            self.rate = cfg["RATE"]
            self.burst_rate = cfg["BURST_RATE"]
            self.burst_duration = cfg["BURST_DURATION"]
            self.quiet_duration = cfg["QUIET_DURATION"]
            self.stochastic = cfg.get("STOCHASTIC", True)
            assert(self.rate >= 0 and self.burst_rate >= 0)
            assert(self.burst_duration > 0 and self.quiet_duration > 0)

            # burst schedule, extended lazily as time advances: the phase starts
            # quiet, and _segment_end is when the current state expires
            self._in_burst = True  # flipped to quiet by the first call
            self._segment_end = 0.0
            self._last_rate_time = 0.0

        # a phase that ends only once it has sent NUM_JOBS jobs never ends if it
        # cannot send any
        assert(self.duration is not None or self.peak_rate() > 0), \
            f"Workload phase {cfg} is bounded by NUM_JOBS but never sends anything"

    def rate_at(self, time_in_phase: float) -> float:
        """Returns the instantaneous send rate (qps) [time_in_phase] ms into the phase.

        BURST phases draw their schedule as they go, so this must be called with a
        non-decreasing argument within a phase.
        """
        assert(time_in_phase >= 0)

        if self.kind == "CONSTANT":
            return self.rate

        elif self.kind == "BURST":
            assert(time_in_phase >= self._last_rate_time), \
                "BURST phase rates must be read in non-decreasing time order"
            self._last_rate_time = time_in_phase

            while time_in_phase >= self._segment_end:
                self._in_burst = not self._in_burst
                self._segment_end += self._draw_sojourn(self._in_burst)

            return self.burst_rate if self._in_burst else self.rate

        raise ValueError(f"Unrecognized workload phase kind {self.kind}")

    def _draw_sojourn(self, in_burst: bool) -> float:
        """Returns how long the BURST phase stays in its current state (ms)."""
        mean = self.burst_duration if in_burst else self.quiet_duration
        return np.random.exponential(mean) if self.stochastic else mean

    def peak_rate(self) -> float:
        """Returns the highest send rate (qps) the phase can reach. Doubles as the
        proposal rate for thinning, so it must be an upper bound on rate_at.
        """
        if self.kind == "CONSTANT":
            return self.rate
        elif self.kind == "BURST":
            return max(self.rate, self.burst_rate)

        raise ValueError(f"Unrecognized workload phase kind {self.kind}")

    def mean_rate(self) -> float:
        """Returns the send rate (qps) averaged over the phase."""
        if self.kind == "CONSTANT":
            return self.rate
        elif self.kind == "BURST":
            duty = self.burst_duration / (self.burst_duration + self.quiet_duration)
            return duty * self.burst_rate + (1 - duty) * self.rate

        raise ValueError(f"Unrecognized workload phase kind {self.kind}")

    def expected_duration(self) -> float:
        """Returns how long the phase is expected to last (ms), which for a phase
        bounded by job count depends on the rate it generates them at.
        """
        if self.duration is not None:
            return self.duration

        mean_rate = self.mean_rate()
        assert(mean_rate > 0), "A workload phase bounded by NUM_JOBS needs a nonzero rate"
        return self.num_jobs / mean_rate * 1000


class Workload:
    """A client's full send pattern: an arrival process and a list of phases run
    back to back.
    """

    def __init__(self, cfg: dict):
        self.arrival_process = cfg.get("ARRIVAL_PROCESS", gcfg.DEFAULT_ARRIVAL_PROCESS)
        assert(self.arrival_process in ARRIVAL_PROCESSES), \
            f"Unknown arrival process {self.arrival_process}"

        # coefficient of variation of the inter-arrival times, GAMMA only
        self.gamma_cv = cfg.get("GAMMA_CV", gcfg.DEFAULT_GAMMA_CV)
        assert(self.gamma_cv > 0)

        self.trace_file_path = cfg.get("TRACE_FILE_PATH", cfg.get("trace_file_path"))
        self.trace_arrival_times: list[float] = []

        if self.arrival_process == "ALITRACE":
            if self.trace_file_path is None:
                repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
                self.trace_file_path = os.path.join(repo_root, "workflow", "azuretrace", "llm_az_processed_trace.csv")
            self.trace_arrival_times = self._load_trace_arrivals(self.trace_file_path)
            duration = max(float(self.trace_arrival_times[-1] - self.trace_arrival_times[0]) if len(self.trace_arrival_times) > 1 else 1.0, 1.0)
            self.phases = [WorkloadPhase({"KIND": "CONSTANT", "RATE": self.mean_rate(), "DURATION": duration})]
            return

        self.phases = [WorkloadPhase(p) for p in cfg["PHASES"]]
        assert(self.phases), "A workload needs at least one phase"

    def _load_trace_arrivals(self, trace_file_path: str) -> list[float]:
        """Loads a real arrival trace from a CSV file and normalizes it to ms.

        Supported columns include creation_time and start_timestamp_ms, which are
        common in the repo's workflow trace datasets.
        """
        expanded_path = os.path.abspath(os.path.expanduser(trace_file_path))
        if not os.path.exists(expanded_path):
            raise FileNotFoundError(f"Trace file not found: {expanded_path}")

        df = pd.read_csv(expanded_path)
        for col_name in ["creation_time", "start_timestamp_ms", "arrival_time", "timestamp_ms"]:
            if col_name in df.columns:
                arrival_times = pd.to_numeric(df[col_name], errors="coerce").dropna().to_numpy(dtype=float)
                break
        else:
            raise ValueError(
                f"Trace file {expanded_path} does not contain a supported arrival-time column; "
                "expected one of: creation_time, start_timestamp_ms, arrival_time, timestamp_ms"
            )

        arrival_times = np.sort(arrival_times)
        if arrival_times.size == 0:
            raise ValueError(f"Trace file {expanded_path} contains no usable arrival times")

        arrival_times = arrival_times - arrival_times.min()
        return arrival_times.astype(float).tolist()

    def generate_arrival_times(self, start_time: float) -> list[float]:
        """Generates the times at which this workload sends jobs.

        Args:
            start_time: Time the workload starts (ms)

        Returns:
            arrival_times: Send times in increasing order (ms)
        """
        if self.arrival_process == "ALITRACE":
            return [float(start_time + arrival_time) for arrival_time in self.trace_arrival_times]

        times: list[float] = []

        phase_start = start_time
        for phase in self.phases:
            phase_times = self._generate_phase(phase, phase_start)
            times.extend(phase_times)

            # a phase bounded by job count ends when its last job is sent
            phase_start = (phase_start + phase.duration if phase.duration is not None
                           else (phase_times[-1] if phase_times else phase_start))

        return times

    def _generate_phase(self, phase: WorkloadPhase, start_time: float) -> list[float]:
        """Generates the send times of one phase, starting at [start_time] (ms)."""
        end_time = start_time + phase.duration if phase.duration is not None else np.inf
        target_jobs = phase.num_jobs if phase.num_jobs is not None else np.inf
        peak_rate = phase.peak_rate()

        times: list[float] = []
        time = start_time
        while len(times) < target_jobs and time < end_time:
            if self.arrival_process == "POISSON":
                if peak_rate <= 0:
                    break

                # thinning: propose arrivals at the phase's peak rate, then keep
                # each one with probability rate(t)/peak_rate. This reproduces the
                # time varying rate exactly, unlike holding the rate fixed between
                # consecutive arrivals
                time += np.random.exponential(1000 / peak_rate)
                if time >= end_time:
                    break

                if np.random.random() * peak_rate <= phase.rate_at(time - start_time):
                    times.append(time)

                continue

            rate = phase.rate_at(time - start_time)
            if rate <= 0:
                # nothing to send right now; step forward and look again
                time += _IDLE_STEP_MS
                continue

            mean_interarrival = 1000 / rate
            if self.arrival_process == "CONSTANT":
                time += mean_interarrival
            elif self.arrival_process == "GAMMA":
                # shape/scale chosen so the draw has mean [mean_interarrival] and
                # coefficient of variation [self.gamma_cv]
                shape = 1 / self.gamma_cv**2
                time += np.random.gamma(shape, mean_interarrival / shape)
            else:
                raise ValueError(f"Unrecognized arrival process {self.arrival_process}")

            if time >= end_time:
                break

            times.append(time)

        return times

    def peak_rate(self) -> float:
        """Returns the highest send rate (qps) reached by any phase."""
        if self.arrival_process == "ALITRACE":
            if not self.trace_arrival_times:
                return 0.0
            diffs = np.diff(np.asarray(self.trace_arrival_times, dtype=float))
            positive = diffs[diffs > 0]
            if positive.size == 0:
                return 0.0
            return 1000.0 / positive.min()

        return max(phase.peak_rate() for phase in self.phases)

    def mean_rate(self) -> float:
        """Returns the send rate (qps) averaged over the whole workload, weighting
        each phase by how long it is expected to last.
        """
        if self.arrival_process == "ALITRACE":
            if not self.trace_arrival_times:
                return 0.0
            trace = np.asarray(self.trace_arrival_times, dtype=float)
            duration = float(trace[-1] - trace[0]) if trace.size > 1 else 1.0
            duration = max(duration, 1e-9)
            return len(trace) * 1000.0 / duration

        total_duration = sum(phase.expected_duration() for phase in self.phases)
        assert(total_duration > 0)

        return sum(phase.mean_rate() * phase.expected_duration()
                   for phase in self.phases) / total_duration


def get_client_workloads(client_configs: list[dict] = None) -> list[tuple[int, Workload, float]]:
    """Parses the configured client workloads.

    Args:
        client_configs: Client configs to parse, defaulting to gcfg.CLIENT_CONFIGS

    Returns:
        client_workloads: (workflow ID, workload, job SLO) for each configured client
    """
    if client_configs is None:
        client_configs = gcfg.CLIENT_CONFIGS

    return [(wid, Workload(cfgw["WORKLOAD"]), cfgw["SLO"])
            for cfg in client_configs for wid, cfgw in cfg.items()]


def get_workflow_mean_rates(client_configs: list[dict] = None) -> dict[int, float]:
    """Returns the total mean send rate (qps) per workflow, summed over the clients
    sending it. This is the long run demand each workflow places on the cluster.
    """
    rates: dict[int, float] = {}
    for wid, workload, _ in get_client_workloads(client_configs):
        rates[wid] = rates.get(wid, 0) + workload.mean_rate()

    return rates


def get_workflow_peak_rates(client_configs: list[dict] = None) -> dict[int, float]:
    """Returns the total peak send rate (qps) per workflow, summed over the clients
    sending it.
    """
    rates: dict[int, float] = {}
    for wid, workload, _ in get_client_workloads(client_configs):
        rates[wid] = rates.get(wid, 0) + workload.peak_rate()

    return rates


def get_workflow_slos(client_configs: list[dict] = None) -> dict[int, float]:
    """Returns the job SLO (ms) configured for each workflow.

    One Workflow object is shared by every client sending it, so clients sending
    the same workflow must agree on its SLO.
    """
    slos: dict[int, float] = {}
    for wid, _, slo in get_client_workloads(client_configs):
        assert(wid not in slos or slos[wid] == slo), \
            f"Workflow {wid} is sent with conflicting SLOs"
        slos[wid] = slo

    return slos
