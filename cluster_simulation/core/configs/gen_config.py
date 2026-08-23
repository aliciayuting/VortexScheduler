import numpy as np

""" --------      Verification Parameters      -------- """

PRODUCE_EVENT_LOG = True

ENABLE_LIVE_VERIFICATION = True

# Runs a trace of produced event logs to verify simulator actions
ENABLE_VERIFICATION = False
ENABLE_TRACE_VERIFICATION = False
VERIFICATION_WINDOW_SIZE = 5000 # only process up to this many events (don't do full trace)

# Print event details for every step of verifier trace
ENABLE_VERIFICATION_DEBUG_LOGGING = False

ENABLE_DUPLICATE_EVENT_CHECK = False
ENABLE_CONSOLE_PRINT = False

""" --------      Worker Machine Parameters      -------- """

GPU_MEMORY_SIZE = 24000000  # in KB, 24GB for NVIDIA A30
MIN_NUM_NODES = 5
MAX_NUM_NODES = 5
VALID_WORKER_SIZES = [24000000, 12000000, 6000000]

MAX_NUM_MODELS_PER_NODE = 4

"""  --------       Workload Parameters    --------  """

# A workload is a list of PHASES run back to back, plus the ARRIVAL_PROCESS that
# turns the rate of the current phase into individual arrivals (CONSTANT | POISSON
# | GAMMA, defaulting to DEFAULT_ARRIVAL_PROCESS below). Every phase ends after
# either DURATION ms or NUM_JOBS jobs, and takes one of these shapes:
#
#   {"KIND": "CONSTANT", "RATE": 40}                      steady at 40 qps
#   {"KIND": "BURST", "RATE": 20, "BURST_RATE": 90,       20 qps, spiking to 90 qps
#    "BURST_DURATION": 2000, "QUIET_DURATION": 8000}      for ~2s out of every ~10s
#
# A BURST phase draws each burst and each gap between bursts from an exponential
# around the configured mean, unless "STOCHASTIC": False makes them exactly that
# long. GAMMA workloads take a "GAMMA_CV" per workload; a CV above 1 clusters
# arrivals more tightly than Poisson does, below 1 spreads them more evenly.

CLIENT_CONFIGS = [
    # WF6/10/11 (textvision variants)
    # {6: {"WORKLOAD": {"ARRIVAL_PROCESS": "POISSON",
    #                   "PHASES": [{"KIND": "CONSTANT", "RATE": 40, "NUM_JOBS": 5000}]},
    #      "SLO": int(62.48 * 5)}},
    # {10: {"WORKLOAD": {"ARRIVAL_PROCESS": "POISSON",
    #                    "PHASES": [{"KIND": "CONSTANT", "RATE": 40, "NUM_JOBS": 5000}]},
    #       "SLO": int(70.48 * 5)}},
    # {11: {"WORKLOAD": {"ARRIVAL_PROCESS": "POISSON",
    #                    "PHASES": [{"KIND": "CONSTANT", "RATE": 40, "NUM_JOBS": 5000}]},
    #       "SLO": int(80.48 * 5)}},

    # WF6/10/11 (textvision variants) with spike
    {6: {"WORKLOAD": {"ARRIVAL_PROCESS": "POISSON",
                      "PHASES": [{"KIND": "BURST", "RATE": 35, "BURST_RATE": 120,
                                  "QUIET_DURATION": 30000, "BURST_DURATION": 10000,
                                  "STOCHASTIC": True, "DURATION": 40000},
                                 {"KIND": "CONSTANT", "RATE": 35, "DURATION": 80000}]},
         "SLO": int(62.48 * 5)}},
    {10: {"WORKLOAD": {"ARRIVAL_PROCESS": "POISSON",
                       "PHASES": [{"KIND": "BURST", "RATE": 35, "BURST_RATE": 120,
                                   "QUIET_DURATION": 60000, "BURST_DURATION": 10000,
                                   "STOCHASTIC": True, "DURATION": 70000},
                                  {"KIND": "CONSTANT", "RATE": 35, "DURATION": 50000}]},
          "SLO": int(70.48 * 5)}},
    {11: {"WORKLOAD": {"ARRIVAL_PROCESS": "POISSON",
                       "PHASES": [{"KIND": "BURST", "RATE": 35, "BURST_RATE": 120,
                                   "QUIET_DURATION": 90000, "BURST_DURATION": 10000,
                                   "STOCHASTIC": True, "DURATION": 100000},
                                  {"KIND": "CONSTANT", "RATE": 35, "DURATION": 20000}]},
          "SLO": int(80.48 * 5)}},

    # WF1/4/5 (multitenant ppl 2)
    # {1: {"WORKLOAD": {"ARRIVAL_PROCESS": "POISSON",
    #                   "PHASES": [{"KIND": "CONSTANT", "RATE": 6, "NUM_JOBS": 5000}]},
    #      "SLO": int(186.3 * 5)}},
    # {4: {"WORKLOAD": {"ARRIVAL_PROCESS": "POISSON",
    #                   "PHASES": [{"KIND": "CONSTANT", "RATE": 6, "NUM_JOBS": 5000}]},
    #      "SLO": int(358.6 * 5)}},
    # {5: {"WORKLOAD": {"ARRIVAL_PROCESS": "POISSON",
    #                   "PHASES": [{"KIND": "CONSTANT", "RATE": 6, "NUM_JOBS": 5000}]},
    #      "SLO": int(326.7 * 5)}},

    # WF12/13/14 (flmr-shared)
    # {12: {"WORKLOAD": {"ARRIVAL_PROCESS": "POISSON",
    #                    "PHASES": [{"KIND": "CONSTANT", "RATE": 44, "NUM_JOBS": 5000}]},
    #       "SLO": int(24.05 * 5)}},
    # {13: {"WORKLOAD": {"ARRIVAL_PROCESS": "POISSON",
    #                    "PHASES": [{"KIND": "CONSTANT", "RATE": 72, "NUM_JOBS": 5000}]},
    #       "SLO": int(47.75 * 5)}},
    # {14: {"WORKLOAD": {"ARRIVAL_PROCESS": "POISSON",
    #                    "PHASES": [{"KIND": "CONSTANT", "RATE": 12, "NUM_JOBS": 5000}]},
    #       "SLO": int(108.25 * 5)}},

    # WF15/16/17 (search_1-shared)
    # {15: {"WORKLOAD": {"ARRIVAL_PROCESS": "POISSON",
    #                    "PHASES": [{"KIND": "CONSTANT", "RATE": 28, "NUM_JOBS": 5000}]},
    #       "SLO": int(124.0 * 5)}},
    # {16: {"WORKLOAD": {"ARRIVAL_PROCESS": "POISSON",
    #                    "PHASES": [{"KIND": "CONSTANT", "RATE": 28, "NUM_JOBS": 5000}]},
    #       "SLO": int(125.56 * 5)}},
    # {17: {"WORKLOAD": {"ARRIVAL_PROCESS": "POISSON",
    #                    "PHASES": [{"KIND": "CONSTANT", "RATE": 28, "NUM_JOBS": 5000}]},
    #       "SLO": int(124.0 * 5)}},
]

DEFAULT_ARRIVAL_PROCESS = "POISSON"  # CONSTANT | POISSON | GAMMA
DEFAULT_GAMMA_CV = 1.5  # coefficient of variation of gamma inter-arrival times

"""  -------        Navigator Parameters  --------- """

LOAD_INFORMATION_STALENESS = 1  # in ms
PLACEMENT_INFORMATION_STALENESS = 1  # in ms
RESCHEDULE_THREASHOLD = 1.5

"""  -------        Shepherd Parameters  --------- """

FLEX_LAMBDA = 3.03
HERD_K = 1.5
HERD_PERIODICITY = 12000    # run HERD every [HERD_PERIODICITY] ms
ENABLE_PREEMPTION = True

"""  -------        Boost Parameters  --------- """

BOOST_PARAMETER = 0.00293596042

# FCFS | EDF | LAXITY | RELATIVE_LAXITY
# JOB_SIZE | REMAINING_EXEC_TIME | REMAINING_TIME_TO_DEADLINE | LAXITY_BOOST | RELATIVE_LAXITY_BOOST
BOOST_POLICY = "FCFS"

""" -------         Inferline Parameters  -------- """

ESTIMATOR_CLIENT_CONFIGS = [ # in ms
    {0: {"WORKLOAD": {"ARRIVAL_PROCESS": "POISSON",
                      "PHASES": [{"KIND": "CONSTANT", "RATE": 55, "NUM_JOBS": 1000},
                                 {"KIND": "CONSTANT", "RATE": 95, "NUM_JOBS": 1000}]},
         "SLO": 1014}},
]
INFERLINE_TUNING_INTERVAL = 15 * 1000 # ms

ENABLE_ESTIMATOR_LOGGING = False

"""  -------        General Scheduling Parameters  --------- """

# ROUND_ROBIN (central or decentral) | SHEPHERD
DISPATCH_POLICY = "ROUND_ROBIN"
ENABLE_PIPELINING = False
ENABLE_NETWORKING_DELAYS = False

# LARGEST | LARGEST_FEASIBLE (largest non-SLO violating batch)
# NOTE: LARGEST_FEASIBLE is not wired up -- see TaskBatcher.get_batch
BATCH_POLICY = "LARGEST"
FALLBACK_TO_LARGEST_BATCH = True
DISABLE_BATCHING = False  # always run batch size 1 when True

# NONE | LAZY | EARLY
DROP_POLICY = "NONE"
SLO_SLACK = 0
SLO_TYPE = "JOB_LEVEL" # JOB_LEVEL | NEXUS

# NONE | PROBABILISTIC | ROUND_ROBIN | TOKEN_BUCKET
ADMISSION_CONTROL_POLICY = "TOKEN_BUCKET"
ADMISSION_DROP_RATE = 0.1  # PROBABILISTIC or ROUND_ROBIN
ADMISSION_TARGET_UTILIZATION = 0.9  # TOKEN_BUCKET: fraction of estimated capacity to admit

# TOKEN_BUCKET: largest burst admitted at once, in jobs. None derives it per
# workflow from the SLO: the burst the cluster can drain before the jobs at the
# back of it run out of slack
ADMISSION_BURST_SIZE = 5

ENABLE_MULTITHREADING = True # allow multiple models on same partition to run at once

# NONE | INFERLINE
AUTOSCALING_POLICY = "NONE"

# HERD | CUSTOM | INFERLINE
ALLOCATION_STRATEGY = "CUSTOM"

# WF6/10/11 alloc (textvision variants)
CUSTOM_ALLOCATION = [
    (24, [1]), (24, [1]), (24, [1]), (6, [3]), (6, [3]), (6, [3]), (6, [0, 2]),
    (6, [14]), (6, [15]), (6, [16]), (6, [])
]

# 10-node multitenant ppl 2 (3 versions) alloc
# CUSTOM_ALLOCATION = [
#     (12, [4]), (6, [5,13]), (6, [5,13]),
#     (12, [6]), (12, [6]),
#     (12, [7]), (12, [7]),
#     (12, [7]), (12, [7]),
#     (12, [8]), (12, [8]),
#     (24, [9]),
#     (24, [9]),
#     (24, [10]),
#     (24, [10]),
#     (24, [10])
# ]

# flmr-shared workflows (WF12/13/14) alloc
# CUSTOM_ALLOCATION = [
#     (6, [2]),
#     (6, [5, 14]),
#     (6, [5]),
#     (6, [0, 12]),
#     (6, [0]),
#     (6, [12]),
#     (6, [15]),
#     (6, [11]),
#     (24, [8, 11]),
#     (24, [17]),
#     (24, [17]),
# ]

# search_1-shared workflows (WF15/16/17) alloc
# CUSTOM_ALLOCATION = [
#     (12, [4]), (12, [4]),
#     (12, [4]), (12, [4]),
#     (6, [11]), (6, [11]), (6, [11]), (6, [11]),
#     (12, [6]), (12, [6]), (12, [6]),
#     (6, [5, 3]), (6, [12, 0]),
# ]

# ppl1 4 node alloc:
# [(24, [1]), (24, [1]), (24, [1]), (6, [3]), (6, [3]), (6, [3]), (6, [0, 2])]

# ppl2 4 node alloc:
# [(12, [4]), (12, [5,6]), (12, [7]), (12, [7]), (12, [7]), (12, [7]), (12, [7]), (12, [7])]
