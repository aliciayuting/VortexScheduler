import numpy as np

""" --------      Simulation Parameters      -------- """

PRODUCE_EVENT_LOG = True

# Runs a trace of produced event logs to verify simulator actions
ENABLE_VERIFICATION = False
ENABLE_TRACE_VERIFICATION = False
VERIFICATION_WINDOW_SIZE = 5000 # only process up to this many events (don't do full trace)

# Print event details for every step of verifier trace
ENABLE_VERIFICATION_DEBUG_LOGGING = True

""" --------      Worker Machine Parameters      -------- """

GPU_MEMORY_SIZE = 24000000  # in KB, 24GB for NVIDIA A30
MIN_NUM_NODES = 5
MAX_NUM_NODES = 5
VALID_WORKER_SIZES = [24000000, 12000000, 6000000]

MAX_NUM_MODELS_PER_NODE = 4

"""  --------       Workload Parameters    --------  """

CLIENT_CONFIGS = [ # in ms
    {6: {"SEND_RATES": [32],
         "JOBS_PER_SEND_RATE": [1000], 
         "SLO": int(62.48 * 5)}},
    {7: {"SEND_RATES": [32],
         "JOBS_PER_SEND_RATE": [1000], 
         "SLO": int(70.48 * 5)}},
    {8: {"SEND_RATES": [32],
         "JOBS_PER_SEND_RATE": [1000], 
         "SLO": int(80.48 * 5)}},

    # {1: {"NUM_JOBS": 5000,
    #      "SEND_RATES": [8],#[12],
    #      "SEND_RATE_CHANGE_INTERVALS": [], 
    #      "SLO": int(256.3*2)}},
    # {4: {"NUM_JOBS": 5000,
    #      "SEND_RATES": [8],#[12],
    #      "SEND_RATE_CHANGE_INTERVALS": [], 
    #      "SLO": int(787.2*2)}},
    # {5: {"NUM_JOBS": 5000,
    #      "SEND_RATES": [8],#[12],
    #      "SEND_RATE_CHANGE_INTERVALS": [], 
    #      "SLO": int(388.7*2)}},
]

WORKLOAD_DISTRIBUTION = "POISSON"  # CONSTANT | POISSON | GAMMA
GAMMA_CV = 10  # Coefficient of variation for gamma distribution


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

BOOST_PARAMETER = 0.00293596042 # 0.00104567474

# FCFS | EDF | LAXITY | JOB_SIZE | REMAINING_JOB_TIME
BOOST_POLICY = "REMAINING_JOB_TIME"

""" -------         Inferline Parameters  -------- """

ESTIMATOR_CLIENT_CONFIGS = [ # in ms
    {0: {"NUM_JOBS": 1000,
         "SEND_RATES": [55, 95],
         "SEND_RATE_CHANGE_INTERVALS": [500], 
         "SLO": 1014}},
]
INFERLINE_TUNING_INTERVAL = 15 * 1000 # ms

ENABLE_ESTIMATOR_LOGGING = False

"""  -------        General Scheduling Parameters  --------- """

# ROUND_ROBIN | QUEUED_ROUND_ROBIN | SHEPHERD | HEFT (central only)
DISPATCH_POLICY = "QUEUED_ROUND_ROBIN"
ENABLE_PIPELINING = False
ENABLE_NETWORKING_DELAYS = False

# LARGEST | LARGEST_FEASIBLE (largest non-SLO violating batch)
BATCH_POLICY = "LARGEST"
FALLBACK_TO_LARGEST_BATCH = False

# OPTIMAL | LATEST_POSSIBLE | CLUSTER_ADMISSION_LIMIT | NONE
DROP_POLICY = "LATEST_POSSIBLE"

SLO_SLACK = 0
SLO_TYPE = "JOB_LEVEL" # JOB_LEVEL | NEXUS

ENABLE_MULTITHREADING = True # allow multiple models on same partition to run at once

# NONE | INFERLINE
AUTOSCALING_POLICY = "NONE" #"INFERLINE"

# HERD | CUSTOM | INFERLINE
ALLOCATION_STRATEGY = "CUSTOM" #"INFERLINE"

# [(partition size in GB, [model ids])]
CUSTOM_ALLOCATION = [
    (24, [1]), (24, [1]), (24, [1]), (6, [3]), (6, [3]), (6, [3]), (6, [0, 2]),
    (6, [14]), (6, [15]), (6, [16]), (6, [])
]

# 12-node mutlitenant ppl 2 (3 versions) alloc
# [
#  (12, [4]), (6, [5,13]), (6, [5,13]),
#  (12, [6]), (12, [6]), 
#  (12, [7]), (12, [7]), 
#  (12, [7]), (12, [7]),
#  (12, [8]), (12, [8]),
#  (24, [9]), 
#  (24, [9]), 
#  (24, [9]), 
#  (24, [9]),
#  (24, [10]),
#  (24, [10]),
#  (24, [10])]

# ppl1 4 node alloc:
# [(24, [1]), (24, [1]), (24, [1]), (6, [3]), (6, [3]), (6, [3]), (6, [0, 2])]

# ppl2 4 node alloc:
# [(12, [4]), (12, [5,6]), (12, [7]), (12, [7]), (12, [7]), (12, [7]), (12, [7]), (12, [7])]