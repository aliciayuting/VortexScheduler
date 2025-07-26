import numpy as np


""" --------      Worker Machine Parameters      -------- """

GPU_MEMORY_SIZE = 24000000  # in KB, 24GB for NVIDIA A30
TOTAL_NUM_OF_NODES = 4
VALID_WORKER_SIZES = [24000000, 12000000, 6000000]


"""  --------       Workload Parameters    --------  """

CLIENT_CONFIGS = [
    {0: {"NUM_JOBS": 3000,
         "SEND_RATES": [95, 125],
         "SEND_RATE_CHANGE_INTERVALS": [1500], 
         "SLO": 100}}, # in ms
    # {0: {"NUM_JOBS": 1000,
    #      "SEND_RATES": [55, 95],
    #      "SEND_RATE_CHANGE_INTERVALS": [500], # in queries since last change
    #      "SLO": 30000}}
]

WORKLOAD_DISTRIBUTION = "CONSTANT"  # CONSTANT | POISSON | GAMMA
GAMMA_CV = 10  # Coefficient of variation for gamma distribution


"""  -------        Navigator Parameters  --------- """

LOAD_INFORMATION_STALENESS = 1  # in ms
PLACEMENT_INFORMATION_STALENESS = 1  # in ms
RESCHEDULE_THREASHOLD = 1.5


"""  -------        Shepherd Parameters  --------- """

FLEX_LAMBDA = 3.03
HERD_K = 1.5
HERD_PERIODICITY = 12000    # run HERD every [HERD_PERIODICITY] ms


"""  -------        General Scheduling Parameters  --------- """

SLO_SLACK = 0.1
SLO_GRANULARITY = "JOB" # TASK | JOB

ENABLE_MULTITHREADING = False # allow multiple models on same partition to run at once
ENABLE_MODEL_PREFETCH = True
ENABLE_DYNAMIC_MODEL_LOADING = True

# HERD | VORTEX | CUSTOM
# NOTE: HERD requires ENABLE_DYNAMIC_MODEL_LOADING
ALLOCATION_STRATEGY = "CUSTOM"

# [(partition size in GB, [model ids])]
CUSTOM_ALLOCATION = [(24, [1]), (24, [1]), (24, [1]), (6, [3]), (6, [3]), (6, [3]), (6, [0, 2])]

# ppl1 4 node real alloc:
# [(24, [1]), (24, [1]), (24, [1]), (6, [3]), (6, [3]), (6, [3]), (6, [0, 2])]

# ppl2 4 node real alloc:
# [(12, [4]), (12, [5,6]), (12, [7]), (12, [7]), (12, [7]), (12, [7]), (12, [7]), (12, [7])]