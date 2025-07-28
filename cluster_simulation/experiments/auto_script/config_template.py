import numpy as np


""" --------      Worker Machine Parameters      -------- """

GPU_MEMORY_SIZE = 24000000  # in KB, 24GB for NVIDIA A30
TOTAL_NUM_OF_NODES = 4
VALID_WORKER_SIZES = [24000000, 12000000, 6000000]


"""  --------       Workload Parameters    --------  """

CLIENT_CONFIGS = []

# CONSTANT | POISSON | GAMMA
WORKLOAD_DISTRIBUTION = "POISSON"

# Coefficient of variation for gamma distribution
GAMMA_CV = 10


"""  -------        Navigator Parameters  --------- """

# in ms
LOAD_INFORMATION_STALENESS = 1
PLACEMENT_INFORMATION_STALENESS = 1

RESCHEDULE_THREASHOLD = 1.5


"""  -------        Shepherd Parameters  --------- """

FLEX_LAMBDA = 3.03
HERD_K = 1.5

# run HERD every [HERD_PERIODICITY] ms
HERD_PERIODICITY = 0


"""  -------        General Scheduling Parameters  --------- """

SLO_SLACK = 0.1

# TASK | JOB
SLO_GRANULARITY = "JOB"

# allow multiple models on same partition to run at once
ENABLE_MULTITHREADING = False

ENABLE_MODEL_PREFETCH = False
ENABLE_DYNAMIC_MODEL_LOADING = True

# HERD | VORTEX | CUSTOM
# NOTE: HERD requires ENABLE_DYNAMIC_MODEL_LOADING
ALLOCATION_STRATEGY = "HERD"

# [(partition size in GB, [model ids])]
CUSTOM_ALLOCATION = []

# static experiment alloc, ppl1:
# [(24, [1]), (24, [1]), (24, [1]), (6, [3]), (6, [3]), (6, [3]), (6, [0, 2])]

# static experiment alloc, ppl2:
# [(12, [4]), (12, [5,6]), (12, [7]), (12, [7]), (12, [7]), (12, [7]), (12, [7]), (12, [7])]