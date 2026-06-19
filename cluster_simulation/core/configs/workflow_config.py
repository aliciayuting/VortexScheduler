import numpy as np


"""  --------       Workflow Parameters     --------  """
# https://keras.io/api/applications/
WORKFLOW_LIST = [
    {"JOB_TYPE": 0,         # ID of the type of workflow (dependency graph)
     "JOB_NAME": "textvision0",
     "TASKS": [
        {"MODEL_ID": 0,
         "TASK_INDEX": 0,
         "PREV_TASK_INDEX": [],
         "NEXT_TASK_INDEX": [2],
         "INPUT_SIZE": 1, # kB
         "OUTPUT_SIZE": 2,
         "SLO": 0}, # ms
        {"MODEL_ID": 1,
         "TASK_INDEX": 1,
         "PREV_TASK_INDEX": [],
         "NEXT_TASK_INDEX": [2],
         "INPUT_SIZE": 10000, # kB
         "OUTPUT_SIZE": 1000,
         "SLO": 0},
        {"MODEL_ID": 2,
         "TASK_INDEX": 2,
         "PREV_TASK_INDEX": [0,1],
         "NEXT_TASK_INDEX": [3],
         "INPUT_SIZE": 1002, # kB
         "OUTPUT_SIZE": 10,
         "SLO": 0},
        {"MODEL_ID": 3,
         "TASK_INDEX": 3,
         "PREV_TASK_INDEX": [2],
         "NEXT_TASK_INDEX": [],
         "INPUT_SIZE": 10, # kB
         "OUTPUT_SIZE": 10,
         "SLO": 0}]
    },
    {"JOB_TYPE": 1,
     "JOB_NAME": "tts",
     "TASKS": [
        {"MODEL_ID": 4,
         "TASK_INDEX": 0,
         "PREV_TASK_INDEX": [],
         "NEXT_TASK_INDEX": [1],
         "INPUT_SIZE": 1000, # kB
         "OUTPUT_SIZE": 10000,
         "SLO": 0},
        {"MODEL_ID": 5,
         "TASK_INDEX": 1,
         "PREV_TASK_INDEX": [0],
         "NEXT_TASK_INDEX": [2,3],
         "INPUT_SIZE": 10000, # kB
         "OUTPUT_SIZE": 20000,
         "SLO": 0},
        {"MODEL_ID": 6,
         "TASK_INDEX": 2,
         "PREV_TASK_INDEX": [1],
         "NEXT_TASK_INDEX": [3],
         "INPUT_SIZE": 20000, # kB
         "OUTPUT_SIZE": 25000,
         "SLO": 0},
        {"MODEL_ID": 7,
         "TASK_INDEX": 3,
         "PREV_TASK_INDEX": [1,2],
         "NEXT_TASK_INDEX": [],
         "INPUT_SIZE": 25000, # kB
         "OUTPUT_SIZE": 30000,
         "SLO": 0}]
    },
    {"JOB_TYPE": 2,
     "JOB_NAME": "textvision1",
     "TASKS": [
        {"MODEL_ID": 0,
         "TASK_INDEX": 0,
         "PREV_TASK_INDEX": [],
         "NEXT_TASK_INDEX": [2],
         "INPUT_SIZE": 1, # kB
         "OUTPUT_SIZE": 2, # intermediate: 10s of MB, starting is 1-2MB
         "SLO": 0}, # ms
        {"MODEL_ID": 1,
         "TASK_INDEX": 1,
         "PREV_TASK_INDEX": [],
         "NEXT_TASK_INDEX": [2],
         "INPUT_SIZE": 10000, # kB
         "OUTPUT_SIZE": 1000,
         "SLO": 0},
        {"MODEL_ID": 2,
         "TASK_INDEX": 2,
         "PREV_TASK_INDEX": [0,1],
         "NEXT_TASK_INDEX": [3],
         "INPUT_SIZE": 1002, # kB
         "OUTPUT_SIZE": 10,
         "SLO": 0},
        {"MODEL_ID": 11,
         "TASK_INDEX": 3,
         "PREV_TASK_INDEX": [2],
         "NEXT_TASK_INDEX": [],
         "INPUT_SIZE": 10, # kB
         "OUTPUT_SIZE": 10,
         "SLO": 0}]
    },
    {"JOB_TYPE": 3,
     "JOB_NAME": "textvision2",
     "TASKS": [
        {"MODEL_ID": 0,
         "TASK_INDEX": 0,
         "PREV_TASK_INDEX": [],
         "NEXT_TASK_INDEX": [2],
         "INPUT_SIZE": 1, # kB
         "OUTPUT_SIZE": 2,
         "SLO": 0}, # ms
        {"MODEL_ID": 1,
         "TASK_INDEX": 1,
         "PREV_TASK_INDEX": [],
         "NEXT_TASK_INDEX": [2],
         "INPUT_SIZE": 10000, # kB
         "OUTPUT_SIZE": 1000,
         "SLO": 0},
        {"MODEL_ID": 2,
         "TASK_INDEX": 2,
         "PREV_TASK_INDEX": [0,1],
         "NEXT_TASK_INDEX": [3],
         "INPUT_SIZE": 1002, # kB
         "OUTPUT_SIZE": 10,
         "SLO": 0},
        {"MODEL_ID": 12,
         "TASK_INDEX": 3,
         "PREV_TASK_INDEX": [2],
         "NEXT_TASK_INDEX": [],
         "INPUT_SIZE": 10, # kB
         "OUTPUT_SIZE": 10,
         "SLO": 0}]
    },
    # WORKFLOW 2 VARIANTS
    {"JOB_TYPE": 4,
     "JOB_NAME": "tts_lang",
     "TASKS": [
        {"MODEL_ID": 4,
         "TASK_INDEX": 0,
         "PREV_TASK_INDEX": [],
         "NEXT_TASK_INDEX": [1],
         "INPUT_SIZE": 1000, # kB
         "OUTPUT_SIZE": 20000,
         "SLO": 0},
        {"MODEL_ID": 5,
         "TASK_INDEX": 1,
         "PREV_TASK_INDEX": [0],
         "NEXT_TASK_INDEX": [2,3,5],
         "INPUT_SIZE": 20000, # kB
         "OUTPUT_SIZE": 20000,
         "SLO": 0},
        {"MODEL_ID": 6,
         "TASK_INDEX": 2,
         "PREV_TASK_INDEX": [1],
         "NEXT_TASK_INDEX": [5],
         "INPUT_SIZE": 20000, # kB
         "OUTPUT_SIZE": 20000,
         "SLO": 0},
        {"MODEL_ID": 8,
         "TASK_INDEX": 3,
         "PREV_TASK_INDEX": [1],
         "NEXT_TASK_INDEX": [4,5],
         "INPUT_SIZE": 20000, # kB
         "OUTPUT_SIZE": 20000,
         "SLO": 0},
        {"MODEL_ID": 9,
         "TASK_INDEX": 4,
         "PREV_TASK_INDEX": [3],
         "NEXT_TASK_INDEX": [5],
         "INPUT_SIZE": 20000, # kB
         "OUTPUT_SIZE": 20000,
         "SLO": 0},
        {"MODEL_ID": 7,
         "TASK_INDEX": 5,
         "PREV_TASK_INDEX": [1,2,3,4],
         "NEXT_TASK_INDEX": [],
         "INPUT_SIZE": 80000, # kB
         "OUTPUT_SIZE": 80000,
         "SLO": 0}]
    },
    {"JOB_TYPE": 5,
     "JOB_NAME": "img_captioning",
     "TASKS": [
        {"MODEL_ID": 10,
         "TASK_INDEX": 0,
         "PREV_TASK_INDEX": [],
         "NEXT_TASK_INDEX": [1,2,3],
         "INPUT_SIZE": 1000, # kB
         "OUTPUT_SIZE": 20000,
         "SLO": 0},
        {"MODEL_ID": 6,
         "TASK_INDEX": 1,
         "PREV_TASK_INDEX": [0],
         "NEXT_TASK_INDEX": [3],
         "INPUT_SIZE": 20000, # kB
         "OUTPUT_SIZE": 20000,
         "SLO": 0},
        {"MODEL_ID": 7,
         "TASK_INDEX": 2,
         "PREV_TASK_INDEX": [0],
         "NEXT_TASK_INDEX": [3],
         "INPUT_SIZE": 20000, # kB
         "OUTPUT_SIZE": 20000,
         "SLO": 0},
        {"MODEL_ID": 13,
         "TASK_INDEX": 3,
         "PREV_TASK_INDEX": [0,1,2],
         "NEXT_TASK_INDEX": [],
         "INPUT_SIZE": 60000, # kB
         "OUTPUT_SIZE": 60000,
         "SLO": 0}]
    },
    # WORKFLOW 1 VARIANTS
    {"JOB_TYPE": 6,         # ID of the type of workflow (dependency graph)
     "JOB_NAME": "textvision0",
     "TASKS": [
        {"MODEL_ID": 0,
         "TASK_INDEX": 0,
         "PREV_TASK_INDEX": [],
         "NEXT_TASK_INDEX": [2],
         "INPUT_SIZE": 1000, # kB
         "OUTPUT_SIZE": 20000,
         "SLO": 0}, # ms
        {"MODEL_ID": 1,
         "TASK_INDEX": 1,
         "PREV_TASK_INDEX": [],
         "NEXT_TASK_INDEX": [2],
         "INPUT_SIZE": 10000, # kB
         "OUTPUT_SIZE": 20000,
         "SLO": 0},
        {"MODEL_ID": 2,
         "TASK_INDEX": 2,
         "PREV_TASK_INDEX": [0,1],
         "NEXT_TASK_INDEX": [3],
         "INPUT_SIZE": 40000, # kB
         "OUTPUT_SIZE": 20000,
         "SLO": 0},
        {"MODEL_ID": 3,
         "TASK_INDEX": 3,
         "PREV_TASK_INDEX": [2],
         "NEXT_TASK_INDEX": [4],
         "INPUT_SIZE": 20000, # kB
         "OUTPUT_SIZE": 30000,
         "SLO": 0},
        {"MODEL_ID": 14,
         "TASK_INDEX": 4,
         "PREV_TASK_INDEX": [3],
         "NEXT_TASK_INDEX": [],
         "INPUT_SIZE": 30000, # kB
         "OUTPUT_SIZE": 1000,
         "SLO": 0}]
    },
    {"JOB_TYPE": 7,         # ID of the type of workflow (dependency graph)
     "JOB_NAME": "textvision1",
     "TASKS": [
        {"MODEL_ID": 0,
         "TASK_INDEX": 0,
         "PREV_TASK_INDEX": [],
         "NEXT_TASK_INDEX": [2],
         "INPUT_SIZE": 1000, # kB
         "OUTPUT_SIZE": 20000,
         "SLO": 0}, # ms
        {"MODEL_ID": 1,
         "TASK_INDEX": 1,
         "PREV_TASK_INDEX": [],
         "NEXT_TASK_INDEX": [2],
         "INPUT_SIZE": 10000, # kB
         "OUTPUT_SIZE": 20000,
         "SLO": 0},
        {"MODEL_ID": 2,
         "TASK_INDEX": 2,
         "PREV_TASK_INDEX": [0,1],
         "NEXT_TASK_INDEX": [3],
         "INPUT_SIZE": 40000, # kB
         "OUTPUT_SIZE": 20000,
         "SLO": 0},
        {"MODEL_ID": 3,
         "TASK_INDEX": 3,
         "PREV_TASK_INDEX": [2],
         "NEXT_TASK_INDEX": [4],
         "INPUT_SIZE": 20000, # kB
         "OUTPUT_SIZE": 30000,
         "SLO": 0},
        {"MODEL_ID": 15,
         "TASK_INDEX": 4,
         "PREV_TASK_INDEX": [3],
         "NEXT_TASK_INDEX": [],
         "INPUT_SIZE": 30000, # kB
         "OUTPUT_SIZE": 1000,
         "SLO": 0}]
    },
    {"JOB_TYPE": 8,         # ID of the type of workflow (dependency graph)
     "JOB_NAME": "textvision2",
     "TASKS": [
        {"MODEL_ID": 0,
         "TASK_INDEX": 0,
         "PREV_TASK_INDEX": [],
         "NEXT_TASK_INDEX": [2],
         "INPUT_SIZE": 1000, # kB
         "OUTPUT_SIZE": 20000,
         "SLO": 0}, # ms
        {"MODEL_ID": 1,
         "TASK_INDEX": 1,
         "PREV_TASK_INDEX": [],
         "NEXT_TASK_INDEX": [2],
         "INPUT_SIZE": 10000, # kB
         "OUTPUT_SIZE": 20000,
         "SLO": 0},
        {"MODEL_ID": 2,
         "TASK_INDEX": 2,
         "PREV_TASK_INDEX": [0,1],
         "NEXT_TASK_INDEX": [3],
         "INPUT_SIZE": 40000, # kB
         "OUTPUT_SIZE": 20000,
         "SLO": 0},
        {"MODEL_ID": 3,
         "TASK_INDEX": 3,
         "PREV_TASK_INDEX": [2],
         "NEXT_TASK_INDEX": [4],
         "INPUT_SIZE": 20000, # kB
         "OUTPUT_SIZE": 30000,
         "SLO": 0},
        {"MODEL_ID": 16,
         "TASK_INDEX": 4,
         "PREV_TASK_INDEX": [3],
         "NEXT_TASK_INDEX": [],
         "INPUT_SIZE": 30000, # kB
         "OUTPUT_SIZE": 1000,
         "SLO": 0}]
    },
    # workflow 1 variants with search stage moved to pipeline start
    {"JOB_TYPE": 9,         # ID of the type of workflow (dependency graph)
     "JOB_NAME": "textvision3",
     "TASKS": [
         {"MODEL_ID": 14,
         "TASK_INDEX": 0,
         "PREV_TASK_INDEX": [],
         "NEXT_TASK_INDEX": [1,2],
         "INPUT_SIZE": 30000, # kB
         "OUTPUT_SIZE": 1000,
         "SLO": 0},
        {"MODEL_ID": 0,
         "TASK_INDEX": 1,
         "PREV_TASK_INDEX": [0],
         "NEXT_TASK_INDEX": [3],
         "INPUT_SIZE": 1000, # kB
         "OUTPUT_SIZE": 20000,
         "SLO": 0}, # ms
        {"MODEL_ID": 1,
         "TASK_INDEX": 2,
         "PREV_TASK_INDEX": [0],
         "NEXT_TASK_INDEX": [3],
         "INPUT_SIZE": 10000, # kB
         "OUTPUT_SIZE": 20000,
         "SLO": 0},
        {"MODEL_ID": 2,
         "TASK_INDEX": 3,
         "PREV_TASK_INDEX": [1,2],
         "NEXT_TASK_INDEX": [4],
         "INPUT_SIZE": 40000, # kB
         "OUTPUT_SIZE": 20000,
         "SLO": 0},
        {"MODEL_ID": 3,
         "TASK_INDEX": 4,
         "PREV_TASK_INDEX": [3],
         "NEXT_TASK_INDEX": [],
         "INPUT_SIZE": 20000, # kB
         "OUTPUT_SIZE": 30000,
         "SLO": 0},]
    },
    {"JOB_TYPE": 10,         # ID of the type of workflow (dependency graph)
     "JOB_NAME": "textvision4",
     "TASKS": [
         {"MODEL_ID": 15,
         "TASK_INDEX": 0,
         "PREV_TASK_INDEX": [],
         "NEXT_TASK_INDEX": [1,2],
         "INPUT_SIZE": 30000, # kB
         "OUTPUT_SIZE": 1000,
         "SLO": 0},
        {"MODEL_ID": 0,
         "TASK_INDEX": 1,
         "PREV_TASK_INDEX": [0],
         "NEXT_TASK_INDEX": [3],
         "INPUT_SIZE": 1000, # kB
         "OUTPUT_SIZE": 20000,
         "SLO": 0}, # ms
        {"MODEL_ID": 1,
         "TASK_INDEX": 2,
         "PREV_TASK_INDEX": [0],
         "NEXT_TASK_INDEX": [3],
         "INPUT_SIZE": 10000, # kB
         "OUTPUT_SIZE": 20000,
         "SLO": 0},
        {"MODEL_ID": 2,
         "TASK_INDEX": 3,
         "PREV_TASK_INDEX": [1,2],
         "NEXT_TASK_INDEX": [4],
         "INPUT_SIZE": 40000, # kB
         "OUTPUT_SIZE": 20000,
         "SLO": 0},
        {"MODEL_ID": 3,
         "TASK_INDEX": 4,
         "PREV_TASK_INDEX": [3],
         "NEXT_TASK_INDEX": [],
         "INPUT_SIZE": 20000, # kB
         "OUTPUT_SIZE": 30000,
         "SLO": 0},]
    },
    {"JOB_TYPE": 11,         # ID of the type of workflow (dependency graph)
     "JOB_NAME": "textvision5",
     "TASKS": [
         {"MODEL_ID": 16,
         "TASK_INDEX": 0,
         "PREV_TASK_INDEX": [],
         "NEXT_TASK_INDEX": [1, 2],
         "INPUT_SIZE": 30000, # kB
         "OUTPUT_SIZE": 1000,
         "SLO": 0},
        {"MODEL_ID": 0,
         "TASK_INDEX": 1,
         "PREV_TASK_INDEX": [0],
         "NEXT_TASK_INDEX": [3],
         "INPUT_SIZE": 1000, # kB
         "OUTPUT_SIZE": 20000,
         "SLO": 0}, # ms
        {"MODEL_ID": 1,
         "TASK_INDEX": 2,
         "PREV_TASK_INDEX": [0],
         "NEXT_TASK_INDEX": [3],
         "INPUT_SIZE": 10000, # kB
         "OUTPUT_SIZE": 20000,
         "SLO": 0},
        {"MODEL_ID": 2,
         "TASK_INDEX": 3,
         "PREV_TASK_INDEX": [1,2],
         "NEXT_TASK_INDEX": [4],
         "INPUT_SIZE": 40000, # kB
         "OUTPUT_SIZE": 20000,
         "SLO": 0},
        {"MODEL_ID": 3,
         "TASK_INDEX": 4,
         "PREV_TASK_INDEX": [3],
         "NEXT_TASK_INDEX": [],
         "INPUT_SIZE": 20000, # kB
         "OUTPUT_SIZE": 30000,
         "SLO": 0},
        ]
    },
    # flmr-shared workflows (model 2 at beginning / middle / end)
    {"JOB_TYPE": 12,
     "JOB_NAME": "fast_retrieval",
     "TASKS": [
        {"MODEL_ID": 2,
         "TASK_INDEX": 0,
         "PREV_TASK_INDEX": [],
         "NEXT_TASK_INDEX": [1],
         "INPUT_SIZE": 500,   # kB
         "OUTPUT_SIZE": 5000,
         "SLO": 0},
        {"MODEL_ID": 5,
         "TASK_INDEX": 1,
         "PREV_TASK_INDEX": [0],
         "NEXT_TASK_INDEX": [2],
         "INPUT_SIZE": 5000,  # kB
         "OUTPUT_SIZE": 3000,
         "SLO": 0},
        {"MODEL_ID": 14,
         "TASK_INDEX": 2,
         "PREV_TASK_INDEX": [1],
         "NEXT_TASK_INDEX": [],
         "INPUT_SIZE": 3000,  # kB
         "OUTPUT_SIZE": 1000,
         "SLO": 0},
    ]},
    {"JOB_TYPE": 13,
     "JOB_NAME": "document_search",
     "TASKS": [
        {"MODEL_ID": 0,
         "TASK_INDEX": 0,
         "PREV_TASK_INDEX": [],
         "NEXT_TASK_INDEX": [1],
         "INPUT_SIZE": 2000,  # kB
         "OUTPUT_SIZE": 4000,
         "SLO": 0},
        {"MODEL_ID": 2,
         "TASK_INDEX": 1,
         "PREV_TASK_INDEX": [0],
         "NEXT_TASK_INDEX": [2],
         "INPUT_SIZE": 4000,  # kB
         "OUTPUT_SIZE": 8000,
         "SLO": 0},
        {"MODEL_ID": 12,
         "TASK_INDEX": 2,
         "PREV_TASK_INDEX": [1],
         "NEXT_TASK_INDEX": [3],
         "INPUT_SIZE": 8000,  # kB
         "OUTPUT_SIZE": 6000,
         "SLO": 0},
        {"MODEL_ID": 15,
         "TASK_INDEX": 3,
         "PREV_TASK_INDEX": [2],
         "NEXT_TASK_INDEX": [],
         "INPUT_SIZE": 6000,  # kB
         "OUTPUT_SIZE": 2000,
         "SLO": 0},
    ]},
    {"JOB_TYPE": 14,
     "JOB_NAME": "cross_lingual_retrieval",
     "TASKS": [
        {"MODEL_ID": 8,
         "TASK_INDEX": 0,
         "PREV_TASK_INDEX": [],
         "NEXT_TASK_INDEX": [1],
         "INPUT_SIZE": 1000,  # kB
         "OUTPUT_SIZE": 2000,
         "SLO": 0},
        {"MODEL_ID": 17,
         "TASK_INDEX": 1,
         "PREV_TASK_INDEX": [0],
         "NEXT_TASK_INDEX": [2],
         "INPUT_SIZE": 2000,  # kB
         "OUTPUT_SIZE": 5000,
         "SLO": 0},
        {"MODEL_ID": 11,
         "TASK_INDEX": 2,
         "PREV_TASK_INDEX": [1],
         "NEXT_TASK_INDEX": [3],
         "INPUT_SIZE": 5000,  # kB
         "OUTPUT_SIZE": 8000,
         "SLO": 0},
        {"MODEL_ID": 2,
         "TASK_INDEX": 3,
         "PREV_TASK_INDEX": [2],
         "NEXT_TASK_INDEX": [],
         "INPUT_SIZE": 8000,  # kB
         "OUTPUT_SIZE": 3000,
         "SLO": 0},
    ]},
]

def get_task_types(job_types: list[int]) -> list[tuple[int,int]]:
    return [(jt, t["TASK_INDEX"]) for jt in job_types for t in WORKFLOW_LIST[jt]["TASKS"]]
def get_model_id_for_task_type(task_type: tuple[int,int]) -> int:
    return WORKFLOW_LIST[task_type[0]]["TASKS"][task_type[1]]["MODEL_ID"]
def get_task_types_for_model(model_id: int) -> list[tuple[int,int]]:
    task_types = []
    for wf in WORKFLOW_LIST:
        for task in wf["TASKS"]:
            if task["MODEL_ID"] == model_id:
                task_types.append((wf["JOB_TYPE"], task["TASK_INDEX"]))
    return task_types
