import numpy as np


def _build_object_defect_tasks():
    """Build the 3-round/8-camera defect-inspection fan-out and fan-in DAG.

    Every captured image is independently sent to both YOLOv5 detectors.  The
    final task is released only after all 24 * 2 detection results are ready.
    ROUND and CAMERA_ID are descriptive metadata; the simulator uses the task
    dependency fields to execute the graph.
    """
    image_size_kb = 1000
    detection_result_size_kb = 10
    detector_tasks = []

    for round_id in range(1, 4):
        for camera_id in range(1, 9):
            for detector, model_id in (("crack", 17), ("hole", 18)):
                task_index = len(detector_tasks)
                detector_tasks.append({
                    "MODEL_ID": model_id,
                    "TASK_INDEX": task_index,
                    "PREV_TASK_INDEX": [],
                    "NEXT_TASK_INDEX": [48],
                    "INPUT_SIZE": image_size_kb,
                    "OUTPUT_SIZE": detection_result_size_kb,
                    "SLO": 0,
                    "ROUND": round_id,
                    "CAMERA_ID": camera_id,
                    "DETECTOR": detector,
                })

    detector_tasks.append({
        "MODEL_ID": 19,
        "TASK_INDEX": 48,
        "PREV_TASK_INDEX": list(range(48)),
        "NEXT_TASK_INDEX": [],
        "INPUT_SIZE": 48 * detection_result_size_kb,
        "OUTPUT_SIZE": detection_result_size_kb,
        "SLO": 0,
        "AGGREGATES_RESULTS": 48,
        "IMAGE_COUNT": 24,
    })
    return detector_tasks


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
    {"JOB_TYPE": 12,     # IIT defect-inspection workflow: 3 rounds x 8 cameras x 2 YOLOv5 detectors
     "JOB_NAME": "object_defect_inspection",
     "DESCRIPTION": ("3 rounds x 8 cameras x 2 YOLOv5 detectors; "
                     "aggregate 48 results from 24 images"),
     "TASKS": _build_object_defect_tasks(),
    },
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
