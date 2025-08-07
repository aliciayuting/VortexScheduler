import numpy as np


"""  --------       Workflow Parameters     --------  """
# https://keras.io/api/applications/

WORKFLOW_LIST = [
    {"JOB_TYPE": 0,         # ID of the type of workflow (dependency graph)
     "JOB_NAME": "textvision",
     # the minimum amount of time necessary to execute the whole job
     "BEST_EXEC_TIME": 51.7,
     "TASKS": [
               {"MODEL_NAME": "text_encoder",
                "MODEL_ID": 0,
                "TASK_INDEX": 0,
                "PREV_TASK_INDEX": [],
                "NEXT_TASK_INDEX": [2],
                "MODEL_SIZE": 5100000, # 5677000,       # in kB
                "INPUT_SIZE": 1,
                "OUTPUT_SIZE": 2,            # in kB
                "EXECUTION_TIME": 10,        # avg time, in ms
                "MAX_BATCH_SIZE": 4,
                "MAX_EMIT_BATCH_SIZE": 4,
                "MAX_WAIT_TIME": 1,         # ms
                "BATCH_SIZES": [1, 2, 3, 4],

                # concurrency
                # "BATCH_EXEC_TIME": [9.15, 12.16, 16.57, 21.50],
                # "MIG_BATCH_EXEC_TIMES": {6: [9.15, 12.16, 16.57, 21.50],
                #                          24: [9.15, 12.16, 16.57, 21.50]},

                # no concurrency
                "BATCH_EXEC_TIME": [7.10, 8.38, 10.29, 13.16],
                "MIG_BATCH_EXEC_TIMES": {6: [7.10, 8.38, 10.29, 13.16],
                                         24: [7.10, 8.38, 10.29, 13.16]},

                "EXEC_TIME_CV": 0.06,
                "SLO": 50},
               {"MODEL_NAME": "vision_encoder",
                "MODEL_ID": 1,
                "TASK_INDEX": 1,
                "PREV_TASK_INDEX": [],
                "NEXT_TASK_INDEX": [2],
                "MODEL_SIZE": 20919000,      # in kB
                "INPUT_SIZE": 10000,
                "OUTPUT_SIZE": 1000,
                "EXECUTION_TIME": 31,        # in ms
                "MAX_BATCH_SIZE": 16,
                "MAX_EMIT_BATCH_SIZE": 4,
                "MAX_WAIT_TIME": 10,        # ms
                "BATCH_SIZES": list(range(1,17)),
                "BATCH_EXEC_TIME": [30.57, 54.38, 77.24, 101.91, 
                                    127.34, 150.61, 166.60, 189.00, 215.82, 
                                    231.59, 253.35, 265.56, 292.49, 313.73,
                                    341.07, 358.86],
                "MIG_BATCH_EXEC_TIMES": {24: [30.57, 54.38, 77.24, 101.91, 
                                              127.34, 150.61, 166.60, 189.00, 215.82, 
                                              231.59, 253.35, 265.56, 292.49, 313.73,
                                              341.07, 358.86]},
                "EXEC_TIME_CV": 0.05,
                "SLO": 155},
               {"MODEL_NAME": "flmr",
                "MODEL_ID": 2,
                "TASK_INDEX": 2,
                "PREV_TASK_INDEX": [0,1],
                "NEXT_TASK_INDEX": [3],
                "MODEL_SIZE": 854000,        # in KB
                "INPUT_SIZE": 1002,
                "OUTPUT_SIZE": 10,
                "EXECUTION_TIME": 1.7,       # in ms
                "MAX_BATCH_SIZE": 8,
                "MAX_EMIT_BATCH_SIZE": 16,
                "MAX_WAIT_TIME": 1,         # ms
                "BATCH_SIZES": list(range(1,9)),
                "BATCH_EXEC_TIME": [8.94, 15.15, 21.92, 27.91,
                                    31.86, 38.95, 45.69, 53.84],
                "MIG_BATCH_EXEC_TIMES": {
                    6: [5.35, 9.45, 13.64, 17.62, 21.70, 26.78, 31.08, 34.30],
                    24: [5.35, 9.45, 13.64, 17.62, 21.70, 26.78, 31.08, 34.30]
                }, # { 6: [8.94, 15.15, 21.92, 27.91, 31.86, 38.95, 45.69, 53.84]},
                "EXEC_TIME_CV": 0.95,
                "SLO": 8.5},
               {"MODEL_NAME": "search",
                "MODEL_ID": 3,
                "TASK_INDEX": 3,
                "PREV_TASK_INDEX": [2],
                "NEXT_TASK_INDEX": [],
                "MODEL_SIZE": 777000,        # in KB
                "INPUT_SIZE": 10,
                "OUTPUT_SIZE": 10,
                "EXECUTION_TIME": 18,        # in ms
                "MAX_BATCH_SIZE": 16,
                "MAX_EMIT_BATCH_SIZE": 0, # NOTE: should not be used
                "MAX_WAIT_TIME": 1,        # ms
                "BATCH_SIZES": list(range(1,17)),
                "BATCH_EXEC_TIME": [24.56, 48.38, 66.22, 86.08,
                                    108.49, 136.14, 153.81, 175.46,
                                    200.74, 228.27, 245.03, 259.32,
                                    283.92, 307.50, 329.19, 347.88],
                "MIG_BATCH_EXEC_TIMES": {6: [24.56, 48.38, 66.22, 86.08,
                                             108.49, 136.14, 153.81, 175.46,
                                             200.74, 228.27, 245.03, 259.32,
                                             283.92, 307.50, 329.19, 347.88],
                                        24: [24.56, 48.38, 66.22, 86.08,
                                             108.49, 136.14, 153.81, 175.46,
                                             200.74, 228.27, 245.03, 259.32,
                                             283.92, 307.50, 329.19, 347.88]},
                "EXEC_TIME_CV": 0.06,
                "SLO": 90}
               ]
     },
     {"JOB_TYPE": 1,
     "JOB_NAME": "tts",
     # the minimum amount of time necessary to execute the whole job
     "BEST_EXEC_TIME": 101.4,
     "TASKS": [{"MODEL_NAME": "audio_det",
                "MODEL_ID": 4,
                "TASK_INDEX": 0,
                "PREV_TASK_INDEX": [],
                "NEXT_TASK_INDEX": [1],
                "MODEL_SIZE": 6093000,      # in kB
                "INPUT_SIZE": 1000,
                "OUTPUT_SIZE": 2,            # in kB
                "EXECUTION_TIME": 65,        # avg time, in ms
                "MAX_BATCH_SIZE": 8,
                "MAX_EMIT_BATCH_SIZE": 10,
                "MAX_WAIT_TIME": 1,          # ms
                "BATCH_SIZES": [1, 2, 4, 8],
                "BATCH_EXEC_TIME": [65.0, 68.0, 69.4, 72.1],
                "MIG_BATCH_EXEC_TIMES": {
                    24: [65.0, 68.0, 69.4, 72.1],
                    12: [65.0, 68.0, 69.4, 72.1] # TODO: update with real nums
                },
                "EXEC_TIME_CV": 0.182,
                "SLO": 325},
               {"MODEL_NAME": "encode_search-ivf",
                "MODEL_ID": 5,
                "TASK_INDEX": 1,
                "PREV_TASK_INDEX": [0],
                "NEXT_TASK_INDEX": [2,3],
                "MODEL_SIZE": 1210000,       # in kB
                "INPUT_SIZE": 2,
                "OUTPUT_SIZE": 2,
                "EXECUTION_TIME": 16.7,      # in ms
                "MAX_BATCH_SIZE": 8,
                "MAX_EMIT_BATCH_SIZE": 10,
                "MAX_WAIT_TIME": 1,          # ms
                "BATCH_SIZES": [1, 2, 4, 8],
                "BATCH_EXEC_TIME": [16.7, 17.2, 17.5, 17.5],
                "MIG_BATCH_EXEC_TIMES": {
                    24: [16.7, 17.2, 17.5, 17.5], # TODO [0.397, 0.405, 0.424, 0.456],
                    12: [16.7, 17.2, 17.5, 17.5],
                    6: [16.5, 16.9, 16.9, 17.3]
                },
                "EXEC_TIME_CV": 0.414,
                "SLO": 83.5},
               {"MODEL_NAME": "text_check",
                "MODEL_ID": 6,
                "TASK_INDEX": 2,
                "PREV_TASK_INDEX": [1],
                "NEXT_TASK_INDEX": [3],
                "MODEL_SIZE": 2101000,       # in kB
                "INPUT_SIZE": 2,
                "OUTPUT_SIZE": 2,
                "EXECUTION_TIME": 3.36,        # in ms
                "MAX_BATCH_SIZE": 2,
                "MAX_EMIT_BATCH_SIZE": 10,
                "MAX_WAIT_TIME": 1,          # ms
                "BATCH_SIZES": [1, 2],
                "BATCH_EXEC_TIME": [17.3, 25.3],
                "MIG_BATCH_EXEC_TIMES": {
                    24: [17.3, 25.3], 12: [25.3, 44.2]
                },
                "EXEC_TIME_CV": 0.65,
                "SLO": 16.8},
               {"MODEL_NAME": "aggregate-tts",
                "MODEL_ID": 7,
                "TASK_INDEX": 3,
                "PREV_TASK_INDEX": [1,2],
                "NEXT_TASK_INDEX": [],
                "MODEL_SIZE": 6135000,             # in kB
                "INPUT_SIZE": 4,
                "OUTPUT_SIZE": 4,
                "EXECUTION_TIME": 87.3,         # in ms
                "MAX_BATCH_SIZE": 1,
                "MAX_EMIT_BATCH_SIZE": 0,
                "MAX_WAIT_TIME": 1,          # ms
                "BATCH_SIZES": [1],
                "BATCH_EXEC_TIME": [87.3, 164],
                "MIG_BATCH_EXEC_TIMES": {24: [87.3, 164], 12: [149.3, 337.6]},
                "EXEC_TIME_CV": 0.389,
                "SLO": 436.5}
               ]
     }
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