from core.workflow import *
from workers.worker import Worker

import json


class HerdAssignment:

    def __init__(self, worker_groups: list[list[Worker]], task_type_to_group: dict[tuple[int,int],int]):
        self.worker_groups = worker_groups
        self.task_type_to_group = task_type_to_group
        self.group_task_types = [[tt for tt, gid in self.task_type_to_group.items() if gid==group] 
                                 for group in range(len(self.worker_groups))]
        self.task_type_to_model = { tt: get_model_id_for_task_type(tt) for tt in self.task_type_to_group.keys() }
        self.group_models = [set(self.task_type_to_model[tt] for tt in gtts) for gtts in self.group_task_types]

    def __str__(self):
        str_tt_to_group = { str(tt): grp for tt, grp in self.task_type_to_group.items() }
        to_dict = {
            "groups_by_worker_ids": [[w.worker_id for w in group] for group in self.worker_groups],
            "task_type_to_group_index": str_tt_to_group,
            "group_model_ids": [ list(models) for models in self.group_models]
        }
        return json.dumps(to_dict)