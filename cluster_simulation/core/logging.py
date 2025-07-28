class TaskLifeCycleTimestamp(object):
    def __init__(self, job_id, task_id, is_initial_task=False):
        self.job_id = job_id
        self.task_id = task_id
        self.is_initial_task = is_initial_task
        self.job_creation_timestamp = 0
        self.task_arrival_at_worker_buffer_timestamp = 0
        self.task_placed_on_worker_queue_timestamp = 0
        # below: time when task(with all dependent tasks' results arrived, but may need to wait for models) gets poped off the queue
        self.task_front_queue_timestamp = 0
        self.task_execution_start_timestamp = 0
        self.task_execution_end_timestamp = 0
        self.task_arrival_at_scheduler_timestamp = 0
        self.task_placed_on_global_queue_timestamp = 0 # QLM

    def set_task_arrival_at_worker_buffer_timestamp(self, timestamp):
        """
        log the timestamp when the FIRST dependent input of this task 
                     arrives at the machine's wait buffer
        """
        if self.task_arrival_at_worker_buffer_timestamp == 0:
            self.task_arrival_at_worker_buffer_timestamp = timestamp

    def set_task_placed_on_worker_queue_timestamp(self, timestamp):
        """
        log the timestamp when the task is placed on the worker queue. 
        i.e. tasks are only placed on the worker queue after all its dependent tasks' results have arrived to the worker
        """
        self.task_placed_on_worker_queue_timestamp = timestamp
        # for initial task since it doesn't have any dependent tasks,
        # once it arrives at the worker, it skips the step of buffer, and is directly placed on the worker queue
        if self.is_initial_task:
            self.task_arrival_at_worker_buffer_timestamp = timestamp

    def get_model_fetch_time(self):
        """
        model fetch time for this task
        """
        assert self.task_front_queue_timestamp != 0 and self.task_execution_start_timestamp != 0
        return self.task_execution_start_timestamp - self.task_front_queue_timestamp

    def get_task_wait_time(self):
        """
        task wait time on the worker's queue
        """
        assert self.task_arrival_at_worker_buffer_timestamp != 0 and self.task_front_queue_timestamp != 0
        return self.task_front_queue_timestamp - self.task_arrival_at_worker_buffer_timestamp

    def toString(self):
        """
        string of time stamps for this task
        """
        result = f"job_id: {self.job_id}\
                 \ntask_id: {self.task_id}\
                 \njob_creation timestamp: {self.job_creation_timestamp}\
                 \ntask_arrival_at_worker_buffer_timestamp: {self.task_arrival_at_worker_buffer_timestamp}\
                 \ntask_placed_on_worker_queue_timestamp: {self.task_placed_on_worker_queue_timestamp}\
                 \ntask_front_queue_timestamp: {self.task_front_queue_timestamp}\
                 \ntask_execution_start_timestamp: {self.task_execution_start_timestamp}\
                 \ntask_execution_end_timestamp: {self.task_execution_end_timestamp}"
        return result
