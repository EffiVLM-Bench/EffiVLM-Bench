import json
import os
import re
from datetime import datetime
from typing import List, Tuple

from accelerate import Accelerator, DistributedType
from loguru import logger as eval_logger
from tqdm import tqdm

from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model


@register_model("from_log")
class FromLog(lmms):
    def __init__(
        self,
        logs: str = "logs",
        model_name: str = None,
        model_args: str = None,
        have_limits: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()

        self.logs = {}
        # read group task config
        self.task_config_key = kwargs["task"]
        self.task_list = []
        self.cur_task = ""

        log_folders = logs.split(",")

        def matched_model(_model_args):
            if model_name and model_name != _model_args["model"]:
                return False

            if model_args:
                _model_args_list = model_args.split(",")

                for _model_arg in _model_args_list:
                    if _model_arg not in _model_args["model_args"]:
                        return False

            # if not have_limits and _model_args["limit"] is not None:
                # return False

            return True

        for log_folder in log_folders:
            for root, dirs, files in os.walk(log_folder):
                for file in files:
                    if file.endswith(".json"):
                        try:
                            log_file = os.path.join(root, file)

                            with open(log_file, "r") as f:
                                log_data = json.load(f)

                            # check if model is matched
                            _model_args = log_data["config"]

                            # read group task config
                            if self.task_config_key in log_data["group_subtasks"]:
                                assert type(log_data["group_subtasks"][self.task_config_key]) == list, "group_subtasks must be a list"
                                if len(log_data["group_subtasks"][self.task_config_key]) == 0:
                                    self.task_list.append(self.task_config_key)
                                else:
                                    self.task_list.extend(log_data["group_subtasks"][self.task_config_key])
                            else:
                                raise Exception(f"{self.task_config_key} Task config not found , your model_args is wrong!")
                            
                            if not matched_model(_model_args):
                                raise Exception("Model not matched")
                        except Exception as e:
                            eval_logger.error(f"Error processing log file {log_file}: {e}")

        for log_folder in log_folders:
            for root, dirs, files in os.walk(log_folder):
                for file in files:          
                    if file.endswith(".jsonl"):
                        try:
                            self.cur_task = ""
                            for task in self.task_list:
                                if task in file:
                                    self.cur_task = task
                            
                            if self.cur_task == "":
                                # skip this file
                                continue
                                
                            log_data_file = os.path.join(root, file)
                            logs_data = {}
                            with open(log_data_file, "r", encoding="utf-8") as f:
                                for line in f:
                                    log_data = json.loads(line)

                                    id = log_data["doc_id"]
                                    response = log_data["resps"][0]
                                    logs_data[id] = response

                                pattern = re.compile(r"\d{4}_\d{4}")

                                if "time" in log_data:
                                    log_time = log_data["time"]
                                elif pattern.search(os.path.abspath(log_file)):
                                    log_time = pattern.findall(os.path.abspath(log_file))[-1]
                                else:
                                    log_time = "unknown"

                                if self.cur_task not in self.logs or (self.logs[self.cur_task]["time"] == "unknown" or datetime.strptime(log_time, "%m%d_%H%M") > datetime.strptime(self.logs[self.cur_task]["time"], "%m%d_%H%M")):
                                    self.logs[self.cur_task] = {"time": log_time, "logs": logs_data}
                        except Exception as e:
                            eval_logger.error(f"Error processing log file {log_data_file}: {e}")

        accelerator = Accelerator()
        if accelerator.num_processes > 1:
            assert accelerator.distributed_type in [DistributedType.FSDP, DistributedType.MULTI_GPU, DistributedType.DEEPSPEED], "Unsupported distributed type provided. Only DDP and FSDP are supported."
            self.accelerator = accelerator
            if self.accelerator.is_local_main_process:
                eval_logger.info(f"Using {accelerator.num_processes} devices with data parallelism")
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes
        else:
            self.accelerator = accelerator
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes

        self.device = self.accelerator.device

    def generate_until(self, requests) -> List[str]:
        res = []
        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")

        for contexts, gen_kwargs, doc_to_visual, doc_id, task, split in [reg.args for reg in requests]:
            response = self.logs[task]["logs"][doc_id]
            res.append(response[0])
            pbar.update(1)
        pbar.close()
        return res

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        # TODO
        assert False, "not support"

    def generate_until_multi_round(self, requests) -> List[str]:
        return generate_until(self, requests)
