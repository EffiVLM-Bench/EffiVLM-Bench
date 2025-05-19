import json
import os

from loguru import logger

from lmms_eval.tasks._task_utils.file_utils import generate_submission_file


def docvqa_doc_to_visual(doc):
    return [doc["image"].convert("RGB")]


def docvqa_doc_to_text(doc, lmms_eval_specific_kwargs):
    question = doc["question"]
    pre_prompt = lmms_eval_specific_kwargs["pre_prompt"]
    post_prompt = lmms_eval_specific_kwargs["post_prompt"]
    return f"{pre_prompt}{question}{post_prompt}"


def docvqa_test_process_results(doc, results):
    pred = results[0]
    questionId = doc["questionId"]
    return {"anls": {"questionId": int(questionId), "answer": pred}, "submission": {"questionId": int(questionId), "answer": pred}}



def parse_model_args(model_args_str: str):
    if not model_args_str:
        return {}

    return {
        item.split("=", 1)[0].strip(): item.split("=", 1)[1].strip()
        for item in model_args_str.split(",") if "=" in item
    }

def docvqa_test_aggregate_results(results, args):
    config_args = args.model_args 
    parsed_dict = parse_model_args(config_args)

    method = parsed_dict.get("method")
    budgets = parsed_dict.get("budgets")

    file_name = "docvqa_test_for_submission"
    if method:
        file_name += f"_{method}"
    if budgets:
        file_name += f"_{budgets}"
    file_name += ".json"

    path = generate_submission_file(file_name, args)
    with open(path, "w") as f:
        json.dump(results, f)
    logger.info(f"Results saved to {path}")


# def docvqa_test_aggregate_results(results, args):
#     # save results as json
#     config_args = args.model_args
#     file_name = "docvqa_test_for_submission"+  +'.json'
#     path = generate_submission_file("docvqa_test_for_submission.json", args)
#     with open(path, "w") as f:
#         json.dump(results, f)
#     logger.info(f"Results saved to {path}")
