import fnmatch
import json
import warnings
import os
import datasets
import torch
import transformers
import argparse
from lm_eval.evaluator import Evaluator
from lm_eval.tasks import ALL_TASKS


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        default="Llama3.1-8B-Instruct",
        help="Model to evaluate, provide a repo name in Hugging Face hub or a local path",
    )
    parser.add_argument(
        "--task_name",
        default=None,
        help="Task to evaluate on, can be a single task",
    )
    parser.add_argument(
        "--load_generations_path",
        type=str,
        default=None,
        help="Path of file with previously generated solutions, if provided generation is skipped and only evaluation is done",
    )
    parser.add_argument(
        "--metric_output_path",
        type=str,
        default="evaluation_results.json",
        help="Path to save the results",
    )
    parser.add_argument(
        "--bertscore",
        action="store_true",
        help="Whether use additional bertscore for xsum/cnndailymail.",
    )
    parser.add_argument(
        "--bert_path",
        type=str,
        default='bert-base-uncased',
        help="The model bert in Hugging Face hub or a local path",
    )
    parser.add_argument(
        "--comet",
        action="store_true",
        help="Whether use additional comet for wmt.",
    )
    parser.add_argument(
        "--postprocessed_output_path",
        type=str,
        default=None,
        help="Path to save the postprocessed generations",
    )
    parser.add_argument(
        "--sft",
        action="store_true",
        help="Whether sft or not.",
    )
    parser.add_argument(
        "--only_calculate_correct_case",
        action="store_true",
        help="Whether diversity or not.",
    )
    parser.add_argument(
        "--diversity",
        action="store_true",
        help="Whether diversity or not.",
    )
    parser.add_argument(
        "--mauve",
        action="store_true",
        help="Whether mauve or not.",
    )
    parser.add_argument(
        "--bleu",
        action="store_true",
        help="Whether bleu or not.",
    )
    parser.add_argument(
        "--coherence",
        action="store_true",
        help="Whether coherence or not.",
    )
    parser.add_argument(
        "--gpt2_path",
        type=str,
        default='gpt2',
        help="The model gpt2 in Hugging Face hub or a local path",
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    print(f"Selected Tasks: {args.task_name}")

    results = {}
    if args.load_generations_path:
        # here we don't generate code but only evaluate previously computed generations
        print("evaluation only mode")
        evaluator = Evaluator(args)
        results["results"] = evaluator.evaluate(args.task_name)
    
    # Save all args to config
    for k,v in vars(args).items():
        results[k] = v
    # Save jsonl
    dumped = json.dumps(results)
    print(dumped)
    # Extract the directory path and create it
    output_dir = os.path.dirname(args.metric_output_path)
    if output_dir:  # If there is a directory path
        os.makedirs(output_dir, exist_ok=True)

    with open(args.metric_output_path, "a+") as f:
        f.write(dumped+"\n")

