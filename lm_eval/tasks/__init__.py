import inspect
from pprint import pprint
from . import (commonsense_qa, strategy_qa,wikitext103)

TASK_REGISTRY = {
    "commonsense_qa": commonsense_qa.CommonsenseQA,
    "strategy_qa": strategy_qa.StrategyQA,
    'wikitext103': wikitext103.WikiText103,
}

ALL_TASKS = sorted(list(TASK_REGISTRY))


def get_task(task_name, args=None):
    try:
        kwargs = {}
        if "postprocessed_output_path" in inspect.signature(TASK_REGISTRY[task_name]).parameters:
            kwargs["postprocessed_output_path"] = args.postprocessed_output_path
        return TASK_REGISTRY[task_name](**kwargs)
    except KeyError:
        print("Available tasks:")
        pprint(TASK_REGISTRY)
        raise KeyError(f"Missing task {task_name}")
