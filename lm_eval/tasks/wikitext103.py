import re

from evaluate import load

from lm_eval.base import Task

import numpy as np
class WikiText103(Task):
    """A task represents an entire benchmark including its dataset, problems,
    answers, generation settings and evaluation methods.
    """

    DATASET_PATH = "data/wikitext103"

    def __init__(self, postprocessed_output_path, sft, bertscore):
        self.postprocessed_output_path = postprocessed_output_path
        self.sft = sft
        self.bertscore = bertscore
        super().__init__()

    def get_dataset(self):
        """Returns dataset for the task or an iterable of any object, that get_prompt can handle"""
        # dataset = self.dataset["test"]
        dataset = self.dataset["validation"]
        assert (
            # len(dataset) == 591
            len(dataset) == 200
        )
        return dataset


    def get_reference(self, doc):
        """Builds the reference solution for the doc (sample from the test dataset)."""
        return doc

    def postprocess_generation(self, generation):
        """Defines the postprocessing for a LM generation.
        :param generation: str
            code generation from LM
        :param idx: int
            index of doc in the dataset to which the generation belongs
        """
        return generation
    
    def process_results(self, generations, references):
        """Takes the list of LM generations and evaluates them against ground truth references,
        returning the metric for the generations.
        :param generations: list(list(str))
            list of lists containing generations
        :param references: list(str)
            list of str containing refrences
        """
        for i in range(len(generations)):
            generations[i]=references[i]['instructions']+' '+generations[i][0]
            # references = references[i]['gold_ref']
        return None, generations, None
