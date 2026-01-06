import inspect
import json
import os
import warnings
from transformers import AutoTokenizer
from lm_eval import tasks
from nltk.util import ngrams
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine
from nltk.translate.bleu_score import corpus_bleu
import torch
import numpy as np
import mauve
from collections import Counter
from typing import Dict, List
_WARNING = """
################################################################################
                                  !!!WARNING!!!
################################################################################
The "code_eval"/"apps_metric" you are about to use, execute untrusted 
model-generated code in Python.
Although it is highly unlikely that model-generated code will do something
overtly malicious in response to this test suite, model-generated code may act
destructively due to a lack of model capability or alignment.
Users are strongly encouraged to sandbox this evaluation suite so that it
does not perform destructive actions on their host or network. For more
information on how OpenAI sandboxes its code, see the paper "Evaluating Large
Language Models Trained on Code" (https://arxiv.org/abs/2107.03374).
Once you have read this disclaimer and taken appropriate precautions, set the argument 
"allow_code_execution" to True.
################################################################################\
"""


class Evaluator:
    def __init__(self, args):
        self.args = args

        # setup arguments
        self.metric_output_path = args.metric_output_path

        # code evaluation permission
        self.allow_code_execution = args.allow_code_execution

    def get_generate_text(self, task_name):
        task = tasks.get_task(task_name, self.args)
        dataset = task.get_dataset()
        # if args.limit is None, use all samples
        n_tasks = len(dataset)
        references = [task.get_reference(dataset[i]) for i in range(n_tasks)]

        if self.args.check_references:
            if "get_solution" in inspect.signature(task.get_reference).parameters:
                solutions = [[task.get_reference(dataset[i], get_solution=True)] for i in range(n_tasks)]
            else:
                solutions = [[ref] for ref in references]
            return solutions, references
        if self.args.load_generations_path:
            generations = []
            with open(self.args.load_generations_path) as fp:
                for line in fp:
                    json_obj = json.loads(line)
                    generations.append(json_obj)
                print(
                    f"generations loaded, {n_tasks} selected from {len(generations)}."
                )
        generations = generations[:n_tasks]
        generations = [[_['completion']] for _ in generations]

        return generations, references

    def evaluate(self, task_name):
        task = tasks.get_task(task_name, self.args)
        if task.requires_execution and not self.allow_code_execution:
            raise ValueError(_WARNING)

        generations, references = self.get_generate_text(task_name)

        # make sure tokenizer plays nice with multiprocessing
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        if self.allow_code_execution and task.requires_execution:
            os.environ["HF_ALLOW_CODE_EVAL"] = "1"
        print("Evaluating generations...")
        # First, evaluate the accuracy of the results
        results = {}
        results['accuracy'],fixed_generations,is_correct = task.process_results(generations, references)
        # Then evaluate the diversity of the results
        if self.args.diversity:
            if self.args.only_calculate_correct_case:
                fixed_generations = [gen for gen, correct in zip(fixed_generations, is_correct) if correct == 1]
            if references:
                diversity_score = self.get_diversity_score(fixed_generations,references)
            else:
                diversity_score = self.get_diversity_score(fixed_generations)
            results['diversity_score'] = diversity_score
        return results
    
    def get_diversity_score(self, generations , references=None):
        diversity_score={
            'ngram_reps':None,
            'diversity':None,
            'mauve':None,
            'rouge':None,
            'bleu':None,
            'coherence':None,
        }
        def calculate_ngram_diversity(tokens,n):
            if isinstance(tokens[0], List):
                # Two-dimensional list: Process each text and then calculate the average value
                diversities = []
                for token_list in tokens:
                    if len(token_list) >= n:  # Make sure there are enough tokens to generate n-grams
                        n_grams = list(ngrams(token_list, n))
                        n_gram_counts = Counter(n_grams)
                        total_n_grams = len(n_grams)
                        unique_n_grams = len(n_gram_counts)

                        if total_n_grams > 0:
                            diversity = (total_n_grams - unique_n_grams) / total_n_grams
                            diversities.append(diversity)

                # Return the average diversity
                return sum(diversities) / len(diversities) if diversities else 0.0
            else:
                # One-dimensional list: Original logic
                n_grams = list(ngrams(tokens, n))
                n_gram_counts = Counter(n_grams)
                total_n_grams = len(n_grams)
                unique_n_grams = len(n_gram_counts)

                if total_n_grams == 0:
                    return 0.0

                diversity = (total_n_grams - unique_n_grams) / total_n_grams
                return diversity
        
        tokenizer = AutoTokenizer.from_pretrained(
            self.args.model, trust_remote_code=True
        )
        tokens = [tokenizer.tokenize(gen) for gen in generations]
        # Achieve a repetition of 2 to 4 grams
        ngram_reps = [calculate_ngram_diversity(tokens, n) for n in range(2, 5)]
        diversity = np.prod([(1 - rep) for rep in ngram_reps])*100
        diversity_score['ngram_reps'] = ngram_reps
        diversity_score['diversity'] = diversity
        if references:
            if type(references[0])==dict:
                references = [ref['gold_ref'] for ref in references]
        if self.args.mauve: 
            out = mauve.compute_mauve(
                p_text=references,
                q_text=generations,
                device_id=0,  
                max_text_length=256, 
                verbose=True,
                batch_size=1024,
                featurize_model_name=self.args.gpt2_path
            )
            diversity_score['mauve'] = out.mauve
        if self.args.bleu:
            # BLEU
            references_blue = [[ref.split()] for ref in references]
            candidates_blue = [gen.split() for gen in generations]
            bleu_score = corpus_bleu(references_blue, candidates_blue)
            diversity_score['bleu'] = bleu_score

        if self.args.coherence:
            model = SentenceTransformer(self.args.bert_path,device='cuda:0')
            coherence_scores = []

            # Convert the generated text and reference text into sentence embeddings
            generation_embeddings = model.encode(generations)
            reference_embeddings = model.encode(references)

            for gen_emb, ref_emb in zip(generation_embeddings, reference_embeddings):
                # Calculate the cosine similarity between each pair of generated text and the reference text
                sim = 1 - cosine(gen_emb, ref_emb)
                coherence_scores.append(sim)
            coherence_score = np.mean(coherence_scores)
            diversity_score['coherence'] = coherence_score
        return diversity_score

