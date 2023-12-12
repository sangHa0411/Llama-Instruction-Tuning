import re
import logging
import pandas as pd
import numpy as np
import multiprocessing
from functools import partial
from typing import Dict, List, Any
from pytz import timezone
from datetime import datetime
from datasets import Dataset, concatenate_datasets
from transformers import LlamaTokenizer
from tqdm import tqdm

logging.basicConfig(format = "[%(asctime)s][%(levelname)s][Message] - %(message)s", level = logging.INFO)
logging.Formatter.converter = lambda *args: datetime.now(tz=timezone("Asia/Seoul")).timetuple()

class EvaluationDatasetPreprocessor :

    def __init__(self,         
        tokenizer: LlamaTokenizer,
        sequence_max_length: int,
        label_pad_token_id: int = -100
    ) :      
        self.tokenizer = tokenizer
        self.sequence_max_length = sequence_max_length
        self.label_pad_token_id = label_pad_token_id
        # self.num_cores = multiprocessing.cpu_count() // 4
        self.num_cores = 1

        self.preprocessors = {
            "ai2_arc" : ArcPreprocessor(tokenizer, sequence_max_length),
            "Rowan/hellaswag" : HellaswagPreprocessor(tokenizer, sequence_max_length),
        }

    def __call__(self, dataset_names: str, num_shots: str, datasets: Dict[str, Dataset]) -> Dict[str, Dataset] :
        dataset_names = dataset_names[1:-1].split(",")
        num_shots = num_shots[1:-1].split(",")
        num_shots = [int(shot) for shot in num_shots]
        
        preprocessed_datasets = {}

        for i, dataset_name in enumerate(dataset_names) :
            dataset = datasets[dataset_name]
            num_shot = num_shots[i]

            if dataset_name in self.preprocessors :
                logging.info(f"Preprocessing and Encoding Dataset | {dataset_name}")
                preprocessor = self.preprocessors[dataset_name]

                preprocess_fn = partial(preprocessor.preprocess, num_shot=num_shot)
                preprocessed = dataset.map(preprocess_fn, batched=True, num_proc=self.num_cores, remove_columns=dataset.column_names)
                preprocessed_datasets[dataset_name] = preprocessed

        return preprocessed_datasets


class ArcPreprocessor :
    def __init__(self, 
        tokenizer: LlamaTokenizer,
        sequence_max_length: int,
    ) :       
        self.tokenizer = tokenizer
        self.sequence_max_length = sequence_max_length

    def make_few_shot_example(self, datasets: List[Dict[str, Any]], sampled_ids: List[int]) :
        questions = datasets["question"]
        choices = datasets["choices"]
        answer_keys = datasets["answerKey"]

        examples = []
        for i in sampled_ids :
            question = questions[i]
            choice = choices[i]
            answer_key = answer_keys[i]

            if ord(answer_key) >= ord("A") :
                target_id = ord(answer_key) - ord("A") 
            else :
                target_id = int(answer_key) - 1

            candidate_answer = " ".join([f"({l}): {t}" for t, l in zip(choice["text"], choice["label"])])
            target_text = choice["text"][target_id]
            input_text = f"### QUESTION:\n{question}\n\n### CANDIDATE ANSWERS:\n{candidate_answer}\n\n### ANSWER:\n{target_text}"
            examples.append(input_text)

        few_shot_example = "\n\n\n\n".join(examples)
        return few_shot_example

    def preprocess(self, datasets: List[Dict[str, Any]], num_shot: int):
        questions = datasets["question"]
        choices = datasets["choices"]
        answer_keys = datasets["answerKey"]

        input_ids, attention_masks, labels = [], [], []

        size = len(questions)
        for i in range(size) :

            question = questions[i]
            choice = choices[i]
            answer_key = answer_keys[i]

            candidate_answer = " ".join([f"({l}): {t}" for t, l in zip(choice["text"], choice["label"])])
            input_text = f"### QUESTION:\n{question}\n\n### CANDIDATE ANSWERS:\n{candidate_answer}\n\n### ANSWER:\n"

            if ord(answer_key) >= ord("A") :
                target_id = ord(answer_key) - ord("A") 
            else :
                target_id = int(answer_key) - 1
            target_text = choice["text"][target_id]

            if num_shot > 0 :
                sampled_ids = np.random.choice(size, num_shot+1, replace=False)
                sampled_ids = list(set(sampled_ids) - set([i]))[:num_shot]
                few_shot_example = self.make_few_shot_example(datasets, sampled_ids)
                input_text = few_shot_example + "\n\n\n\n" + input_text

            input_id = self.tokenizer(
                input_text, 
                max_length=self.sequence_max_length,
                truncation='do_not_truncate',
                add_special_tokens=False
            ).input_ids
            attention_mask = [1]*len(input_id)

            input_ids.append(input_id)
            attention_masks.append(attention_mask)
            labels.append(target_text)

        datasets["input_ids"] = input_ids
        datasets["attention_mask"] = attention_masks
        datasets["labels"] = labels

        return datasets


class HellaswagPreprocessor :
    def __init__(self, 
        tokenizer: LlamaTokenizer,
        sequence_max_length: int,
    ) :       
        self.tokenizer = tokenizer
        self.sequence_max_length = sequence_max_length

    def make_few_shot_example(self, datasets: List[Dict[str, Any]], sampled_ids: List[int]) :
        ctxs = datasets["ctx"]
        endings = datasets["endings"]
        answers = datasets["label"]

        examples = []
        for i in sampled_ids :
            context = ctxs[i]
            ending = endings[i]
            answer = int(answers[i])

            candidate_ending = " ".join([f"({i}): {e}" for i, e in enumerate(ending)])
            target_text = ending[answer]
            input_text = f"### CONTEXT:\n{context}\n\n### CANDIDATE ENDINGS:\n{candidate_ending}\n\n### ANSWER:\n{target_text}"
            examples.append(input_text)

        few_shot_example = "\n\n\n\n".join(examples)
        return few_shot_example

    def preprocess(self, datasets: List[Dict[str, Any]], num_shot: int) :
        ctxs = datasets["ctx"]
        endings = datasets["endings"]
        answers = datasets["label"]

        input_ids, attention_masks, labels = [], [], []

        size = len(ctxs)
        for i in range(size) :
            context = ctxs[i]
            ending = endings[i]
            answer = int(answers[i])
            
            candidate_ending = " ".join([f"({i}): {e}" for i, e in enumerate(ending)])

            input_text = f"### CONTEXT:\n{context}\n\n### CANDIDATE ENDINGS:\n{candidate_ending}\n\n### ANSWER:\n"
            target_text = ending[answer]

            if num_shot > 0 :
                sampled_ids = np.random.choice(size, num_shot+1, replace=False)
                sampled_ids = list(set(sampled_ids) - set([i]))[:num_shot]
                few_shot_example = self.make_few_shot_example(datasets, sampled_ids)
                input_text = few_shot_example + "\n\n\n\n" + input_text

            input_id = self.tokenizer(
                input_text, 
                max_length=self.sequence_max_length,
                truncation='do_not_truncate',
                add_special_tokens=False
            ).input_ids
            attention_mask = [1]*len(input_id)

            input_ids.append(input_id)
            attention_masks.append(attention_mask)
            labels.append(target_text)

        datasets["input_ids"] = input_ids
        datasets["attention_mask"] = attention_masks
        datasets["labels"] = labels

        return datasets

