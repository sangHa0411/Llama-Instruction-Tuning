import re
import logging
import pandas as pd
import multiprocessing
from typing import Dict
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
            # "Rowan/hellaswag" : CoTCollectionPreprocessor(tokenizer, sequence_max_length, label_pad_token_id),
            # "gsm8k" : SlimOrcaPreprocessor(tokenizer, sequence_max_length, label_pad_token_id)
        }

    def __call__(self, dataset_names: str, num_shots: str, datasets: Dict[str, Dataset]) -> Dataset :
        dataset_names = dataset_names[1:-1].split(",")
        num_shots = num_shots[1:-1].split(",")
        num_shots = [int(shot) for shot in num_shots]
        
        preprocessed_datasets = []

        for i, dataset_name in enumerate(dataset_names) :
            dataset = datasets[dataset_name]
            num_shot = num_shots[i]

            if dataset_name in self.preprocessors :
                logging.info(f"Preprocessing and Encoding Dataset | {dataset_name}")
                preprocessor = self.preprocessors[dataset_name]

                preprocessed = dataset.map(preprocessor, batched=True, num_proc=self.num_cores, remove_columns=dataset.column_names)
                preprocessed_datasets.append(preprocessed)  

        preprocessed_datasets = concatenate_datasets(preprocessed_datasets)
        return preprocessed_datasets


class ArcPreprocessor :
    def __init__(self, 
        tokenizer: LlamaTokenizer,
        sequence_max_length: int,
    ) :       
        self.tokenizer = tokenizer
        self.sequence_max_length = sequence_max_length

    def __call__(self, datasets):

        questions = datasets["question"]
        choices = datasets["choices"]
        answer_keys = datasets["answerKey"]

        input_ids, attention_masks, labels = [], [], []

        size = len(questions)
        for i in range(size) :

            question = questions[i]
            choice = choices[i]
            answer_key = answer_keys[i]
            
            candidate_answer = " ".join([f"({l}) : {t}" for t, l in zip(choice["text"], choice["label"])])

            input_text = f"### QUESTION:\n{question}\n\n### CANDIDATE ANSWERS:\n{input_text}\n\n### ANSWER:\n"
            target_id = ord(answer_key) - ord("A")
            target_text = choice["text"][target_id]

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

