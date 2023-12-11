import re
import logging
import pandas as pd
import multiprocessing
from typing import Dict, List
from pytz import timezone
from datetime import datetime
from datasets import Dataset, concatenate_datasets
from transformers import LlamaTokenizer
from tqdm import tqdm

logging.basicConfig(format = "[%(asctime)s][%(levelname)s][Message] - %(message)s", level = logging.INFO)
logging.Formatter.converter = lambda *args: datetime.now(tz=timezone("Asia/Seoul")).timetuple()

class InstructionDatasetPreprocessor :

    def __init__(self,         
        tokenizer: LlamaTokenizer,
        sequence_max_length: int,
        label_pad_token_id: int = -100
    ) :      
        self.tokenizer = tokenizer
        self.sequence_max_length = sequence_max_length
        self.label_pad_token_id = label_pad_token_id
        self.num_cores = multiprocessing.cpu_count() // 4

        self.preprocessors = {
            "tatsu-lab/alpaca" : AlpacaPreprocessor(tokenizer, sequence_max_length, label_pad_token_id),
            "kaist-ai/CoT-Collection" : CoTCollectionPreprocessor(tokenizer, sequence_max_length, label_pad_token_id),
            "Open-Orca/SlimOrca" : SlimOrcaPreprocessor(tokenizer, sequence_max_length, label_pad_token_id)
        }

    def __call__(self, dataset_names: str, datasets: Dict[str, Dataset]) -> Dataset :
        dataset_names = dataset_names[1:-1].split(",")

        preprocessed_datasets = []
        for dataset_name in dataset_names :
            dataset = datasets[dataset_name]

            if dataset_name in self.preprocessors :
                logging.info(f"Preprocessing and Encoding Dataset | {dataset_name}")
                preprocessor = self.preprocessors[dataset_name]

                if dataset_name == "Open-Orca/SlimOrca" :
                    dataset = preprocessor.split(dataset)

                preprocessed = dataset.map(preprocessor, batched=True, num_proc=self.num_cores, remove_columns=dataset.column_names)
                preprocessed_datasets.append(preprocessed)  

        preprocessed_datasets = concatenate_datasets(preprocessed_datasets)
        return preprocessed_datasets


class AlpacaPreprocessor :
    def __init__(self, 
        tokenizer: LlamaTokenizer,
        sequence_max_length: int,
        label_pad_token_id: int = -100
    ) :       
        self.tokenizer = tokenizer
        self.sequence_max_length = sequence_max_length
        self.label_pad_token_id = label_pad_token_id
        self.template = {
            "prompt_input" : "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.",
            "prompte_no_input" : "Below is an instruction that describes a task. Write a response that appropriately completes the request."
        }

    def __call__(self, datasets):

        instructions = datasets["instruction"]
        input_texts = datasets["input"]
        output_texts = datasets["output"]

        input_ids, attention_masks, labels = [], [], []

        size = len(instructions)
        for i in range(size) :

            instruction = instructions[i]
            input_text = input_texts[i]
            output_text = output_texts[i]

            if input_text != "" :
                all_text = f"{self.template['prompt_input']}\n\n### INSTRUCTION:\n{instruction}\n\n### INPUT:\n{input_text}\n\n### RESPONSE:\n{output_text}"
                source_text = f"{self.template['prompt_input']}\n\n### INSTRUCTION:\n{instruction}\n\n### INPUT:\n{input_text}\n\n### RESPONSE:\n"
            else :
                all_text = f"{self.template['prompte_no_input']}\n\n### INSTRUCTION:\n{instruction}\n\n### RESPONSE:\n{output_text}"
                source_text = f"{self.template['prompte_no_input']}\n\n### INSTRUCTION:\n{instruction}\n\n### RESPONSE:\n"

            all_input_id = self.tokenizer(
                all_text, 
                max_length=self.sequence_max_length,
                truncation='do_not_truncate',
                add_special_tokens=False
            ).input_ids
            all_input_id = all_input_id + [self.tokenizer.eos_token_id]
            attention_mask = [1]*len(all_input_id)

            source_input_id = self.tokenizer(
                source_text, 
                max_length=self.sequence_max_length,
                truncation='do_not_truncate',
                add_special_tokens=False
            ).input_ids
            source_input_id_length = len(source_input_id)
            label = [self.label_pad_token_id] * source_input_id_length + all_input_id[source_input_id_length:]

            input_ids.append(all_input_id)
            attention_masks.append(attention_mask)
            labels.append(label)

        datasets["input_ids"] = input_ids
        datasets["attention_mask"] = attention_masks
        datasets["labels"] = labels

        return datasets


class CoTCollectionPreprocessor :
    def __init__(self, 
        tokenizer: LlamaTokenizer,
        sequence_max_length: int,
        label_pad_token_id: int = -100
    ) :       
        self.tokenizer = tokenizer
        self.sequence_max_length = sequence_max_length
        self.label_pad_token_id = label_pad_token_id
        self.template = {
            "prompt" : "Below is context which has instruction and context. Write a response that appropriately with step by step reasonable thoughts. "
        }

    def __call__(self, datasets):
        sources = datasets["source"]
        rationales = datasets["rationale"]
        targets = datasets["target"]

        input_ids, attention_masks, labels = [], [], []

        size = len(sources)
        for i in range(size) :

            source = sources[i]
            rationale = rationales[i]
            target = targets[i]

            all_text = f"{self.template['prompt']}\n\n### SOURCE:\n{source}\n\n### RATIONALE:\n{rationale}\n\n### TARGET:\n{target}"
            source_text = f"{self.template['prompt']}\n\n### SOURCE:\n{source}\n\n### RATIONALE:\n"
           
            all_input_id = self.tokenizer(
                all_text, 
                max_length=self.sequence_max_length,
                truncation='do_not_truncate',
                add_special_tokens=False
            ).input_ids
            all_input_id = all_input_id + [self.tokenizer.eos_token_id]
            attention_mask = [1]*len(all_input_id)

            source_input_id = self.tokenizer(
                source_text, 
                max_length=self.sequence_max_length,
                truncation='do_not_truncate',
                add_special_tokens=False
            ).input_ids
            source_input_id_length = len(source_input_id)
            label = [self.label_pad_token_id] * source_input_id_length + all_input_id[source_input_id_length:]

            input_ids.append(all_input_id)
            attention_masks.append(attention_mask)
            labels.append(label)

        datasets["input_ids"] = input_ids
        datasets["attention_mask"] = attention_masks
        datasets["labels"] = labels

        return datasets



class SlimOrcaPreprocessor :
    def __init__(self, 
        tokenizer: LlamaTokenizer,
        sequence_max_length: int,
        label_pad_token_id: int = -100
    ) :       
        self.tokenizer = tokenizer
        self.sequence_max_length = sequence_max_length
        self.label_pad_token_id = label_pad_token_id
        self.template = {
            "prompt" : "Below is Dialouge between human and machine. Understand dialogue history, follow human's instruction and reponse that appropriately "
        }

    def split(self, dataset: Dataset) :
        contexts, responses = [], []

        for data in tqdm(dataset) :
            conversations = data["conversations"]
            conversations = eval(re.sub("} *\n *{", "},{", conversations))

            assert conversations[-1]["from"] == "gpt"
            
            gpt_chat = conversations[-1]
            gpt_response = gpt_chat["value"]
            
            history = []
            for i in range(len(conversations)-1) :
                subject = conversations[i]["from"]
                value = conversations[i]["value"]
                
                chat = f"### FROM:\n{subject}\n\n### VALUE:\n{value}\n\n"
                history.append(chat)
            context = "".join(history)

            contexts.append(context)
            responses.append(gpt_response)

        splited_dataset = Dataset.from_pandas(
            pd.DataFrame({"context" : contexts, "response" : responses})
        )

        return splited_dataset

    def __call__(self, datasets):
        contexts = datasets["context"]
        responses = datasets["response"]

        input_ids, attention_masks, labels = [], [], []

        size = len(contexts)
        for i in range(size) :

            context = contexts[i]
            response = responses[i]

            all_text = f"{self.template['prompt']}\n\n" + context + f"### GPT:\n{response}"
            source_text = f"{self.template['prompt']}\n\n" + context + "### GPT:\n"
           
            all_input_id = self.tokenizer(
                all_text, 
                max_length=self.sequence_max_length,
                truncation='do_not_truncate',
                add_special_tokens=False
            ).input_ids
            all_input_id = all_input_id + [self.tokenizer.eos_token_id]
            attention_mask = [1]*len(all_input_id)

            source_input_id = self.tokenizer(
                source_text, 
                max_length=self.sequence_max_length,
                truncation='do_not_truncate',
                add_special_tokens=False
            ).input_ids
            source_input_id_length = len(source_input_id)
            label = [self.label_pad_token_id] * source_input_id_length + all_input_id[source_input_id_length:]

            input_ids.append(all_input_id)
            attention_masks.append(attention_mask)
            labels.append(label)

        datasets["input_ids"] = input_ids
        datasets["attention_mask"] = attention_masks
        datasets["labels"] = labels

        return datasets
