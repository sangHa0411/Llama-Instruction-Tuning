import re
import logging
import pandas as pd
import multiprocessing
from typing import Dict, List, Any
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
        self.num_cores = multiprocessing.cpu_count() // 3

        self.preprocessors = {
            "alpaca" : AlpacaPreprocessor(tokenizer, sequence_max_length, label_pad_token_id),
            "cot-collections" : CoTCollectionPreprocessor(tokenizer, sequence_max_length, label_pad_token_id),
            "slimorca" : SlimOrcaPreprocessor(tokenizer, sequence_max_length, label_pad_token_id),
            "openorca-multiplechoice" : OpenOrcaMCPreprocessor(tokenizer, sequence_max_length, label_pad_token_id),
            "arc" : ArcPreprocessor(tokenizer, sequence_max_length, label_pad_token_id),
            "gsm8k" : GSM8KPreprocessor(tokenizer, sequence_max_length, label_pad_token_id),
            "winogrande" : WinograndePreprocessor(tokenizer, sequence_max_length, label_pad_token_id)
        }

    def __call__(self, datasets: Dict[str, Dataset]) -> Dataset :
        dataset_names = list(datasets.keys())

        preprocessed_datasets = []
        for dataset_name in dataset_names :
            dataset = datasets[dataset_name]

            if dataset_name in self.preprocessors :
                logging.info(f"Preprocessing and Encoding Dataset | {dataset_name}")
                preprocessor = self.preprocessors[dataset_name]

                preprocess_fn = preprocessor.preprocess
                preprocessed = dataset.map(preprocess_fn, batched=True, num_proc=self.num_cores, remove_columns=dataset.column_names)
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


    def preprocess(self, datasets: List[Dict[str, Any]]):
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
                all_text = f"### INSTRUCTION:\n{instruction}\n\n### INPUT:\n{input_text}\n\n### RESPONSE:\n{output_text}"
                source_text = f"### INSTRUCTION:\n{instruction}\n\n### INPUT:\n{input_text}\n\n### RESPONSE:\n"
            else :
                all_text = f"### INSTRUCTION:\n{instruction}\n\n### RESPONSE:\n{output_text}"
                source_text = f"### INSTRUCTION:\n{instruction}\n\n### RESPONSE:\n"

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

    def preprocess(self, datasets: List[Dict[str, Any]]):
        sources = datasets["source"]
        rationales = datasets["rationale"]
        targets = datasets["target"]

        input_ids, attention_masks, labels = [], [], []

        size = len(sources)
        for i in range(size) :

            source = sources[i]
            rationale = rationales[i]
            target = targets[i]

            all_text = f"### SOURCE:\n{source}\n\n### RATIONALE:\n{rationale}\n\n### TARGET:\n{target}"
            source_text = f"### SOURCE:\n{source}\n\n### RATIONALE:\n"
           
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

    def _split(self, conversation: str) -> Dict[str, str]:
        conversation = eval(re.sub("} *\n *{", "},{", conversation))
        assert conversation[-1]["from"] == "gpt"
            
        gpt_chat = conversation[-1]
        gpt_response = gpt_chat["value"]
            
        history = []
        for i in range(len(conversation)-1) :
            subject = conversation[i]["from"]
            value = conversation[i]["value"]
            
            chat = f"### FROM:\n{subject}\n\n### VALUE:\n{value}\n\n"
            history.append(chat)
        context = "".join(history)

        return {
            "context" : context,
            "response" : gpt_response
        }

    def preprocess(self, datasets):
        conversations = datasets["conversations"]

        input_ids, attention_masks, labels = [], [], []

        size = len(conversations)
        for i in range(size) :

            splited = self._split(conversations[i])
            context = splited["context"]
            response = splited["response"]

            all_text = context + f"### GPT:\n{response}"
            source_text = context + "### GPT:\n"
           
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


class OpenOrcaMCPreprocessor :
    def __init__(self, 
        tokenizer: LlamaTokenizer,
        sequence_max_length: int,
        label_pad_token_id: int = -100
    ) :       
        self.tokenizer = tokenizer
        self.sequence_max_length = sequence_max_length
        self.label_pad_token_id = label_pad_token_id

    def preprocess(self, datasets):
        prompts = datasets["system_prompt"]
        questions = datasets["question"]
        responses = datasets["response"]

        input_ids, attention_masks, labels = [], [], []

        size = len(prompts)
        for i in range(size) :
            prompt = prompts[i]
            question = questions[i]
            response = responses[i]

            all_text = f"### INSTRUCTION:\n{prompt}\n\n### QUESTION:\n{question}\n\n### RESPONSE:\n{response}"
            source_text = f"### INSTRUCTION:\n{prompt}\n\n### INPUT:\n{question}\n\n### RESPONSE:\n"
           
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


class ArcPreprocessor :
    def __init__(self, 
        tokenizer: LlamaTokenizer,
        sequence_max_length: int,
        label_pad_token_id: int = -100
    ) :       
        self.tokenizer = tokenizer
        self.sequence_max_length = sequence_max_length
        self.label_pad_token_id = label_pad_token_id

    def preprocess(self, datasets: List[Dict[str, Any]]):
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
            if ord(answer_key) >= ord("A") :
                target_id = ord(answer_key) - ord("A") 
            else :
                target_id = int(answer_key) - 1

            target_text = choice["text"][target_id]

            all_text = f"### QUESTION:\n{question}\n\n### CANDIDATE ANSWERS:\n{candidate_answer}\n\n### ANSWER:\n{target_text}"
            source_text = f"### QUESTION:\n{question}\n\n### CANDIDATE ANSWERS:\n{candidate_answer}\n\n### ANSWER:\n"

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


class GSM8KPreprocessor :
    def __init__(self, 
        tokenizer: LlamaTokenizer,
        sequence_max_length: int,
        label_pad_token_id: int = -100
    ) :       
        self.tokenizer = tokenizer
        self.sequence_max_length = sequence_max_length
        self.label_pad_token_id = label_pad_token_id

    def preprocess(self, datasets: List[Dict[str, Any]]):
        questions = datasets["question"]
        answers = datasets["answer"]

        input_ids, attention_masks, labels = [], [], []

        size = len(questions)
        for i in range(size) :
            question = questions[i]
            answer = answers[i]
            
            all_text = f"### QUESTION:\n{question}\n\n### ANSWER:\n{answer}"
            source_text = f"### QUESTION:\n{question}\n\n### ANSWER:\n"

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


class WinograndePreprocessor :
    def __init__(self, 
        tokenizer: LlamaTokenizer,
        sequence_max_length: int,
        label_pad_token_id: int = -100
    ) :       
        self.tokenizer = tokenizer
        self.sequence_max_length = sequence_max_length
        self.label_pad_token_id = label_pad_token_id

    def preprocess(self, datasets: List[Dict[str, Any]]):
        sentences = datasets["sentence"]
        option1s = datasets["option1"]
        option2s = datasets["option2"]
        answers = datasets["answer"]

        input_ids, attention_masks, labels = [], [], []

        size = len(sentences)
        for i in range(size) :
            sentence = sentences[i]
            option1 = option1s[i]
            option2 = option2s[i]
            answer = answers[i]
            answer_text = option1 if answer == 1 else option2
            
            all_text = f"### SENTENCE:\n{sentence}\n\n### OPTION1:\n{option1}\n\n### OPTION2:\n{option2}\n\n### ANSWER:\n{answer_text}"
            source_text = f"### SENTENCE:\n{sentence}\n\n### OPTION1:\n{option1}\n\n### OPTION2:\n{option2}\n\n### ANSWER:\n"

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
