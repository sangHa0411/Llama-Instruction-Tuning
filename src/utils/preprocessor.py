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
        self.num_cores = max(multiprocessing.cpu_count() // 3, 1)

        self.preprocessors = {
            "alpaca" : AlpacaPreprocessor(tokenizer, sequence_max_length, label_pad_token_id),
            "cot-collections" : CoTCollectionPreprocessor(tokenizer, sequence_max_length, label_pad_token_id),
            "slimorca" : SlimOrcaPreprocessor(tokenizer, sequence_max_length, label_pad_token_id),
            "openorca-mc10k" : OpenOrcaMCPreprocessor(tokenizer, sequence_max_length, label_pad_token_id),
            "wizardlm" : WizardLMPreprocessor(tokenizer, sequence_max_length, label_pad_token_id),
            "open-platypus" : OpenPlatypusPreprocessor(tokenizer, sequence_max_length, label_pad_token_id),
            "arc" : ArcPreprocessor(tokenizer, sequence_max_length, label_pad_token_id),
            "mmlu" : MmluPreprocessor(tokenizer, sequence_max_length, label_pad_token_id),
            "hellaswag" : HellaswagPreprocessor(tokenizer, sequence_max_length, label_pad_token_id),
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

                # Preprocessing and encoding dataset
                preprocess_fn = preprocessor.preprocess
                preprocessed = dataset.map(preprocess_fn, batched=True, num_proc=self.num_cores, remove_columns=dataset.column_names)

                # Count the data which length is longer then sequence_max_length
                data_longer_then_sequence_max_length = 0
                for d in preprocessed :
                    if len(d["input_ids"]) > self.sequence_max_length :
                        data_longer_then_sequence_max_length += 1
                logging.info(f"### The number of data which length is longer then sequence_max_length\n{data_longer_then_sequence_max_length}\n")

                # Logging preprocessed dataset's input_id and label example
                preprocessed_input_id = preprocessed[0]["input_ids"]
                preprocessed_input_text = self.tokenizer.decode(preprocessed_input_id)

                preprocessed_label = preprocessed[0]["labels"]
                preprocessed_label = [l for l in preprocessed_label if l >= 0]
                preprocessed_label_text = self.tokenizer.decode(preprocessed_label)

                logging.info(f"Preprocessed dataset | {dataset_name}\n### Input\n{preprocessed_input_text}\n### Label\n{preprocessed_label_text}\n\n")

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
                all_text = f"Instruction: {instruction}\nInput: {input_text}\nResponse: {output_text}"
                source_text = f"Instruction: {instruction}\nInput: {input_text}\nResponse: "
            else :
                all_text = f"Instruction: {instruction}\nResponse: {output_text}"
                source_text = f"Instruction: {instruction}\nResponse: "

            all_input_id = self.tokenizer(
                all_text, 
                max_length=self.sequence_max_length,
                truncation='do_not_truncate',
                add_special_tokens=False
            ).input_ids
            attention_mask = [1]*len(all_input_id)

            source_input_id = self.tokenizer(
                source_text, 
                max_length=self.sequence_max_length,
                truncation='do_not_truncate',
                add_special_tokens=False
            ).input_ids
            source_input_id_length = len(source_input_id) - 1
            label = [self.label_pad_token_id] * source_input_id_length + all_input_id[source_input_id_length:]
            label = label[1:] + [self.tokenizer.eos_token_id]

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

            all_text = f"Question: {source}\nRationale: {rationale}\nAnswer: {target}"
            source_text = f"Question: {source}\nRationale: "
           
            all_input_id = self.tokenizer(
                all_text, 
                max_length=self.sequence_max_length,
                truncation='do_not_truncate',
                add_special_tokens=False
            ).input_ids
            attention_mask = [1]*len(all_input_id)

            source_input_id = self.tokenizer(
                source_text, 
                max_length=self.sequence_max_length,
                truncation='do_not_truncate',
                add_special_tokens=False
            ).input_ids
            source_input_id_length = len(source_input_id) - 1
            label = [self.label_pad_token_id] * source_input_id_length + all_input_id[source_input_id_length:]
            label = label[1:] + [self.tokenizer.eos_token_id]

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
        assert conversation[-1]["from"] == "gpt"
            
        gpt_chat = conversation[-1]
        gpt_response = gpt_chat["value"]
            
        history = []
        for i in range(len(conversation)-1) :
            subject = conversation[i]["from"]
            value = conversation[i]["value"]
            
            chat = f"<|im_start|>{subject}\n{value}<|im_end|>"
            history.append(chat)
        context = "\n".join(history)

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

            all_text = context + f"\n<|im_start|>gpt\n{response}<|im_end|>"
            source_text = context + "\n<|im_start|>gpt\n"
           
            all_input_id = self.tokenizer(
                all_text, 
                max_length=self.sequence_max_length,
                truncation='do_not_truncate',
                add_special_tokens=False
            ).input_ids
            attention_mask = [1]*len(all_input_id)

            source_input_id = self.tokenizer(
                source_text, 
                max_length=self.sequence_max_length,
                truncation='do_not_truncate',
                add_special_tokens=False
            ).input_ids
            source_input_id_length = len(source_input_id) - 1
            label = [self.label_pad_token_id] * source_input_id_length + all_input_id[source_input_id_length:]
            label = label[1:] + [self.tokenizer.eos_token_id]

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

            all_text = f"Instruction: {prompt}\nQuestion: {question}\nResponse: {response}"
            source_text = f"Instruction: {prompt}\nQuestion: {question}\nResponse: "
           
            all_input_id = self.tokenizer(
                all_text, 
                max_length=self.sequence_max_length,
                truncation='do_not_truncate',
                add_special_tokens=False
            ).input_ids
            attention_mask = [1]*len(all_input_id)

            source_input_id = self.tokenizer(
                source_text, 
                max_length=self.sequence_max_length,
                truncation='do_not_truncate',
                add_special_tokens=False
            ).input_ids
            source_input_id_length = len(source_input_id) - 1
            label = [self.label_pad_token_id] * source_input_id_length + all_input_id[source_input_id_length:]
            label = label[1:] + [self.tokenizer.eos_token_id]

            input_ids.append(all_input_id)
            attention_masks.append(attention_mask)
            labels.append(label)

        datasets["input_ids"] = input_ids
        datasets["attention_mask"] = attention_masks
        datasets["labels"] = labels

        return datasets


class WizardLMPreprocessor :
    def __init__(self, 
        tokenizer: LlamaTokenizer,
        sequence_max_length: int,
        label_pad_token_id: int = -100
    ) :       
        self.tokenizer = tokenizer
        self.sequence_max_length = sequence_max_length
        self.label_pad_token_id = label_pad_token_id

    def preprocess(self, datasets):
        instructions = datasets["instruction"]
        outputs = datasets["output"]

        input_ids, attention_masks, labels = [], [], []

        size = len(instructions)
        for i in range(size) :
            instruction = instructions[i]
            output = outputs[i]

            all_text = f"Instruction: {instruction}\nOutput: {output}"
            source_text = f"Instruction: {instruction}\nOutput: "
           
            all_input_id = self.tokenizer(
                all_text, 
                max_length=self.sequence_max_length,
                truncation='do_not_truncate',
                add_special_tokens=False
            ).input_ids
            attention_mask = [1]*len(all_input_id)

            source_input_id = self.tokenizer(
                source_text, 
                max_length=self.sequence_max_length,
                truncation='do_not_truncate',
                add_special_tokens=False
            ).input_ids
            source_input_id_length = len(source_input_id) - 1
            label = [self.label_pad_token_id] * source_input_id_length + all_input_id[source_input_id_length:]
            label = label[1:] + [self.tokenizer.eos_token_id]

            input_ids.append(all_input_id)
            attention_masks.append(attention_mask)
            labels.append(label)

        datasets["input_ids"] = input_ids
        datasets["attention_mask"] = attention_masks
        datasets["labels"] = labels

        return datasets


class OpenPlatypusPreprocessor :
    def __init__(self, 
        tokenizer: LlamaTokenizer,
        sequence_max_length: int,
        label_pad_token_id: int = -100
    ) :       
        self.tokenizer = tokenizer
        self.sequence_max_length = sequence_max_length
        self.label_pad_token_id = label_pad_token_id

    def preprocess(self, datasets):
        instructions = datasets["instruction"]
        outputs = datasets["output"]

        input_ids, attention_masks, labels = [], [], []

        size = len(instructions)
        for i in range(size) :
            instruction = instructions[i]
            output = outputs[i]

            all_text = f"Instruction: {instruction}\nOutput: {output}"
            source_text = f"Instruction: {instruction}\nOutput: "
           
            all_input_id = self.tokenizer(
                all_text, 
                max_length=self.sequence_max_length,
                truncation='do_not_truncate',
                add_special_tokens=False
            ).input_ids
            attention_mask = [1]*len(all_input_id)

            source_input_id = self.tokenizer(
                source_text, 
                max_length=self.sequence_max_length,
                truncation='do_not_truncate',
                add_special_tokens=False
            ).input_ids
            source_input_id_length = len(source_input_id) - 1
            label = [self.label_pad_token_id] * source_input_id_length + all_input_id[source_input_id_length:]
            label = label[1:] + [self.tokenizer.eos_token_id]

            input_ids.append(all_input_id)
            attention_masks.append(attention_mask)
            labels.append(label)

        datasets["input_ids"] = input_ids
        datasets["attention_mask"] = attention_masks
        datasets["labels"] = labels

        return datasets


class MmluPreprocessor :
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
        answers = datasets["answer"]

        input_ids, attention_masks, labels = [], [], []

        size = len(questions)
        for i in range(size) :

            question = questions[i]
            choice = choices[i]
            answer = answers[i]

            candidate_answer = "\n".join([f"{i}. {c}" for i, c in enumerate(choice)])
            target_text = choice[answer]

            all_text = f"Question: {question}\nChoices:\n{candidate_answer}\nAnswer: {target_text}"
            source_text = f"Question: {question}\nChoices:\n{candidate_answer}\nAnswer: "

            all_input_id = self.tokenizer(
                all_text, 
                max_length=self.sequence_max_length,
                truncation='do_not_truncate',
                add_special_tokens=False
            ).input_ids
            attention_mask = [1]*len(all_input_id)

            source_input_id = self.tokenizer(
                source_text, 
                max_length=self.sequence_max_length,
                truncation='do_not_truncate',
                add_special_tokens=False
            ).input_ids
            source_input_id_length = len(source_input_id) - 1 
            label = [self.label_pad_token_id] * source_input_id_length + all_input_id[source_input_id_length:]
            label = label[1:] + [self.tokenizer.eos_token_id]

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
            if ord(answer_key) >= ord("A") :
                target_id = ord(answer_key) - ord("A") 
            else :
                target_id = int(answer_key) - 1

            target_text = choice["text"][target_id]
            all_text = f"Question: {question}\nAnswer: {target_text}"
            source_text = f"Question: {question}\nAnswer: "

            all_input_id = self.tokenizer(
                all_text, 
                max_length=self.sequence_max_length,
                truncation='do_not_truncate',
                add_special_tokens=False
            ).input_ids
            attention_mask = [1]*len(all_input_id)

            source_input_id = self.tokenizer(
                source_text, 
                max_length=self.sequence_max_length,
                truncation='do_not_truncate',
                add_special_tokens=False
            ).input_ids
            source_input_id_length = len(source_input_id) - 1 
            label = [self.label_pad_token_id] * source_input_id_length + all_input_id[source_input_id_length:]
            label = label[1:] + [self.tokenizer.eos_token_id]

            input_ids.append(all_input_id)
            attention_masks.append(attention_mask)
            labels.append(label)

        datasets["input_ids"] = input_ids
        datasets["attention_mask"] = attention_masks
        datasets["labels"] = labels

        return datasets


class HellaswagPreprocessor :
    def __init__(self, 
        tokenizer: LlamaTokenizer,
        sequence_max_length: int,
        label_pad_token_id: int = -100
    ) :       
        self.tokenizer = tokenizer
        self.sequence_max_length = sequence_max_length
        self.label_pad_token_id = label_pad_token_id

    def preprocess(self, datasets: List[Dict[str, Any]]):
        activity_labels = datasets["activity_label"]
        ctxs = datasets["ctx"]
        endings = datasets["endings"]
        answers = datasets["label"]

        input_ids, attention_masks, labels = [], [], []

        size = len(ctxs)
        for i in range(size) :
            context = activity_labels[i] + " " + ctxs[i]
            ending = endings[i]
            answer = int(answers[i])            
            target_text = ending[answer]

            all_text = f"Context: {context}\nAnswer: {target_text}"
            source_text = f"Context: {context}\nAnswer: "

            all_input_id = self.tokenizer(
                all_text, 
                max_length=self.sequence_max_length,
                truncation='do_not_truncate',
                add_special_tokens=False
            ).input_ids
            attention_mask = [1]*len(all_input_id)

            source_input_id = self.tokenizer(
                source_text, 
                max_length=self.sequence_max_length,
                truncation='do_not_truncate',
                add_special_tokens=False
            ).input_ids
            source_input_id_length = len(source_input_id) - 1
            label = [self.label_pad_token_id] * source_input_id_length + all_input_id[source_input_id_length:]
            label = label[1:] + [self.tokenizer.eos_token_id]

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
            
            all_text = f"Question: {question}\nAnswer: {answer}"
            source_text = f"Question: {question}\nAnswer: "

            all_input_id = self.tokenizer(
                all_text, 
                max_length=self.sequence_max_length,
                truncation='do_not_truncate',
                add_special_tokens=False
            ).input_ids
            attention_mask = [1]*len(all_input_id)

            source_input_id = self.tokenizer(
                source_text, 
                max_length=self.sequence_max_length,
                truncation='do_not_truncate',
                add_special_tokens=False
            ).input_ids
            source_input_id_length = len(source_input_id) - 1
            label = [self.label_pad_token_id] * source_input_id_length + all_input_id[source_input_id_length:]
            label = label[1:] + [self.tokenizer.eos_token_id]

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

            option_idx = sentence.index("_")
            prefix_text = sentence[:option_idx]
            target_text = answer_text + sentence[option_idx+1:]

            all_text = f"Sentence: {prefix_text}{target_text}"
            source_text = f"Sentence: {prefix_text}"

            all_input_id = self.tokenizer(
                all_text, 
                max_length=self.sequence_max_length,
                truncation='do_not_truncate',
                add_special_tokens=False
            ).input_ids
            attention_mask = [1]*len(all_input_id)

            source_input_id = self.tokenizer(
                source_text, 
                max_length=self.sequence_max_length,
                truncation='do_not_truncate',
                add_special_tokens=False
            ).input_ids
            source_input_id_length = len(source_input_id) - 1
            label = [self.label_pad_token_id] * source_input_id_length + all_input_id[source_input_id_length:]
            label = label[1:] + [self.tokenizer.eos_token_id]

            input_ids.append(all_input_id)
            attention_masks.append(attention_mask)
            labels.append(label)

        datasets["input_ids"] = input_ids
        datasets["attention_mask"] = attention_masks
        datasets["labels"] = labels

        return datasets
