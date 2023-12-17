import logging
import numpy as np
import multiprocessing
from functools import partial
from typing import Dict, List, Any
from pytz import timezone
from datetime import datetime
from datasets import Dataset
from transformers import LlamaTokenizer

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
        self.num_cores = multiprocessing.cpu_count() // 3

        self.preprocessors = {
            "arc" : EvalArcPreprocessor(tokenizer, sequence_max_length),
            "mmlu" : EvalMmluPreprocessor(tokenizer, sequence_max_length),
            "hellaswag" : EvalHellaswagPreprocessor(tokenizer, sequence_max_length),
            "gsm8k" : EvalGSM8KPreprocessor(tokenizer, sequence_max_length),
            "truthful_qa-generation" : EvalTruthfulQAGenerationPreprocessor(tokenizer, sequence_max_length),
            "truthful_qa-multiple_choice" : EvalTruthfulQAMultipleChoicePreprocessor(tokenizer, sequence_max_length),
            "winogrande" : EvalWinograndePreprocessor(tokenizer, sequence_max_length),
        }

    def __call__(self, num_shots: str, datasets: Dict[str, Dataset]) -> Dict[str, Dataset] :
        num_shots = num_shots[1:-1].split(",")
        num_shots = [int(shot) for shot in num_shots]

        dataset_names = list(datasets.keys())
        assert len(dataset_names) == len(num_shots)
        
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


class EvalArcPreprocessor :
    def __init__(self, 
        tokenizer: LlamaTokenizer,
        sequence_max_length: int,
    ) :       
        self.tokenizer = tokenizer
        self.sequence_max_length = sequence_max_length

    def _make_few_shot_example(self, datasets: List[Dict[str, Any]], sampled_ids: List[int]) :
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

    def _truncate(self, input_ids: List[int]) :
        input_ids = input_ids[-self.sequence_max_length:]
        input_string = self.tokenizer.decode(input_ids)

        input_shots = input_string.split("\n\n\n\n")
        if input_shots[0][:3] != "###" :
            input_shots = input_shots[1:]

        num_shots = len(input_shots) - 1
        truncated_input_string = "\n\n\n\n".join(input_shots)
        truncated_input_id = self.tokenizer(
            truncated_input_string, 
            max_length=self.sequence_max_length,
            truncation='do_not_truncate',
            add_special_tokens=False
        ).input_ids

        return truncated_input_id, num_shots

    def preprocess(self, datasets: List[Dict[str, Any]], num_shot: int):
        questions = datasets["question"]
        choices = datasets["choices"]
        answer_keys = datasets["answerKey"]

        input_ids, attention_masks, labels = [], [], []
        num_shots = []

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
                few_shot_example = self._make_few_shot_example(datasets, sampled_ids)
                input_text = few_shot_example + "\n\n\n\n" + input_text

            input_id = self.tokenizer(
                input_text, 
                max_length=self.sequence_max_length,
                truncation='do_not_truncate',
                add_special_tokens=False
            ).input_ids
            if num_shot > 0 :
                input_id, num_used_shot = self._truncate(input_id)
            else :
                num_used_shot = 0
            attention_mask = [1]*len(input_id)
            num_shots.append(num_used_shot)

            input_ids.append(input_id)
            attention_masks.append(attention_mask)
            labels.append(target_text)

        datasets["input_ids"] = input_ids
        datasets["attention_mask"] = attention_masks
        datasets["labels"] = labels

        datasets["num_shot"] = num_shots

        return datasets


class EvalMmluPreprocessor :
    def __init__(self, 
        tokenizer: LlamaTokenizer,
        sequence_max_length: int,
    ) :       
        self.tokenizer = tokenizer
        self.sequence_max_length = sequence_max_length

    def _make_few_shot_example(self, datasets: List[Dict[str, Any]], sampled_ids: List[int]) :
        questions = datasets["question"]
        choices = datasets["choices"]
        answers = datasets["answer"]

        examples = []
        for i in sampled_ids :
            question = questions[i]
            choice = choices[i]
            answer = answers[i]

            candidate_answer = " ".join([f"({i}): {c}" for i, c in enumerate(choice)])
            target_text = choice[answer]

            input_text = f"### QUESTION:\n{question}\n\n### CHOICES:\n{candidate_answer}\n\n### ANSWER:\n{target_text}"
            examples.append(input_text)

        few_shot_example = "\n\n\n\n".join(examples)
        return few_shot_example

    def _truncate(self, input_ids: List[int]) :
        input_ids = input_ids[-self.sequence_max_length:]
        input_string = self.tokenizer.decode(input_ids)

        input_shots = input_string.split("\n\n\n\n")
        if input_shots[0][:3] != "###" :
            input_shots = input_shots[1:]

        num_shots = len(input_shots) - 1
        truncated_input_string = "\n\n\n\n".join(input_shots)
        truncated_input_id = self.tokenizer(
            truncated_input_string, 
            max_length=self.sequence_max_length,
            truncation='do_not_truncate',
            add_special_tokens=False
        ).input_ids

        return truncated_input_id, num_shots

    def preprocess(self, datasets: List[Dict[str, Any]], num_shot: int):
        questions = datasets["question"]
        choices = datasets["choices"]
        answers = datasets["answer"]

        input_ids, attention_masks, labels = [], [], []
        num_shots = []

        size = len(questions)
        for i in range(size) :

            question = questions[i]
            choice = choices[i]
            answer = answers[i]

            candidate_answer = " ".join([f"({i}): {c}" for i, c in enumerate(choice)])
            target_text = choice[answer]

            input_text = f"### QUESTION:\n{question}\n\n### CHOICES:\n{candidate_answer}\n\n### ANSWER:\n"

            if num_shot > 0 :
                sampled_ids = np.random.choice(size, num_shot+1, replace=False)
                sampled_ids = list(set(sampled_ids) - set([i]))[:num_shot]
                few_shot_example = self._make_few_shot_example(datasets, sampled_ids)
                input_text = few_shot_example + "\n\n\n\n" + input_text

            input_id = self.tokenizer(
                input_text, 
                max_length=self.sequence_max_length,
                truncation='do_not_truncate',
                add_special_tokens=False
            ).input_ids
            if num_shot > 0 :
                input_id, num_used_shot = self._truncate(input_id)
            else :
                num_used_shot = 0
            attention_mask = [1]*len(input_id)

            num_shots.append(num_used_shot)

            input_ids.append(input_id)
            attention_masks.append(attention_mask)
            labels.append(target_text)

        datasets["input_ids"] = input_ids
        datasets["attention_mask"] = attention_masks
        datasets["labels"] = labels

        datasets["num_shot"] = num_shots

        return datasets


class EvalHellaswagPreprocessor :
    def __init__(self, 
        tokenizer: LlamaTokenizer,
        sequence_max_length: int,
    ) :       
        self.tokenizer = tokenizer
        self.sequence_max_length = sequence_max_length

    def _make_few_shot_example(self, datasets: List[Dict[str, Any]], sampled_ids: List[int]) :
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

    def _truncate(self, input_ids: List[int]) :
        input_ids = input_ids[-self.sequence_max_length:]
        input_string = self.tokenizer.decode(input_ids)

        input_shots = input_string.split("\n\n\n\n")
        if input_shots[0][:3] != "###" :
            input_shots = input_shots[1:]
        
        num_shots = len(input_shots) - 1
        truncated_input_string = "\n\n\n\n".join(input_shots)
        truncated_input_id = self.tokenizer(
            truncated_input_string, 
            max_length=self.sequence_max_length,
            truncation='do_not_truncate',
            add_special_tokens=False
        ).input_ids

        return truncated_input_id, num_shots

    def preprocess(self, datasets: List[Dict[str, Any]], num_shot: int) :
        ctxs = datasets["ctx"]
        endings = datasets["endings"]
        answers = datasets["label"]

        input_ids, attention_masks, labels = [], [], []
        num_shots = []

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
                few_shot_example = self._make_few_shot_example(datasets, sampled_ids)
                input_text = few_shot_example + "\n\n\n\n" + input_text

            input_id = self.tokenizer(
                input_text, 
                max_length=self.sequence_max_length,
                truncation='do_not_truncate',
                add_special_tokens=False
            ).input_ids
            if num_shot > 0 :
                input_id, num_used_shot = self._truncate(input_id)
            else :
                num_used_shot = 0
            attention_mask = [1]*len(input_id)

            num_shots.append(num_used_shot)

            input_ids.append(input_id)
            attention_masks.append(attention_mask)
            labels.append(target_text)

        datasets["input_ids"] = input_ids
        datasets["attention_mask"] = attention_masks
        datasets["labels"] = labels

        datasets["num_shot"] = num_shots

        return datasets


class EvalGSM8KPreprocessor :
    def __init__(self, 
        tokenizer: LlamaTokenizer,
        sequence_max_length: int,
    ) :       
        self.tokenizer = tokenizer
        self.sequence_max_length = sequence_max_length

    def _make_few_shot_example(self, datasets: List[Dict[str, Any]], sampled_ids: List[int]) :
        questions = datasets["question"]
        answers = datasets["answer"]

        examples = []
        for i in sampled_ids :
            question = questions[i]
            answer = answers[i]

            input_text = f"### QUESTION:\n{question}\n\n### ANSWER:\n{answer}"
            examples.append(input_text)

        few_shot_example = "\n\n\n\n".join(examples)
        return few_shot_example

    def _truncate(self, input_ids: List[int]) :
        input_ids = input_ids[-self.sequence_max_length:]
        input_string = self.tokenizer.decode(input_ids)

        input_shots = input_string.split("\n\n\n\n")
        if input_shots[0][:3] != "###" :
            input_shots = input_shots[1:]

        num_shots = len(input_shots) - 1
        truncated_input_string = "\n\n\n\n".join(input_shots)
        truncated_input_id = self.tokenizer(
            truncated_input_string, 
            max_length=self.sequence_max_length,
            truncation='do_not_truncate',
            add_special_tokens=False
        ).input_ids

        return truncated_input_id, num_shots

    def preprocess(self, datasets: List[Dict[str, Any]], num_shot: int) :
        questions = datasets["question"]
        answers = datasets["answer"]
        
        input_ids, attention_masks, labels = [], [], []
        num_shots = []

        size = len(questions)
        for i in range(size) :
            question = questions[i]
            answer = answers[i]
            
            input_text = f"### QUESTION:\n{question}\n\n### ANSWER:\n"
            target_text = answer

            if num_shot > 0 :
                sampled_ids = np.random.choice(size, num_shot+1, replace=False)
                sampled_ids = list(set(sampled_ids) - set([i]))[:num_shot]
                few_shot_example = self._make_few_shot_example(datasets, sampled_ids)
                input_text = few_shot_example + "\n\n\n\n" + input_text

            input_id = self.tokenizer(
                input_text, 
                max_length=self.sequence_max_length,
                truncation='do_not_truncate',
                add_special_tokens=False
            ).input_ids
            if num_shot > 0 :
                input_id, num_used_shot = self._truncate(input_id)
            else :
                num_used_shot = 0
            attention_mask = [1]*len(input_id)

            num_shots.append(num_used_shot)

            input_ids.append(input_id)
            attention_masks.append(attention_mask)
            labels.append(target_text)

        datasets["input_ids"] = input_ids
        datasets["attention_mask"] = attention_masks
        datasets["labels"] = labels

        datasets["num_shot"] = num_shots

        return datasets



class EvalTruthfulQAGenerationPreprocessor :
    def __init__(self, 
        tokenizer: LlamaTokenizer,
        sequence_max_length: int,
    ) :       
        self.tokenizer = tokenizer
        self.sequence_max_length = sequence_max_length

    def _make_few_shot_example(self, datasets: List[Dict[str, Any]], sampled_ids: List[int]) :
        questions = datasets["question"]
        answers = datasets["best_answer"]

        examples = []
        for i in sampled_ids :
            question = questions[i]
            answer = answers[i]

            input_text = f"### QUESTION:\n{question}\n\n### ANSWER:\n{answer}"
            examples.append(input_text)

        few_shot_example = "\n\n\n\n".join(examples)
        return few_shot_example

    def _truncate(self, input_ids: List[int]) :
        input_ids = input_ids[-self.sequence_max_length:]
        input_string = self.tokenizer.decode(input_ids)

        input_shots = input_string.split("\n\n\n\n")
        if input_shots[0][:3] != "###" :
            input_shots = input_shots[1:]

        num_shots = len(input_shots) - 1
        truncated_input_string = "\n\n\n\n".join(input_shots)
        truncated_input_id = self.tokenizer(
            truncated_input_string, 
            max_length=self.sequence_max_length,
            truncation='do_not_truncate',
            add_special_tokens=False
        ).input_ids

        return truncated_input_id, num_shots

    def preprocess(self, datasets: List[Dict[str, Any]], num_shot: int) :
        questions = datasets["question"]
        correct_answers = datasets["correct_answers"]

        input_ids, attention_masks, labels = [], [], []
        num_shots = []

        size = len(questions)
        for i in range(size) :
            
            question = questions[i]
            correct_answer = correct_answers[i]

            input_text = f"### QUESTION:\n{question}\n\n### ANSWER:\n"
            target_texts = correct_answer

            if num_shot > 0 :
                sampled_ids = np.random.choice(size, num_shot+1, replace=False)
                sampled_ids = list(set(sampled_ids) - set([i]))[:num_shot]
                few_shot_example = self._make_few_shot_example(datasets, sampled_ids)
                input_text = few_shot_example + "\n\n\n\n" + input_text

            input_id = self.tokenizer(
                input_text, 
                max_length=self.sequence_max_length,
                truncation='do_not_truncate',
                add_special_tokens=False
            ).input_ids
            if num_shot > 0 :
                input_id, num_used_shot = self._truncate(input_id)
            else :
                num_used_shot = 0
            attention_mask = [1]*len(input_id)

            num_shots.append(num_used_shot)

            input_ids.append(input_id)
            attention_masks.append(attention_mask)
            labels.append(target_texts)

        datasets["input_ids"] = input_ids
        datasets["attention_mask"] = attention_masks
        datasets["labels"] = labels

        datasets["num_shot"] = num_shots

        return datasets


class EvalTruthfulQAMultipleChoicePreprocessor :
    def __init__(self, 
        tokenizer: LlamaTokenizer,
        sequence_max_length: int,
    ) :       
        self.tokenizer = tokenizer
        self.sequence_max_length = sequence_max_length

    def _make_few_shot_example(self, datasets: List[Dict[str, Any]], sampled_ids: List[int]) :
        questions = datasets["question"]
        mc1_targets = datasets["mc1_targets"]

        examples = []
        for i in sampled_ids :
            question = questions[i]
            mc1_target = mc1_targets[i]

            candidates = " ".join([f"({i}): {e}" for i, e in enumerate(mc1_target["choices"])])
            target_id = mc1_target["labels"].index(1)
            target_text = mc1_target["choices"][target_id]

            input_text = f"### QUESTION:\n{question}\n\n### CHOICES:\n{candidates}### ANSWER:\n{target_text}"
            examples.append(input_text)

        few_shot_example = "\n\n\n\n".join(examples)
        return few_shot_example

    def _truncate(self, input_ids: List[int]) :
        input_ids = input_ids[-self.sequence_max_length:]
        input_string = self.tokenizer.decode(input_ids)

        input_shots = input_string.split("\n\n\n\n")
        if input_shots[0][:3] != "###" :
            input_shots = input_shots[1:]

        num_shots = len(input_shots) - 1
        truncated_input_string = "\n\n\n\n".join(input_shots)
        truncated_input_id = self.tokenizer(
            truncated_input_string, 
            max_length=self.sequence_max_length,
            truncation='do_not_truncate',
            add_special_tokens=False
        ).input_ids

        return truncated_input_id, num_shots

    def preprocess(self, datasets: List[Dict[str, Any]], num_shot: int) :
        questions = datasets["question"]
        mc1_targets = datasets["mc1_targets"]

        input_ids, attention_masks, labels = [], [], []
        num_shots = []

        size = len(questions)
        for i in range(size) :
            question = questions[i]
            mc1_target = mc1_targets[i]
            candidates = " ".join([f"({i}): {e}" for i, e in enumerate(mc1_target["choices"])])

            input_text = f"### QUESTION:\n{question}\n\n### CHOICES:\n{candidates}### ANSWER:\n"
            target_id = mc1_target["labels"].index(1)
            target_text = mc1_target["choices"][target_id]

            if num_shot > 0 :
                sampled_ids = np.random.choice(size, num_shot+1, replace=False)
                sampled_ids = list(set(sampled_ids) - set([i]))[:num_shot]
                few_shot_example = self._make_few_shot_example(datasets, sampled_ids)
                input_text = few_shot_example + "\n\n\n\n" + input_text

            input_id = self.tokenizer(
                input_text, 
                max_length=self.sequence_max_length,
                truncation='do_not_truncate',
                add_special_tokens=False
            ).input_ids
            if num_shot > 0 :
                input_id, num_used_shot = self._truncate(input_id)
            else :
                num_used_shot = 0
            attention_mask = [1]*len(input_id)

            num_shots.append(num_used_shot)

            input_ids.append(input_id)
            attention_masks.append(attention_mask)
            labels.append(target_text)

        datasets["input_ids"] = input_ids
        datasets["attention_mask"] = attention_masks
        datasets["labels"] = labels

        datasets["num_shot"] = num_shots

        return datasets


class EvalWinograndePreprocessor :
    def __init__(self, 
        tokenizer: LlamaTokenizer,
        sequence_max_length: int,
    ) :       
        self.tokenizer = tokenizer
        self.sequence_max_length = sequence_max_length

    def _make_few_shot_example(self, datasets: List[Dict[str, Any]], sampled_ids: List[int]) :
        sentences = datasets["sentence"]
        option1s = datasets["option1"]
        option2s = datasets["option2"]
        answers = datasets["answer"]

        examples = []
        for i in sampled_ids :
            sentence = sentences[i]
            option1 = option1s[i]
            option2 = option2s[i]
            answer = answers[i]
            answer_text = option1 if answer == 1 else option2

            input_text = f"### SENTENCE:\n{sentence}\n\n### OPTION1:\n{option1}\n\n### OPTION2:\n{option2}\n\n### ANSWER:\n{answer_text}"
            examples.append(input_text)

        few_shot_example = "\n\n\n\n".join(examples)
        return few_shot_example

    def _truncate(self, input_ids: List[int]) :
        input_ids = input_ids[-self.sequence_max_length:]
        input_string = self.tokenizer.decode(input_ids)

        input_shots = input_string.split("\n\n\n\n")
        if input_shots[0][:3] != "###" :
            input_shots = input_shots[1:]

        num_shots = len(input_shots) - 1
        truncated_input_string = "\n\n\n\n".join(input_shots)
        truncated_input_id = self.tokenizer(
            truncated_input_string, 
            max_length=self.sequence_max_length,
            truncation='do_not_truncate',
            add_special_tokens=False
        ).input_ids

        return truncated_input_id, num_shots

    def preprocess(self, datasets: List[Dict[str, Any]], num_shot: int) :
        sentences = datasets["sentence"]
        option1s = datasets["option1"]
        option2s = datasets["option2"]
        answers = datasets["answer"]

        input_ids, attention_masks, labels = [], [], []
        num_shots = []

        size = len(sentences)
        for i in range(size) :
            sentence = sentences[i]
            option1 = option1s[i]
            option2 = option2s[i]
            answer = answers[i]
            answer_text = option1 if answer == 1 else option2
            
            input_text = f"### SENTENCE:\n{sentence}\n\n### OPTION1:\n{option1}\n\n### OPTION2:\n{option2}\n\n### ANSWER:\n"

            if num_shot > 0 :
                sampled_ids = np.random.choice(size, num_shot+1, replace=False)
                sampled_ids = list(set(sampled_ids) - set([i]))[:num_shot]
                few_shot_example = self._make_few_shot_example(datasets, sampled_ids)
                input_text = few_shot_example + "\n\n\n\n" + input_text

            input_id = self.tokenizer(
                input_text, 
                max_length=self.sequence_max_length,
                truncation='do_not_truncate',
                add_special_tokens=False
            ).input_ids
            if num_shot > 0 :
                input_id, num_used_shot = self._truncate(input_id)
            else :
                num_used_shot = 0
            attention_mask = [1]*len(input_id)

            num_shots.append(num_used_shot)

            input_ids.append(input_id)
            attention_masks.append(attention_mask)
            labels.append(answer_text)

        datasets["input_ids"] = input_ids
        datasets["attention_mask"] = attention_masks
        datasets["labels"] = labels

        datasets["num_shot"] = num_shots

        return datasets
