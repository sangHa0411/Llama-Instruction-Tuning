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
        self.num_cores = max(multiprocessing.cpu_count() // 3, 1)

        self.preprocessors = {
            "arc" : EvalArcPreprocessor(tokenizer, sequence_max_length),
            "mmlu" : EvalMmluPreprocessor(tokenizer, sequence_max_length),
            "hellaswag" : EvalHellaswagPreprocessor(tokenizer, sequence_max_length),
            "gsm8k" : EvalGSM8KPreprocessor(tokenizer, sequence_max_length),
            "truthful_qa" : EvalTruthfulQAPreprocessor(tokenizer, sequence_max_length),
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
                remove_column_names = list(set(dataset.column_names) - set(["id"]))
                preprocessed = dataset.map(preprocess_fn, batched=True, num_proc=self.num_cores, remove_columns=remove_column_names)

                preprocessed_example = preprocessed[0]["input_ids"]
                preprocessed_label = preprocessed[0]["labels"]

                if isinstance(preprocessed_example[0], int) :
                    num_sequence = 1
                    preprocessed_example = self.tokenizer.decode(preprocessed_example)
                else :
                    num_sequence = len(preprocessed_example)
                    preprocessed_example = self.tokenizer.decode(preprocessed_example[0])

                # ARC, Hellaswag, MMLU, Winogrande
                if isinstance(preprocessed_label, int) :
                    logging.info(f"Preprocessed dataset | {dataset_name}\n### The number of candidate in data: {num_sequence}\n\n### First candidate in data\n{preprocessed_example}\n\n### Answer candidate index\n{preprocessed_label}\n\n")
                else :
                    # TruthfulQA
                    if isinstance(preprocessed_label, list) :
                        logging.info(f"Preprocessed dataset | {dataset_name}\n### The number of candidate in data: {num_sequence}\n\n### First candidate in data\n{preprocessed_example}\n\n### Candidates' label\n{preprocessed_label}\n\n")
                    # GSM8K
                    else :
                        logging.info(f"Preprocessed dataset | {dataset_name}\n### The number of candidate in data: {num_sequence}\n\n### Input text\n{preprocessed_example}\n\n### Answer\n{preprocessed_label}\n\n")
                    
                preprocessed_datasets[dataset_name] = preprocessed
        return preprocessed_datasets


class EvalArcPreprocessor :
    def __init__(self, 
        tokenizer: LlamaTokenizer,
        sequence_max_length: int,
    ) :       
        self.tokenizer = tokenizer
        self.sequence_max_length = sequence_max_length

    # Make few shot example prompt
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
            target_text = choice["text"][target_id]

            input_text = f"Question: {question}\nAnswer: {target_text}"
            examples.append(input_text)

        few_shot_example = "\n\n".join(examples)
        return few_shot_example

    # Delete if first shot is truncated
    def _truncate(self, input_ids: List[int]) :
        input_ids = input_ids[-self.sequence_max_length:]
        input_string = self.tokenizer.decode(input_ids)

        input_shots = input_string.split("\n\n")
        start_span = "Question: "
        if input_shots[0][:len(start_span)] != start_span :
            input_shots = input_shots[1:]

        truncated_input_string = "\n\n".join(input_shots)
        truncated_input_id = self.tokenizer(
            truncated_input_string, 
            max_length=self.sequence_max_length,
            truncation='do_not_truncate',
            add_special_tokens=False
        ).input_ids

        return truncated_input_id

    def preprocess(self, datasets: List[Dict[str, Any]], num_shot: int):
        prev_data_ids = datasets["id"]
        questions = datasets["question"]
        choices = datasets["choices"]
        answer_keys = datasets["answerKey"]

        cur_data_ids = []
        input_ids, attention_masks, labels = [], [], []
        candidate_lengths = []

        size = len(questions)
        for i in range(size) :
            data_id = prev_data_ids[i]
            question = questions[i]
            choice = choices[i]
            answer_key = answer_keys[i]

            input_text = f"Question: {question}\nAnswer: "
            # if answer_key is one of A, B, C or D, change answer key to integer
            if ord(answer_key) >= ord("A") :
                target_id = ord(answer_key) - ord("A") 
            # Answer key is one of 1, 2, 3 or 4
            else :
                target_id = int(answer_key) - 1
            # Answer candidate index
            labels.append(target_id)

            if num_shot > 0 :
                sampled_ids = np.random.choice(size, num_shot+1, replace=False)
                sampled_ids = list(set(sampled_ids) - set([i]))[:num_shot]
                few_shot_example = self._make_few_shot_example(datasets, sampled_ids)
                input_text = few_shot_example + "\n\n" + input_text

            sub_data_ids, sub_input_ids, sub_attention_mask = [], [], []
            sub_candidate_lengths = []
            # Preprocess all candidates using same prompt
            for j in range(len(choice["text"])) :
                candidate = choice["text"][j]
                input_text = input_text + candidate

                input_id = self.tokenizer(
                    input_text, 
                    max_length=self.sequence_max_length,
                    truncation='do_not_truncate',
                    add_special_tokens=False
                ).input_ids
                if num_shot > 0 :
                    input_id = self._truncate(input_id)
                attention_mask = [1]*len(input_id)

                sub_input_ids.append(input_id)
                sub_attention_mask.append(attention_mask)

                sub_data_ids.append(data_id+f"-{j}")

                # Store candidate's token length for acc_norm
                candidate_ids = self.tokenizer(
                    candidate,
                    max_length=self.sequence_max_length,
                    truncation='do_not_truncate',
                    add_special_tokens=False
                ).input_ids
                sub_candidate_lengths.append(len(candidate_ids))

            cur_data_ids.append(sub_data_ids)

            input_ids.append(sub_input_ids)
            attention_masks.append(sub_attention_mask)

            candidate_lengths.append(sub_candidate_lengths)
 
        datasets["id"] = cur_data_ids

        datasets["input_ids"] = input_ids
        datasets["attention_mask"] = attention_masks
        datasets["labels"] = labels

        datasets["candidate_length"] = candidate_lengths

        return datasets


class EvalMmluPreprocessor :
    def __init__(self, 
        tokenizer: LlamaTokenizer,
        sequence_max_length: int,
    ) :       
        self.tokenizer = tokenizer
        self.sequence_max_length = sequence_max_length

    # Make few shot example prompt
    def _make_few_shot_example(self, datasets: List[Dict[str, Any]], sampled_ids: List[int]) :
        questions = datasets["question"]
        choices = datasets["choices"]
        answers = datasets["answer"]

        examples = []
        for i in sampled_ids :
            question = questions[i]
            choice = choices[i]
            answer = answers[i]

            candidate_answer = "\n".join([f"{chr(i+65)}. {c}" for i, c in enumerate(choice)])
            target_text = choice[answer]

            input_text = f"Question: {question}\nChoices:\n{candidate_answer}\nAnswer: {target_text}"
            examples.append(input_text)

        few_shot_example = "\n\n".join(examples)
        return few_shot_example

    # Delete if first shot is truncated
    def _truncate(self, input_ids: List[int]) :
        input_ids = input_ids[-self.sequence_max_length:]
        input_string = self.tokenizer.decode(input_ids)

        input_shots = input_string.split("\n\n")
        start_span = "Question: "
        if input_shots[0][:len(start_span)] != start_span :
            input_shots = input_shots[1:]

        truncated_input_string = "\n\n".join(input_shots)
        truncated_input_id = self.tokenizer(
            truncated_input_string, 
            max_length=self.sequence_max_length,
            truncation='do_not_truncate',
            add_special_tokens=False
        ).input_ids

        return truncated_input_id

    def preprocess(self, datasets: List[Dict[str, Any]], num_shot: int):
        prev_data_ids = datasets["id"]
        questions = datasets["question"]
        choices = datasets["choices"]
        answers = datasets["answer"]

        cur_data_ids = []
        input_ids, attention_masks, labels = [], [], []
        candidate_lengths = []

        size = len(questions)
        for i in range(size) :
            data_id = prev_data_ids[i]
            question = questions[i]
            choice = choices[i]

            answer = answers[i]
            labels.append(answer)
            
            # Choice should start from one of A, B, C or D
            candidate_answer = "\n".join([f"{chr(i + 65)}. {c}" for i, c in enumerate(choice)])
            input_text = f"Question: {question}\nChoices:\n{candidate_answer}\nAnswer: "

            if num_shot > 0 :
                sampled_ids = np.random.choice(size, num_shot+1, replace=False)
                sampled_ids = list(set(sampled_ids) - set([i]))[:num_shot]
                few_shot_example = self._make_few_shot_example(datasets, sampled_ids)
                input_text = few_shot_example + "\n\n" + input_text

            sub_data_ids, sub_input_ids, sub_attention_mask = [], [], []
            sub_candidate_lengths = []
            # Preprocess all candidates using same prompt
            for j in range(len(choice)) :
                candidate = choice[j]
                input_text = input_text + candidate

                input_id = self.tokenizer(
                    input_text, 
                    max_length=self.sequence_max_length,
                    truncation='do_not_truncate',
                    add_special_tokens=False
                ).input_ids
                if num_shot > 0 :
                    input_id = self._truncate(input_id)
                attention_mask = [1]*len(input_id)

                sub_input_ids.append(input_id)
                sub_attention_mask.append(attention_mask)

                sub_data_ids.append(data_id+f"-{j}")    

                # Store candidate's token length for acc_norm
                candidate_ids = self.tokenizer(
                    candidate,
                    max_length=self.sequence_max_length,
                    truncation='do_not_truncate',
                    add_special_tokens=False
                ).input_ids
                sub_candidate_lengths.append(len(candidate_ids))

            cur_data_ids.append(sub_data_ids)

            input_ids.append(sub_input_ids)
            attention_masks.append(sub_attention_mask)

            candidate_lengths.append(sub_candidate_lengths)
 
        datasets["id"] = cur_data_ids

        datasets["input_ids"] = input_ids
        datasets["attention_mask"] = attention_masks
        datasets["labels"] = labels

        datasets["candidate_length"] = candidate_lengths

        return datasets


class EvalHellaswagPreprocessor :
    def __init__(self, 
        tokenizer: LlamaTokenizer,
        sequence_max_length: int,
    ) :       
        self.tokenizer = tokenizer
        self.sequence_max_length = sequence_max_length

    # Make few shot example prompt
    def _make_few_shot_example(self, datasets: List[Dict[str, Any]], sampled_ids: List[int]) :
        activity_labels = datasets["activity_label"]
        ctxs = datasets["ctx"]
        endings = datasets["endings"]
        answers = datasets["label"]

        examples = []
        for i in sampled_ids :
            context = activity_labels[i] + ": " + ctxs[i]
            ending = endings[i]
            answer = int(answers[i])

            target_text = ending[answer]
            input_text = context + " " + target_text
            examples.append(input_text)

        few_shot_example = "\n\n".join(examples)
        return few_shot_example

    # Delete if first shot is truncated
    def _truncate(self, input_ids: List[int], activity_labels: List[str]) :
        input_ids = input_ids[-self.sequence_max_length:]
        input_string = self.tokenizer.decode(input_ids)

        input_shots = input_string.split("\n\n")
        first_shot = input_shots[0]
        if ":" not in first_shot :
            input_shots = input_shots[1:]
        else :
            activity_label_pos = first_shot.index(":") 
            activity_labels_name = first_shot[:activity_label_pos]

            if activity_labels_name not in activity_labels :
                input_shots = input_shots[1:]
        
        truncated_input_string = "\n\n".join(input_shots)
        truncated_input_id = self.tokenizer(
            truncated_input_string, 
            max_length=self.sequence_max_length,
            truncation='do_not_truncate',
            add_special_tokens=False
        ).input_ids

        return truncated_input_id

    def preprocess(self, datasets: List[Dict[str, Any]], num_shot: int) :
        prev_data_ids = datasets["id"]
        activity_labels = datasets["activity_label"]
        ctxs = datasets["ctx"]
        endings = datasets["endings"]
        answers = datasets["label"]

        cur_data_ids = []
        input_ids, attention_masks, labels = [], [], []
        candidate_lengths = []

        size = len(ctxs)
        for i in range(size) :
            data_id = prev_data_ids[i]
            input_text = activity_labels[i] + ": " + ctxs[i]
            ending = endings[i]

            answer = int(answers[i])
            labels.append(answer)

            # Make few shot prompt
            if num_shot > 0 :
                sampled_ids = np.random.choice(size, num_shot+1, replace=False)
                sampled_ids = list(set(sampled_ids) - set([i]))[:num_shot]
                few_shot_example = self._make_few_shot_example(datasets, sampled_ids)
                input_text = few_shot_example + "\n\n" + input_text

            # Preprocess all endings
            sub_data_ids, sub_input_ids, sub_attention_mask = [], [], []
            sub_candidate_lengths = []
            for j in range(len(ending)) :
                candidate = ending[j]
                input_text = input_text + " " + candidate

                input_id = self.tokenizer(
                    input_text, 
                    max_length=self.sequence_max_length,
                    truncation='do_not_truncate',
                    add_special_tokens=False
                ).input_ids
                if num_shot > 0 :
                    input_id = self._truncate(input_id, activity_labels)

                attention_mask = [1]*len(input_id)
                sub_data_ids.append(data_id+f"-{j}")

                sub_input_ids.append(input_id)
                sub_attention_mask.append(attention_mask)

                # Store candidate's token length for acc_norm
                candidate_ids = self.tokenizer(
                    candidate,
                    max_length=self.sequence_max_length,
                    truncation='do_not_truncate',
                    add_special_tokens=False
                ).input_ids
                sub_candidate_lengths.append(len(candidate_ids))

            cur_data_ids.append(sub_data_ids)

            input_ids.append(sub_input_ids)
            attention_masks.append(sub_attention_mask)

            candidate_lengths.append(sub_candidate_lengths)
 
        datasets["id"] = cur_data_ids

        datasets["input_ids"] = input_ids
        datasets["attention_mask"] = attention_masks
        datasets["labels"] = labels

        datasets["candidate_length"] = candidate_lengths

        return datasets


class EvalGSM8KPreprocessor :
    def __init__(self, 
        tokenizer: LlamaTokenizer,
        sequence_max_length: int,
    ) :       
        self.tokenizer = tokenizer
        self.sequence_max_length = sequence_max_length

    # Make few shot example prompt
    def _make_few_shot_example(self, datasets: List[Dict[str, Any]], sampled_ids: List[int]) :
        questions = datasets["question"]
        answers = datasets["answer"]

        examples = []
        for i in sampled_ids :
            question = questions[i]
            answer = answers[i]

            input_text = f"Question: {question}\nAnswer: {answer}"
            examples.append(input_text)

        few_shot_example = "\n\n".join(examples)
        return few_shot_example

    # Delete if first shot is truncated
    def _truncate(self, input_ids: List[int]) :
        input_ids = input_ids[-self.sequence_max_length:]
        input_string = self.tokenizer.decode(input_ids)

        input_shots = input_string.split("\n\n")
        start_span = "Question: "
        if input_shots[0][:len(start_span)] != start_span :
            input_shots = input_shots[1:]

        truncated_input_string = "\n\n".join(input_shots)
        truncated_input_id = self.tokenizer(
            truncated_input_string, 
            max_length=self.sequence_max_length,
            truncation='do_not_truncate',
            add_special_tokens=False
        ).input_ids

        return truncated_input_id

    def preprocess(self, datasets: List[Dict[str, Any]], num_shot: int) :
        questions = datasets["question"]
        answers = datasets["answer"]
        
        input_ids, attention_masks, labels = [], [], []

        size = len(questions)
        for i in range(size) :
            question = questions[i]
            answer = answers[i]
            
            input_text = f"Question: {question}\nAnswer: "
            target_text = answer

            # Make few shot prompt
            if num_shot > 0 :
                sampled_ids = np.random.choice(size, num_shot+1, replace=False)
                sampled_ids = list(set(sampled_ids) - set([i]))[:num_shot]
                few_shot_example = self._make_few_shot_example(datasets, sampled_ids)
                input_text = few_shot_example + "\n\n" + input_text

            # Preprocess input_text and answer
            input_id = self.tokenizer(
                input_text, 
                max_length=self.sequence_max_length,
                truncation='do_not_truncate',
                add_special_tokens=False
            ).input_ids
            if num_shot > 0 :
                input_id = self._truncate(input_id)
            attention_mask = [1]*len(input_id)

            input_ids.append(input_id)
            attention_masks.append(attention_mask)
            labels.append(target_text)

        datasets["input_ids"] = input_ids
        datasets["attention_mask"] = attention_masks
        datasets["labels"] = labels

        return datasets


class EvalTruthfulQAPreprocessor :
    def __init__(self, 
        tokenizer: LlamaTokenizer,
        sequence_max_length: int,
    ) :       
        self.tokenizer = tokenizer
        self.sequence_max_length = sequence_max_length

    def preprocess(self, datasets: List[Dict[str, Any]], num_shot: int) :
        assert num_shot == 0
        prev_data_ids = datasets["id"]
        questions = datasets["question"]
        targets_list = datasets["mc2_targets"]

        cur_data_ids = []
        input_ids, attention_masks, labels = [], [], []
        candidate_lengths = []

        size = len(questions)
        for i in range(size) :
            data_id = prev_data_ids[i]
            question = questions[i]

            targets = targets_list[i]
            target_choices, target_labels = targets["choices"], targets["labels"]

            sub_data_ids = []
            sub_input_ids, sub_attention_masks, sub_labels = [], [], []
            sub_candidate_lengths = []

            data_id_ptr = 0
            # Preprocess correct answers
            for candidate, label in zip(target_choices, target_labels) :
                input_text = question + " " + candidate
                sub_labels.append(label)

                input_id = self.tokenizer(
                    input_text, 
                    max_length=self.sequence_max_length,
                    truncation='do_not_truncate',
                    add_special_tokens=False
                ).input_ids
                attention_mask = [1]*len(input_id)

                sub_input_ids.append(input_id)
                sub_attention_masks.append(attention_mask)

                # Store candidate's token length for mc2
                candidate_ids = self.tokenizer(
                    candidate,
                    max_length=self.sequence_max_length,
                    truncation='do_not_truncate',
                    add_special_tokens=False
                ).input_ids
                sub_candidate_lengths.append(len(candidate_ids))

                sub_data_ids.append(data_id+f"-{data_id_ptr}")
                data_id_ptr += 1
        
            cur_data_ids.append(sub_data_ids)

            input_ids.append(sub_input_ids)
            attention_masks.append(sub_attention_masks)
            labels.append(sub_labels)

            candidate_lengths.append(sub_candidate_lengths)

        datasets["id"] = cur_data_ids

        datasets["input_ids"] = input_ids
        datasets["attention_mask"] = attention_masks
        datasets["labels"] = labels

        datasets["candidate_length"] = candidate_lengths

        return datasets


class EvalWinograndePreprocessor :
    def __init__(self, 
        tokenizer: LlamaTokenizer,
        sequence_max_length: int,
    ) :       
        self.tokenizer = tokenizer
        self.sequence_max_length = sequence_max_length

    # Make few shot example prompt
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
            if answer == 1 :
                input_text = sentence.replace("_", option1)
            else :
                input_text = sentence.replace("_", option2)

            input_text = "Sentence: " + input_text 
            examples.append(input_text)

        few_shot_example = "\n\n".join(examples)
        return few_shot_example

    # Delete if first shot is truncated
    def _truncate(self, input_ids: List[int]) :
        input_ids = input_ids[-self.sequence_max_length:]
        input_string = self.tokenizer.decode(input_ids)

        input_shots = input_string.split("\n\n")
        start_span = "Sentence: "
        if input_shots[0][:len(start_span)] != len(start_span) :
            input_shots = input_shots[1:]

        truncated_input_string = "\n\n".join(input_shots)

        truncated_input_string = truncated_input_string.replace("Sentence: ", "")
        truncated_input_id = self.tokenizer(
            truncated_input_string, 
            max_length=self.sequence_max_length,
            truncation='do_not_truncate',
            add_special_tokens=False
        ).input_ids

        return truncated_input_id

    def preprocess(self, datasets: List[Dict[str, Any]], num_shot: int) :
        prev_data_ids = datasets["id"]
        sentences = datasets["sentence"]
        option1s = datasets["option1"]
        option2s = datasets["option2"]
        answers = datasets["answer"]

        cur_data_ids = []
        input_ids, attention_masks, labels = [], [], []
        candidate_lengths = []

        size = len(sentences)
        for i in range(size) :
            data_id = prev_data_ids[i]
            sentence = sentences[i]
            option1 = option1s[i]
            option2 = option2s[i]

            answer = int(answers[i]) - 1
            labels.append(answer)
            
            # Make 2 sequences using option1 and option2
            candidate_idx = sentence.index("_")

            prefix_text = sentence[:candidate_idx] 
            input_text_a = option1 + sentence[candidate_idx+1:]
            input_text_b = option2 + sentence[candidate_idx+1:]

            sentence_a = self.tokenizer(
                input_text_a, 
                max_length=self.sequence_max_length,
                truncation='do_not_truncate',
                add_special_tokens=False
            ).input_ids
            sentence_b = self.tokenizer(
                input_text_b, 
                max_length=self.sequence_max_length,
                truncation='do_not_truncate',
                add_special_tokens=False
            ).input_ids
            candidate_lengths.append([len(sentence_a), len(sentence_b)])

            # Add few shot prompt
            if num_shot > 0 :
                sampled_ids = np.random.choice(size, num_shot+1, replace=False)
                sampled_ids = list(set(sampled_ids) - set([i]))[:num_shot]
                few_shot_example = self._make_few_shot_example(datasets, sampled_ids)

                input_text_a = few_shot_example + "\n\n" + prefix_text + input_text_a
                input_text_b = few_shot_example + "\n\n" + prefix_text + input_text_b

            # Preprocess input_text | option a
            input_id_a = self.tokenizer(
                input_text_a, 
                max_length=self.sequence_max_length,
                truncation='do_not_truncate',
                add_special_tokens=False
            ).input_ids
            if num_shot > 0 :
                input_id_a = self._truncate(input_id_a)
            attention_mask_a = [1]*len(input_id_a)

            # Preprocess input_text | option b
            input_id_b = self.tokenizer(
                input_text_b, 
                max_length=self.sequence_max_length,
                truncation='do_not_truncate',
                add_special_tokens=False
            ).input_ids
            if num_shot > 0 :
                input_id_b = self._truncate(input_id_b)
            attention_mask_b = [1]*len(input_id_b)

            cur_data_ids.append([data_id+f"-{0}", data_id+f"-{1}"])

            input_ids.append([input_id_a, input_id_b])
            attention_masks.append([attention_mask_a, attention_mask_b])

        datasets["id"] = cur_data_ids

        datasets["input_ids"] = input_ids
        datasets["attention_mask"] = attention_masks
        datasets["labels"] = labels

        datasets["candidate_length"] = candidate_lengths
        
        return datasets
