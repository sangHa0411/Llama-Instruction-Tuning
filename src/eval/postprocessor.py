import logging
from typing import Dict, List, Any
from pytz import timezone
from datetime import datetime
from datasets import Dataset
from transformers import LlamaTokenizer

logging.basicConfig(format = "[%(asctime)s][%(levelname)s][Message] - %(message)s", level = logging.INFO)
logging.Formatter.converter = lambda *args: datetime.now(tz=timezone("Asia/Seoul")).timetuple()

class Postprocessor :

    def __init__(self, tokenizer: LlamaTokenizer) :
        self.tokenizer = tokenizer

    def postprocess_generations(
        self, 
        dataset: Dataset, 
        predictions: Dict[str, Any], 
        labels: List[str]
    ) -> Dict[str, Any]:
        results = {}

        generations = predictions["generation"]
        for i in range(len(generations)) :
            data = dataset[i]

            data_id = data["id"]
            generation = generations[i]
            label = labels[i]

            results[data_id] = {"generation" : generation, "label" : label}

        return results

    def postprocess_multiplce_choice(
        self, 
        dataset: Dataset,
        predictions: Dict[str, Any],
        labels: List[str],
    ) -> Dict[str, Any] :
        results = {}
        sequence_log_probs = predictions["sequence_log_prob"]
        label_ptr = 0
        for i in range(len(sequence_log_probs)) :
            data = dataset[i]

            data_id = data["id"]
            dataset_name, data_num, data_count = data_id.split("-")
            data_count = int(data_count)

            candidate_length, input_ids = data["candidate_length"], data["input_ids"]

            candidate_ids = input_ids[-candidate_length:]
            candidate_tokens = self.tokenizer.decode(candidate_ids)

            byte_length = len(candidate_tokens)

            sequence_log_prob = sequence_log_probs[i][-candidate_length:]
            sequence_log_prob = -sequence_log_prob.sum().item()

            map_key = f"{dataset_name}-{data_num}"
            if data_count == 0 :
                results[map_key] = {
                    "log_prob" : [sequence_log_prob],
                    "normalized_log_prob" : [sequence_log_prob / byte_length],
                    "label" : labels[label_ptr]
                }

                label_ptr += 1
            else :
                results[map_key]["log_prob"].append(sequence_log_prob)   
                results[map_key]["normalized_log_prob"].append(sequence_log_prob / byte_length)   

        return results 

    def postprocess_tfqa_mc2(
        self, 
        dataset: Dataset,
        predictions: Dict[str, Any],
        labels: List[str],
    ) -> Dict[str, Any] :
        labels = sum(labels, [])

        results = {}
        sequence_log_probs = predictions["sequence_log_prob"]
        for i in range(len(sequence_log_probs)) :
            data = dataset[i]

            data_id = data["id"]
            dataset_name, data_id, data_count = data_id.split("-")
            data_count = int(data_count)

            candidate_length, input_ids = data["candidate_length"], data["input_ids"]

            sequence_log_prob = sequence_log_probs[i][-candidate_length:]
            sequence_log_prob = -sequence_log_prob.sum().item()

            candidate_ids = input_ids[-candidate_length:]
            candidate_tokens = self.tokenizer.decode(candidate_ids)

            byte_length = len(candidate_tokens)

            sequence_log_prob = sequence_log_prob / byte_length
            label = labels[i]

            map_key = f"{dataset_name}-{data_id}"
            if data_count == 0 :
                results[map_key] = {
                    "log_prob" : [sequence_log_prob],
                    "label" : [label]
                }
            else :
                results[map_key]["log_prob"].append(sequence_log_prob)   
                results[map_key]["label"].append(label)   

        return results 