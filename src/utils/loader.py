
import os
import logging
import pandas as pd
from datasets import Dataset
from typing import List, Optional
from pytz import timezone
from datetime import datetime
from datasets import load_dataset, concatenate_datasets

logging.basicConfig(format = "[%(asctime)s][%(levelname)s][Message] - %(message)s", level = logging.INFO)
logging.Formatter.converter = lambda *args: datetime.now(tz=timezone("Asia/Seoul")).timetuple()

class InstructionDatasetLoader :

    def __init__(self, random_seed: int, datasets: str, dataset_sizes: str, cache_dir: Optional[str]=None) :
        self.random_seed = random_seed
        self.datasets = datasets[1:-1].split(",")
        self.dataset_sizes = dataset_sizes[1:-1].split(",")
        self.cache_dir = cache_dir

        assert len(self.datasets) == len(self.ratios)

    def load(self) :
        datasets = {}
        for i, dataset_name in enumerate(self.datasets) :
            logging.info(f"Loading instruction dataset | {dataset_name}")

            if "alpaca" in dataset_name :
                dataset_path = "tatsu-lab/alpaca"
                if self.cache_dir is not None :
                    dataset = load_dataset(dataset_path, split="train", cache_dir=self.cache_dir)
                else :
                    dataset = load_dataset(dataset_path, split="train")

                dataset = dataset.shuffle(self.random_seed)

            elif "cot-collection" in dataset_name :
                dataset_path = "kaist-ai/CoT-Collection"
                if self.cache_dir is not None :
                    dataset = load_dataset(dataset_path, split="train", cache_dir=self.cache_dir)
                else :
                    dataset = load_dataset(dataset_path, split="train")
                dataset = dataset.shuffle(self.random_seed)

            elif "slimorca" in dataset_name :
                dataset_path = "Open-Orca/SlimOrca"
                if self.cache_dir is not None :
                    dataset = load_dataset(dataset_path, split="train", cache_dir=self.cache_dir)
                else :
                    dataset = load_dataset(dataset_path, split="train")
                dataset = dataset.shuffle(self.random_seed)

            elif "openorca-multiplechoice" == dataset_name :
                dataset_path = "beaugogh/openorca-multiplechoice-10k"
                if self.cache_dir is not None :
                    dataset = load_dataset(dataset_path, split="train", cache_dir=self.cache_dir)
                else :
                    dataset = load_dataset(dataset_path, split="train")

                dataset = dataset.shuffle(self.random_seed)

            elif "alpaca" in dataset_name :
                dataset_path = "tatsu-lab/alpaca"
                if self.cache_dir is not None :
                    dataset = load_dataset(dataset_path, split="train", cache_dir=self.cache_dir)
                else :
                    dataset = load_dataset(dataset_path, split="train")

                dataset = dataset.shuffle(self.random_seed)

            elif "mmlu" in dataset_name :
                dataset_path = "cais/mmlu"
                if self.cache_dir is not None :
                    dataset = load_dataset(dataset_path, "all", cache_dir=self.cache_dir)
                else :
                    dataset = load_dataset(dataset_path, "all")

                dataset = dataset["auxiliary_train"]
                dataset = dataset.shuffle(self.random_seed)

            elif "arc" in dataset_name :
                dataset_path = "ai2_arc"
                if self.cache_dir is not None :
                    challenge_dataset = load_dataset(dataset_path, "ARC-Challenge", cache_dir=self.cache_dir)
                else :
                    challenge_dataset = load_dataset(dataset_path, "ARC-Challenge")
                challenge_dataset = challenge_dataset["train"]

                if self.cache_dir is not None :
                    easy_dataset = load_dataset(dataset_path, "ARC-Easy", cache_dir=self.cache_dir)
                else :
                    easy_dataset = load_dataset(dataset_path, "ARC-Easy")
                easy_dataset = easy_dataset["train"]

                dataset = concatenate_datasets([challenge_dataset, easy_dataset])
                dataset = dataset.shuffle(self.random_seed)

            elif "gsm8k" in dataset_name :
                dataset_path = "gsm8k"
                if self.cache_dir is not None :
                    dataset = load_dataset(dataset_path, "main", cache_dir=self.cache_dir)
                else :
                    dataset = load_dataset(dataset_path, "main")

                dataset = dataset["train"]
                dataset = dataset.shuffle(self.random_seed)

            elif "winogrande" in dataset_name :
                assert dataset_name in ["winogrande_xs", "winogrande_s", "winogrande_m", "winogrande_l", "winogrande_xl"]

                dataset_path = "winogrande"
                if self.cache_dir is not None :
                    dataset = load_dataset(dataset_path, dataset_name, cache_dir=self.cache_dir)
                else :
                    dataset = load_dataset(dataset_path, dataset_name)

                dataset = dataset["train"]
                dataset = dataset.shuffle(self.random_seed)
                dataset_name = "winogrande"

            else :
                raise NameError("Not valid dataset name")

            if self.ratios[i] == "all" :
                num_data = len(dataset)
            elif self.ratios[i][-1] == "%" :
                num_data = int(len(dataset) * float(self.ratios[i][:-1]))
            else :  
                num_data = int(self.ratios[i])

            dataset = dataset.select(range(num_data))
            datasets[dataset_name] = dataset
        return datasets

    

class EvalDatasetLoader :

    def __init__(self, datasets: str, cache_dir: Optional[str]=None) :
        self.datasets = datasets[1:-1].split(",")
        self.cache_dir = cache_dir
                
    def load(self) :
        datasets = {}
        for i, dataset_name in enumerate(self.datasets) : 
            logging.info(f"Loading evaluation dataset | {dataset_name}")
            
            if "arc" in dataset_name :
                dataset_path = "ai2_arc"
                if self.cache_dir is not None :
                    challenge_dataset = load_dataset(dataset_path, "ARC-Challenge", cache_dir=self.cache_dir)
                else :
                    challenge_dataset = load_dataset(dataset_path, "ARC-Challenge")
                challenge_dataset = challenge_dataset["test"]

                if self.cache_dir is not None :
                    easy_dataset = load_dataset(dataset_path, "ARC-Easy", cache_dir=self.cache_dir)
                else :
                    easy_dataset = load_dataset(dataset_path, "ARC-Easy")
                easy_dataset = easy_dataset["test"]

                dataset = concatenate_datasets([challenge_dataset, easy_dataset])

            elif "mmlu" in dataset_name :
                dataset_path = "cais/mmlu"
                if self.cache_dir is not None :
                    dataset = load_dataset(dataset_path, "all", cache_dir=self.cache_dir)
                else :
                    dataset = load_dataset(dataset_path, "all")

                dataset = dataset["test"]

            elif "hellaswag" in dataset_name :
                dataset_path = "Rowan/hellaswag"
                if self.cache_dir is not None :
                    dataset = load_dataset(dataset_path, split="test", cache_dir=self.cache_dir)
                else :
                    dataset = load_dataset(dataset_path, split="test")

            elif "gsm8k" in dataset_name :
                dataset_path = "gsm8k"
                if self.cache_dir is not None :
                    dataset = load_dataset(dataset_path, "main", cache_dir=self.cache_dir)
                else :
                    dataset = load_dataset(dataset_path, "main")
                dataset = dataset["test"]

            elif "truthful_qa" in dataset_name :
                assert dataset_name in ["truthful_qa-generation", "truthful_qa-multiple_choice"]

                category = dataset_name.split("-")[1]
                dataset_path = "truthful_qa"
                if self.cache_dir is not None :
                    dataset = load_dataset(dataset_path, category, cache_dir=self.cache_dir)
                else :
                    dataset = load_dataset(dataset_path, category)

                dataset = dataset["test"]

            elif "winogrande" in dataset_name :
                assert dataset_name in ["winogrande_xs", "winogrande_s", "winogrande_m", "winogrande_l", "winogrande_xl"]

                dataset_path = "winogrande"
                if self.cache_dir is not None :
                    dataset = load_dataset(dataset_path, dataset_name, cache_dir=self.cache_dir)
                else :
                    dataset = load_dataset(dataset_path, dataset_name)

                dataset = dataset["test"]
                dataset_name = "winogrande"

            else :
                raise NameError("Not valid dataset name")

            datasets[dataset_name] = dataset
            
        return datasets

    

