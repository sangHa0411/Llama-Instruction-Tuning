
import os
import logging
import pandas as pd
from datasets import Dataset
from typing import List
from pytz import timezone
from datetime import datetime
from datasets import load_dataset, concatenate_datasets

logging.basicConfig(format = "[%(asctime)s][%(levelname)s][Message] - %(message)s", level = logging.INFO)
logging.Formatter.converter = lambda *args: datetime.now(tz=timezone("Asia/Seoul")).timetuple()

class InstructionDatasetLoader :

    def __init__(self, random_seed: int, datasets : str, ratios : str) :
        self.random_seed = random_seed
        self.datasets = datasets[1:-1].split(",")
        self.ratios = ratios[1:-1].split(",")

        assert len(self.datasets) == len(self.ratios)

    def load(self) :
        datasets = {}
        for i, data in enumerate(self.datasets) :
 
            logging.info(f"Loading instruction dataset | {data}")

            if self.ratios[i][-1] == "%" :
                num_data = int(len(dataset) * float(self.ratios[i][:-1]))
            else :  
                num_data = int(self.ratios[i])

            if "tatsu-lab/alpaca" == data :
                dataset_path = "tatsu-lab/alpaca"
                dataset = load_dataset(dataset_path, split="train", cache_dir="/mnt/disks-standard/persist/huggingface")
                dataset = dataset.shuffle(self.random_seed).select(range(num_data))

            elif "kaist-ai/CoT-Collection" == data :
                dataset_path = "kaist-ai/CoT-Collection"
                dataset = load_dataset(dataset_path, split="train", cache_dir="/mnt/disks-standard/persist/huggingface")
                dataset = dataset.shuffle(self.random_seed).select(range(num_data))

            elif "Open-Orca/SlimOrca" == data :
                dataset_path = "Open-Orca/SlimOrca"

                # train.csv from https://www.kaggle.com/datasets/thedevastator/open-orca-slimorca-gpt-4-completions
                raw_csv_path = os.path.join("/mnt/disks-standard/persist", "slimorca", "train.csv")

                train_df = pd.read_csv(raw_csv_path)
                dataset = Dataset.from_pandas(train_df)
                dataset = dataset.shuffle(self.random_seed).select(range(num_data))

            else :
                raise NameError("Not valid dataset name")
            
            datasets[dataset_path] = dataset
        return datasets

    

class EvalDatasetLoader :

    def __init__(self, random_seed: int, datasets : str) :
        self.random_seed = random_seed
        self.datasets = datasets[1:-1].split(",")
                
    def load(self) :
        datasets = {}
        for i, data in enumerate(self.datasets) : 
            logging.info(f"Loading evaluation dataset | {data}")
            
            if "ai2_arc" == data :
                dataset_path = "ai2_arc"
                challenge_dataset = load_dataset(dataset_path, "ARC-Challenge", cache_dir="/mnt/disks-standard/persist/huggingface")
                challenge_dataset = challenge_dataset["validation"]

                easy_dataset = load_dataset(dataset_path, "ARC-Easy", cache_dir="/mnt/disks-standard/persist/huggingface")
                easy_dataset = easy_dataset["validation"]
                
                dataset = concatenate_datasets([challenge_dataset, easy_dataset])
                dataset = dataset.shuffle(self.random_seed)

            elif "Rowan/hellaswag" == data :
                dataset_path = "Rowan/hellaswag"
                dataset = load_dataset(dataset_path, split="validation", cache_dir="/mnt/disks-standard/persist/huggingface")
                dataset = dataset.shuffle(self.random_seed)

            elif "gsm8k" == data :
                dataset_path = "gsm8k"

                dataset = load_dataset(dataset_path, "main", cache_dir="/mnt/disks-standard/persist/huggingface")
                dataset = dataset["test"]
                dataset = dataset.shuffle(self.random_seed)
            else :
                raise NameError("Not valid dataset name")
            
            datasets[dataset_path] = dataset

        return datasets

    

