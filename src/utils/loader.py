
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

        assert len(self.datasets) == len(self.dataset_sizes)

    def load(self) :
        datasets = {}
        for i, dataset_name in enumerate(self.datasets) :
            logging.info(f"Loading instruction dataset | {dataset_name}")

            # This dataset is a dataset of 52,000 instructions and demonstrations generated by OpenAI's text-davinci-003 engine. 
            if "alpaca" in dataset_name :
                dataset_path = "tatsu-lab/alpaca"
                if self.cache_dir is not None :
                    dataset = load_dataset(dataset_path, split="train", cache_dir=self.cache_dir)
                else :
                    dataset = load_dataset(dataset_path, split="train")

                dataset = dataset.shuffle(self.random_seed)

            # This dataset provides 1.84 million Chain-of-Thoughts augmented across 1060 tasks from the Flan Collection.
            elif "cot-collection" in dataset_name :
                dataset_path = "kaist-ai/CoT-Collection"
                if self.cache_dir is not None :
                    dataset = load_dataset(dataset_path, split="train", cache_dir=self.cache_dir)
                else :
                    dataset = load_dataset(dataset_path, split="train")
                dataset = dataset.shuffle(self.random_seed)

            # This dataset is a new curated subset of our OpenOrca data. It has GPT's reaction to human's instruction
            elif "slimorca" in dataset_name :
                dataset_path = "Open-Orca/SlimOrca"
                if self.cache_dir is not None :
                    dataset = load_dataset(dataset_path, split="train", cache_dir=self.cache_dir)
                else :
                    dataset = load_dataset(dataset_path, split="train")
                dataset = dataset.shuffle(self.random_seed)

            # This dataset is a 10k subset of OpenOrca dataset, focusing on multiple choice questions.
            elif "openorca-mc10k" == dataset_name :
                dataset_path = "beaugogh/openorca-multiplechoice-10k"
                if self.cache_dir is not None :
                    dataset = load_dataset(dataset_path, split="train", cache_dir=self.cache_dir)
                else :
                    dataset = load_dataset(dataset_path, split="train")

                dataset = dataset.shuffle(self.random_seed)

            # This dataset is training dataset of WizardLM (Empowering large language models to follow complex instructions)
            elif "wizardlm" in dataset_name :
                dataset_path = "WizardLM/WizardLM_evol_instruct_70k"
                if self.cache_dir is not None :
                    dataset = load_dataset(dataset_path, split="train", cache_dir=self.cache_dir)
                else :
                    dataset = load_dataset(dataset_path, split="train")

                dataset = dataset.shuffle(self.random_seed)

            # This dataset is focused on improving LLM logical reasoning skills and was used to train the Platypus2 models.
            elif "open-platypus" in dataset_name :
                dataset_path = "garage-bAInd/Open-Platypus"
                if self.cache_dir is not None :
                    dataset = load_dataset(dataset_path, split="train", cache_dir=self.cache_dir)
                else :
                    dataset = load_dataset(dataset_path, split="train")

                dataset = dataset.shuffle(self.random_seed) 

            # This dataset is a massive multitask test consisting of multiple-choice questions from various branches of knowledge.
            elif "mmlu" in dataset_name :
                dataset_path = "cais/mmlu"
                if self.cache_dir is not None :
                    dataset = load_dataset(dataset_path, "all", cache_dir=self.cache_dir)
                else :
                    dataset = load_dataset(dataset_path, "all")

                dataset = dataset["auxiliary_train"]
                dataset = dataset.shuffle(self.random_seed)

            # This dataset dataset consists of 7,787 genuine grade-school level, multiple-choice science questions, assembled to encourage research in advanced question-answering.
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

            # This dataset is for commonsense NLI. (Can a Machine Really Finish Your Sentence?)
            elif "hellaswag" in dataset_name :
                dataset_path = "Rowan/hellaswag"
                if self.cache_dir is not None :
                    dataset = load_dataset(dataset_path, split="train", cache_dir=self.cache_dir)
                else :
                    dataset = load_dataset(dataset_path, split="train")

                dataset = dataset.shuffle(self.random_seed)

            # This dataset consisits of 8.5K high quality linguistically diverse grade school math word problems
            elif "gsm8k" in dataset_name :
                dataset_path = "gsm8k"
                if self.cache_dir is not None :
                    dataset = load_dataset(dataset_path, "main", cache_dir=self.cache_dir)
                else :
                    dataset = load_dataset(dataset_path, "main")

                dataset = dataset["train"]
                dataset = dataset.shuffle(self.random_seed)

            # This dataset's goal is to choose the right option for a given sentence which requires commonsense reasoning. (Formulated as a fill-in-a-blank task with binary options)
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

            if self.dataset_sizes[i] == "all" :
                num_data = len(dataset)
            elif self.dataset_sizes[i][-1] == "%" :
                num_data = int(len(dataset) * (float(self.dataset_sizes[i][:-1])/ 100))
            else :  
                num_data = int(self.dataset_sizes[i])

            dataset = dataset.select(range(num_data))
            datasets[dataset_name] = dataset
        return datasets

    

class EvaluationDatasetLoader :

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
                challenge_dataset = challenge_dataset["validation"]

                if self.cache_dir is not None :
                    easy_dataset = load_dataset(dataset_path, "ARC-Easy", cache_dir=self.cache_dir)
                else :
                    easy_dataset = load_dataset(dataset_path, "ARC-Easy")
                easy_dataset = easy_dataset["validation"]

                dataset = concatenate_datasets([challenge_dataset, easy_dataset])

            elif "mmlu" in dataset_name :
                dataset_path = "cais/mmlu"
                if self.cache_dir is not None :
                    dataset = load_dataset(dataset_path, "all", cache_dir=self.cache_dir)
                else :
                    dataset = load_dataset(dataset_path, "all")

                dataset = dataset["validation"]

            elif "hellaswag" in dataset_name :
                dataset_path = "Rowan/hellaswag"
                if self.cache_dir is not None :
                    dataset = load_dataset(dataset_path, split="validation", cache_dir=self.cache_dir)
                else :
                    dataset = load_dataset(dataset_path, split="validation")

            elif "gsm8k" in dataset_name :
                dataset_path = "gsm8k"
                if self.cache_dir is not None :
                    dataset = load_dataset(dataset_path, "main", cache_dir=self.cache_dir)
                else :
                    dataset = load_dataset(dataset_path, "main")
                dataset = dataset["test"]

            elif "truthful_qa" in dataset_name :

                dataset_path = "truthful_qa"
                if self.cache_dir is not None :
                    dataset = load_dataset(dataset_path, "multiple_choice", cache_dir=self.cache_dir)
                else :
                    dataset = load_dataset(dataset_path, "multiple_choice")
                dataset = dataset["validation"]

            elif "winogrande" in dataset_name :
                assert dataset_name in ["winogrande_xs", "winogrande_s", "winogrande_m", "winogrande_l", "winogrande_xl"]

                dataset_path = "winogrande"
                if self.cache_dir is not None :
                    dataset = load_dataset(dataset_path, dataset_name, cache_dir=self.cache_dir)
                else :
                    dataset = load_dataset(dataset_path, dataset_name)

                dataset = dataset["validation"]
                dataset_name = "winogrande"

            else :
                raise NameError("Not valid dataset name")

            # Reset data_id for evaluation dataset
            if "id" in dataset.column_names :
                dataset = dataset.remove_columns(["id"])
            dataset_ids = [f"{dataset_name}-{i}" for i in range(len(dataset))]
            dataset = dataset.add_column("id", dataset_ids)

            datasets[dataset_name] = dataset
            
        return datasets

    