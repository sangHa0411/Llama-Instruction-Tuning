
import sys
from datasets import load_dataset, concatenate_datasets

# Add source code path to system paths
sys.path.append("/home/sangha110495/project/Llama-Instruction-Tuning/src")

from utils.loader import InstructionDatasetLoader, EvaluationDatasetLoader

def test_instruction_dataset_loader() :

    random_seed = 42
    datasets = "[alpaca,cot-collection,slimorca,openorca-mc10k,wizardlm,open-platypus,mmlu,arc,hellaswag,gsm8k,winogrande_l]"
    dataset_sizes = "[all,100,10%,100,20%,100,200,20%,15%,300,all]"
    cache_dir = "/mnt/disks-standard/persist/huggingface"

    dataset_dict = {
        "alpaca" : load_dataset("tatsu-lab/alpaca", split="train", cache_dir=cache_dir),
        "cot-collection" : load_dataset("kaist-ai/CoT-Collection", split="train", cache_dir=cache_dir),
        "slimorca" : load_dataset("Open-Orca/SlimOrca", split="train", cache_dir=cache_dir),
        "openorca-mc10k" : load_dataset("beaugogh/openorca-multiplechoice-10k", split="train", cache_dir=cache_dir),
        "wizardlm" : load_dataset("WizardLM/WizardLM_evol_instruct_70k", split="train", cache_dir=cache_dir),
        "open-platypus" : load_dataset("garage-bAInd/Open-Platypus", split="train", cache_dir=cache_dir),
        "mmlu" : load_dataset("cais/mmlu", "all", cache_dir=cache_dir)["auxiliary_train"],
        "arc" : concatenate_datasets(
            [
                load_dataset("ai2_arc", "ARC-Challenge", cache_dir=cache_dir)["train"],
                load_dataset("ai2_arc", "ARC-Easy", cache_dir=cache_dir)["train"]
            ]
        ),
        "hellaswag" : load_dataset("Rowan/hellaswag", split="train", cache_dir=cache_dir),
        "gsm8k" : load_dataset("gsm8k", "main", cache_dir=cache_dir)["train"],
        "winogrande" : load_dataset("winogrande", "winogrande_l", cache_dir=cache_dir)["train"]
    }

    # Test if loader get dataset without error
    train_dataset_loader = InstructionDatasetLoader(
        random_seed=random_seed, 
        datasets=datasets, 
        dataset_sizes=dataset_sizes, 
        cache_dir=cache_dir
    )
    train_datasets = train_dataset_loader.load()

    dataset_size_splited = dataset_sizes[1:-1].split(",")
    for i, dataset_name in enumerate(train_datasets) :
        size = dataset_size_splited[i]

        # Compare to original dataset and test if datasets are corretly sampled
        org_dataset = dataset_dict[dataset_name]

        if size == "all" :
            assert len(train_datasets[dataset_name]) == len(org_dataset)
        elif size[-1] == "%" :
            ratio = int(size[:-1]) / 100
            assert len(train_datasets[dataset_name]) == int(len(org_dataset) * ratio)
        else :
            assert len(train_datasets[dataset_name]) == int(size)



def test_eval_dataset_loader() :

    datasets = "[mmlu,arc,hellaswag,gsm8k,truthful_qa-multiple_choice,winogrande_l]"
    cache_dir = "/mnt/disks-standard/persist/huggingface"

    dataset_dict = {
        "mmlu" : load_dataset("cais/mmlu", "all", cache_dir=cache_dir)["test"],
        "arc" : concatenate_datasets(
            [
                load_dataset("ai2_arc", "ARC-Challenge", cache_dir=cache_dir)["test"],
                load_dataset("ai2_arc", "ARC-Easy", cache_dir=cache_dir)["test"]
            ]
        ),
        "hellaswag" : load_dataset("Rowan/hellaswag", split="test", cache_dir=cache_dir),
        "gsm8k" : load_dataset("gsm8k", "main", cache_dir=cache_dir)["test"],
        "truthful_qa-multiple_choice" : load_dataset("truthful_qa", "multiple_choice", cache_dir=cache_dir)["validation"],
        "winogrande" : load_dataset("winogrande", "winogrande_l", cache_dir=cache_dir)["test"]
    }

    # Test if loader get dataset without error
    eval_dataset_loader = EvaluationDatasetLoader(
        datasets=datasets, 
        cache_dir=cache_dir
    )
    evaluation_datasets = eval_dataset_loader.load()

    for dataset_name in evaluation_datasets :
        # Compare to original dataset and test if datasets are fully loaded
        org_dataset = dataset_dict[dataset_name]
        assert len(evaluation_datasets[dataset_name]) == len(org_dataset)


        