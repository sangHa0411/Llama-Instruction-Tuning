
import sys
from transformers import LlamaTokenizer

# Add source code path to system paths
sys.path.append("/home/sangha110495/project/Llama-Instruction-Tuning/src")

from utils.loader import InstructionDatasetLoader
from utils.preprocessor import InstructionDatasetPreprocessor


def test_preprocessor() :

    random_seed = 42
    sequence_max_length = 1024
    datasets = "[alpaca,cot-collection,slimorca,openorca-mc10k,wizardlm,open-platypus,mmlu,arc,hellaswag,gsm8k,winogrande_l]"
    dataset_sizes = "[100,100,100,100,100,100,100,100,100,100,100]"
    model_path = "/mnt/disks-standard/persist/llama/llama-2-7b-hf"
    cache_dir = "/mnt/disks-standard/persist/huggingface"

    # Load tokenizer
    tokenizer = LlamaTokenizer.from_pretrained(model_path)
    tokenizer.pad_token_id = 0
    tokenizer.padding_side = "left"

    train_dataset_loader = InstructionDatasetLoader(
        random_seed=random_seed, 
        datasets=datasets, 
        dataset_sizes=dataset_sizes, 
        cache_dir=cache_dir
    )
    train_datasets = train_dataset_loader.load()
    train_prerprocessor = InstructionDatasetPreprocessor(tokenizer, sequence_max_length)
    train_encoded_dataset = train_prerprocessor(train_datasets)

    for data in train_encoded_dataset :
        input_ids = data["input_ids"]
        attention_mask = data["attention_mask"]
        label = data["labels"]

        # Test if length of dataset content are all same
        assert len(input_ids) == len(attention_mask)
        assert len(input_ids) == len(label)

        sequence_max_length = len(input_ids)
        for i in range(1, sequence_max_length) :
            # Test if labels format is autoregressive given input_ids
            if label[i-1] != -100 :
                assert input_ids[i] == label[i-1]
