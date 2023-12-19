
import sys
from transformers import LlamaTokenizer
from datasets import load_dataset

# Add source code path to system paths
sys.path.append("/home/sangha110495/project/Llama-Instruction-Tuning/src")

from utils.loader import EvaluationDatasetLoader
from eval.eval_preprocessor import EvaluationDatasetPreprocessor


def test_eval_preprocessor() :

    sequence_max_length = 1024
    model_path = "/mnt/disks-standard/persist/llama/llama-2-7b-hf"

    datasets = "[mmlu,arc,hellaswag,gsm8k,truthful_qa-multiple_choice,winogrande_l]"
    num_shots = "[5,5,5,5,5,5]"
    cache_dir = "/mnt/disks-standard/persist/huggingface"

    # Test if loader get dataset without error
    eval_dataset_loader = EvaluationDatasetLoader(
        datasets=datasets, 
        cache_dir=cache_dir
    )
    evaluation_datasets = eval_dataset_loader.load()

    # Load tokenizer
    tokenizer = LlamaTokenizer.from_pretrained(model_path)
    tokenizer.pad_token_id = 0
    tokenizer.padding_side = "left"

    eval_preprocessor = EvaluationDatasetPreprocessor(tokenizer=tokenizer, sequence_max_length=sequence_max_length)
    eval_encoded_datasets = eval_preprocessor(num_shots, evaluation_datasets)

    for dataset_name in eval_encoded_datasets :
        dataset = eval_encoded_datasets[dataset_name]
        dataset = dataset.select(range(100))

        input_ids = dataset["input_ids"]
        attention_masks = dataset["attention_mask"]
        labels = dataset["labels"]

        for i in range(len(dataset)) :
            input_id = input_ids[i]
            attention_mask = attention_masks[i]
            label = labels[i]

            # Test if length of input_ids and attention_mask is same
            assert len(input_id) == len(attention_mask)
            # Test if type of label is list of string or string
            if isinstance(label, list) :
                assert isinstance(label[0], str)
            else :
                assert isinstance(label, str)

            # Test if all few-shot example are correctly truncated and formatted well
            decoded = tokenizer.decode(input_id)
            first_char, last_char = decoded[0], decoded[-2:]
            assert first_char.isupper()
            assert last_char == ": "
