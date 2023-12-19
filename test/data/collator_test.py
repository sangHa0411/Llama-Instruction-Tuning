import sys
from transformers import LlamaTokenizer

# Add source code path to system paths
sys.path.append("/home/sangha110495/project/Llama-Instruction-Tuning/src")

from data.collator import Seq2SeqCollator

def test_dataset_collator() :

    sequence_max_length = 1024
    model_path = "/mnt/disks-standard/persist/llama/llama-2-7b-hf"

    # Load tokenizer
    tokenizer = LlamaTokenizer.from_pretrained(model_path)
    tokenizer.pad_token_id = 0
    tokenizer.padding_side = "left"

    # Tokenizing input example
    examples = ["Who are you?", "I like you"]
    tokenized_examples = tokenizer(examples)

    data_collator = Seq2SeqCollator(tokenizer, sequence_max_length=sequence_max_length)
    collated_example = data_collator(tokenized_examples)

    # Test if tokenized example padded correctly | padding side is left
    assert collated_example["input_ids"].shape[1] == sequence_max_length
    assert collated_example["attention_mask"].shape[1] == sequence_max_length

    for i in range(len(collated_example["input_ids"])) :
        input_id = collated_example["input_ids"][i]
        attention_mask = collated_example["attention_mask"][i]

        padding_size = sequence_max_length - len(tokenized_examples["input_ids"][i])
        for k in range(padding_size) :
            assert input_id[k] == tokenizer.pad_token_id
            assert attention_mask[k] == 0

    # Test if tokenized example padded correctly | padding side is right
    tokenizer.padding_side = "right"
    data_collator = Seq2SeqCollator(tokenizer, sequence_max_length=sequence_max_length)
    collated_example = data_collator(tokenized_examples)

    for i in range(len(collated_example["input_ids"])) :
        input_id = collated_example["input_ids"][i]
        attention_mask = collated_example["attention_mask"][i]

        for k in range(len(tokenized_examples["input_ids"][i]), sequence_max_length) :
            assert input_id[k] == tokenizer.pad_token_id
            assert attention_mask[k] == 0
