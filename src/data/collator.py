import jax.numpy as jnp
from dataclasses import dataclass
from transformers import T5Tokenizer

@dataclass
class Seq2SeqCollator:
    tokenizer: T5Tokenizer
    sequence_max_length: int = 2048
    label_pad_token_id : int = -100

    def pad(self, seq_ids, pad_value, max_length):

        seq_ids_len = len(seq_ids)
        if seq_ids_len < max_length :
            pad_len = max_length - seq_ids_len

            pad_ids = [pad_value for _ in range(pad_len)]
            if self.tokenizer.padding_side == "right" :
                seq_ids = seq_ids + pad_ids
            elif self.tokenizer.padding_side == "left" :
                seq_ids = pad_ids + seq_ids
            else :
                raise NameError("Tokenizer padding size should be left or right")

        else :
            seq_ids = seq_ids[-max_length:]

        return seq_ids
    
    def __call__(self, features):
        collated = {}

        columns = list(features.keys())
        for column in columns :
            batch = features[column]

            if column == "input_ids" :
                pad_token_id = self.tokenizer.pad_token_id
                max_length = self.sequence_max_length
            elif column == "attention_mask" :
                pad_token_id = 0
                max_length = self.sequence_max_length
            elif column == "labels" :
                pad_token_id = self.label_pad_token_id
                max_length = self.sequence_max_length
            else :
                continue
            
            batch_ids = [self.pad(b, pad_token_id, max_length) for b in batch]
            batch_array = jnp.array(batch_ids, dtype=jnp.int32)

            collated[column] = batch_array

        return collated
