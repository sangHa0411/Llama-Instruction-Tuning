
import jax
import math
import numpy as np
from datasets import Dataset
from utils.collator import Seq2SeqCollator


def data_loader(
    rng: jax.random.PRNGKey, 
    dataset: Dataset, 
    data_collator: Seq2SeqCollator, 
    batch_size: int, 
    shuffle: bool = False, 
    drop_last=True
    ) :

    if shuffle:
        batch_idx = jax.random.permutation(rng, len(dataset))
        batch_idx = np.asarray(batch_idx)
    else:
        batch_idx = np.arange(len(dataset))

    if drop_last:
        steps_per_epoch = len(dataset) // batch_size
        batch_idx = batch_idx[: steps_per_epoch * batch_size]  # Skip incomplete batch.
        batch_idx = batch_idx.reshape((steps_per_epoch, batch_size))
    else:
        steps_per_epoch = math.ceil(len(dataset) / batch_size)
        batch_idx = np.array_split(batch_idx, steps_per_epoch)

    for idx in batch_idx:
        batch = dataset[idx]
        batch = data_collator(batch)
        batch = {k: np.array(v) for k, v in batch.items()}

        yield batch