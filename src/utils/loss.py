
import optax
import jax.numpy as jnp
from flax.training.common_utils import onehot

def loss_fn(logits, labels):
    vocab_size = logits.shape[-1]
    labels = onehot(labels, vocab_size)
    loss = optax.softmax_cross_entropy(logits, labels)

    loss = loss.sum() / jnp.count_nonzero(loss)
    return loss
