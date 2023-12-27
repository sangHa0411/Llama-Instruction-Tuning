
import optax
import jax.numpy as jnp
from flax.training.common_utils import onehot

def loss_fn(logits, labels):
    vocab_size = logits.shape[-1]
    label_tokens = labels

    labels = onehot(labels, vocab_size)
    loss = optax.softmax_cross_entropy(logits, labels)
    loss = loss.sum() / jnp.count_nonzero(loss)

    pred_tokens = jnp.argmax(logits, -1)
    
    valid_tokens = label_tokens != -100
    correct = pred_tokens == label_tokens

    correct = correct * valid_tokens
    acc = correct.sum() / valid_tokens.sum()
    return loss, acc
