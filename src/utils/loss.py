
import optax
from flax.training.common_utils import onehot

def loss_fn(logits, labels):
    target_tokens = labels != -100

    vocab_size = logits.shape[-1]
    labels = onehot(labels, vocab_size)
    loss = optax.softmax_cross_entropy(logits, labels)

    loss = loss.sum() / target_tokens.sum()
    return loss
