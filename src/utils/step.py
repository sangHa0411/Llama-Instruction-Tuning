
import jax
import optax
import numpy as np
import jax.numpy as jnp
from functools import partial
from utils.loss import loss_fn
from optax import GradientTransformation
from flax.training.train_state import TrainState
from transformers import LlamaTokenizer
from model.llama_model import FlaxLlaMaForCausalLM
from typing import Callable, Dict, Any

def train_step(
    params,
    opt_state,
    model: FlaxLlaMaForCausalLM,
    dropout_rng: jax.random.PRNGKey,
    batch: Dict[str, np.ndarray], 
    optimizer: GradientTransformation,
    ):

    dropout_rng, new_dropout_rng = jax.random.split(dropout_rng)

    def compute_loss(params):
        labels = batch.pop("labels")
        logits = model(**batch, params=params, dropout_rng=dropout_rng, train=True)[0]
        padding_mask = labels == -100
        loss = loss_fn(logits, labels, padding_mask)
        return loss

    grad_fn = jax.value_and_grad(compute_loss)
    loss, grads = grad_fn(params)

    updates, new_opt_state = optimizer.update(grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)
    new_params = params

    metrics = {"loss": loss}
    return new_params, tuple(new_opt_state), new_dropout_rng, metrics


# Define eval fn
def eval_step(
    params, 
    batch: Dict[str, np.ndarray], 
    model: FlaxLlaMaForCausalLM 
    ):

    labels = batch.pop("labels")
    logits = model(**batch, params=params, train=False)[0]

    loss = loss_fn(logits, labels, batch["decoder_attention_mask"])
    metrics = {"loss": loss}
    return metrics


# Define generation function
def generate_step(
    params, 
    batch: Dict[str, np.ndarray], 
    model: FlaxLlaMaForCausalLM, 
    gen_kwargs: Dict[str, Any]
    ):

    model.params = params
    output_ids = model.generate(
        batch["input_ids"], 
        attention_mask=batch["attention_mask"],
        **gen_kwargs,
    )
    return output_ids.sequences