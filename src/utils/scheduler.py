
import optax
import jax.numpy as jnp
from typing import Callable

def create_linear_decay_lr_scheduler(
    train_ds_size: int, 
    train_batch_size: int, 
    num_train_epochs: int, 
    warmup_ratio: float, 
    learning_rate: float
) -> Callable[[int], jnp.ndarray]:

    steps_per_epoch = train_ds_size // train_batch_size
    num_train_steps = steps_per_epoch * num_train_epochs
    num_warmup_steps = int(num_train_steps * warmup_ratio)

    warmup_fn = optax.linear_schedule(init_value=0.0, end_value=learning_rate, transition_steps=num_warmup_steps)

    decay_fn = optax.linear_schedule(
        init_value=learning_rate, end_value=0, transition_steps=num_train_steps - num_warmup_steps
    )

    schedule_fn = optax.join_schedules(schedules=[warmup_fn, decay_fn], boundaries=[num_warmup_steps])
    return schedule_fn

def create_constant_lr_scheduler(
    train_ds_size: int, 
    train_batch_size: int, 
    num_train_epochs: int, 
    warmup_ratio: float, 
    learning_rate: float
) -> Callable[[int], jnp.ndarray]:

    steps_per_epoch = train_ds_size // train_batch_size
    num_train_steps = steps_per_epoch * num_train_epochs
    num_warmup_steps = int(num_train_steps * warmup_ratio)

    warmup_fn = optax.linear_schedule(init_value=0.0, end_value=learning_rate, transition_steps=num_warmup_steps)
    constant_fn = optax.linear_schedule(init_value=learning_rate, end_value=learning_rate, transition_steps=num_train_steps)

    schedule_fn = optax.join_schedules(schedules=[warmup_fn, constant_fn], boundaries=[num_warmup_steps])
    return schedule_fn