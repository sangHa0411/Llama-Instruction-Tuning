import jax
import optax
import logging
import numpy as np
import jax.numpy as jnp
from typing import Dict
from pytz import timezone
from datetime import datetime
from datasets import Dataset
from jaxtyping import PyTree
from utils.loss import loss_fn
from data.data_loader import data_loader
from data.collator import Seq2SeqCollator
from model.llama_model import FlaxLlaMaForCausalLM
from utils.scheduler import create_constant_lr_scheduler, create_linear_decay_lr_scheduler
from tqdm import tqdm

logging.basicConfig(format = "[%(asctime)s][%(levelname)s][Message] - %(message)s", level = logging.INFO)
logging.Formatter.converter = lambda *args: datetime.now(tz=timezone("Asia/Seoul")).timetuple()


class Trainer :

    def __init__(
        self, 
        args, 
        model: FlaxLlaMaForCausalLM, 
        params: PyTree[np.ndarray], 
        dataset: Dataset,
        data_collator: Seq2SeqCollator 
    ) :
        self.args = args
        self.model = model
        self.params = params
        self.dataset = dataset
        self.data_collator = data_collator

        # Train & Eval Batch size
        train_batch_size = args.per_device_train_batch_size

        # Scheduler
        if args.lr_scheduler_type == "constant" :
            lr_scheduler = create_constant_lr_scheduler(
                len(dataset),
                train_batch_size=train_batch_size,
                num_train_epochs=args.num_train_epochs,
                warmup_ratio=args.warmup_ratio,
                learning_rate=args.learning_rate
            )
        elif args.lr_scheduler_type == "linear" :
            lr_scheduler = create_linear_decay_lr_scheduler(
                len(dataset),
                train_batch_size=train_batch_size,
                num_train_epochs=args.num_train_epochs,
                warmup_ratio=args.warmup_ratio,
                learning_rate=args.learning_rate
            )
        else :
            raise NotImplementedError("Not implemented lr scheduelr type")
        self.lr_scheduler = lr_scheduler

        # Optimizer
        optimizer = optax.adamw(
            learning_rate=lr_scheduler,
            weight_decay=args.weight_decay,
            mu_dtype=jnp.bfloat16
        )
        self.optimizer = optimizer

    def train(self, ) :

        optimizer_update = self.optimizer.update
        opt_state = self.optimizer.init(self.params)
        jax_params = self.params
        jax_model = self.model

        @jax.jit
        def train_step(
            params,
            opt_state,
            dropout_rng: jax.random.PRNGKey,
            batch: Dict[str, np.ndarray], 
            step: int,
            ):
            dropout_rng, new_dropout_rng = jax.random.split(dropout_rng)

            def compute_loss(params):
                labels = batch.pop("labels")
                logits = jax_model(**batch, params=params, dropout_rng=dropout_rng, train=True)[0]
                loss = loss_fn(logits, labels)
                return loss

            grad_fn = jax.value_and_grad(compute_loss)
            loss, grads = grad_fn(params)

            updates, new_opt_state = optimizer_update(grads, opt_state, params)
            new_params = optax.apply_updates(params, updates)

            metrics = {"loss": loss, "learning_rate": self.lr_scheduler(step)}
            return new_params, tuple(new_opt_state), new_dropout_rng, metrics

        rng = jax.random.PRNGKey(self.args.random_seed)
        rng, dropout_rng = jax.random.split(rng)

        training_step_ptr = 0

        num_epoch = self.args.num_train_epochs
        train_batch_size = self.args.per_device_train_batch_size

        logging.info("Training Starts")
        for epoch in tqdm(range(num_epoch)) :

            rng, dropout_rng = jax.random.split(rng)
            train_loader = data_loader(
                rng=dropout_rng, 
                dataset=self.dataset, 
                data_collator=self.data_collator,
                batch_size=train_batch_size, 
                shuffle=True,
                drop_last=True
            )

            train_steps = len(self.dataset["input_ids"]) // train_batch_size
            with tqdm(total=train_steps, desc="Training", leave=False) as progress_bar_train :
                for train_data in train_loader :
                    
                    jax_params, opt_state, dropout_rng, train_metric = train_step(
                        params=jax_params, 
                        opt_state=opt_state, 
                        dropout_rng=dropout_rng,
                        batch=train_data, 
                        step=training_step_ptr
                    )

                    training_step_ptr += 1
                    progress_bar_train.update(1)

                    if training_step_ptr % self.args.logging_steps == 0 :
                        logging.info(f"Train [Step : {training_step_ptr}] | Loss: {round(train_metric['loss'].mean(), 3)}, Learning Rate: {round(train_metric['learning_rate'].mean(), 9)}")

