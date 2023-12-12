import jax
import optax
import logging
import numpy as np
import jax.numpy as jnp
from typing import Dict, Any, Tuple
from pytz import timezone
from datetime import datetime
from datasets import Dataset
from jaxtyping import PyTree
from utils.loss import loss_fn
from data.data_loader import data_loader
from data.collator import Seq2SeqCollator
from eval.metrics import InstructionMetrics
from model.llama_model import FlaxLlaMaForCausalLM
from transformers import LlamaTokenizer
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
        tokenizer: LlamaTokenizer,
        dataset: Dataset,
        eval_datasets: Dict[str, Dataset],
        data_collator: Seq2SeqCollator 
    ) :

        self.args = args
        self.model = model
        self.params = params
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.eval_datasets = eval_datasets
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

    def evaluate(self, ) :
        jax_params = self.params
        jax_model = self.model

        insturction_metrics = InstructionMetrics()

        @jax.jit
        def generate_step(params, input_ids: jnp.ndarray, sequence_length: int):

            pad_ids = jnp.ones((1, self.args.generation_max_length), dtype=jnp.int32) * self.tokenizer.pad_token_id
            input_ids = jnp.concatenate((input_ids, pad_ids), axis=1)

            def body_fn(n: int, input_state: Tuple[jnp.ndarray, int]) :
                sub_input_ids, sequence_last_pos = input_state
                logits = jax_model(input_ids=sub_input_ids, params=params, train=False)[0]

                pred = jnp.argmax(logits[0, sequence_last_pos, :])
                sub_input_ids = sub_input_ids.at[0, sequence_last_pos+1].set(pred)

                input_state = (sub_input_ids, sequence_last_pos+1)
                return input_state

            input_state = (input_ids, sequence_length-1)
            output_state = jax.lax.fori_loop(lower=0, upper=self.args.generation_max_length, body_fun=body_fn, init_val=input_state)
            return output_state

        rng = jax.random.PRNGKey(self.args.random_seed)
        rng, dropout_rng = jax.random.split(rng)

        logging.info("Evaluation Starts")
        for dataset_name in self.eval_datasets :
            eval_dataset = self.eval_datasets[dataset_name]
            logging.info(f"Evaluation dataset name : {dataset_name} | dataset information : {eval_dataset}")

            eval_labels = eval_dataset["labels"]
            eval_dataset = eval_dataset.remove_columns(["labels"])

            eval_loader = data_loader(
                rng=dropout_rng, 
                dataset=eval_dataset, 
                data_collator=self.data_collator,
                batch_size=1, 
                shuffle=False,
                drop_last=False
            )

            eval_predictions = []

            eval_steps = len(eval_labels)
            with tqdm(total=eval_steps, desc="Evaluation", leave=False) as progress_bar_eval :

                for eval_data in tqdm(eval_loader) :
                    input_ids = eval_data["input_ids"]
                    sequence_length = input_ids[0].tolist().index(self.tokenizer.pad_token_id) - 1

                    output_state = generate_step(jax_params, input_ids, sequence_length)

                    output_ids, sequence_end_length = output_state
                    output_sequence = self.tokenizer.decode(output_ids[0][sequence_length+1:sequence_end_length])
                    output_sequence = output_sequence.split("\n\n\n\n")[0]

                    eval_predictions.append(output_sequence)

                    progress_bar_eval.update(1)

                if dataset_name == "ai2_arc" or dataset_name == "Rowan/hellaswag" :
                    metric = insturction_metrics.get_multiple_choice_accuracy(eval_predictions, eval_labels)
                elif dataset_name == "gsm8k" :
                    metric = insturction_metrics.get_gsm8k_accuracy(eval_predictions, eval_labels)
                elif dataset_name == "truthful_qa-generation" :
                    metric = insturction_metrics.get_truthful_qa_blue(eval_predictions, eval_labels)
                else :
                    raise NameError("Not valid evaluation dataset name")

                logging.info(f"Evaluation dataset name : {dataset_name} | Accuracy : {metric}")

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
                        logging.info(f"Train [Step : %s] | Loss: %e, Learning Rate: %e" %(training_step_ptr, train_metric["loss"].mean(), train_metric["learning_rate"]))

                    if self.args.evaluation_strategy == "steps" :
                        if training_step_ptr % self.args.eval_steps == 0 :
                            self.evaluate()

            if self.args.evaluation_strategy == "epoch" :
                self.evaluate()
