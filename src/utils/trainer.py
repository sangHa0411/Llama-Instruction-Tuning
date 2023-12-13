import jax
import optax
import logging
import numpy as np
import jax.numpy as jnp
from typing import Dict, Any, Tuple
from jax.sharding import Mesh
from pytz import timezone
from datetime import datetime
from datasets import Dataset
from jaxtyping import PyTree
from utils.loss import loss_fn
from jax.sharding import PartitionSpec as PS
from data.data_loader import data_loader
from data.collator import Seq2SeqCollator
from eval.metrics import InstructionMetrics
from model.llama_model import FlaxLlaMaForCausalLM
from transformers import LlamaTokenizer
from transformers.generation import GenerationConfig
from model.partitions import with_named_sharding_constraint
from utils.scheduler import create_constant_lr_scheduler, create_linear_decay_lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

logging.basicConfig(format = "[%(asctime)s][%(levelname)s][Message] - %(message)s", level = logging.INFO)
logging.Formatter.converter = lambda *args: datetime.now(tz=timezone("Asia/Seoul")).timetuple()


class Trainer :

    def __init__(
        self, 
        args, 
        model: FlaxLlaMaForCausalLM, 
        params: PyTree[np.ndarray], 
        mesh: Mesh,
        tokenizer: LlamaTokenizer,
        dataset: Dataset,
        eval_datasets: Dict[str, Dataset],
        data_collator: Seq2SeqCollator 
    ) :

        self.args = args
        self.model = model
        self.params = params
        self.mesh = mesh
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.eval_datasets = eval_datasets
        self.data_collator = data_collator

        # Train Batch Size
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

        # tensorboard for logging
        self.writer = SummaryWriter(log_dir=args.logging_path)

    def evaluate(self, trainin_step: int = 0) :
        jax_params = self.params
        jax_model = self.model

        insturction_metrics = InstructionMetrics()

        @jax.jit
        def generate_step(params, input_ids: jnp.ndarray, attention_mask: jnp.array) :
            pad_ids = jnp.ones(
                (input_ids.shape[0], self.args.generation_max_length), 
                dtype=jnp.int32
            )
            input_ids = jnp.concatenate((pad_ids * self.tokenizer.pad_token_id, input_ids), axis=1)
            attention_mask = jnp.concatenate((pad_ids * 0, attention_mask), axis=1)
            
            def body_fn(_: int, input_state: Tuple[jnp.ndarray, jnp.ndarray]) :
                input_ids, attention_mask = input_state

                logits = jax_model(input_ids=input_ids, attention_mask=attention_mask, params=params, train=False)[0]
                pred = jnp.argmax(logits[:, -1, :], axis=-1)
                attn = jnp.ones((input_ids.shape[0], ), dtype=jnp.int32)

                input_ids = jnp.roll(input_ids, -1, 1).at[:, -1].set(pred)
                attention_mask = jnp.roll(attention_mask, -1, 1).at[:, -1].set(attn)

                input_state = (input_ids, attention_mask)
                return input_state

            input_state = (input_ids, attention_mask)
            output_ids = jax.lax.fori_loop(lower=0, upper=self.args.generation_max_length, body_fun=body_fn, init_val=input_state)
            return output_ids

        rng = jax.random.PRNGKey(self.args.random_seed)
        rng, dropout_rng = jax.random.split(rng)

        logging.info("Evaluation Starts")
        for dataset_name in self.eval_datasets :
            eval_dataset = self.eval_datasets[dataset_name]
            logging.info(f"Evaluation dataset name : {dataset_name} | dataset information : {eval_dataset}\n")

            eval_labels = eval_dataset["labels"]
            eval_dataset = eval_dataset.remove_columns(["labels"])

            eval_loader = data_loader(
                rng=dropout_rng, 
                dataset=eval_dataset, 
                data_collator=self.data_collator,
                batch_size=self.args.per_device_eval_batch_size, 
                shuffle=False,
                drop_last=False
            )
            eval_predictions = []

            eval_steps = len(eval_labels)
            with tqdm(total=eval_steps, desc="Evaluation", leave=False) as progress_bar_eval :

                for eval_data in eval_loader :
                    input_ids = eval_data["input_ids"]
                    attention_mask = eval_data["attention_mask"]

                    output_state = generate_step(jax_params, input_ids, attention_mask)
                    output_ids, _ = output_state
                    output_ids = output_ids[:, -self.args.generation_max_length:]

                    output_sequences = [seq.split("\n\n\n\n")[0] for seq in self.tokenizer.batch_decode(output_ids)]
                    eval_predictions.extend(output_sequences)

                    progress_bar_eval.update(1)

                if dataset_name in ["arc", "hellaswag", "truthful_qa-multiple_choice"] :
                    metric = insturction_metrics.get_multiple_choice_accuracy(eval_predictions, eval_labels)
                elif dataset_name == "gsm8k" :
                    metric = insturction_metrics.get_gsm8k_accuracy(eval_predictions, eval_labels)
                elif dataset_name == "truthful_qa-generation" :
                    metric = insturction_metrics.get_truthful_qa_blue(eval_predictions, eval_labels)
                else :
                    raise NameError("Not valid evaluation dataset name")

                logging.info(f"Evaluation dataset name : {dataset_name} | Accuracy : {metric}\n")

                for key in metric :
                    score = metric[key]
                    self.writer.add_scalar(f"{dataset_name}/{key}", score, global_step=trainin_step)


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

        logging.info("Training Starts\n")
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
                        train_loss = train_metric["loss"].mean().item()
                        learning_rate = train_metric["learning_rate"].item()
                        logging.info(f"Train [Step : %s] | Loss: %.8f & Learning Rate: %e\n" %(training_step_ptr, train_loss, learning_rate))

                        self.writer.add_scalar("train/loss", train_loss, global_step=training_step_ptr)
                        self.writer.add_scalar("train/learning_rate", learning_rate, global_step=training_step_ptr)

                    if self.args.evaluation_strategy == "steps" :
                        if training_step_ptr % self.args.eval_steps == 0 :
                            self.evaluate(training_step_ptr)

            if self.args.evaluation_strategy == "epoch" :
                self.evaluate(training_step_ptr)
