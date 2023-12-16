import os
import jax
import json
import torch
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
from eval.metrics import InstructionMetrics
from model.llama_model import FlaxLlaMaForCausalLM
from transformers import LlamaTokenizer, LlamaConfig, LlamaForCausalLM
from transformers.generation import GenerationConfig
from flax.traverse_util import flatten_dict
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

        # directory path for checkpoints
        os.makedirs(self.args.output_dir, exist_ok=True)

        # directory path for logging
        self.logging_path = os.path.join(self.args.logging_dir, args.run_name)
        os.makedirs(self.logging_path, exist_ok=True)
        self.writer = SummaryWriter(log_dir=self.logging_path)

    def save_model(self, params, num_training_step: int, output_dir: str) :
        logging.info(f"Saving trained weights [Step : {num_training_step}] | Directory: {output_dir}\n")
        flattend_params = flatten_dict(params)

        config = LlamaConfig.from_pretrained(self.args.model_path)

        model_state_dict = {}
        for keys_tuple in flattend_params :
            key_string = ".".join(list(keys_tuple))
            # Formatting parameter name to huggingface model
            if "kernel" in key_string :
                key_string = key_string.replace("kernel", "weight")
            elif "embedding" in key_string :
                key_string = key_string.replace("embedding", "weight")

            # Formatting paramter value to huggingfacce model
            jnp_array = flattend_params[keys_tuple]
            if "norm" in key_string or "embed_tokens" in key_string :
                weight_tensor = torch.from_numpy(np.array(jnp_array,  dtype=np.float32))
            else :
                weight_tensor = torch.from_numpy(np.array(jnp_array,  dtype=np.float32)).transpose(0, 1)

                if "self_attn.q_proj" in key_string or "self_attn.k_proj" in key_string :
                    n_heads = config.num_attention_heads
                    hidden_size = config.hidden_size

                    reshaped_weight_tensor = weight_tensor.reshape(n_heads, hidden_size // n_heads // 2, 2, hidden_size)
                    transposed_weight_tensor = reshaped_weight_tensor.transpose(1, 2)
                    inverted_weight_tensor = transposed_weight_tensor.reshape(hidden_size, hidden_size)

                    weight_tensor = inverted_weight_tensor

            model_state_dict[key_string] = weight_tensor

        # Set parameters to huggingface Llama model (LlamaForCausalLM)
        model = LlamaForCausalLM(config)
        model.load_state_dict(model_state_dict)
        model.save_pretrained(os.path.join(output_dir, f"checkpoint-{num_training_step}"))


    def evaluate(self, params, num_trainin_step: int = 0) :
        jax_params = params
        jax_model = self.model

        insturction_metrics = InstructionMetrics()

        @jax.jit
        def generate_step(params, input_ids: jnp.ndarray, attention_mask: jnp.array) :
            generations = jax_model.generate(
                input_ids=input_ids, 
                attention_mask=attention_mask, 
                params=params, 
                generation_config=GenerationConfig(
                    num_beams=1, 
                    do_sample=False, 
                    max_length=input_ids.shape[1]+self.args.generation_max_length, 
                    pad_token_id=self.tokenizer.pad_token_id, 
                    eos_token_id=self.tokenizer.eos_token_id, 
                ), 
            )
            out_tokens = generations.sequences
            return out_tokens

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
                drop_last=True
            )
            eval_predictions = []

            if len(eval_labels) % self.args.per_device_eval_batch_size > 0 :
                eval_steps = len(eval_labels) // self.args.per_device_eval_batch_size + 1
            else :
                eval_steps = len(eval_labels) // self.args.per_device_eval_batch_size
    
            with tqdm(total=eval_steps, desc="Evaluation", leave=False) as progress_bar_eval :

                for eval_data in eval_loader :
                    input_ids = eval_data["input_ids"]
                    attention_mask = eval_data["attention_mask"]

                    output_tokens = generate_step(jax_params, input_ids, attention_mask)
                    generated = output_tokens[:, -self.args.generation_max_length:]
                    generated_sequences = [seq.split("</s>")[0].strip() for seq in self.tokenizer.batch_decode(generated)]
                    eval_predictions.extend(generated_sequences)

                    progress_bar_eval.update(1)

                # Making evaluation results json file and save
                eval_labels = eval_labels[:len(eval_predictions)]
                eval_results = {}
                for i, (pred, label) in enumerate(zip(eval_predictions, eval_labels)) :
                    eval_results[i] = {
                        "prediction" : pred,
                        "label" : label
                    }

                results_logging_dir = os.path.join(self.logging_path, f"{dataset_name}")
                os.makedirs(results_logging_dir, exist_ok=True)

                results_logging_path = os.path.join(results_logging_dir, f"checkpoint-{num_trainin_step}.json")
                with open(results_logging_path, "w") as f :
                    json.dump(eval_results, f, ensure_ascii=False, indent=4)

                # Get appropirate metric for each evaluation dataset
                if dataset_name in ["arc", "mmlu", "hellaswag", "truthful_qa-multiple_choice", "winogrande"] :
                    metric = insturction_metrics.get_multiple_exact_match(eval_predictions, eval_labels)
                elif dataset_name == "gsm8k" :
                    metric = insturction_metrics.get_gsm8k_accuracy(eval_predictions, eval_labels)
                elif dataset_name == "truthful_qa-generation" :
                    metric = insturction_metrics.get_truthful_qa_blue(eval_predictions, eval_labels)
                else :
                    raise NameError("Not valid evaluation dataset name")

                logging.info(f"Evaluation dataset name : {dataset_name} | Score : {metric}\n")
                for key in metric :
                    score = metric[key]
                    self.writer.add_scalar(f"{dataset_name}/{key}", score, global_step=num_trainin_step)


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

                    if training_step_ptr % (self.args.logging_steps) == 0 :
                        train_loss = train_metric["loss"].mean().item()
                        learning_rate = train_metric["learning_rate"].item()
                        logging.info(f"Train [Step : %s] | Loss: %.8f & Learning Rate: %e\n" %(training_step_ptr, train_loss, learning_rate))

                        self.writer.add_scalar("train/loss", train_loss, global_step=training_step_ptr)
                        self.writer.add_scalar("train/learning_rate", learning_rate, global_step=training_step_ptr)

                    if self.args.evaluation_strategy == "steps" :
                        if training_step_ptr % self.args.eval_steps == 0 :
                            self.evaluate(jax_params, training_step_ptr)

                    if self.args.save_strategy == "steps" :
                        if training_step_ptr % self.args.save_steps == 0 :
                            self.save_model(jax_params, training_step_ptr, self.arge.output_dir)

            if self.args.evaluation_strategy == "epoch" :
                self.evaluate(jax_params, training_step_ptr)

            if self.args.save_strategy == "epoch" :
                self.save_model(jax_params, training_step_ptr, self.args.output_dir)

