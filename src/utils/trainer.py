import os
import jax
import json
import torch
import optax
import logging
import numpy as np
import jax.numpy as jnp
from pytz import timezone
from datetime import datetime
from datasets import Dataset
from jaxtyping import PyTree
from utils.loss import loss_fn
from data.data_loader import data_loader
from data.collator import Seq2SeqCollator
from eval.metrics import InstructionMetrics
from eval.postprocessor import Postprocessor
from model.llama_model import FlaxLlaMaForCausalLM
from transformers.generation import GenerationConfig
from flax.traverse_util import flatten_dict
from utils.optimizer import OptimizerFactory
from torch.utils.tensorboard import SummaryWriter
from transformers import LlamaTokenizer, LlamaConfig, LlamaForCausalLM
from typing import Dict, List, Tuple, Any
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

        opt_factory = OptimizerFactory(args, num_data=len(dataset))
        lr_scheudler, optimizer = opt_factory.get_optimizer(params)
        self.lr_scheduler = lr_scheudler
        self.optimizer = optimizer

        # directory path for checkpoints
        os.makedirs(self.args.output_dir, exist_ok=True)

        # directory path for logging
        self.logging_path = os.path.join(self.args.logging_dir, args.run_name)
        os.makedirs(self.logging_path, exist_ok=True)
        self.writer = SummaryWriter(log_dir=self.logging_path)

    # Save trained parameters after converting parameter to huggingface model's weight
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

    # Postprocess evaluation results and score metrics
    def score_prediction(
        self, 
        dataset_name: str, 
        dataset: Dataset, 
        prediction: Dict[str, Any], 
        labels:List[Any]
    ) -> Dict[str, float]:
    
        insturction_metrics = InstructionMetrics()
        postprocessor = Postprocessor(tokenizer=self.tokenizer)

        # Postprocess results for each evaluation dataset
        if dataset_name == "mmlu" :
            results = postprocessor.postprocess_multiplce_choice(dataset, prediction, labels)
            metric = insturction_metrics.get_multiple_choice_acc(results)

        elif dataset_name == "hellaswag" :
            results = postprocessor.postprocess_multiplce_choice(dataset, prediction, labels)
            metric = insturction_metrics.get_multiple_choice_acc(results)

        elif dataset_name == "arc" :
            results = postprocessor.postprocess_multiplce_choice(dataset, prediction, labels)
            metric = insturction_metrics.get_multiple_choice_acc(results)

        elif dataset_name == "truthful_qa" :
            results = postprocessor.postprocess_tfqa_mc2(dataset, prediction, labels)
            metric = insturction_metrics.get_truthful_qa_mc2(results)

        elif dataset_name == "winogrande" :
            results = postprocessor.postprocess_multiplce_choice(dataset, prediction, labels)
            metric = insturction_metrics.get_multiple_choice_acc(results)

        elif dataset_name == "gsm8k" :
            results = postprocessor.postprocess_generations(dataset, prediction, labels)
            metric = insturction_metrics.get_gsm8k_acc(results)

        else :
            raise NameError("Not valid evaluation dataset name")

        return metric

    # Flatten dataset's content for evaluation
    def prepare_evaluation_datasets(self, eval_dataset: Dataset) -> Tuple[Dataset, List[Any], str] :
        if isinstance(eval_dataset["input_ids"][0][0], list) : 
            data_ids = eval_dataset["id"]
            input_ids = eval_dataset["input_ids"]
            attention_mask = eval_dataset["attention_mask"]
            candidate_lengths = eval_dataset["candidate_length"]

            data_ids = sum(data_ids, [])
            input_ids = sum(input_ids, [])
            attention_mask = sum(attention_mask, [])
            candidate_lengths = sum(candidate_lengths, [])

            eval_dataset = Dataset.from_dict(
                {
                    "id" : data_ids,
                    "input_ids" : input_ids, 
                    "attention_mask" : attention_mask,
                    "candidate_length" : candidate_lengths
                }
            )
            eval_category = "multiple_choice"
        else :          
            eval_category = "generation"      

        return eval_dataset, eval_category

    # Decode generated token to string
    def decode_output_tokens(self, output_tokens: jnp.ndarray) -> List[str]:
        generated = output_tokens[:, -self.args.generation_max_length:]
        generated_sequences = [seq.split("</s>")[0] for seq in self.tokenizer.batch_decode(generated)]

        for i, generated_seq in enumerate(generated_sequences) :
            if "\n\n" in generated_seq :
                generated_seq = generated_seq.split("\n\n")[0]
                generated_sequences[i] = generated_seq
                
        return generated_sequences

    # Evaluation function
    def evaluate(self, params, num_trainin_step: int = 0) :
        jax_params = params
        jax_model = self.model

        @jax.jit
        # Generate sequence from input_ids
        def generate_step(params, input_ids: jnp.ndarray, attention_mask: jnp.array) :
            generations = jax_model.generate(
                input_ids=input_ids, 
                attention_mask=attention_mask, 
                params=params, 
                generation_config=GenerationConfig(
                    do_sample=False, 
                    early_stopping=True,
                    max_length=input_ids.shape[1]+self.args.generation_max_length, 
                    pad_token_id=self.tokenizer.pad_token_id, 
                    eos_token_id=self.tokenizer.eos_token_id, 
                ), 
            )
            generated_tokens = generations.sequences
            return generated_tokens

        @jax.jit
        # Get sequence logit from input_ids
        def forward_step(params, input_ids: jnp.ndarray, attention_mask: jnp.array) :
            batch_size = input_ids.shape[0]

            outputs = jax_model(
                input_ids=input_ids, 
                attention_mask=attention_mask,
                params=params,
                train=False
            )
            output_logits = outputs.logits

            # Output logit and log probabilities for input sequence
            sequence_logits = output_logits[:, :-1]
            sequence_log_probs = jax.nn.log_softmax(sequence_logits, axis=-1)

            # Answer token for input_sequence
            sequence_labels = input_ids[:, 1:]

            sequence_log_prob_results = []
            for i in range(batch_size) :
                sequence_label = sequence_labels[i]
                sequence_id = jnp.arange(len(sequence_label))

                # Sequence log probability for label token
                sequence_log_prob = sequence_log_probs[i][sequence_id, sequence_label]
                sequence_log_prob_results.append(sequence_log_prob)
                
            return sequence_log_prob_results

        rng = jax.random.PRNGKey(self.args.random_seed)
        rng, dropout_rng = jax.random.split(rng)

        eval_data_collator = Seq2SeqCollator(self.tokenizer, self.args.eval_sequence_max_length)

        logging.info("Evaluation Starts")
        for dataset_name in self.eval_datasets :
            eval_dataset = self.eval_datasets[dataset_name]
            eval_labels = eval_dataset["labels"]

            eval_dataset = eval_dataset.remove_columns(["labels"])

            eval_dataset, eval_category = self.prepare_evaluation_datasets(eval_dataset)
            logging.info(f"Evaluation dataset name : {dataset_name} | dataset category: {eval_category}\ndataset category: {eval_dataset}")

            eval_batch_size = self.args.per_device_eval_forward_batch_size \
                if eval_category == "multiple_choice" else \
                self.args.per_device_eval_generate_batch_size
            
            eval_loader = data_loader(
                rng=dropout_rng, 
                dataset=eval_dataset, 
                data_collator=eval_data_collator,
                batch_size=eval_batch_size, 
                shuffle=False,
                drop_last=True
            )
            eval_predictions = {
                "sequence_log_prob" : [], 
                "generation" : []
            }
            eval_steps = len(eval_dataset) // eval_batch_size
    
            with tqdm(total=eval_steps, desc="Evaluation", leave=False) as progress_bar_eval :
                for eval_data in eval_loader :
                    input_ids = eval_data["input_ids"]
                    attention_mask = eval_data["attention_mask"]
                    
                    if eval_category == "generation" :
                        output_tokens = generate_step(jax_params, input_ids, attention_mask)
                        generated_sequences = self.decode_output_tokens(output_tokens)

                        eval_predictions["generation"].extend(generated_sequences)
                    else :
                        sequence_log_probs = forward_step(jax_params, input_ids, attention_mask)
                        eval_predictions["sequence_log_prob"].extend(sequence_log_probs)

                    progress_bar_eval.update(1)

                metric = self.score_prediction(dataset_name, eval_dataset, eval_predictions, eval_labels)
                logging.info(f"Evaluation dataset name : {dataset_name} | Score : {metric}\n")
                for key in metric :
                    score = metric[key]
                    self.writer.add_scalar(f"{dataset_name}/{key}", score, global_step=num_trainin_step)

    # Training function
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
                loss, acc = loss_fn(logits, labels)
                return loss, acc

            grad_fn = jax.value_and_grad(compute_loss, has_aux=True)
            (loss, acc), grads = grad_fn(params)

            updates, new_opt_state = optimizer_update(grads, opt_state, params)
            new_params = optax.apply_updates(params, updates)

            metrics = {"loss": loss, "accuracy": acc, "learning_rate": self.lr_scheduler(step)}
            return new_params, new_opt_state, new_dropout_rng, metrics

        rng = jax.random.PRNGKey(self.args.random_seed)
        rng, dropout_rng = jax.random.split(rng)

        training_step_ptr = 0
        train_metric_ptr = {"loss": 0.0, "accuracy": 0.0, "num_step": 0}

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

                    train_metric_ptr["loss"] += train_metric["loss"].mean().item()
                    train_metric_ptr["accuracy"] += train_metric["accuracy"].mean().item()
                    train_metric_ptr["num_step"] += 1

                    if training_step_ptr % (self.args.logging_steps) == 0 :
                        train_loss = train_metric_ptr["loss"] / train_metric_ptr["num_step"]
                        train_acc = train_metric_ptr["accuracy"] / train_metric_ptr["num_step"]

                        learning_rate = train_metric["learning_rate"].item()
                        logging.info(f"Train [Step : %s] | Loss: %.8f & Accuracy: %.8f & Learning Rate: %e\n" %(training_step_ptr, train_loss, train_acc, learning_rate))

                        self.writer.add_scalar("train/loss", train_loss, global_step=training_step_ptr)
                        self.writer.add_scalar("train/accuracy", train_acc, global_step=training_step_ptr)
                        self.writer.add_scalar("train/learning_rate", learning_rate, global_step=training_step_ptr)

                        train_metric_ptr["loss"] = 0.0
                        train_metric_ptr["accuracy"] = 0.0
                        train_metric_ptr["num_step"] = 0

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

