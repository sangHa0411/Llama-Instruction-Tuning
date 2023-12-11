import os
import jax
import torch
import random
import logging
import numpy as np
from pytz import timezone
from typing import Tuple
from datetime import datetime
from trainer import Trainer
from jax.sharding import Mesh
from jax.experimental import mesh_utils
from utils.collator import Seq2SeqCollator
from utils.convert import ParameterConvertor
from utils.loader import InstructionDatasetLoader
from utils.preprocessor import InstructionDatasetPreprocessor
from utils.collator import Seq2SeqCollator
from model.llama_model import FlaxLlaMaForCausalLM
from datasets import disable_caching
from transformers import (
    LlamaConfig,
    LlamaTokenizer,
    LlamaForCausalLM
)

logging.basicConfig(format = "[%(asctime)s][%(levelname)s][Message] - %(message)s", level = logging.INFO)
logging.Formatter.converter = lambda *args: datetime.now(tz=timezone("Asia/Seoul")).timetuple()

def get_model_and_tokenizer(args) -> Tuple[LlamaConfig, LlamaForCausalLM, LlamaTokenizer]:

    logging.info(f"model | {args.model_path} | tokenizer: {args.tokenizer_path}")
    model_path = args.model_path

    # Load config
    config = LlamaConfig.from_pretrained(model_path)

    if args.gradient_checkpointing :
        config.gradient_checkpointing = True

    config.embd_pdrop = args.dropout_rate
    config.resid_pdrop = args.dropout_rate
    config.attn_pdrop = args.dropout_rate
    
    # Load tokenizer
    tokenizer = LlamaTokenizer.from_pretrained(args.tokenizer_path)
    tokenizer.pad_token_id = 0
    tokenizer.padding_side = "right"

    # Load torch model
    logging.info("Load huggingface model")
    torch_model = LlamaForCausalLM.from_pretrained(model_path, cache_dir="/mnt/disks-standard/persist/huggingface")

    return config, torch_model, tokenizer

def train(args):

    os.environ["WANDB_PROJECT"] = args.project_name

    # Loading config, model and tokenizer
    config, torch_model, tokenizer = get_model_and_tokenizer(args)

    # Setting random seed
    seed_everything(args.random_seed)

    disable_caching()

    # Loading Dataset
    loader = InstructionDatasetLoader(args.random_seed, args.datasets, args.ratios)
    instruction_dataset = loader.load()
    logging.info(f"Instruction dataset:{instruction_dataset}")

    # Preprocessing and Encodign Dataset
    preprocessor = InstructionDatasetPreprocessor(tokenizer=tokenizer, sequence_max_length=args.sequence_max_length)
    encoded_instruction_dataset = preprocessor(instruction_dataset)
    logging.info(f"Encoded dataset:{encoded_instruction_dataset}")

    # Setting Device & Model mesh
    num_tpu_device = jax.device_count()
    tpu_devices = jax.local_devices()
    devices = mesh_utils.create_device_mesh((1, len(jax.devices())))
    mesh = Mesh(devices, axis_names=('dp', 'mp'))
    logging.info(f"The number of tpu device:{num_tpu_device}")
    logging.info(f"Tpu devices:{tpu_devices}")

    # Extracting model weights from huggingface model
    parameter_convertor = ParameterConvertor(mesh, config, tokenizer)
    params = parameter_convertor(torch_model)
    
    # Data Collator
    data_collator = Seq2SeqCollator(tokenizer, sequence_max_length=args.sequence_max_length)

    # Model
    model = FlaxLlaMaForCausalLM(config, _do_init=False)

    # Trainer
    trainer = Trainer(
        args=args, 
        model=model, 
        params=params, 
        dataset=encoded_instruction_dataset, 
        data_collator=data_collator
    )

    trainer.train()

   
def seed_everything(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    np.random.default_rng(seed)
    random.seed(seed)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="lg-t5")

    # Dataset names
    parser.add_argument("--datasets", type=str, default="[alpaca,cot-collection]", help="instruction datasets")
    parser.add_argument("--ratios", type=str, default="[1.0,0.1]", help="instruction dataset ratios")

    # Wandb logging name
    parser.add_argument("--entity_name", type=str, default="sangha0411", help="wandb entity name")
    parser.add_argument("--project_name", type=str, default="llama-instruction-tuning", help="project's name")
    parser.add_argument("--model_name", type=str, default="llama/llama-2-7b-hf", help="model's name")
    parser.add_argument("--group_name", type=str, default="instruction-tuning", help="group's name")
    parser.add_argument("--run_name", type=str, default=None, help="A descriptor for the run. used for wandb logging")
    parser.add_argument("--report_to", type=str, default=None, help="""The list of integrations to report the results and logs to. Supported platforms are ['azure_ml, 'comet_ml, 'mlflow, tensorboard", 'wandb']""")

    # Random Seed
    parser.add_argument("--random_seed", type=int, default=42, help="fix random seed in torch, random, numpy")

    # Gradient checkpointing
    parser.add_argument("--gradient_checkpointing", type=bool, default=False, help="use gradient checkpointing for training")

    # Sequence Length and Generation Length
    parser.add_argument("--dropout_rate", type=float, default=0.0, help="dropout rate for llm training")

    # Sequence Length and Generation Length
    parser.add_argument("--sequence_max_length", type=int, default=2048, help="llm model max sequence length")
    parser.add_argument("--generation_max_length", type=int, default=1024, help="generation max length")

    # Data & Logging Path
    parser.add_argument("--logging_path", type=str, default="/project/llama-instruction-tuning/exps/logging", help="path for evaluation prediction results")
    parser.add_argument("--output_dir", type=str, default="/mnt/disks-standard/persist/t5/llama-alpaca/exps/checkpoints", help="model checkpoint path")

    # Model evaluation & save strategy
    parser.add_argument("--do_model_save", type=bool, default=False, help="do model saving during training")
    parser.add_argument("--do_model_evaluate", type=bool, default=False, help="do model evalution during training")

    # Model & Tokenizer path
    parser.add_argument("--tokenizer_path", type=str, default="/mnt/disks-standard/persist/llama/llama-2-7b-hf", help="path for evaluation prediction results")
    parser.add_argument("--model_path", type=str, default="/mnt/disks-standard/persist/llama/llama-2-7b-hf", help="path for evaluation prediction results")

    # Epoch & Batch size
    parser.add_argument("--num_train_epochs", type=int, default=3, help="num_train_epochs for training")
    parser.add_argument("--per_device_train_batch_size", type=int, default=1, help="training batch size")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=1, help="evaluation batch size, if none, use batch_size")

    # Optimizer
    parser.add_argument("--learning_rate", type=float, default=3e-5, help="dataset")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="dataset")

    # Scheduler
    parser.add_argument("--lr_scheduler_type", type=str, default="constant", help="type of learning rate scheduler")
    parser.add_argument("--warmup_ratio", type=float, default=0.05, help="warmup ratio of linear learning rate scheduler")

    # Logging
    parser.add_argument("--logging_steps", type=int, default=100, help="Number of update steps between two logs if logging_strategy is 'step")

    args = parser.parse_args()

    args.run_name = f"MODEL_NAME:{args.model_name}-EP:{args.num_train_epochs}-LR:{args.learning_rate}-BS:{args.per_device_train_batch_size}-WR:{args.warmup_ratio}-WD:{args.weight_decay}"
    args.output_dir = f"{args.output_dir}/{args.run_name}"

    logging.info(f"Training arguments: {args}")
    train(args)

