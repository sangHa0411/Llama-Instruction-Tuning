import sys
import jax
import torch
import numpy as np
import jax.numpy as jnp
from flax.traverse_util import flatten_dict
from jax.sharding import Mesh
from jax.experimental import mesh_utils
from transformers import (
    LlamaConfig,
    LlamaTokenizer,
    LlamaForCausalLM
)
from transformers.generation import GenerationConfig

# Add source code path to system paths
sys.path.append("/home/sangha110495/project/Llama-Instruction-Tuning/src")

from model.convert import ParameterConvertor

def test_save_model() :

    model_path = "/mnt/disks-standard/persist/llama/llama-2-7b-hf"
    config = LlamaConfig.from_pretrained(model_path)

    config.embd_pdrop = 0.0
    config.resid_pdrop = 0.0
    config.attn_pdrop = 0.0
    
    # Load tokenizer
    tokenizer = LlamaTokenizer.from_pretrained(model_path)
    tokenizer.pad_token_id = 0
    tokenizer.padding_side = "left"

    # Load huggingface model
    hf_model = LlamaForCausalLM.from_pretrained(model_path)

    sequence_max_length = 32
    example = "Who are you?"
    tokenized_example = tokenizer(example)

    # Generated sample from huggingface model
    org_sample = hf_model.generate(
        input_ids = torch.tensor([tokenized_example["input_ids"]]),
        attention_mask = torch.tensor([tokenized_example["attention_mask"]]),
         generation_config=GenerationConfig(
            do_sample=False, 
            early_stopping=True,
            max_length=sequence_max_length, 
            pad_token_id=tokenizer.pad_token_id, 
            eos_token_id=tokenizer.eos_token_id, 
        ), 
    )
    org_sample_decoded = tokenizer.batch_decode(org_sample)[0]

    # Device Setting
    devices = mesh_utils.create_device_mesh((1, len(jax.devices())))
    mesh = Mesh(devices, axis_names=('dp', 'mp'))

    # Converted parameters
    parameter_convertor = ParameterConvertor(mesh=mesh, config=config, tokenizer=tokenizer)
    params = parameter_convertor(hf_model=hf_model)

    flattend_params = flatten_dict(params)

    # Load Huggingface weight from converted parameters
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

    tgt_sample = model.generate(
        input_ids = torch.tensor([tokenized_example["input_ids"]]),
        attention_mask = torch.tensor([tokenized_example["attention_mask"]]),
         generation_config=GenerationConfig(
            do_sample=False, 
            early_stopping=True,
            max_length=sequence_max_length, 
            pad_token_id=tokenizer.pad_token_id, 
            eos_token_id=tokenizer.eos_token_id, 
        ), 
    )
    tgt_sample_decoded = tokenizer.batch_decode(tgt_sample)[0]

    # Test above logic is correct for formatting huggingface model from converted parameters
    assert org_sample_decoded == tgt_sample_decoded
