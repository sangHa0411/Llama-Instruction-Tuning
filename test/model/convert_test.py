
import sys
import jax
import jax.numpy as jnp
from flax.traverse_util import flatten_dict
from jax.sharding import Mesh
from jax.experimental import mesh_utils
from transformers import (
    LlamaConfig,
    LlamaTokenizer,
    LlamaForCausalLM
)

# Add source code path to system paths
sys.path.append("/home/sangha110495/project/Llama-Instruction-Tuning/src")

from model.convert import ParameterConvertor

def test_convert_parameters() :
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

    # Device Setting
    devices = mesh_utils.create_device_mesh((1, len(jax.devices())))
    mesh = Mesh(devices, axis_names=('dp', 'mp'))

    parameter_convertor = ParameterConvertor(mesh=mesh, config=config, tokenizer=tokenizer)
    params = parameter_convertor(hf_model=hf_model)
    
    # Huggingface model parameters and converted parameters
    flattend_params = flatten_dict(params)
    hf_model_state_dict = hf_model.state_dict()

    for keys_tuple in flattend_params :           
        key_string = ".".join(list(keys_tuple))
        # Formatting parameter's name to huggingface model
        if "kernel" in key_string :
            key_string = key_string.replace("kernel", "weight")
        elif "embedding" in key_string :
            key_string = key_string.replace("embedding", "weight")

        # Test if parameter's name in huggingface model
        assert key_string in hf_model_state_dict
        # Test if parameter's type in bfloat16
        assert flattend_params[keys_tuple].dtype == jnp.bfloat16

    # Test if all parameters are exactly converted
    assert len(flattend_params) == len(hf_model_state_dict)
