# This script is largely borrow from https://github.com/Sea-Snell/JAX_llama

import jax
import torch
import numpy as np
import jax.numpy as jnp
from typing import Dict
from jaxtyping import PyTree
from jax.sharding import Mesh, NamedSharding
from flax.core.frozen_dict import freeze, unfreeze
from model.partitions import get_llama_param_partition_spec
from transformers import LlamaConfig, LlamaTokenizerFast, LlamaForCausalLM


class ParameterConvertor :

    def __init__(self, mesh: Mesh, config: LlamaConfig, tokenizer: LlamaTokenizerFast) :
        self.mesh = mesh
        self.config = config 
        self.tokenizer = tokenizer

    def inverse_permute(self, w: torch.Tensor) -> torch.Tensor :
        n_heads = self.config.num_attention_heads
        hidden_size = self.config.hidden_size

        reshaped_w = w.reshape(n_heads, 2, hidden_size // n_heads // 2, hidden_size)
        transposed_w = reshaped_w.transpose(0, 2, 1, 3)

        inverted_w = transposed_w.reshape(hidden_size, hidden_size)
        return inverted_w

    def convert_from_hf_model(self, params: Dict[str, torch.Tensor]) -> PyTree[np.ndarray]:
        num_hidden_layers = getattr(self.config, "num_hidden_layers")

        jax_weights = {
            "model" : {
                "embed_tokens" : {"embedding" : params["model.embed_tokens.weight"].numpy()},
                "norm" : {"kernel" : params["model.norm.weight"].numpy()},
                "layers" : {
                    str(layer) : {
                            "self_attn" : {
                                "q_proj" : {"kernel" : self.inverse_permute(params[f"model.layers.{layer}.self_attn.q_proj.weight"].numpy()).transpose()},
                                "k_proj" : {"kernel" : self.inverse_permute(params[f"model.layers.{layer}.self_attn.k_proj.weight"].numpy()).transpose()},
                                "v_proj" : {"kernel" : params[f"model.layers.{layer}.self_attn.v_proj.weight"].numpy().transpose()},
                                "o_proj" : {"kernel" : params[f"model.layers.{layer}.self_attn.o_proj.weight"].numpy().transpose()},
                            },
                            "mlp" : {
                                "gate_proj" : {"kernel" : params[f"model.layers.{layer}.mlp.gate_proj.weight"].numpy().transpose()},
                                "up_proj" : {"kernel" : params[f"model.layers.{layer}.mlp.up_proj.weight"].numpy().transpose()},
                                "down_proj" : {"kernel" : params[f"model.layers.{layer}.mlp.down_proj.weight"].numpy().transpose()},
                            },
                            "input_layernorm" : {"kernel" : params[f"model.layers.{layer}.input_layernorm.weight"].numpy()},
                            "post_attention_layernorm" : {"kernel" : params[f"model.layers.{layer}.post_attention_layernorm.weight"].numpy()},
                        }
                    for layer in range(num_hidden_layers)
                },
            },
            "lm_head" : {"kernel" : params["lm_head.weight"].numpy().transpose()}
        }
        return jax_weights

    def to_bf16(self, t: PyTree[jnp.ndarray]) -> PyTree[jnp.ndarray]:
        return jax.tree_map(lambda x: x.astype(jnp.bfloat16) if x.dtype == jnp.float32 else x, t)

    def __call__(self, hf_model: LlamaForCausalLM) :
        hf_params = hf_model.state_dict()
        jax_params = self.convert_from_hf_model(params=hf_params)

        with jax.default_device(jax.devices('cpu')[0]):
            jax_params = freeze(jax.tree_map(lambda x: jnp.asarray(x), jax_params))

        param_spec = freeze(get_llama_param_partition_spec(unfreeze(jax_params)))
        jax_params = jax.tree_util.tree_map(
            lambda param, spec: jax.device_put(param, NamedSharding(self.mesh, spec)), jax_params, param_spec
        )
        jax_params = self.to_bf16(jax_params)

        return jax_params
