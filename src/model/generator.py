# This script is largely borrow from https://github.com/Sea-Snell/JAX_llama

import jax
import jax.numpy as jnp
from jax.sharding import Mesh
from jaxtyping import PyTree
from flax import struct
from functools import partial
from typing import List, Optional
from jax.sharding import PartitionSpec as P
from transformers.generation import GenerationConfig
from model.llama_model import FlaxLlaMaForCausalLM
from model.partitions import with_named_sharding_constraint
from transformers import LlamaTokenizerFast


class LLaMA(struct.PyTreeNode):
    params: PyTree
    model: FlaxLlaMaForCausalLM = struct.field(pytree_node=False)
    tokenizer: LlamaTokenizerFast = struct.field(pytree_node=False)
    mesh: Optional[Mesh] = struct.field(pytree_node=False, default=None)

    @partial(jax.jit, static_argnums=(3,4,5))
    def generate(
        self, 
        tokens: jnp.ndarray, 
        attention_mask: jnp.ndarray, 
        max_gen_len: int, 
        temperature: float = 1.0, 
        top_p: float = 1.0
    ) -> jnp.ndarray:

        tokens = with_named_sharding_constraint(tokens, self.mesh, P("dp", None))
        attention_mask = with_named_sharding_constraint(attention_mask, self.mesh, P("dp", None))

        generations = self.model.generate(
            input_ids=tokens, 
            attention_mask=attention_mask, 
            params=self.params, 
            generation_config=GenerationConfig(
                num_beams=1, 
                do_sample=True, 
                max_length=max_gen_len+tokens.shape[1], 
                pad_token_id=self.tokenizer.eos_token_id, 
                eos_token_id=self.tokenizer.eos_token_id, 
                temperature=temperature, 
                top_p=top_p, 
            ), 
        )
        out_tokens = generations.sequences
        
        out_tokens = with_named_sharding_constraint(out_tokens, self.mesh, P("dp", None))
        return out_tokens
    
    def generate_from_str(self, prompts: List[str], max_gen_len: int, temperature: float = 0.8, top_p: float = 0.95):
        prompt_tokens = [self.tokenizer.encode(x) for x in prompts]

        max_prompt_size = max([len(t) for t in prompt_tokens])

        tokens = jnp.full((len(prompts), max_prompt_size), self.tokenizer.eos_token_id).astype(jnp.int32)
        for i, t in enumerate(prompt_tokens):
            tokens = tokens.at[i, -len(t):].set(t) # left pad
        attention_mask = (tokens != self.tokenizer.eos_token_id).astype(jnp.int32)

        out_tokens = self.generate(tokens, attention_mask, max_gen_len, temperature, top_p)

        decoded = []
        for i, t in enumerate(out_tokens.tolist()):
            # cut to max gen len
            t = t[t.index(self.tokenizer.bos_token_id):]
            t = t[:(len(prompt_tokens[i])+max_gen_len)]
            # cut to eos tok if any
            try:
                t = t[:t.index(self.tokenizer.eos_token_id)]
            except ValueError:
                pass
            decoded.append(self.tokenizer.decode(t))

        return decoded