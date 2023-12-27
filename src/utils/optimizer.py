import jax
import optax
import jax.numpy as jnp
from jaxtyping import PyTree
from flax.core import frozen_dict
from flax.core.frozen_dict import FrozenDict
from typing import Callable, List, Tuple
from optax import GradientTransformationExtraArgs
from utils.scheduler import create_constant_lr_scheduler, create_linear_decay_lr_scheduler

class OptimizerFactory :

    def __init__(self, args, num_data: int) :
        self.args = args
        self.num_data = num_data

    def create_mask(self, params: PyTree[jnp.ndarray]) -> PyTree[str]:
        def freeze_mask(param_name: List[str]) -> bool :
            for p_name in param_name :
                if "embedding" in p_name :
                    return True
                if "norm" in p_name :
                    return True
                if "lm_head" in p_name :
                    return True
            return False

        def _map(params, mask, routes):
            for k in params :
                if isinstance(params[k], FrozenDict):
                    mask[k] = {}
                    _map(params[k], mask[k], routes+[k])
                else:
                    paran_name = routes + [k]
                    if freeze_mask(paran_name) :
                        mask[k] = "frozen"
                    else :
                        mask[k] = "trainable"

        parm_masks = {}
        _map(params, parm_masks, [])
        return frozen_dict.freeze(parm_masks)

    def get_lr_scheduler(self, ) -> Callable[[int], jnp.ndarray]:
        args = self.args
        train_batch_size = args.per_device_train_batch_size

        # Scheduler
        if args.lr_scheduler_type == "constant" :
            lr_scheduler = create_constant_lr_scheduler(
                self.num_data,
                train_batch_size=train_batch_size,
                num_train_epochs=args.num_train_epochs,
                warmup_ratio=args.warmup_ratio,
                learning_rate=args.learning_rate
            )
        elif args.lr_scheduler_type == "linear" :
            lr_scheduler = create_linear_decay_lr_scheduler(
                self.num_data,
                train_batch_size=train_batch_size,
                num_train_epochs=args.num_train_epochs,
                warmup_ratio=args.warmup_ratio,
                learning_rate=args.learning_rate
            )
        else :
            raise NotImplementedError("Not implemented lr scheduelr type")
        
        return lr_scheduler

    def get_optimizer(self, params: PyTree[jnp.ndarray]) -> \
        Tuple[Callable[[int], jnp.ndarray], GradientTransformationExtraArgs]:
        
        args = self.args
        lr_scheduler = self.get_lr_scheduler()
        
        adamw_optimizer = optax.adamw(
            learning_rate=lr_scheduler,
            weight_decay=args.weight_decay,
            mu_dtype=jnp.bfloat16
        )

        optimizer = optax.multi_transform(
            {
                "trainable": optax.chain(
                    optax.clip_by_global_norm(1.0),
                    adamw_optimizer
                ), 
                "frozen": optax.set_to_zero()
            }, 
            param_labels = self.create_mask(params)
        )
        return lr_scheduler, optimizer
