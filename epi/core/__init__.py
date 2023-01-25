# import multiprocessing

import jax
from jax.config import config

config.update("jax_enable_x64", True)

# TODO: Remove restriction to cpu
# multiprocessing.set_start_method("spawn", force=True)
jax.config.update("jax_platform_name", "cpu")  # Restricting to cpu for now
