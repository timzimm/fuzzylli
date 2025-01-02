import math
import pickle
import logging
import os

import jax.numpy as jnp

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def load_model(cache_dir, name):
    logger.info(f"Loading {name}...")
    with open(f"{cache_dir}/{name}", mode="rb") as file:
        model = pickle.load(file)
    return model


def save_model(overwrite, cache_dir, name, model):
    if not os.path.exists(f"{cache_dir}/{name}") or overwrite:
        logger.info(f"Saving {name}...")
        with open(f"{cache_dir}/{name}", mode="wb") as file:
            pickle.dump(model, file)


def compute_and_save_model(cache_dir, name_of, compute, **kwargs):
    compute_args = kwargs["for_compute"]
    name_args = kwargs["for_name"]
    model = compute(*compute_args)
    name = name_of(*name_args)
    save_model(True, cache_dir, name, model)
    return name, model


def load_or_compute_model(load_if_available, cache_dir, name_of, compute, **kwargs):
    compute_args = kwargs["for_compute"]
    name_args = kwargs["for_name"]
    name = name_of(*name_args)
    if os.path.exists(f"{cache_dir}/{name}") and load_if_available:
        return name, load_model(cache_dir, name)
    else:
        logger.info("No model found/Recompute enforced. Compute it...")
        return name, compute(*compute_args)


def hash_to_int64(digested_hash):
    num = int(digested_hash, 16)
    num = num // 10 ** (int(math.log(num, 10)) - 18 + 1)
    return jnp.int64(num)
