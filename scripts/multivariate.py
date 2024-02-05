# ---
# 
# 
# ---

import jax
from jax import numpy as jnp
from functools import wraps
import time

rng_key = jax.random.key(seed=98347134)


# test sample from this distribution
mu = jnp.zeros(shape=(2,))
Sigma = jnp.array([[1.0, 0.4], [0.4, 1.0]])

initial_state = {"x": 0, "y": 0}


def timeit(func):
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        print(f"Function {func.__name__}{args} {kwargs} took {total_time:.4f} seconds")
        return result

    return timeit_wrapper


def kernel(rng_key, state):
    rng_key, rng_key_, rng_key__= jax.random.split(key=rng_key, num=3)

    state = state.copy()

    mean = mu[0] + Sigma[1, 0] / Sigma[0, 0] * (state["y"] - mu[1])
    sigma = Sigma[0, 0] - Sigma[1, 0] / Sigma[1, 1] * Sigma[1, 0]

    state["x"] = jax.random.normal(rng_key_) * jnp.sqrt(sigma) + mean

    mean = mu[1] + Sigma[0, 1] / Sigma[1, 1] * (state["x"] - mu[0])
    sigma = Sigma[1, 1] - Sigma[0, 1] / Sigma[0, 0] * Sigma[0, 1]

    state["y"] = jax.random.normal(rng_key__) * jnp.sqrt(sigma) + mean

    return state


# inference
@timeit
def inference_loop(rng_key, initial_state, num_samples):
    @jax.jit
    def one_step(state, rng_key):
        state = kernel(rng_key=rng_key, state=state)
        positions = {k: state[k] for k in state.keys()}
        return state, positions

    keys = jax.random.split(rng_key, num_samples)
    _, positions = jax.lax.scan(one_step, initial_state, keys)

    return positions


rng_key, sample_key = jax.random.split(rng_key)
positions = inference_loop(sample_key, initial_state, 100_000)

data = jnp.concatenate([k.reshape(-1, 1) for k in positions.values()], axis=1)
data = data[500:,]

print(jnp.mean(data, axis=0))
print(jnp.cov(data, rowvar=False))


with jax.disable_jit():
    rng_key, sample_key = jax.random.split(rng_key)
    positions2 = inference_loop(sample_key, initial_state, 10_000)
