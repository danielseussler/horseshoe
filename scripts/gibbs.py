# ---
# A simple sampler for the horseshoe estimator written in Jax
# https://arxiv.org/pdf/1508.03884.pdf
# ---

import argparse
import sys
import time
from functools import wraps
from typing import Dict, List

import blackjax
import jax
import numpy as np
import pandas as pd
from jax import config
from jax import numpy as jnp
from sklearn.datasets import make_sparse_uncorrelated

sys.argv = [""]

config.update("jax_disable_jit", False)
config.update("jax_debug_nans", True)


def timeit(func):
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        # print(f"Function {func.__name__}{args} {kwargs} took {total_time:.4f} seconds")
        print(f"Function {func.__name__} took {total_time:.4f} seconds")
        return result

    return timeit_wrapper


def create_dataset(args):
    data, target = make_sparse_uncorrelated(
        n_samples=args.num_data, n_features=args.num_features, random_state=args.seed
    )

    x = jnp.asarray(data)
    y = jnp.asarray(target)
    n, p = x.shape

    return x, y, n, p


def kernel(rng_key: jax.Array, state: Dict) -> Dict:
    """
    A simple sampler for the horseshoe estimator
    https://arxiv.org/pdf/1508.03884.pdf

    """    
    state = state.copy()

    inv_lambda_s = jnp.diag(1 / state["lambda2"]) / state["tau2"]
    inv_a = jnp.linalg.inv(jnp.matmul(x.T, x) + inv_lambda_s)

    state["beta"] = jax.random.multivariate_normal(
        key=rng_key[0],
        mean=jnp.dot(jnp.matmul(inv_a, x.T), y),
        cov=state["sigma2"] * inv_a,
        method="cholesky",
    )

    err = y - jnp.dot(x, state["beta"])
    sigma2_scale = (
        jnp.dot(err.T, err) / 2.0
        + (jnp.sum(jnp.square(state["beta"]) / state["lambda2"]) / state["tau2"]) / 2.0
    )
    state["sigma2"] = sigma2_scale / jax.random.gamma(rng_key[1], a=(n + p) / 2.0)

    lambda2_scale = 1.0 / state["nu"] + jnp.square(state["beta"]) / (
        2.0 * state["tau2"] * state["sigma2"]
    )
    state["lambda2"] = lambda2_scale / jax.random.exponential(
        rng_key[2], shape=state["lambda2"].shape
    )

    tau2_scale = 1.0 / state["xi"] + jnp.sum(
        jnp.square(state["beta"]) / state["lambda2"]
    ) / (2.0 * state["sigma2"])
    state["tau2"] = tau2_scale / jax.random.gamma(rng_key[3], a=(p + 1.0) / 2.0)

    state["nu"] = 1.0 / (
        jax.random.exponential(rng_key[4], shape=state["lambda2"].shape)
        * (state["lambda2"] / (state["lambda2"] + 1))
    )
    state["xi"] = 1.0 / (
        jax.random.exponential(rng_key[5]) * (state["tau2"] / (state["tau2"] + 1))
    )

    return state


@timeit
def inference_loop(rng_key: jax.Array, initial_state: Dict, num_iter: int) -> Dict:
    """
    Creates `num_samples` number of samples using the above specified custom gibbs kernel.

    rng_key: a random number generator key
    initial_state: the initial, starting state of the Markov Chain
    num_iter: number of states in the Markov Chain to generate

    returns: dictionary with arrays of variables num_iter x dim covariate
    """

    @jax.jit
    def one_step(state, rng_key):
        state = kernel(rng_key=rng_key, state=state)
        positions = {k: state[k] for k in state.keys()}
        return state, positions

    keys = jax.random.split(rng_key, (num_iter, 6))
    _, positions = jax.lax.scan(one_step, initial_state, keys)

    return positions


def concatenate_chains(chains: List[Dict]):
    """Concatenate a list of dictionarys of variables to a dictionary of variables with chain x samples x dim."""
    results = {}

    for k in chains[0].keys():
        results[k] = np.concatenate(
            [jnp.expand_dims(chain[k], axis=0) for chain in chains], axis=0
        )

    return results


def apply_burnin_thinning(vars: Dict, num_burnin: int, thin: int) -> Dict:
    """Thin samples by keeping only samples defined by stepsize `thin`"""
    correct_vars = {
        k: vars[k][..., np.newaxis] if vars[k].ndim == 2 else vars[k]
        for k in vars.keys()
    }
    thinned_vars = {
        k: correct_vars[k][:, num_burnin::thin, :] for k in correct_vars.keys()
    }
    return thinned_vars


def create_summary(vars: Dict) -> pd.DataFrame:
    """Makeshift function to obtain summary statistics aggragated over the chains."""
    conc = np.concatenate(
        [
            jnp.expand_dims(vars[k], axis=-1) if vars[k].ndim == 2 else vars[k]
            for k in vars.keys()
        ],
        axis=2,
    )
    names_list = []

    for key in vars.keys():
        if vars[key].ndim == 2:
            names_list.append(f"{key}[{0}]")
        else:
            for i in range(vars[key].shape[2]):
                names_list.append(f"{key}[{i}]")

    summary = {
        "name": names_list,
        "mean": jnp.mean(conc, axis=(0, 1)),
        "p05": jnp.quantile(conc, q=0.05, axis=(0, 1)),
        "p50": jnp.quantile(conc, q=0.50, axis=(0, 1)),
        "p95": jnp.quantile(conc, q=0.95, axis=(0, 1)),
        "ess": blackjax.ess(conc),
        "rhat": blackjax.rhat(conc),
    }

    return pd.DataFrame(summary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--num-samples", nargs="?", default=1_000, type=int)
    parser.add_argument("--num-burnin", nargs="?", default=100, type=int)
    parser.add_argument("--num-chains", nargs="?", default=4, type=int)
    parser.add_argument("--num-data", nargs="?", default=1_000, type=int)
    parser.add_argument("--num-features", nargs="?", default=200, type=int)
    parser.add_argument("--thin", nargs="?", default=5, type=int)
    parser.add_argument("--seed", nargs="?", default=823743009, type=int)

    args = parser.parse_args()

    rng_key: jax.Array = jax.random.key(seed=args.seed)
    rng_key, rng_key_ = jax.random.split(key=rng_key)

    x, y, n, p = create_dataset(args)

    states = []

    # for more complex kernels / longer chains this can be run in parallel
    for i in range(args.num_chains):
        rng_key, rng_key_ = jax.random.split(rng_key)

        initial_state = {
            "beta": jnp.zeros((p,)),
            "sigma2": 1.0,
            "lambda2": jax.random.uniform(rng_key_, shape=(p,)),
            "tau2": 1.0,
            "nu": jnp.ones((p,)),
            "xi": 1.0,
        }

        rng_key, rng_key_ = jax.random.split(rng_key)
        st = inference_loop(rng_key_, initial_state, args.num_samples)

        states.append(st)

    states = concatenate_chains(states)
    states = apply_burnin_thinning(states, args.num_burnin, args.thin)
    summary = create_summary(states)

    print(summary)
