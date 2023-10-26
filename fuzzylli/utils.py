import numpy as np
from jax import jit
import jax.numpy as jnp
import jax.random as random
import jax.lax as lax
from blackjax import nuts


def quad(f, a, b):
    """
    Fixed order (order=12) Gauss-Legendre quadrature for integration of f(x)
    from x=a to x=b
    """
    glx, glw = np.polynomial.legendre.leggauss(12)
    glx = jnp.asarray(0.5 * (glx + 1))
    glw = jnp.asarray(0.5 * glw)
    x_i = a + (b - a) * glx
    return (b - a) * f(x_i) @ glw


def inference_loop(rng_key, initial_states, tuned_params, log_prob_fn, num_samples):
    """
    HMC boiler plate. Code taken from blackjax documentation.
    """
    step_fn = nuts.kernel()

    def kernel(key, state, **params):
        return step_fn(key, state, log_prob_fn, **params)

    @jit
    def one_step(states, rng_key):
        states, infos = kernel(rng_key, states, **tuned_params)
        return states, (states, infos)

    keys = random.split(rng_key, num_samples)
    _, (states, infos) = lax.scan(one_step, initial_states, keys)

    return (states, infos)
