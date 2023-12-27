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


"""
functions are taken from https://github.com/yt-project/yt
"""


def SIEVE_PRIMES(x):
    return x and x[:1] + SIEVE_PRIMES([n for n in x if n % x[0]])


def decompose_to_primes(max_prime):
    """Decompose number into the primes"""
    for prime in SIEVE_PRIMES(list(range(2, max_prime))):
        if prime * prime > max_prime:
            break
        while max_prime % prime == 0:
            yield prime
            max_prime //= prime
    if max_prime > 1:
        yield max_prime


def factorize_number(pieces):
    """Return array consisting of prime, its power and number of different
    decompositions in three dimensions for this prime
    """
    factors = np.array(list(decompose_to_primes(pieces)))
    temp = np.bincount(factors)
    return np.array(
        [
            (prime, temp[prime], (temp[prime] + 1) * (temp[prime] + 2) // 2)
            for prime in np.unique(factors)
        ]
    ).astype(np.int64)


def evaluate_domain_decomposition(n_d, pieces, ldom):
    """Evaluate longest to shortest edge ratio
    BEWARE: lot's of magic here"""
    eff_dim = (n_d > 1).sum()
    exp = float(eff_dim - 1) / float(eff_dim)
    ideal_bsize = eff_dim * pieces ** (1.0 / eff_dim) * np.prod(n_d) ** exp
    mask = np.where(n_d > 1)
    nd_arr = np.array(n_d, dtype=np.float64)[mask]
    bsize = int(np.sum(ldom[mask] / nd_arr * np.prod(nd_arr)))
    load_balance = float(np.prod(n_d)) / (
        float(pieces) * np.prod((n_d - 1) // ldom + 1)
    )

    # 0.25 is magic number
    quality = load_balance / (1 + 0.25 * (bsize / ideal_bsize - 1.0))
    # \todo add a factor that estimates lower cost when x-direction is
    # not chopped too much
    # \deprecated estimate these magic numbers
    quality *= 1.0 - (0.001 * ldom[0] + 0.0001 * ldom[1]) / pieces
    if np.any(ldom > n_d):
        quality = 0

    return quality


def get_pencil_size(n_d, pieces):
    """Calculate the best division of array into px*py*pz subarrays.
    The goal is to minimize the ratio of longest to shortest edge
    to minimize the amount of inter-process communication.
    """
    fac = factorize_number(pieces)
    nfactors = len(fac[:, 2])
    best = 0.0
    p_size = np.ones(2, dtype=np.int64)
    if pieces == 1:
        return p_size

    while np.all(fac[:, 2] > 0):
        ldom = np.ones(2, dtype=np.int64)
        for nfac in range(nfactors):
            i = int(np.sqrt(0.25 + 2 * (fac[nfac, 2] - 1)) - 0.5)
            k = fac[nfac, 2] - (1 + i * (i + 1) // 2)
            i = fac[nfac, 1] - i
            ldom *= fac[nfac, 0] ** np.array([i, k])

        quality = evaluate_domain_decomposition(n_d, pieces, ldom)
        if quality > best:
            best = quality
            p_size = ldom
        # search for next unique combination
        for j in range(nfactors):
            if fac[j, 2] > 1:
                fac[j, 2] -= 1
                break
            else:
                if j < nfactors - 1:
                    fac[j, 2] = int((fac[j, 1] + 1) * (fac[j, 1] + 2) / 2)
                else:
                    fac[:, 2] = 0  # no more combinations to try

    return p_size
