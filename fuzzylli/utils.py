import builtins
from functools import wraps, partial
from typing import Any, Callable, TypeVar
from collections.abc import Sequence

import numpy as np
import jax
import jax.numpy as jnp


from jax.tree_util import tree_unflatten, tree_flatten, tree_map


T = TypeVar("T")
F = TypeVar("F", bound=Callable[..., Any])


def _tree_map_multi_output(f, *args):
    """Like tree_map, but for functions that return tuples."""
    leaves, treedefs = zip(*builtins.map(tree_flatten, args))
    if any(treedef != treedefs[0] for treedef in treedefs):
        raise ValueError(f"argument treedefs do not match {treedefs=}")
    outputs = zip(*builtins.map(f, *leaves))
    return tuple(tree_unflatten(treedefs[0], out) for out in outputs)


def _lax_map(f, *xs, **kwargs):
    """Like lax.map, but supports multiple arguments like the built-in map."""
    _, ys = jax.lax.scan(lambda _, x: ((), f(*x, **kwargs)), (), xs)
    return ys


def map_vmap(
    f: F,
    in_axes: int | None | Sequence[Any] = 0,
    out_axes: Any = 0,
    *,
    batch_size: int,
) -> F:
    """jax.vmap, but looping when the batch dimension exceeds batch_size."""

    def preprocess(x, in_axis):
        batch_count = x.shape[in_axis] // batch_size
        x = jax.numpy.moveaxis(x, in_axis, 0)
        loop_elements = batch_count * batch_size
        x_loop = x[:loop_elements].reshape((batch_count, batch_size) + x.shape[1:])
        x_tail = x[loop_elements:]
        return x_loop, x_tail

    def postprocess(x_loop, x_tail, out_axis):
        shape = x_loop.shape
        x_loop = x_loop.reshape((shape[0] * shape[1],) + shape[2:])
        x = jax.numpy.concatenate([x_loop, x_tail], axis=0)
        return jax.numpy.moveaxis(x, 0, out_axis)

    @wraps(f)
    def _batch_vmap_wrapper(*args, **kwargs):
        if isinstance(in_axes, int) or in_axes is None:
            in_axes_tuple = (in_axes,) * len(args)
        else:
            in_axes_tuple = tuple(in_axes)

        broadcasts_args: list[Any] = []
        batched_args: list[Any] = []
        remainder_args: list[Any] = []

        for i, (arg, in_axis) in enumerate(zip(args, in_axes_tuple)):
            if in_axis is None:
                broadcasts_args.append((i, arg))
            elif isinstance(in_axis, int):
                loop_arg, tail_arg = _tree_map_multi_output(
                    partial(preprocess, in_axis=in_axis), arg
                )
                batched_args.append(loop_arg)
                remainder_args.append(tail_arg)
            else:
                loop_arg, tail_arg = _tree_map_multi_output(preprocess, arg, in_axis)
                batched_args.append(loop_arg)
                remainder_args.append(tail_arg)

        def vmap_f(*args, **kwargs):
            args2 = list(args)
            for i, arg in broadcasts_args:
                args2.insert(i, arg)
            return f(*args2, **kwargs)

        loop_out = _lax_map(jax.vmap(vmap_f), *batched_args, **kwargs)
        tail_out = jax.vmap(vmap_f)(*remainder_args, **kwargs)
        if isinstance(out_axes, int):
            out = tree_map(partial(postprocess, out_axis=out_axes), loop_out, tail_out)
        else:
            out = tree_map(postprocess, loop_out, tail_out, out_axes)
        return out

    return _batch_vmap_wrapper  # type: ignore


def quad(f, a, b):
    """
    Fixed order (order=16) Gauss-Legendre quadrature for integration of f(x)
    from x=a to x=b
    """
    glx, glw = np.polynomial.legendre.leggauss(16)
    glx = jnp.asarray(0.5 * (glx + 1))
    glw = jnp.asarray(0.5 * glw)
    x_i = a + (b - a) * glx
    return (b - a) * f(x_i) @ glw


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
