from distributed import get_client, secede, rejoin

import jax
from collections import namedtuple
import hashlib
import jax.numpy as jnp
from scipy.fft import fht
import numpy as np

from fuzzylli.interpolation_jax import init_1d_interpolation_params, eval_interp1d
from fuzzylli.io_utils import hash_to_int64
from fuzzylli.eigenstates import (
    eval_eigenstate,
)
from fuzzylli.wavefunction import _reinit_random_phases
from fuzzylli.utils import map_vmap

_normalised_polar_powerspectrum = namedtuple(
    "normalised_polar_powerspectrum", ["P_params"]
)


class normalised_polar_powerspectrum(_normalised_polar_powerspectrum):
    @classmethod
    def compute_name(cls, psi_params_name, r_min, r_max, L, N_H, offset, bias):
        combined = hashlib.sha256()
        combined.update(hashlib.md5(jnp.array(psi_params_name)).digest())
        combined.update(hashlib.md5(jnp.array(r_min)).digest())
        combined.update(hashlib.md5(jnp.array(r_max)).digest())
        combined.update(hashlib.md5(jnp.array(L)).digest())
        combined.update(hashlib.md5(jnp.array(N_H)).digest())
        combined.update(hashlib.md5(jnp.array(offset)).digest())
        combined.update(hashlib.md5(jnp.array(bias)).digest())
        return hash_to_int64(combined.hexdigest())


def init_normalised_polar_powerspectrum(
    psi_params, r_min, r_max, L, N_H, offset, bias, keys
):
    client = get_client()
    r = jnp.logspace(jnp.log10(r_min), jnp.log10(r_max), N_H)
    K = jnp.exp(offset) / r[::-1]

    u_k = client.map(
        lambda key: compute_normalised_polar_powerspectrum(
            K, psi_params, L, offset, bias, key
        ),
        keys,
    )
    secede()
    u_k = jnp.array(client.gather(u_k))
    rejoin()

    P_iso = jnp.mean(jnp.mean(jnp.abs(u_k) ** 2, axis=-1), axis=0)
    logK = jnp.log10(K)
    return normalised_polar_powerspectrum(
        P_params=init_1d_interpolation_params(logK[0], logK[1] - logK[0], P_iso)
    )


def eval_normalised_polar_powerspectrum(K, normalised_polar_powerspectrum):
    return eval_interp1d(jnp.log10(K), normalised_polar_powerspectrum.P_params)


_eval_library = jax.vmap(eval_eigenstate, in_axes=(None, 0))
_correlate = jax.vmap(
    jax.vmap(lambda a, b: jnp.correlate(a, b, mode="full"), in_axes=(0, None)),
    in_axes=(None, 0),
)


def compute_normalised_polar_powerspectrum(K, psi_params, L, offset, bias, key):
    def _construct_f_m(r, aR_nl):
        def populate(j, aR_nl):
            aR_nl = aR_nl.at[n_of_j[j], l_max - l_of_j[j]].add(
                a_j[j] * R_j[j] * phase_j[j, 0]
            )
            aR_nl = aR_nl.at[n_of_j[j], l_max + l_of_j[j]].add(
                a_j[j] * R_j[j] * phase_j[j, 1]
            )

            return aR_nl

        R_j = jnp.sqrt(psi_params.total_mass / (2 * jnp.pi)) * _eval_library(
            r, lib.R_j_params
        )
        aR_nl_r = jax.lax.fori_loop(0, lib.J, populate, aR_nl)

        f_m = jnp.sum(_correlate(aR_nl_r, aR_nl_r), axis=(0, 1))

        return f_m

    def _H_m(cmplx_m, m, bias):
        cmplx_m = np.asarray(cmplx_m)
        real = fht(cmplx_m.real, dln, mu=m, bias=bias)
        imag = fht(cmplx_m.imag, dln, mu=m, bias=bias)
        return jnp.asarray(real + 1.0j * imag)

    def _construct_u_m(r_f_m, m, rho, bias):
        hankel_f_m_rho = _H_m(r_f_m, m, bias)
        return 2 * jnp.pi * 1.0j**-m * jnp.array(hankel_f_m_rho) / rho

    client = get_client()
    psi_params = psi_params.persist()
    psi_params = psi_params.compute()

    lib = psi_params.eigenstate_library
    n_of_j = lib.n_of_j
    l_of_j = (lib.l_of_j).astype(int)
    n_max = jnp.max(n_of_j)
    l_max = l_of_j[-1]
    a_j = psi_params.a_j
    phase_new = _reinit_random_phases(lib, key)
    phase_j = psi_params.phase_j * phase_new

    r = jnp.exp(offset) / K[::-1]
    dln = jnp.log(r[1] / r[0])

    # # Compute azimuthal Fourier coefficients for normalised |psi|^2
    aR_nl = jnp.zeros((n_max + 1, 2 * l_max + 1), dtype=complex)
    f_m = map_vmap(lambda r: _construct_f_m(r, aR_nl), batch_size=16)(r)
    # f_m = jax.lax.map(lambda r: _construct_f_m(r, aR_nl), r)

    # Hankel transform each coefficient to get spectral representation in log uniform rho
    m_max = (f_m.shape[-1] - 1) // 2
    m = jnp.arange(m_max + 1)

    u_m_rho = client.map(
        lambda *args: _construct_u_m(*args, K, bias), r * f_m[:, m_max:].T, m
    )
    secede()
    u_m_rho = jnp.array(client.gather(u_m_rho)).T
    rejoin()

    u_m_neg_rho = (-1) ** m[None, :0:-1] * jnp.conjugate(u_m_rho[:, :0:-1])
    u_m_rho = jnp.hstack([u_m_rho, u_m_neg_rho])

    # Sum over all m modes to get spectral representation in (rho, omega)
    u_mn = jnp.fft.ifft(u_m_rho, norm="forward", axis=-1)
    return u_mn / (psi_params.total_mass * L)
