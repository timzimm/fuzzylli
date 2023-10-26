import gc
import logging
import warnings
import jax
import jax.numpy as jnp
from mpi4py import MPI
import mpi4jax
import numpy as np
from numba import njit
from jax_tqdm import scan_tqdm
from scipy.optimize import root_scalar

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def density_contrast(rho, comm):
    eps = 1e-12
    rank = comm.Get_rank()
    if rank == 0:
        logger.info("Compute density contrast")
    rho_sum_loc = np.sum(rho)
    N_loc = np.prod(rho.shape)
    rho_mean = 0.0
    rho_sum = comm.reduce(rho_sum_loc, op=MPI.SUM, root=0)
    N = comm.reduce(N_loc, op=MPI.SUM, root=0)
    if rank == 0:
        rho_mean = rho_sum / N
        logger.info(f"Mean density {rho_mean}")
    rho_mean = comm.bcast(rho_mean, root=0)
    return rho / rho_mean - 1.0 if rho_mean > eps else rho


@njit
def _bin_delta_k(delta_k, mode_limits):
    n = delta_k.shape[0]
    ikmin = int(-n / 2)
    iksqrmax = int(3 * ikmin**2)
    nloc = (mode_limits[:, 1] - mode_limits[:, 0]) + 1
    ptot2, nptot2 = np.zeros(iksqrmax + 1), np.zeros(iksqrmax + 1)
    for ikx in range(mode_limits[0][0], mode_limits[0][1] + 1):
        for iky in range(mode_limits[1][0], mode_limits[1][1] + 1):
            # xy plane
            ik = np.array([ikx, iky, mode_limits[2][0]])
            ik = np.where(ik < n // 2, ik, ik - n)
            ik2 = np.sum(ik**2)
            prefactor = 2 if ik[2] != 0 else 1
            ik = ik % nloc
            ptot2[ik2] += prefactor * np.abs(delta_k[ik[0], ik[1], ik[2]]) ** 2
            nptot2[ik2] += prefactor
            for ikz in range(mode_limits[2][0] + 1, mode_limits[2][1]):
                ik = np.array([ikx, iky, ikz])
                ik = np.where(ik < n / 2, ik, ik - n)
                ik2 = np.sum(ik**2)
                ik = ik % nloc
                ptot2[ik2] += 2 * np.abs(delta_k[ik[0], ik[1], ik[2]]) ** 2
                nptot2[ik2] += 2
    return ptot2, nptot2


def powerspectrum(delta_k, L, comm):
    rank = comm.Get_rank()
    size = comm.Get_size()
    ij = comm.Get_coords(rank)
    n = delta_k.shape[0]
    chunks_i = n // delta_k.shape[1]
    chunks_j = size // chunks_i
    n_i = np.split(np.arange(n), chunks_i)[ij[0]]
    n_j = np.array_split(np.arange(n // 2 + 1), chunks_j)[ij[1]]
    mode_limits = np.array(
        [
            [0, n - 1],
            [n_i[0], n_i[-1]],
            [n_j[0], n_j[-1]],
        ],
        dtype=int,
    )
    dk = 2 * np.pi / L
    ikmax = int((n - 1) / 2)

    ptot2, nptot2 = _bin_delta_k(delta_k, mode_limits)
    ptot2_all = None
    nptot2_all = None
    if rank == 0:
        ptot, nptot, kptot = (
            np.zeros(ikmax + 1),
            np.zeros(ikmax + 1),
            np.zeros(ikmax + 1),
        )
        ptot2_all = np.zeros_like(ptot2)
        nptot2_all = np.zeros_like(nptot2)

    comm.Reduce(ptot2, ptot2_all, op=MPI.SUM, root=0)
    comm.Reduce(nptot2, nptot2_all, op=MPI.SUM, root=0)

    if rank == 0:
        for iksqr in range(1, int(np.ceil((ikmax + 0.5) ** 2))):
            ik = int(np.rint(np.sqrt(iksqr)))
            ptot[ik] += ptot2_all[iksqr]
            nptot[ik] += nptot2_all[iksqr]
            ik_float = ((iksqr - 1) ** (3 / 2) + (iksqr + 1) ** (3 / 2)) / (2.0 * iksqr)
            kptot[ik] += ik_float * nptot2_all[iksqr]

        P_k = np.where(nptot[1:] > 0, ptot[1:] / nptot[1:], 0.0)
        k = np.where(nptot[1:] > 0, kptot[1:] / nptot[1:], np.arange(1, ikmax + 1)) * dk
    else:
        k = np.empty(ikmax)
        P_k = np.empty(ikmax)
    comm.Bcast([k, MPI.DOUBLE], root=0)
    comm.Bcast([P_k, MPI.DOUBLE], root=0)
    return k, P_k * L**3


def lyman_alpha_flux_powerspectrum(delta_k, k_filter, z, L, comm, pfft_plan):
    """
    Simple Lyman-alpha model based on 1810.01915 (local Gunn-Peterson approx)
    """

    @njit
    def _filter_delta_k_in_place(delta_k, k_filter, dk, mode_limits):
        nloc = (mode_limits[:, 1] - mode_limits[:, 0]) + 1
        for ikx in range(mode_limits[0][0], mode_limits[0][1] + 1):
            for iky in range(mode_limits[1][0], mode_limits[1][1] + 1):
                for ikz in range(mode_limits[2][0], mode_limits[2][1] + 1):
                    ik = np.array([ikx, iky, ikz])
                    ik = np.where(ik < n // 2, ik, ik - n)
                    k = dk * ik
                    ik = ik % nloc
                    delta_k[ik[0], ik[1], ik[2]] *= np.exp(-k @ k / k_filter**2)

    def mean_flux(A, F_at_A_1, comm):
        """
        Compute mean flux across all processes. This routine is basically the
        sam as density_contrast() above
        """
        F = F_at_A_1**A
        F_sum_loc = np.sum(F)
        N_loc = np.prod(F_at_A_1.shape)
        F_sum = comm.reduce(F_sum_loc, op=MPI.SUM, root=0)
        N = comm.reduce(N_loc, op=MPI.SUM, root=0)
        F_mean = 0.0
        if rank == 0:
            F_mean = F_sum / N
        F_mean = comm.bcast(F_mean, root=0)
        return F_mean

    def mean_flux_target(z):
        """
        Ground truth taken from 1306.2314
        """
        return 0.751 * ((1 + z) / 4.5) ** (2.9) - 0.132

    rank = comm.Get_rank()
    size = comm.Get_size()
    ij = comm.Get_coords(rank)
    n = delta_k.shape[0]
    chunks_i = n // delta_k.shape[1]
    chunks_j = size // chunks_i
    n_i = np.split(np.arange(n), chunks_i)[ij[0]]
    n_j = np.array_split(np.arange(n // 2 + 1), chunks_j)[ij[1]]
    mode_limits = np.array(
        [
            [0, n - 1],
            [n_i[0], n_i[-1]],
            [n_j[0], n_j[-1]],
        ],
        dtype=int,
    )
    dk = 2 * np.pi / L
    if rank == 0:
        logger.info(f"Filter powerspectrum with k_f = {k_filter}")
    _filter_delta_k_in_place(delta_k, k_filter, dk, mode_limits)
    comm.Barrier()

    if rank == 0:
        logger.info("Compute optical depth tau according to FPGA")
    F_at_A_1 = pfft_plan.backward(delta_k)
    F_at_A_1[:] = np.exp(-((F_at_A_1[:] + 1) ** 2))
    comm.Barrier()

    if rank == 0:
        logger.info(f"Normalize to mean flux at z={z}")
    sol = root_scalar(
        lambda A: mean_flux(A, F_at_A_1, comm) - mean_flux_target(z),
        bracket=(0.0, 1.0),
        method="brentq",
    )
    A = sol.root
    if rank == 0:
        logger.info(f"Normalization A={A}")
    comm.Barrier()

    if rank == 0:
        logger.info("Compute 3D flux powerspectrum")
    delta_F = density_contrast(F_at_A_1**A, comm)
    delta_k = pfft_plan.forward(delta_F)
    k, Pf_3D = powerspectrum(delta_k, L, comm)
    comm.Barrier()

    if rank == 0:
        logger.info("Compute 1D flux powerspectrum")
    Pf_1D = (
        1.0
        / (2 * np.pi)
        * np.cumsum((np.diff(k) * (k[:-1] * Pf_3D[:-1] + k[1:] * Pf_3D[1:]) / 2)[::-1])
    )
    return k[:-1], Pf_1D[::-1]


def xi(r, delta_k, L, comm):
    rank = comm.Get_rank()
    size = comm.Get_size()
    ij = comm.Get_coords(rank)

    n = delta_k.shape[0]
    chunks_i = n // delta_k.shape[1]
    chunks_j = size // chunks_i
    n_i = np.split(np.arange(n), chunks_i)[ij[0]]
    n_j = np.array_split(np.arange(n // 2 + 1), chunks_j)[ij[1]]
    mode_limits = np.array(
        [
            [0, n - 1],
            [n_i[0], n_i[-1]],
            [n_j[0], n_j[-1]],
        ],
        dtype=int,
    )

    dk = 2 * np.pi / L
    n = delta_k.shape[0]
    ikmin = int(-n / 2)
    iksqrmax = int(3 * ikmin**2)
    ptot2, nptot2 = _bin_delta_k(delta_k, mode_limits)

    ptot2_all = None
    nptot2_all = None
    if rank == 0:
        ptot2_all = np.zeros_like(ptot2)
        nptot2_all = np.zeros_like(nptot2)
    comm.Reduce(ptot2, ptot2_all, op=MPI.SUM, root=0)
    comm.Reduce(nptot2, nptot2_all, op=MPI.SUM, root=0)

    if rank == 0:
        iksq = np.arange(1, iksqrmax + 1)
        xi = np.sum(
            np.sinc(1.0 / np.pi * r[:, np.newaxis] * np.sqrt(iksq) * dk)
            * ptot2_all[np.newaxis, 1:],
            axis=1,
        )
    else:
        xi = np.zeros_like(r)
    comm.Bcast([xi, MPI.DOUBLE], root=0)
    return xi


def sample_S3(key, N):
    """
    Sample N points uniformly in S^3
    """
    xyz_key, r_key, key = jax.random.split(key, 3)
    xyz = jax.random.normal(xyz_key, shape=(N, 3))
    r = jax.random.uniform(r_key, shape=(N, 1))
    samples = r ** (1 / 3) * xyz / jnp.linalg.norm(xyz, axis=-1, keepdims=True)
    return key, samples


def compute_phase_field(rho_k):
    epsilon = 10 * np.finfo(rho_k.dtype).eps
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        phase_k = np.where(np.abs(rho_k) > epsilon, rho_k / np.abs(rho_k), 0.0)
    return phase_k


def line_correlator_stochastic(r, eps, domain, comm, M=200, N=100000, seed=42):
    @jax.vmap
    def mc_integral(r, k, q, eps_k, eps_q, eps_mkmq, k_max_idx, N_modes):
        def mc_integrand(i, l_r):
            # A piecewise constant aproximation of w(|k-q|r)*B(k,q)
            weight_kq = jax.lax.cond(
                jnp.linalg.norm(k[i] + q[i]) <= k_max_idx,
                jnp.sinc,
                lambda x: 0.0,
                2.0 / k_max_idx * jnp.linalg.norm(k[i] - q[i]),
            )
            # Note that FFTs mode ordering works with negative indexing from the
            # back of the array, so no reordering necessary
            bispectrum_kq_mkmq = jnp.real(eps_k[i] * eps_q[i] * eps_mkmq[i])
            return weight_kq * bispectrum_kq_mkmq + l_r

        return (
            jax.lax.fori_loop(0, N, mc_integrand, 0.0)
            / N
            * N_modes**2
            * r ** (9 / 2)
            / domain.V ** (3 / 2)
        )

    def gather_local_phases(k_int, q_int, eps, extend):
        def gather_local_phase(_, k_int_q_int):
            k_int, q_int = k_int_q_int

            nnn = jnp.array([n, n, n], dtype=jnp.int32)
            mkmq_int = (-k_int - q_int) % nnn

            def acquire(_, samples_idx):
                idx = samples_idx
                idx_loc = idx - extend[:, 0]
                eps_i_loc_real = jax.lax.cond(
                    jnp.all(jnp.logical_and(extend[:, 0] <= idx, idx <= extend[:, 1])),
                    lambda i: eps[i[0], i[1], i[2]].real,
                    lambda i: -1.0,
                    idx_loc,
                )
                eps_i_loc_imag = jax.lax.cond(
                    jnp.all(jnp.logical_and(extend[:, 0] <= idx, idx <= extend[:, 1])),
                    lambda i: eps[i[0], i[1], i[2]].imag,
                    lambda i: -1.0,
                    idx_loc,
                )

                idx = (nnn - samples_idx) % nnn
                idx_loc = idx - extend[:, 0]
                eps_i_loc_real = jax.lax.cond(
                    jnp.all(jnp.logical_and(extend[:, 0] <= idx, idx <= extend[:, 1])),
                    lambda i: eps[i[0], i[1], i[2]].real,
                    lambda i: eps_i_loc_real,
                    idx_loc,
                )
                eps_i_loc_imag = jax.lax.cond(
                    jnp.all(jnp.logical_and(extend[:, 0] <= idx, idx <= extend[:, 1])),
                    lambda i: -eps[i[0], i[1], i[2]].imag,
                    lambda i: eps_i_loc_imag,
                    idx_loc,
                )
                eps_i_loc = jnp.array([eps_i_loc_real, eps_i_loc_imag])
                return None, eps_i_loc

            _, eps_loc = jax.lax.scan(
                acquire, None, jnp.array([k_int, q_int, mkmq_int])
            )
            return None, (eps_loc[0], eps_loc[1], eps_loc[2])

        return jax.lax.scan(gather_local_phase, None, (k_int, q_int))[1]

    @scan_tqdm(M, print_rate=1)
    def lr_realisation(key_token, n):
        (key, token) = key_token
        key, k = sample_S3(key, R * N)
        key, q = sample_S3(key, R * N)
        k = (k_max_idx[:, None, None] * k.reshape(R, N, 3)).reshape(R * N, 3)
        q = (k_max_idx[:, None, None] * q.reshape(R, N, 3)).reshape(R * N, 3)

        k_int = (nnn + jnp.rint(k).astype(jnp.int32)) % nnn
        q_int = (nnn + jnp.rint(q).astype(jnp.int32)) % nnn

        eps_k_loc, eps_q_loc, eps_mkmq_loc = gather_local_phases(
            k_int, q_int, eps, extend
        )

        eps_k, token = mpi4jax.reduce(eps_k_loc, MPI.MAX, 0, comm=comm, token=token)
        eps_q, token = mpi4jax.reduce(eps_q_loc, MPI.MAX, 0, comm=comm, token=token)
        eps_mkmq, token = mpi4jax.reduce(
            eps_mkmq_loc, MPI.MAX, 0, comm=comm, token=token
        )

        lr = 0
        if rank == 0:
            k = k.reshape(R, N, 3)
            q = q.reshape(R, N, 3)
            eps_k = (eps_k[:, 0] + 1.0j * eps_k[:, 1]).reshape(R, N)
            eps_q = (eps_q[:, 0] + 1.0j * eps_q[:, 1]).reshape(R, N)
            eps_mkmq = (eps_mkmq[:, 0] + 1.0j * eps_mkmq[:, 1]).reshape(R, N)

            lr = mc_integral(r, k, q, eps_k, eps_q, eps_mkmq, k_max_idx, N_modes)
        return (key, token), lr

    rank = comm.Get_rank()
    size = comm.Get_size()
    ij = comm.Get_coords(rank)
    key = jax.random.PRNGKey(seed)

    eps = jnp.asarray(eps)

    R = r.shape[0]
    n = eps.shape[0]
    chunks_i = n // eps.shape[1]
    chunks_j = size // chunks_i
    n_i = np.split(np.arange(n), chunks_i)[ij[0]]
    n_j = np.array_split(np.arange(n // 2 + 1), chunks_j)[ij[1]]
    extend = jnp.array(
        [
            [0, n - 1],
            [n_i[0], n_i[-1]],
            [n_j[0], n_j[-1]],
        ]
    )

    k_max_idx = jnp.minimum(
        (2 * jnp.pi / r) / domain.dki[0],
        (n - 1) // 2,
    )
    N_modes = (4 * jnp.pi / 3) * k_max_idx**3

    nnn = jnp.array([n, n, n], dtype=jnp.int32)
    (key, token), l_r = jax.lax.scan(
        lr_realisation, (key, jax.lax.create_token()), np.arange(M)
    )

    l_r_mean = jnp.empty(R)
    l_r_std = jnp.empty(R)
    if rank == 0:
        l_r_mean = jnp.mean(l_r, axis=0)
        l_r_std = jnp.std(l_r, axis=0) / np.sqrt(M)

    l_r_mean, token = mpi4jax.bcast(l_r_mean, 0, comm=comm, token=token)
    l_r_std, token = mpi4jax.bcast(l_r_std, 0, comm=comm, token=token)
    return l_r_mean, l_r_std
