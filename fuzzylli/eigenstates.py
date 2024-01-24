import hashlib
import logging
from collections import namedtuple

import numpy as np
import jax.numpy as jnp
from jax import vmap
from scipy.linalg import eig
from jaxopt import ProjectedGradient, Bisection
from jaxopt.projection import projection_non_negative

from fuzzylli.interpolation_jax import init_1d_interpolation_params
from fuzzylli.potential import E_c
from fuzzylli.utils import quad
from fuzzylli.special import lambertw
from fuzzylli.chebyshev import (
    chebyshev_pts,
    chebyshev_dx,
    chebyshev_d2x,
    barycentric_interpolation,
    clenshaw_curtis_weights,
)
from fuzzylli.interpolation_jax import eval_interp1d

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

_rescaled_cheb_params = namedtuple("_rescaled_cheb_params", ["R_k", "R_min", "R_max"])

_eigenstate_library = namedtuple(
    "eigenstate_library", ["R_j_params", "E_j", "l_of_j", "n_of_j"]
)


class eigenstate_library(_eigenstate_library):
    def j_of_nl(self, n, l):
        n = int(n)
        l_index = jnp.searchsorted(self.l_of_j, l)
        n_index = jnp.argmax(self.n_of_j[l_index:] == n)
        return l_index + n_index

    @property
    def J(self):
        return self.E_j.shape[0]

    @classmethod
    def compute_name(cls, scalefactor, V, E_max, N):
        return hashlib.md5(
            np.array([scalefactor]).tobytes()
            + np.array([V(1.0)]).tobytes()
            + np.array([E_max]).tobytes()
            + np.array([N]).tobytes()
        )


def L(l):
    """
    A heuristic mapping from quantum l to classical L. Results, i.e. fit
    accuracy seems mainly affected by the l=0 case, especially for radially
    biased (beta > 0) dispersion. In this case the DF diverges as L^(-beta)
    """
    return jnp.where(l > 0, l, 0.1)


def wkb_estimate_of_Rmax(V, Emax, l, scalefactor):
    def barrier(R):
        return (l**2 - 1.0 / 4) / (2 * scalefactor**2 * (1e-10 + R**2))

    def R_classical_Veff(R):
        return V(R) + barrier(R) - Emax

    def R_classical_V(R):
        return V(R) - Emax

    def wkb_condition_Veff(R_lower, R_upper):
        return (
            jnp.sqrt(2)
            * scalefactor
            * quad(lambda R: jnp.sqrt(R_classical_Veff(R)), R_lower, R_upper)
            - 20
        )

    # Determine classically allowed region (ignoring barrier)
    pg = ProjectedGradient(
        fun=lambda R: R_classical_V(R) ** 2, projection=projection_non_negative
    )
    Rmax = 1.05 * pg.run(jnp.array(1.0)).params

    # Determine radius according to WKB decay in forbidden region
    bisec = Bisection(
        optimality_fun=lambda R: wkb_condition_Veff(Rmax, R),
        lower=Rmax,
        upper=10 * Rmax,
    )
    Rmax = bisec.run().params

    return Rmax


def check_mode_heath(E_n, E_min, E_max):
    if E_n.shape[0] == 0:
        logger.error(f"No modes inside [{E_min:.2f}, {E_max:2f}]")
        raise Exception()
    if np.any(E_n.imag > 1e-10 * E_n.real):
        logger.error("Eigenvalue with significant imaginary part found")
        raise Exception()
    if np.any(np.unique(E_n, return_counts=True)[1] > 1):
        logger.error("Degeneracy detected. This is impossible in 1D.")
        raise Exception()


def init_eigenstate_library(scalefactor, V, E_max, N):
    """
    Compute the eigenstate Library for axialsymmetric potential V.

    """

    init_mult_spline_params = vmap(init_1d_interpolation_params, in_axes=(0, 0, 0))

    @vmap
    def init_mult_cheb_params(R_k, R_min, R_max):
        return _rescaled_cheb_params(R_k=R_k, R_min=R_min, R_max=R_max)

    eval_cheb_eigenstates = vmap(
        vmap(
            lambda R, R_j_params: barycentric_interpolation(
                -1.0
                + 2
                * (
                    jnp.clip(
                        R,
                        R_j_params.R_min,
                        R_j_params.R_max,
                    )
                    - R_j_params.R_min
                )
                / (R_j_params.R_max - R_j_params.R_min),
                chebyshev_pts(R_j_params.R_k.shape[0]),
                R_j_params.R_k,
            ),
            in_axes=(0, None),
        ),
        in_axes=(None, 0),
    )

    def E_min(L):
        """
        Lower bound for eigenvalue search based on circular orbit energy
        plus neglible offset to bypass exactly vanishing DFs at E_c(L)
        """
        epsilon = 10 * jnp.finfo(jnp.float64).eps
        return (1 + epsilon) * E_c(L, V, scalefactor)

    # Discetized radial eigenstate
    R_j_R = []
    # Eigenvalues
    E_j = []
    # Quantum numbers
    l_of_j = []
    n_of_j = []
    R_0 = []
    # R_max_j = []
    dR = []

    l_max = 0
    while E_min(L(l_max)) < E_max:
        l_max += 1
    l_max -= 1
    logger.info(f"l_max = {l_max}")

    # Construct chebyshev derivative operators in normalized x space
    x, d_dx = chebyshev_dx(N)
    _, d2_dx = chebyshev_d2x(N)

    # Impose Dirichlet boundary condition
    x = x[1:-1]
    d_dx = d_dx[1:-1, 1:-1]
    d2_dx = d2_dx[1:-1, 1:-1]

    # Clenshaw-Curis weights for Dirichlet BC ( f(-1) = f(1) = 0 )
    weights = jnp.asarray(clenshaw_curtis_weights(N)[1:-1])

    for l in range(l_max):
        E_min_l = E_min(L(l))
        R_max = wkb_estimate_of_Rmax(V, E_max, l, scalefactor)
        R_min = 1e-6
        X_max = (R_max / 1.01) + jnp.log(R_max / 1.01)
        X_min = (1.01 * R_min) + jnp.log(1.01 * R_min)

        logger.info(f"Hamiltonian domain = [{0.0:.1e},{R_max:.1e}]")
        R_nonuniform = -R_max + (x + 1) * R_max
        X_uniform_sampling = jnp.linspace(X_min, X_max, N)
        R_nonuniform_sampling = lambertw(jnp.exp(X_uniform_sampling))

        # Rescale derivative from normalized x-space [-1, 1] to [-R_max, R_max]
        d_dX = 1.0 / R_max * d_dx
        d2_dX = 1.0 / R_max**2 * d2_dx
        H = construct_chebyshev_hamiltonian_parity(
            l, scalefactor, V, d2_dX, d_dX, R_nonuniform
        )

        E_n, R_n = eig(H)
        check_mode_heath(E_n, E_min_l, E_max)
        order = jnp.argsort(E_n)
        E_n = E_n[order].real
        R_n = R_n[:, order].real
        R_n = R_n[:, E_n <= E_max]
        E_n = E_n[E_n <= E_max]

        # Extend mode beyond R=0 according to parity property on the extended
        # polar domain and normalise all modes
        R_n = np.vstack([R_n, (-1) ** l * R_n[::-1]])
        norm = R_max / 2 * weights @ (np.abs(R_nonuniform[:, np.newaxis]) * R_n**2)
        R_n = R_n / jnp.sqrt(norm)

        # Ideally we evaluate the eigenmodes Chebyshev series directly. This
        # works and is highly accurate and memory efficient. However,
        # Clenshaw's algorithm (see chebyshev.py) is O(N). Since
        # this evaluation represents the most inner loop in every application
        # that involves psi, it is not efficient. We therefore resample
        # and use the O(1) Taylor series interpolation routines as
        # alternative (see interpolation.py).
        # This is less accurate, but tractable.
        # To increase accuracy, we will not sample uniformly in R but on a
        # loglinear grid X = logR + R.
        R_n_params = init_mult_cheb_params(
            R_n.T, jnp.repeat(-R_max, E_n.shape[0]), jnp.repeat(R_max, E_n.shape[0])
        )

        # Barycentric interpolation onto a uniform domain
        R_n = eval_cheb_eigenstates(R_nonuniform_sampling, R_n_params)

        logger.info(
            f"l={l}: V_eff_min = {E_min_l:.2f} <= "
            f"E_0 = {E_n[0]:.2f} <= "
            f"E_max={E_max:.2f}"
        )

        R_j_R.append(jnp.asarray(R_n))
        E_j.append(jnp.asarray(E_n))
        l_of_j.append(l * jnp.ones_like(E_n))
        n_of_j.append(jnp.arange(E_n.shape[0]))
        R_0.append(jnp.repeat(X_uniform_sampling[0], E_n.shape[0]))
        dR.append(
            jnp.repeat(X_uniform_sampling[1] - X_uniform_sampling[0], E_n.shape[0])
        )
        logger.info(f"{E_j[-1].shape[0]} eigenfunctions found")

    R_j_R = jnp.concatenate(R_j_R)
    E_j = jnp.concatenate(E_j)
    l_of_j = jnp.concatenate(l_of_j)
    n_of_j = jnp.concatenate(n_of_j)
    R_0 = jnp.concatenate(R_0)
    dR = jnp.concatenate(dR)
    R_j_params = init_mult_spline_params(R_0, dR, R_j_R)

    return eigenstate_library(
        R_j_params=R_j_params,
        E_j=E_j,
        l_of_j=l_of_j,
        n_of_j=n_of_j,
    )


def construct_chebyshev_hamiltonian_parity(l, scalefactor, V, d2_dX, d_dX, X):
    Np = (X.shape[0] + 1) // 2

    H_full = (
        -1.0
        / (2 * scalefactor**2)
        * (d2_dX + np.diag(1.0 / X) @ d_dX + jnp.diag(-(l**2) / X**2))
    ) + jnp.diag(V(np.abs(X)))

    H1 = H_full[:Np, :Np]
    H2 = H_full[:Np, Np:]
    H2 = H2[:, ::-1]
    return H1 + (-1) ** l * H2


def eval_eigenstate(R, R_j_params):
    return eval_interp1d(R + jnp.log(R), R_j_params)
