import hashlib
import logging
from collections import namedtuple

import numpy as np
import jax.numpy as jnp
from jax import vmap, grad
from scipy.linalg import eig, eigh_tridiagonal
from jaxopt import Broyden, Bisection

from fuzzylli.domain import UniformHypercube
from fuzzylli.interpolation_jax import init_1d_interpolation_params
from fuzzylli.potential import E_c
from fuzzylli.wavefunction import L
from fuzzylli.utils import quad
from fuzzylli.special import lambertw
from fuzzylli.chebyshev import (
    chebyshev_pts,
    chebyshev_d2x,
    barycentric_interpolation,
    clenshaw_curtis_weights,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

_R_j_params = namedtuple("_R_j_params", ["R_k", "X_min", "X_max", "a", "b"])

_eigenstate_library = namedtuple(
    "eigenstate_library", ["R_j_params", "E_j", "l_of_j", "n_of_j"]
)

a = 1.0
b = 0.0


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
    def compute_name(cls, scalefactor, V, R, N):
        H_domain = UniformHypercube([N], np.array([0, 2 * R]))
        r = H_domain.cell_interfaces[0]
        H_diag, H_off_diag = construct_tridiagonal_hamiltonian(r, 0, scalefactor, V)
        return hashlib.md5(
            H_diag.tobytes()
            + np.array([scalefactor]).tobytes()
            + np.array([V(R)]).tobytes()
        )


def wkb_estimate_of_R(V, Emax, l, scalefactor):
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
            - 36
        )

    # Determine classically allowed region (ignoring barrier)
    broyden = Broyden(fun=R_classical_V)
    Rmax = 1.05 * broyden.run(jnp.array(1.0)).params
    Rmin = 0.95 * jnp.nan_to_num(
        jnp.sqrt((l**2 - 1.0 / 4) / (2 * scalefactor**2 * Emax))
    )

    # Determine radius according to WKB decay in forbidden region
    bisec = Bisection(
        optimality_fun=lambda R: wkb_condition_Veff(Rmax, R),
        lower=Rmax,
        upper=10 * Rmax,
    )
    Rmax = bisec.run().params
    if wkb_condition_Veff(0, Rmin) > 0:
        bisec = Bisection(
            optimality_fun=lambda R: wkb_condition_Veff(R, Rmin),
            lower=0,
            upper=Rmin,
        )
        Rmin = bisec.run(Rmin).params
    else:
        Rmin = jnp.finfo(float).eps ** (2 / (2 * l + 1)) * Rmin
    Rmin = max(Rmin, 1e-6)

    return Rmin, Rmax


def check_mode_heath(E_n, E_min, E_max):
    if E_n.shape[0] == 0:
        logger.error(f"No modes inside [{E_min:.2f}, {E_max:2f}]")
        raise Exception()
    if np.any(np.unique(E_n, return_counts=True)[1] > 1):
        logger.error("Degeneracy detected. This is impossible in 1D.")
        raise Exception()


def init_eigenstate_library(scalefactor, V, R, N):
    """
    Compute the eigenstate Library for axialsymmetric potential V.

    """

    @vmap
    def init_mult_R_j_params(R_k, X_min, X_max, a, b):
        return _R_j_params(R_k=R_k, X_min=X_min, X_max=X_max, a=a, b=b)

    init_mult_spline_params = vmap(init_1d_interpolation_params, in_axes=(0, 0, 0))
    eval_cheb_eigenstates = vmap(
        vmap(
            lambda R, R_j_params: barycentric_interpolation(
                x_of_X(
                    jnp.clip(
                        X_of_R(R, R_j_params.a, R_j_params.b),
                        R_j_params.X_min,
                        R_j_params.X_max,
                    ),
                    R_j_params.X_min,
                    R_j_params.X_max,
                ),
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

    dXdR = vmap(grad(lambda R: X_of_R(R, a, b)))

    # Discetized radial eigenstate
    R_j_R = []
    # Eigenvalues
    E_j = []
    # Quantum numbers
    l_of_j = []
    n_of_j = []
    R_0 = []
    dR = []

    E_max = V(R)
    l_max = 0
    while E_min(L(l_max)) < E_max:
        l_max += 1
    l_max -= 1
    logger.info(f"l_max = {l_max}")

    # Construct chebyshev derivative operator in normalized x space
    # (log-linear grid)
    N_cheb = 2**10
    x, d2dx = chebyshev_d2x(N_cheb)
    # Impose boundary condition
    x = x[1:-1]
    d2dx = d2dx[1:-1, 1:-1]
    # Clenshaw-Curis weights for Dirichlet BC ( f(-1) = f(1) = 0 )
    weights = jnp.asarray(clenshaw_curtis_weights(N_cheb)[1:-1])

    for l in range(l_max):
        E_min_l = E_min(L(l))
        R_min, R_max = wkb_estimate_of_R(V, E_max, l, scalefactor)
        X_min, X_max = X_of_R(R_min, a, b), X_of_R(R_max, a, b)
        H_domain = UniformHypercube([N], np.array([R_min, R_max]))

        logger.info(f"Hamiltonian domain = [{R_min:.1e},{R_max:.1e}]")
        R_nonuniform = R_of_X(X_of_x(x, X_min, X_max), a, b)
        R_uniform = H_domain.cell_interfaces[0]

        if l < 10:
            H_diag, H_off_diag = construct_tridiagonal_hamiltonian(
                R_uniform, l, scalefactor, V
            )

            E_n, u_n = eigh_tridiagonal(
                H_diag, H_off_diag, select="v", select_range=(E_min_l, E_max)
            )
            check_mode_heath(E_n, E_min_l, E_max)

            R_n = u_n / np.sqrt(
                R_uniform[:, np.newaxis] * (R_uniform[1] - R_uniform[0])
            )
            R_n = R_n.T

        else:
            # Rescale derivative from normalized x-space [-1, 1] to [X_min, X_max]
            d2dX = 4.0 / (X_max - X_min) ** 2 * d2dx
            H = construct_chebyshev_hamiltonian(R_nonuniform, l, scalefactor, V, d2dX)

            E_n, t_n = eig(H)
            check_mode_heath(E_n, E_min_l, E_max)

            order = jnp.argsort(E_n)
            E_n = E_n[order].real
            t_n = t_n[:, order].real
            t_n = t_n[:, E_n <= E_max]
            E_n = E_n[E_n <= E_max]

            norm = (X_max - X_min) / 2 * weights @ (t_n**2)
            R_n = t_n / jnp.sqrt(
                norm * (dXdR(R_nonuniform) * R_nonuniform)[:, jnp.newaxis]
            )

            R_n_params = init_mult_R_j_params(
                R_n.T,
                jnp.repeat(X_min, E_n.shape[0]),
                jnp.repeat(X_max, E_n.shape[0]),
                jnp.repeat(a, E_n.shape[0]),
                jnp.repeat(b, E_n.shape[0]),
            )
            R_n = eval_cheb_eigenstates(R_uniform, R_n_params)

        logger.info(
            f"l={l}: V_eff_min = {E_min_l:.2f} <= "
            f"E_0 = {E_n[0]:.2f} <= "
            f"E_max={E_max:.2f}"
        )

        # Add noise floor to all modes so that the number of roots is equal
        # to n
        u_n = u_n + 10 * jnp.finfo(jnp.float64).eps

        R_j_R.append(jnp.asarray(R_n))
        E_j.append(jnp.asarray(E_n))
        l_of_j.append(l * jnp.ones_like(E_n))
        n_of_j.append(jnp.arange(E_n.shape[0]))
        R_0.append(jnp.repeat(R_uniform[0], E_n.shape[0]))
        dR.append(jnp.repeat(R_uniform[1] - R_uniform[0], E_n.shape[0]))
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


def construct_tridiagonal_hamiltonian(R, l, scalefactor, V):
    N = R.shape[0]
    dR = R[1] - R[0]
    n = np.arange(1, N + 1)
    H_off_diag = -1 / (2 * scalefactor**2 * dR**2) * np.ones(N - 1)
    # Construct diagonal elements of the Hamlitonian with modification of the
    # angular momentum barrier to make the finite difference approximation
    # accurate at r=0. (The solution goes as u ~ r^{ll+1/2} for r -> 0,
    # i.e. non-polynomial and thus hard to approximate with standard finite
    # differences formulas which are derived via Taylor expansion
    # (polynomials!)). The diagonal term modification is Eq. 11. from
    # 1807.01392.
    H_diag = (
        -1 / (2 * scalefactor**2 * dR**2) * -2 * np.ones(N)
        + 0.5
        * (n**2 * ((1 - 1 / n) ** (l + 1 / 2) + (1 + 1 / n) ** (l + 1 / 2) - 2))
        / (scalefactor * R) ** 2
        + V(R)
    )
    return H_diag, H_off_diag


def construct_chebyshev_hamiltonian(R, l, scalefactor, V, d2dX):
    return (
        -1.0
        / (2 * scalefactor**2)
        * (
            np.diag((b + a * R) ** 2 / R**2) @ d2dX
            + jnp.diag(
                (0.25 - l**2) / R**2
                - b * (b + 4 * a * R) / (4 * (b + a * R) ** 2 * R**2)
            )
        )
    ) + jnp.diag(V(R))


def x_of_X(X, X_min, X_max):
    return -1.0 + 2 * (X - X_min) / (X_max - X_min)


def X_of_x(x, X_min, X_max):
    return X_min + (x + 1) * (X_max - X_min) / 2


def X_of_R(R, a, b):
    return a * R + b * jnp.log(R)


def R_of_X(X, a, b):
    if b == 0.0 and a > 0:
        return X / a
    if a == 0.0 and b > 0:
        return jnp.exp(X / b)
    return b / a * lambertw(a / b * jnp.exp(X / b))


dRdX = grad(lambda X: R_of_X(X, a, b))
dXdR = grad(lambda R: X_of_R(R, a, b))
