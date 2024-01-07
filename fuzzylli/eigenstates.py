import hashlib
import logging
from collections import namedtuple

import numpy as np
import jax.numpy as jnp
from jax import vmap
from scipy.linalg import eigh_tridiagonal
from jaxopt import Broyden, Bisection

from fuzzylli.domain import UniformHypercube
from fuzzylli.interpolation_jax import init_1d_interpolation_params
from fuzzylli.potential import E_c
from fuzzylli.wavefunction import L
from fuzzylli.utils import quad

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

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
    def compute_name(cls, scalefactor, V, R, N):
        H_domain = UniformHypercube([N], np.array([0, 2 * R]))
        r = H_domain.cell_interfaces[0]
        H_diag, H_off_diag = construct_cylindrical_hamiltonian(r, 0, scalefactor, V)
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
            - 18
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

    return Rmin, Rmax


def init_eigenstate_library(scalefactor, V, R, N):
    """
    Compute the eigenstate Library for axialsymmetric potential V.

    """

    init_mult_spline_params = vmap(init_1d_interpolation_params, in_axes=(0, 0, 0))

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
    dr = []

    ll = 0
    E_max = V(R)
    while True:
        R_min, R_max = wkb_estimate_of_R(V, E_max, ll, scalefactor)
        H_domain = UniformHypercube([N], np.array([R_min, R_max]))
        logger.info(f"Hamiltonian domain = {H_domain.extends}")
        r = H_domain.cell_interfaces[0]

        H_diag, H_off_diag = construct_cylindrical_hamiltonian(r, ll, scalefactor, V)
        E_min_ll = E_min(L(ll))

        # If circular orbit above energy cutoff...stop
        if E_min_ll > E_max:
            break
        logger.info(f"l={ll}, E_min = {E_min_ll:4f}, E_max={E_max:4f}")
        E_n, u_n = eigh_tridiagonal(
            H_diag, H_off_diag, select="v", select_range=(E_min_ll, E_max)
        )
        # If no mode exists in interval...stop
        if E_n.shape[0] == 0:
            logger.info(f"No modes inside [{E_min_ll:4f}, {E_max:4f}]")
            break

        # Add noise floor to all modes so that the number of roots is equal
        # to n
        u_n = u_n + 10 * jnp.finfo(jnp.float64).eps

        # Check if states are degenerate (this is impossible in 1D and bound,
        # normalizable states. If it happens, numerics is tha cause.
        # TODO: Maybe
        if np.any(np.unique(E_n, return_counts=True)[1] > 1):
            logger.warning(
                "Degeneracy detected. This is impossible in 1D. "
                "Consider tweaking N/L."
            )
            break

        R_n = u_n / np.sqrt(r[:, np.newaxis] * (r[1] - r[0]))
        R_j_R.append(jnp.asarray(R_n.T))
        E_j.append(jnp.asarray(E_n))
        l_of_j.append(ll * jnp.ones_like(E_n))
        n_of_j.append(jnp.arange(E_n.shape[0]))
        R_0.append(jnp.repeat(r[0], E_j[-1].shape[0]))
        dr.append(jnp.repeat(r[1] - r[0], E_j[-1].shape[0]))
        logger.info(f"{E_j[-1].shape[0]} eigenfunctions found")

        ll += 1

    R_j_R = jnp.concatenate(R_j_R)
    E_j = jnp.concatenate(E_j)
    l_of_j = jnp.concatenate(l_of_j)
    n_of_j = jnp.concatenate(n_of_j)
    R_0 = jnp.concatenate(R_0)
    dr = jnp.concatenate(dr)
    R_j_params = init_mult_spline_params(R_0, dr, R_j_R)

    return eigenstate_library(
        R_j_params=R_j_params,
        E_j=E_j,
        l_of_j=l_of_j,
        n_of_j=n_of_j,
    )


def construct_cylindrical_hamiltonian(r, ll, scalefactor, V):
    N = r.shape[0]
    dr = r[1] - r[0]
    n = np.arange(1, N + 1)
    H_off_diag = -1 / (2 * scalefactor**2 * dr**2) * np.ones(N - 1)
    # Construct diagonal elements of the Hamlitonian with modification of the
    # angular momentum barrier to make the finite difference approximation
    # accurate at r=0. (The solution goes as u ~ r^{ll+1/2} for r -> 0,
    # i.e. non-polynomial and thus hard to approximate with standard finite
    # differences formulas which are derived via Taylor expansion
    # (polynomials!)). The diagonal term modification is Eq. 11. from
    # 1807.01392.
    H_diag = (
        -1 / (2 * scalefactor**2 * dr**2) * -2 * np.ones(N)
        + 0.5
        * (n**2 * ((1 - 1 / n) ** (ll + 1 / 2) + (1 + 1 / n) ** (ll + 1 / 2) - 2))
        / (scalefactor * r) ** 2
        + V(r)
    )
    return H_diag, H_off_diag
