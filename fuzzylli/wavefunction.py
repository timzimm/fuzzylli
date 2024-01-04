import hashlib
import logging
from collections import namedtuple
import matplotlib.pyplot as plt

import numpy as np
import jax
from jax import vmap, random, lax, jit, grad
from jax.scipy.special import erf
from jaxopt._src.tree_util import tree_map
import jax.numpy as jnp

from jaxopt import ProximalGradient, GradientDescent, LBFGS
from jaxopt.prox import prox_non_negative_lasso

from fuzzylli.interpolation_jax import eval_interp1d as evaluate_spline

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

_wavefunction_params = namedtuple(
    "wavefunction_params",
    ["total_mass", "R_fit", "a_j", "phase_j", "eigenstate_library"],
)


class wavefunction_params(_wavefunction_params):
    """
    Named tuple with hashing capability (used for caching)
    """

    @classmethod
    def compute_name(cls, eigenstate_library, rho_target, R_fit, seed):
        R = jnp.linspace(R_fit / 10, R_fit, 10)
        # jax arrays are not hashable, so wrap in numpy
        return hashlib.md5(
            np.array(eigenstate_library.R_j_params.f).tobytes()
            + np.array([rho_target(R)]).tobytes()
            + np.array([R_fit] + np.array([R_fit])).tobytes()
        )


def L(l):
    """
    A heuristic mapping from quantum l to classical L. Results, i.e. fit
    accuracy seems mainly affected by the l=0 case, especially for radially
    biased (beta > 0) dispersion. In this case the DF diverges as L^(-beta)
    """
    return jnp.where(l > 0, jnp.sqrt(l**2 - 1 / 4), 0.1)


def data_generation_gaussian_noise(rho_target, R_fit, N_fit, key, sigma):
    """
    Data generative process for regularized least square fit. Here we assume
                    log(rho) ~ N(log(rho_target), sigma^2)
    Radial coordinate is sampled at positions such that each radial shell
    encloses the same mass (line density). High mass, small radii regions are
    thus as important as low mass, large radii regions.
    """
    Rmax = R_fit

    # Find equal mass shell grid
    mu_inside_R_fit = rho_target.enclosed_mass(R_fit)

    def next_point(r_ip1, x):
        r_i = r_ip1 - mu_inside_R_fit / (2 * jnp.pi * N_fit * r_ip1 * rho_target(r_ip1))
        return r_i, r_i

    _, R = lax.scan(
        next_point,
        Rmax,
        None,
        length=N_fit,
        reverse=True,
    )
    R = R[R > 0]
    N_fit = R.shape[0]

    log_rho_R = jnp.log(rho_target(R))

    log_rho_R_noisy = log_rho_R + random.normal(key, shape=(R.shape[0],)) * sigma
    return R, log_rho_R_noisy


def data_generation_poisson_process(rho_target, R_fit, M, key):
    """
    UNUSED.
    """
    # Simulate non-homogenous Poisson process (per unit length)
    N_samples = random.poisson(key, rho_target.enclosed_mass(R_fit) / M)
    logger.info(f"Simulate NHPP with N={N_samples} points")
    R_samples = rho_target.sample(N_samples, highR=R_fit)
    return R_samples


evaluate_multiple_splines = vmap(evaluate_spline, in_axes=(None, 0))
eval_mult_splines_mult_x = vmap(evaluate_multiple_splines, in_axes=(0, None))


def truncated_spline(R, R_cut, spline_params):
    """
    Smooth truncation of "spline" defined by "spline_params" beyond R_cut
    """

    def mask(x, x_cut):
        transition_scale = x_cut / 100
        cut_scale = x_cut - 5 * transition_scale
        return 1 / 2 * (erf((cut_scale - x) / transition_scale) + 1)

    return mask(R, R_cut) * evaluate_spline(R, spline_params)


evaluate_multiple_truncated_splines = vmap(truncated_spline, in_axes=(None, None, 0))


def while_loop(cond_fun, body_fun, init_val):
    val = init_val
    while cond_fun(val):
        val = body_fun(val)
    return val


def spiral_tap(
    mod_poisson_log_likelihood,
    tau,
    f0,
    eta=1.1,
    sigma=0.9,
    maxiter=10000,
    err=1e-4,
):
    """
    Optimizer for the reconstruction of a non-homogenous Poisson process
    intensity function.
    See arXiv:1005.4274 for more information
    """

    def total_objective(f):
        return mod_poisson_log_likelihood(f) + tau * jnp.sum(jnp.abs(f))

    grad_log_likelihood = jax.grad(mod_poisson_log_likelihood)
    hessian_log_likelihood = jax.hessian(mod_poisson_log_likelihood)

    def not_converged(state):
        f_kp1, f_k, err_k, k = state
        return jnp.logical_and(err_k > err, k < maxiter)

    def find_solution_of_quadratic_approximation(state):
        def find_next_step_size(state):
            f_kp1, alpha_k = state
            alpha_k = eta * alpha_k
            f_kp1 = jax.nn.relu(
                f_k - 1.0 / alpha_k * grad_log_likelihood_k - tau / alpha_k
            )
            return f_kp1, alpha_k

        def not_accept_step(state):
            f_kp1, alpha_k = state

            return total_objective(
                f_kp1
            ) > total_objective_k - sigma * alpha_k / 2 * jnp.sum((f_kp1 - f_k) ** 2)

        f_k, f_km1, _, k = state
        delta_k = f_k - f_km1
        grad_log_likelihood_k = grad_log_likelihood(f_k)
        total_objective_k = total_objective(f_k)
        # Assuming the log_likelihood is convex, we have only positive EV and
        # the rayleigh quotient is thus postive as well.
        alpha_k = jnp.max(
            jnp.array(
                [
                    delta_k.T
                    @ hessian_log_likelihood(f_k)
                    @ delta_k
                    / jnp.dot(delta_k, delta_k),
                    10 * jnp.finfo(jnp.float64).eps,
                ]
            )
        )
        f_kp1 = jax.nn.relu(f_k - 1.0 / alpha_k * grad_log_likelihood_k - tau / alpha_k)

        f_kp1, alpha_k = jax.lax.while_loop(
            not_accept_step, find_next_step_size, (f_kp1, alpha_k)
        )
        # This is the same error metric that JaxOpt's ProximalGradient uses:
        # https://github.com/google/jaxopt/blob/main/jaxopt/_src/proximal_gradient.py#L204
        err_k = jnp.linalg.norm(f_kp1 - f_k) * alpha_k
        return f_kp1, f_k, err_k, k + 1

    return jax.lax.while_loop(
        not_converged,
        find_solution_of_quadratic_approximation,
        (f0, f0 + err * jnp.linalg.norm(f0), 1.0, 0),
    )


def init_wavefunction_params_poisson_process(
    eigenstate_library,
    rho_target,
    df,
    R_fit,
    mu_cdm,
    seed,
):
    fig, ax = plt.subplots()
    ax.set_xscale("log")
    ax.set_yscale("log")
    rho_psi = jax.vmap(rho, in_axes=(0, None))
    r = np.logspace(jnp.log10(R_fit) - 3, jnp.log10(R_fit), 100)
    ax.plot(r, rho_target(r))

    M = mu_cdm
    bin_count = 128
    total_mass = rho_target.total_mass

    seed_sampling, seed_phase = random.split(random.PRNGKey(seed), 2)

    R_samples = data_generation_poisson_process(rho_target, R_fit, M, seed_sampling)
    y, e = jnp.histogram(R_samples, bins=bin_count)
    # logbins = (jnp.roll(loge, -1) + loge)[:-1] / 2.0
    bins = 0.5 * (e[:-1] + e[1:])
    # e = jnp.exp(loge)
    # rho_sample = mu_cdm * y / (2 * jnp.pi * bins**2 * (loge[1] - loge[0]))
    # ax.plot(bins, rho_sample, drawstyle="steps-mid")
    ax.plot(bins, y, drawstyle="steps-mid")
    glx, glw = np.polynomial.legendre.leggauss(16)
    glx = jnp.asarray(0.5 * (glx + 1))
    glw = jnp.asarray(0.5 * glw)
    R_i = (e[:-1, jnp.newaxis] + jnp.diff(e)[:, jnp.newaxis] * glx).ravel()

    prior_lambda_j = (
        total_mass
        * (2 * jnp.pi) ** -2
        * df(
            eigenstate_library.E_j,
            L(eigenstate_library.l_of_j),
        )
        ** (-1)
    )
    prefactor = jnp.where(eigenstate_library.l_of_j > 0, 2.0, 1.0)
    R_j2_R = (
        R_i[:, jnp.newaxis]
        * eval_mult_splines_mult_x(R_i, eigenstate_library.R_j_params) ** 2
    ).reshape(bin_count, glx.shape[0], -1)

    # This matrix maps eigenstate coefficients to the expected number of
    # fiducial line mass particles within bins R_i
    A = (
        jnp.diff(e)[:, jnp.newaxis]
        * jnp.tensordot(R_j2_R, glw, axes=(1, 0))
        * prefactor
        * total_mass
        / prior_lambda_j
        / mu_cdm
    )

    def neg_log_likelihood(aj_2):
        Af = A @ aj_2
        return jnp.sum(Af) - jnp.sum(y * jnp.log(Af + 1e-12))

    global_lambda = 1.0
    logger.info(
        f"Initial -logL = {neg_log_likelihood(jnp.ones_like(prior_lambda_j)):.3f} "
    )

    for i in [140000]:
        res, _, error, iter_num = spiral_tap(
            neg_log_likelihood,
            global_lambda,
            jnp.ones_like(prior_lambda_j),
            maxiter=i,
        )
        aj_2 = res / prior_lambda_j

        logger.info(
            f"Spiral stopped after {iter_num}, "
            f"-logL = {neg_log_likelihood(res):.3f}, "
            f"error = {error:.1e}"
        )

        non_zero_modes_j = jnp.nonzero(aj_2)
        aj_2 = aj_2[non_zero_modes_j]
        reduced_library = tree_map(
            lambda coefs: coefs[non_zero_modes_j], eigenstate_library
        )
        # Each nl gets a random phase but modes with same l alwazs have the same
        # |a_nl|
        phase_j = jnp.where(
            (reduced_library.l_of_j > 0).reshape(-1, 1),
            jnp.exp(
                1.0j
                * random.uniform(
                    seed_phase,
                    shape=(reduced_library.J, 2),
                    maxval=2 * jnp.pi,
                )
            ),
            1.0 / 2,
        )
        logger.info(
            f"{non_zero_modes_j[0].shape[0]}/{eigenstate_library.J} modes "
            f"have non-vanishing coefficents"
        )
        params = wavefunction_params(
            total_mass=total_mass,
            eigenstate_library=reduced_library,
            R_fit=R_fit,
            a_j=jnp.sqrt(aj_2),
            phase_j=phase_j,
        )
        ax.plot(r, rho_psi(r, params).ravel(), label=f"{iter_num}")
        ax.set_ylim([1e-2, 100000])
    ax.legend(*ax.get_legend_handles_labels(), loc="lower left")
    plt.show()


def init_wavefunction_params_least_square_naive(
    eigenstate_library,
    rho_target,
    df,
    R_fit,
    seed,
):
    """
    Initializes the wave function coefficients via an adaptive LASSO reggression
    fit. The posterior mode is found via proximal gradient descent.
    """
    # TODO: Move this somewhere else
    # Creates system matrix. Observations along axis 0 and states j along axis 1
    # These parameters have to be fixed artificially under gaussian noise
    # conditions.
    N_fit = 4 * eigenstate_library.J
    sigma = 0.1

    seed_sampling, seed_phase = random.split(random.PRNGKey(seed), 2)
    R, log_rho_R_noisy = data_generation_gaussian_noise(
        rho_target, R_fit, N_fit, seed_sampling, sigma
    )
    logger.info(f"fit interval = [{R[0]},{R[-1]}]")
    N_fit = R.shape[0]

    total_mass = rho_target.total_mass

    # WKB asymptote as scale of the exponential prior. Postivity of all
    # coefficients (prior support) is enforced by the proximal operator.
    prior_lambda_j = (
        total_mass
        * (2 * jnp.pi) ** -2
        * df(
            eigenstate_library.E_j,
            L(eigenstate_library.l_of_j),
        )
        ** (-1)
    )

    # System matrix. Note that the eigenstate library only contains R_j modes for
    # l>=0 since the Hamiltoniain is invariant under l -> -l. Since we assume
    # |a_nl| = |a_n-l| we get un overall factor of 2 for l>0 modes which we
    # account for in the system matrix
    prefactor = jnp.where(eigenstate_library.l_of_j > 0, 2.0, 1.0)
    R_j2_R = (
        prefactor[jnp.newaxis, :]
        * total_mass
        / (2 * jnp.pi)
        * eval_mult_splines_mult_x(R, eigenstate_library.R_j_params) ** 2
    )

    @jit
    def neg_log_likelihood(aj_2):
        log_rho_psi = jnp.log(R_j2_R @ aj_2)
        return jnp.mean((log_rho_psi - log_rho_R_noisy) ** 2)

    logger.info("Running least square optimization (LBFGS)...")
    R_j2_R = R_j2_R / prior_lambda_j

    solver = LBFGS(
        fun=neg_log_likelihood,
        maxiter=1000,
        tol=1e-4,
    )
    res = solver.run(1.0 * jnp.ones_like(prior_lambda_j))
    aj_2 = res.params / prior_lambda_j

    logger.info(
        f"Optimization stopped after {res.state.iter_num} "
        f"iterations (error = {res.state.error:.5f})"
    )
    non_zero_modes_j = jnp.nonzero(aj_2)
    aj_2 = aj_2[non_zero_modes_j]
    reduced_library = tree_map(
        lambda coefs: coefs[non_zero_modes_j], eigenstate_library
    )
    # Each nl gets a random phase but modes with same l always have the same
    # |a_nl|. We divide the l=0 mode to allow for more straightforward
    # computation in rho(...) below
    phase_j = jnp.where(
        (reduced_library.l_of_j > 0).reshape(-1, 1),
        jnp.exp(
            1.0j
            * random.uniform(
                seed_phase,
                shape=(reduced_library.J, 2),
                maxval=2 * jnp.pi,
            )
        ),
        1.0 / 2,
    )
    logger.info(
        f"{non_zero_modes_j[0].shape[0]}/{eigenstate_library.J} modes have non-vanishing coefficents"
    )
    return wavefunction_params(
        total_mass=total_mass,
        eigenstate_library=reduced_library,
        R_fit=R_fit,
        a_j=jnp.sqrt(aj_2),
        phase_j=phase_j,
    )


def init_wavefunction_params_least_square(
    eigenstate_library,
    rho_target,
    df,
    R_fit,
    seed,
):
    """
    Initializes the wave function coefficients via an adaptive LASSO reggression
    fit. The posterior mode is found via proximal gradient descent.
    """

    @jit
    def neg_log_likelihood(aj_2):
        log_rho_psi = jnp.log(R_j2_R @ aj_2)
        return jnp.sum((log_rho_psi - log_rho_R_noisy) ** 2)

    # TODO: Move this somewhere else
    # Creates system matrix. Observations along axis 0 and states j along axis 1
    # These parameters have to be fixed artificially under gaussian noise
    # conditions.
    N_fit = 4 * eigenstate_library.J
    sigma = 0.1

    seed_sampling, seed_phase = random.split(random.PRNGKey(seed), 2)
    R, log_rho_R_noisy = data_generation_gaussian_noise(
        rho_target, R_fit, N_fit, seed_sampling, sigma
    )
    logger.info(f"fit interval = [{R[0]},{R[-1]}]")
    N_fit = R.shape[0]

    total_mass = rho_target.total_mass

    # This is the leading factor in the gaussian log likelihood
    # taken out of the argmin and thus transferred to the prior term
    global_lambda = 2 * sigma**2

    active_library = eigenstate_library
    epsilon = 1
    N = 1
    solver = ProximalGradient(
        fun=neg_log_likelihood,
        prox=prox_non_negative_lasso,
        maxiter=10000,
        tol=1e-4,
        acceleration=True,
    )
    logger.info("Running l1-regularized optimization...")
    aj_2 = 1.0 * jnp.ones_like(active_library.E_j)
    while epsilon > 1e-4:
        # WKB asymptote as scale of the exponential prior. Postivity of all
        # coefficients (prior support) is enforced by the proximal operator.
        prior_lambda_j = (
            total_mass
            * (2 * jnp.pi) ** -2
            * df(active_library.E_j, L(active_library.l_of_j)) ** (-1)
        )

        # System matrix. Note that the eigenstate library only contains R_j modes for
        # l>=0 since the Hamiltoniain is invariant under l -> -l. Since we assume
        # |a_nl| = |a_n-l| we get un overall factor of 2 for l>0 modes which we
        # account for in the system matrix
        prefactor = jnp.where(active_library.l_of_j > 0, 2.0, 1.0)
        R_j2_R = (
            prefactor[jnp.newaxis, :]
            * total_mass
            / (2 * jnp.pi)
            * eval_mult_splines_mult_x(R, active_library.R_j_params) ** 2
        )

        R_j2_R = R_j2_R / prior_lambda_j

        res = solver.run(aj_2, hyperparams_prox=global_lambda)
        aj_2 = res.params
        non_zero_modes_j = jnp.nonzero(aj_2)

        epsilon = res.state.error
        N += 1
        aj_2 = aj_2[non_zero_modes_j]
        active_library = tree_map(lambda coefs: coefs[non_zero_modes_j], active_library)
        logger.info(
            f"Super iteration {N} stopped after {res.state.iter_num} iterations "
            f"(error = {res.state.error:.5f}, active set = {active_library.J})"
        )
        if N == 100:
            break

    aj_2 = aj_2 / prior_lambda_j[non_zero_modes_j]
    # Each nl gets a random phase but modes with same l always have the same
    # |a_nl|. We divide the l=0 mode to allow for more straightforward
    # computation in rho(...) below
    phase_j = jnp.where(
        (active_library.l_of_j > 0).reshape(-1, 1),
        jnp.exp(
            1.0j
            * random.uniform(
                seed_phase,
                shape=(active_library.J, 2),
                maxval=2 * jnp.pi,
            )
        ),
        1.0 / 2,
    )
    logger.info(
        f"{active_library.J}/{eigenstate_library.J} modes have non-vanishing coefficents"
    )
    return wavefunction_params(
        total_mass=total_mass,
        eigenstate_library=active_library,
        R_fit=R_fit,
        a_j=jnp.sqrt(aj_2),
        phase_j=phase_j,
    )


def psi(R, phi, t, wavefunction_params):
    """Computes the evolution of the wavefunction psi(R,phi,t)"""

    R_j_R = evaluate_multiple_truncated_splines(
        R, wavefunction_params.R_fit, wavefunction_params.eigenstate_library.R_j_params
    )
    e_ilphi_milphi = jnp.c_[
        jnp.exp(1.0j * wavefunction_params.eigenstate_library.l_of_j * phi),
        jnp.exp(-1.0j * wavefunction_params.eigenstate_library.l_of_j * phi),
    ]
    return jnp.sqrt(wavefunction_params.total_mass / (2 * jnp.pi)) * jnp.sum(
        jnp.sum((wavefunction_params.phase_j * e_ilphi_milphi), axis=-1)
        * wavefunction_params.a_j
        * R_j_R
        * jnp.exp(-1.0j * wavefunction_params.eigenstate_library.E_j * t)
    )


def grad_psi(R, phi, t, wavefunction_params):
    """Computes the gradient of the wavefunction psi(R,phi,t)"""
    grad_Re_psi = grad(
        lambda R, phi: jnp.real(psi(R, phi, t, wavefunction_params)),
        argnums=(0, 1),
    )
    grad_Im_psi = grad(
        lambda R, phi: jnp.imag(psi(R, phi, t, wavefunction_params)),
        argnums=(0, 1),
    )
    del_R_psi_Rphi = grad_Re_psi(R, phi)[0] + 1.0j * grad_Im_psi(R, phi)[0]
    inv_R_del_phi_psi_Rphi = (
        1.0 / R * (grad_Re_psi(R, phi)[1] + 1.0j * grad_Im_psi(R, phi)[1])
    )
    return jnp.c_[del_R_psi_Rphi, inv_R_del_phi_psi_Rphi]


def velocity(R, phi, t, wavefunction_params):
    """Computes the conjugate velocty u. The peculiar velocity v=u/a"""
    psi_Rphit = psi(R, phi, t, wavefunction_params)
    del_psi_Rphit = grad_psi(R, phi, t, wavefunction_params)
    v = jnp.imag(jnp.conjugate(psi_Rphit) * del_psi_Rphit) / jnp.abs(psi_Rphit) ** 2
    return v.squeeze()


def vorticity(R, phi, t, wavefunction_params):
    """Computes the vorticity omega = nabla x u."""
    v_Rphit = velocity(R, phi, t, wavefunction_params)
    del_R_v_phi = grad(lambda R: velocity(R, phi, t, wavefunction_params)[1])(R)
    del_phi_v_R = grad(lambda phi: velocity(R, phi, t, wavefunction_params)[0])(phi)
    return 1.0 / R * (v_Rphit[1] + R * del_R_v_phi - del_phi_v_R)


def trace_dispersion(R, phi, t, wavefunction_params):
    """Computes the trace of the Wigner dispersion tensor. Experimental."""

    def log_psi2(R, phi, t):
        return jnp.log(jnp.abs(psi(R, phi, t, wavefunction_params)) ** 2)

    del_R_log_psi2 = grad(lambda R: log_psi2(R, phi, t))
    del_RR_log_psi2 = grad(del_R_log_psi2)
    del_phiphi_log_psi2 = grad(grad(lambda phi: log_psi2(R, phi, t)))
    return (
        -1.0
        / 4
        * (
            1.0 / R * del_R_log_psi2(R)
            + del_RR_log_psi2(R)
            + 1.0 / R**2 * del_phiphi_log_psi2(phi)
        )
    )


def rho(R, wavefunction_params):
    """
    Computes the time static part of the wavefunction density (i.e. the contribution we fit)
    """
    prefactor = jnp.where(wavefunction_params.eigenstate_library.l_of_j > 0, 2.0, 1.0)
    R_j2_R = (
        prefactor[jnp.newaxis, :]
        * wavefunction_params.total_mass
        / (2 * jnp.pi)
        * evaluate_multiple_truncated_splines(
            R,
            wavefunction_params.R_fit,
            wavefunction_params.eigenstate_library.R_j_params,
        )
        ** 2
    )
    return R_j2_R @ jnp.abs(wavefunction_params.a_j) ** 2


def gamma(R, t, wavefunction_params):
    """
    Computes the time correlation function of psi. Experimental.
    """
    prefactor = jnp.where(wavefunction_params.eigenstate_library.l_of_j > 0, 2.0, 1.0)
    R_j2_R = (
        prefactor[jnp.newaxis, :]
        * wavefunction_params.total_mass
        / (2 * jnp.pi)
        * evaluate_multiple_truncated_splines(
            R,
            wavefunction_params.R_fit,
            wavefunction_params.eigenstate_library.R_j_params,
        )
        ** 2
    )
    R_j2_R_n2_R = R_j2_R * R_j2_R[:, jnp.newaxis]
    a_j_2_a_n_2 = (
        jnp.abs(wavefunction_params.a_j) ** 2
        * jnp.abs(wavefunction_params.a_j[:, jnp.newaxis]) ** 2
    )
    E_jmE_n = (
        wavefunction_params.eigenstate_library.E_j
        - wavefunction_params.eigenstate_library.E_j[:, jnp.newaxis]
    )

    M_jn = a_j_2_a_n_2 * R_j2_R_n2_R
    N_jn = M_jn * jnp.exp(1.0j * E_jmE_n * t)

    exp_rho = jnp.sum(jnp.abs(wavefunction_params.a_j) ** 2 * R_j2_R)
    exp_rho2 = jnp.sum(jnp.abs(wavefunction_params.a_j) ** 4 * R_j2_R**2) + 2 * (
        jnp.sum(M_jn) - jnp.trace(M_jn)
    )

    return 1 + (jnp.sum(N_jn - M_jn) - jnp.trace(N_jn - M_jn)) / (
        exp_rho2 - exp_rho**2
    )
