import logging
import hashlib

import jax
import jax.numpy as jnp
from jax.lax import cond
from jax.scipy.special import xlogy

from jaxopt import Bisection, ScipyBoundedMinimize

from fuzzylli.utils import quad
from fuzzylli.io_utils import hash_to_int64
from fuzzylli.spline import init_spline_params, evaluate_spline
from fuzzylli.special import bessel_int_J0


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

quad = jax.vmap(quad, in_axes=(None, 0, 0))


def V_eff(R, L, V, scalefactor):
    """
    Effective potential, i.e. gravitational potential plus angular momentum
    barrier
    """
    return V(R) + L**2 / (2 * scalefactor**2 * R**2)


def R_g(L, V, scalefactor):
    """
    Guiding radius of circular orbit
    """
    solver = Bisection(
        optimality_fun=jax.jit(jax.grad(lambda R: V_eff(R, L, V, scalefactor))),
        lower=jnp.finfo(float).eps,
        upper=V.R_max,
        check_bracket=False,
    )

    def compute_R_via_bisection(L):
        return solver.run(V.R_min).params

    return cond(
        L == 0.0,
        lambda _: 0.0,
        compute_R_via_bisection,
        L + 10 * jnp.finfo(float).eps,
    )


def E_c(L, V, scalefactor):
    """
    Energy of circular orbit --- the minimum of the effective potential
    """
    return V_eff(R_g(L, V, scalefactor), L, V, scalefactor)


class AxialSymmetricPotential:
    """
    Solves Poisson equation for axial symmetric sources in 2D for free space
    boundary conditions.

    Design heavily inspired by github.com/GalacticDynamics-Oxford/Agama (1802.08239)
    """

    def __init__(self, source, R_max, R_min=0, N=None):
        self.rho = source
        self.R_min, self.R_max = self.__set_grid_boundaries(R_min, R_max)

        self.N = N if N is not None else int(50 * jnp.log10(self.R_max / self.R_min))
        logger.info(f"Using N = {self.N} interpolation points")

        # Points at which potential is computed via quadrature
        R = jnp.logspace(jnp.log10(self.R_min), jnp.log10(self.R_max), self.N)

        self.V_params, self.F_params = self.__compute_potential_force(R)
        self.__compute_potential_force_asymptotic()

        self._V0 = self._V_asymptote_small(0.0)

        result_shape = jax.ShapeDtypeStruct((), jnp.int64)
        self.name = jax.pure_callback(
            self.compute_name, result_shape, source.name, R_max, R_min=R_min, N=N
        )

    @classmethod
    def compute_name(cls, source_name, R_max, R_min=0, N=None):
        combined = hashlib.sha256()
        combined.update(hashlib.md5(jnp.array(source_name)).digest())
        combined.update(hashlib.md5(jnp.array(R_max)).digest())
        combined.update(hashlib.md5(jnp.array(R_min)).digest())
        combined.update(hashlib.md5(jnp.array(N)).digest())
        return hash_to_int64(combined.hexdigest())

    def __call__(self, R):
        """
        Computes V(R)
        """
        return (
            jnp.piecewise(
                R / 1.0,
                [
                    R <= self.R_min,
                    jnp.logical_and(R < self.R_max, R > self.R_min),
                    R >= self.R_max,
                ],
                [
                    self._V_asymptote_small,
                    lambda R: evaluate_spline(R, self.V_params),
                    self._V_asymptote_big,
                ],
            )
            - self._V0
        )

    def force(self, R):
        """
        Computes -del_R V(R)
        """
        return jnp.piecewise(
            R,
            [
                R <= self.R_min,
                jnp.logical_and(R < self.R_max, R > self.R_min),
                R >= self.R_max,
            ],
            [
                self._F_asymptote_small,
                lambda R: evaluate_spline(R, self.F_params),
                self._F_asymptote_big,
            ],
        )

    def __set_grid_boundaries(self, R_min, R_max):
        """
        Determines "sane" boundaries for the log interpolation grid on which the
        exact solution is found
        """

        if not R_min > 0 or R_min > R_max:
            # Near the origin V(R) = int_R^inf rlog(r)rho(r) dr dominates the exact
            # solution. We set the lower bound by comparing its rate of change
            # against the maximum rate of change and deem dV(R_min)/DR = 0.01
            # (dV/DR)_max as reasonable lower bound.
            @jax.jit
            def negative_dV_dR(R):
                return xlogy(R, R) * self.rho(R)

            minimizer = ScipyBoundedMinimize(fun=negative_dV_dR, method="l-bfgs-b")
            R_init = 1e-2
            bounds = (jnp.finfo(float).eps, jnp.inf)
            minimize_res = minimizer.run(R_init, bounds=bounds)
            if not minimize_res.state.success:
                logger.warning("Maximum value of dV/dR at low R not found")
            dV_dR_max = minimize_res.params
            logger.info(f"dV/dR_max = {dV_dR_max:.3f}")

            @jax.jit
            def objective(R):
                return jnp.abs(negative_dV_dR(R) / negative_dV_dR(dV_dR_max)) - 0.01

            bisec_res = Bisection(
                optimality_fun=objective, lower=jnp.finfo(float).eps, upper=dV_dR_max
            ).run()
            R_min = bisec_res.params
            logger.info(f"Rmin = {R_min:.3f}, Rmax = {R_max:.3f}")

        return R_min, R_max

    def __compute_potential_force_asymptotic(self):
        """
        Initializes the small and large radii asymptotes for V(R) and -del_R
        V(R). These asymptotes are beased on the assumption that rho(R) ~
        R^alpha (R->0) and rho(R)~R^beta (R->inf)
        """

        # Crued approximation of the exponents at far field and close to the origin
        # limited by integrability considerations
        alpha = (
            1.0 / jnp.log(2) * jnp.log(self.rho(self.R_min) / self.rho(self.R_min / 2))
        )
        beta = -1.0 * min(
            (
                1.0
                / jnp.log(2)
                * jnp.log(self.rho(self.R_max) / self.rho(self.R_max / 2))
            ),
            -3.0,
        )
        logger.info(
            f"Asymptotic exponenets of rho: alpha = {alpha:.3f}, beta ={beta:.3f}"
        )

        # Monopole + 1/r^|beta| density far field approximation:
        # V(R) ~ X * log(R) + Y * R^(2-|beta|)/(|beta|-2)^2*((|beta|-2)log(R) + 1)
        # coefficients found by requiring continuity in V and del_V
        R_max = self.R_max
        del_V_1 = -evaluate_spline(R_max, self.F_params)
        V_1 = evaluate_spline(R_max, self.V_params)

        Y = (V_1 - R_max * del_V_1 * jnp.log(R_max)) / (
            R_max ** (2 - beta)
            * (
                jnp.log(R_max) ** 2
                + ((beta - 2) * jnp.log(R_max) + 1) / (beta - 2) ** 2
            )
        )
        self.X = R_max * del_V_1 + Y * jnp.log(R_max) * R_max ** (2 - beta)

        self._V_asymptote_big = lambda R: self.X * jnp.log(R) + Y * (
            R ** (2 - beta) * ((beta - 2) * jnp.log(R) + 1) / (beta - 2) ** 2
        )
        self._F_asymptote_big = lambda R: -self.X / R + Y * jnp.log(R) * R ** (1 - beta)

        # Constant + r^alpha density approximation for small R:
        # V(R) ~ U + 1/(alpha + 2) * W * log(R) * R^(alpha+2)
        # coefficients found by requiring continuity in V and del_V
        R_min = self.R_min
        del_V_0 = -evaluate_spline(R_min, self.F_params)
        V_0 = evaluate_spline(R_min, self.V_params)

        W = del_V_0 / (
            R_min ** (alpha + 1) / (alpha + 2) + R_min**alpha * xlogy(R_min, R_min)
        )
        U = V_0 - W / (alpha + 2) * R_min ** (alpha + 1) * xlogy(R_min, R_min)

        self._V_asymptote_small = lambda R: U + W / (alpha + 2) * xlogy(R, R) * R ** (
            alpha + 1
        )
        self._F_asymptote_small = lambda R: -W * (
            xlogy(R, R) * R**alpha + R ** (alpha + 1) / (alpha + 2)
        )

    def __compute_potential_force(self, R):
        """
        Initializes the interpolation points V(R) and -del_R V(R) by solving:

                     V(R) = log(R) int_0^R ds s rho(s) + int_r^inf ds s log(s) rho(s)
                -del_V(R) = -1/R int_0^R ds s rho(s)
        """

        rk = R

        # Includes innermost segment starting at zero
        rk_m_1 = jnp.concatenate((jnp.array([0]), R))

        def Q_int_integrand(r):
            return self.rho(r) * r

        def Q_ext_integrand(r):
            return self.rho(r) * xlogy(r, r)

        def Q_ext_transformed(t):
            return (
                rk[-1]
                / t**3
                * self.rho(rk[-1] / t)
                * rk[-1]
                / t
                * jnp.log(rk[-1] / t)
            )

        # Interior segment potentials excluding last one (not interior to
        # anything)
        Q_int = quad(Q_int_integrand, rk_m_1[:-1], R)

        # Exterior segment potentials excluding first one
        # (not exterior to anything) and last one (special treatment)
        Q_ext = quad(Q_ext_integrand, R[:-1], R[1:])
        Q_ext_last = quad(Q_ext_transformed, jnp.array([0]), jnp.array([1]))

        Q_ext = jnp.concatenate((Q_ext, Q_ext_last))

        # Sum interior and exterior potentials using the linear non-homogenous
        # recurrence relations...
        # P_int[k+1] = P_int[k]*log(r_k+1/r_k) + Q_int[k]*log(r_k+1),
        a0 = Q_int[0] * jnp.log(R[0])
        fn = jnp.log(R[1:]) / jnp.log(R[:-1])
        gn = Q_int[1:] * jnp.log(R[1:])

        P_ext = jnp.cumsum(Q_ext[::-1])

        P_int = self.__solve_recurrence(a0, fn, gn)

        V_R = P_int + P_ext[::-1]

        # Force computation
        F_R = -1.0 / (R * jnp.log(R)) * P_int

        return init_spline_params(R, V_R), init_spline_params(R, F_R)

    @staticmethod
    def __solve_recurrence(a_0, f_n, g_n):
        """
        Explicit solution to linear recurrences of the form
                    a_n+1 = a*n * f_n + g_n
        """
        pi_fn = jnp.cumprod(f_n)
        a_n = pi_fn * (a_0 + jnp.cumsum(g_n / pi_fn))
        return jnp.concatenate((jnp.array([a_0]), a_n))


class NonAxialSymmetricPotential:
    """
    Solves Poisson equation in cartesian 2D coordinates and free-space
    conditions. No assupmtion on the symmetry ( in particular no axial symmetry)
    of rho is made.
    """

    def __init__(self, dx, N):
        self.gamma = 0.5772156649015329
        self.Ji0 = jax.vmap(bessel_int_J0)
        self.dx = dx
        self.N = N
        self.G_k = self.__init_greens_function(dx, N)

    def __init_greens_function(self, dx, N):
        dist_x = jnp.roll(dx * jnp.arange(-N, N, dtype=jnp.float64), -N)
        x, y = jnp.meshgrid(dist_x, dist_x, sparse=True)
        r = jnp.sqrt(x**2 + y**2)
        G = self.__precompute_kernel_matrix(r, dx, N)

        return jnp.fft.rfft2(G)

    def __precompute_kernel_matrix(self, r, dx, N):
        def greens_funcion(rho, sigma):
            """
            See 1704.00704 and github.com/mmhej/poissonsolver
            """
            return (
                -1.0 / (2 * jnp.pi) * (self.Ji0(rho) + jnp.log(2 * sigma) - self.gamma)
            )

        sigma = dx / jnp.pi
        rho = r / sigma
        return greens_funcion(rho.ravel(), sigma).reshape(r.shape)

    def solve(self, rho, dx):
        assert rho.shape[0] == rho.shape[1] == self.N
        assert self.dx == dx

        rho_k = jnp.fft.rfft2(rho, s=(2 * self.N, 2 * self.N))
        return -(dx**2) * jnp.fft.irfft2(self.G_k * rho_k)[: self.N, : self.N]
