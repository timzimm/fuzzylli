from abc import ABC, abstractmethod

from scipy.integrate import quad as scipy_quad
from jax import jit, grad
import jax.numpy as jnp
from jax.random import split, PRNGKey, uniform
from jax.scipy.special import erf
from jaxopt import ScipyRootFinding
import blackjax

from fuzzylli.utils import quad, inference_loop


class RadialDensity(ABC):
    def __init__(self, r_cut=0, seed=42):
        self.r_cut = r_cut
        self.seed = PRNGKey(seed)

    def __density_polar(self, R):
        return 2 * jnp.pi * R * self(R)

    @abstractmethod
    def __call__(self, R):
        pass

    def truncated_density(self, R, R_cut):
        def mask(x, x_cut):
            transition_scale = x_cut / 100
            cut_scale = x_cut - 5 * transition_scale
            return 1 / 2 * (erf((cut_scale - x) / transition_scale) + 1)

        return self.__call__(R) * mask(R, R_cut)

    @property
    def total_mass(self):
        if not hasattr(self, "_total_mass"):
            self._total_mass = scipy_quad(self.__density_polar, 0, jnp.inf)[0]
        return self._total_mass

    @property
    def R99(self):
        """
        Radius enclosing 99% of total mass
        """
        if not hasattr(self, "_R99"):
            self._R99 = self.enclosing_radius(0.99)
        return self._R99

    @property
    def R999(self):
        """
        Radius enclosing 99.9% of total mass
        """
        if not hasattr(self, "_R999"):
            self._R999 = self.enclosing_radius(0.999)
        return self._R999

    def enclosed_mass(self, R):
        """
        Finds the enclosed mass up to radius R.
        A default integration grid is provided to keep quadrature errors
        small and comparable.
        """
        r = jnp.logspace(-2, 3, 6)
        r_bounds = jnp.pad(jnp.where(r < R, x=r, y=R), pad_width=(1, 0))
        return jnp.sum(quad(self.__density_polar, r_bounds[:-1], r_bounds[1:]))

    def enclosing_radius(self, mass_fraction):
        """
        The inverse of enclosed_mass with mass_fraction=mass/total_mass
        """
        M = self.total_mass

        @jit
        def objective(R):
            return self.enclosed_mass(R) / M - mass_fraction

        r0 = 1.0
        res = ScipyRootFinding(optimality_fun=objective, method="hybr").run(r0)
        return res.params

    def sample(self, N):
        """
        Fallback sampling via MCMC (if no direct inversion is possible)
        """

        def log_jacobian_fn(logR):
            return jnp.log(jnp.abs(grad(lambda t: jnp.exp(2 * t))(logR)))

        def log_rho(logR):
            R = jnp.exp(logR)
            return jnp.log(self(R)) + log_jacobian_fn(logR)

        warmup = blackjax.window_adaptation(blackjax.nuts, log_rho)
        seed_warmup, seed_sample = split(self.seed, 2)
        (state, tuned_params), _ = warmup.run(seed_warmup, 0.0, 1000)

        samples, infos = inference_loop(seed_sample, state, tuned_params, log_rho, N)
        return jnp.exp(samples.position)


class SteadyStateCylinder(RadialDensity):
    """
    Eisenstein et al (1997) solution of the steady state density profile from
    radial Jeans equation -- here written in comoving form i.e. rho(x) measures
    density in a comoving volume and comoving position x
    """

    def __init__(self, r0, beta, sigma2, scalefactor, *args, **kwargs):
        """
        r0 - scale radias at a = scalefactor (physical)
        beta - anisotropy parameter in [0,1)
        sigma2 - radial velocity dispersion at a = scalefactor
        scalefactor - scalefactor at which density is evaluated
        """
        super().__init__(*args, **kwargs)
        self.r0 = r0
        self.beta = beta
        self.sigma2 = sigma2
        self.scalefactor = scalefactor

    def __call__(self, R):
        """
        Returns comoving density evaluated at comoving radial coordinate R.
        """
        x = R * self.scalefactor / self.r0
        return (
            2
            * self.scalefactor**3
            * (2 - self.beta) ** 2
            * self.sigma2
            / self.r0**2
            * x ** (-self.beta)
            / (x ** (2 - self.beta) + 1) ** 2
        )

    def enclosed_mass(self, R):
        x = R * self.scalefactor / self.r0
        return (
            4
            * jnp.pi
            * self.scalefactor
            * self.sigma2
            * (2 - self.beta)
            * (x ** (2 - self.beta) / (x ** (2 - self.beta) + 1))
        )

    @property
    def total_mass(self):
        return 4 * jnp.pi * self.scalefactor * self.sigma2 * (2 - self.beta)

    def sample(self, N, lowR=None, highR=None):
        lowu = 0
        highu = 1.0
        if lowR is not None:
            lowu = self.enclosed_mass(lowR) / self.total_mass
        if highR is not None:
            highu = self.enclosed_mass(highR) / self.total_mass
        self.seed, sample_seed = split(self.seed)
        mu = uniform(sample_seed, shape=(N,), minval=lowu, maxval=highu)
        return (mu / (1 - mu)) ** (1.0 / (2 - self.beta)) * self.r0 / self.scalefactor
