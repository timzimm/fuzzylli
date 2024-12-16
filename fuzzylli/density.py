from abc import ABC, abstractmethod
import hashlib

from scipy.integrate import quad as scipy_quad
import jax
import jax.numpy as jnp
from jax.random import split, PRNGKey, uniform
from jax.scipy.special import erf
from jaxopt import ScipyRootFinding
import numpy as np

from fuzzylli.utils import quad
from fuzzylli.io_utils import hash_to_int64


class RadialDensity(ABC):
    def __init__(self, r_cut=0, seed=42):
        self.r_cut = r_cut
        self.seed = PRNGKey(seed)

    @classmethod
    def compute_name(cls, r_cut=0, seed=42):
        combined = hashlib.sha256()
        combined.update(hashlib.md5(np.array(r_cut)).digest())
        combined.update(hashlib.md5(np.array(seed)).digest())
        return hash_to_int64(combined.hexdigest())

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
    def R90(self):
        """
        Radius enclosing 90% of total mass
        """
        if not hasattr(self, "_R90"):
            self._R90 = self.enclosing_radius(0.90)
        return self._R90

    @property
    def R95(self):
        """
        Radius enclosing 95% of total mass
        """
        if not hasattr(self, "_R95"):
            self._R95 = self.enclosing_radius(0.95)
        return self._R95

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

        @jax.jit
        def objective(R):
            return self.enclosed_mass(R) / M - mass_fraction

        r0 = 1.0
        res = ScipyRootFinding(optimality_fun=objective, method="hybr").run(r0)
        return res.params


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
        result_shape = jax.ShapeDtypeStruct((), jnp.int64)
        self.name = jax.pure_callback(
            self.compute_name,
            result_shape,
            r0,
            beta,
            sigma2,
            scalefactor,
            *args,
            **kwargs
        )

    @classmethod
    def compute_name(cls, r0, beta, sigma2, scalefactor, *args, **kwargs):
        combined = hashlib.sha256()
        combined.update(hashlib.md5(jnp.array(r0)).digest())
        combined.update(hashlib.md5(jnp.array(beta)).digest())
        combined.update(hashlib.md5(jnp.array(sigma2)).digest())
        combined.update(hashlib.md5(jnp.array(scalefactor)).digest())
        combined.update(
            hashlib.md5(jnp.array(super().compute_name(*args, **kwargs))).digest()
        )
        return hash_to_int64(combined.hexdigest())

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
