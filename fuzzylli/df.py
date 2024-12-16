from functools import partial
import logging
from abc import ABC, abstractmethod
import hashlib

from jax import vmap
import jax.numpy as jnp
from jax.scipy.special import gammaln

from fuzzylli.mlp import init_mlp_params, evaluate_mlp, mlp_optimization
from fuzzylli.utils import quad
from fuzzylli.io_utils import hash_to_int64
from fuzzylli.potential import E_c

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class CylindricalDistributionFunction(ABC):
    """
    Base Class of Cylindrical DFs. All children of CylindircalDistributionFunction
    solve an inverse problem by adjusting the parameters of a parametrized internal
    representation until the target denisty rho is recovered.
    Specification of the forward computation (_logrho_from_df) is left to the
    derived class to implement.
    """

    def __init__(
        self,
        scalefactor,
        potential,
        rho,
        epochs=2000,
        R_min=None,
        R_max=None,
        N_optimization=None,
        df_params=None,
    ):
        self.scalefactor = scalefactor
        self.potential = potential
        self.rho_target = rho
        self.E_c = vmap(E_c, in_axes=(0, None, None))

        self.epochs = epochs
        self.R_min = R_min if R_min is not None else self.potential.R_min
        self.R_max = R_max if R_max is not None else self.potential.R_max
        self.N_optimization = (
            N_optimization
            if N_optimization is not None
            else int(20 * jnp.log10(self.potential.R_max / self.potential.R_min))
        )

        self._logrho_from_df_batched = vmap(self._logrho_from_df, in_axes=(0, None))

        self.logR = jnp.linspace(
            jnp.log(self.R_min),
            jnp.log(self.R_max),
            self.N_optimization,
        )
        self.logrho_target_R = jnp.log(self.rho_target(jnp.exp(self.logR)))

        if df_params is None:
            self.df_params = mlp_optimization(
                self.epochs,
                self.init_params,
                self._logrho_from_df_batched,
                self.logR,
                self.logrho_target_R,
            )
        else:
            self.df_params = df_params

    @classmethod
    def compute_name(
        cls,
        scalefactor,
        potential,
        rho,
        epochs=2000,
        R_min=None,
        R_max=None,
        N_optimization=None,
        df_params=None,
    ):
        combined = hashlib.sha256()
        combined.update(hashlib.md5(jnp.array(scalefactor)).digest())
        combined.update(hashlib.md5(jnp.array(potential.name)).digest())
        combined.update(hashlib.md5(jnp.array(rho.name)).digest())
        combined.update(hashlib.md5(jnp.array(epochs)).digest())
        combined.update(hashlib.md5(jnp.array(R_min)).digest())
        combined.update(hashlib.md5(jnp.array(R_max)).digest())
        combined.update(hashlib.md5(jnp.array(N_optimization)).digest())
        return hash_to_int64(combined.hexdigest())

    @abstractmethod
    def _eval(self, E, L):
        """
        Value of the optimized DF at energy E and angular momentum L

        Arguments:
        E (jnp.array) - Values of the specific energy
        L (jnp.array) - Values of the specific angular momentum

        Returns:
        DF (jnp.array) - Value of the optimized DF
        """
        pass

    def __call__(self, E, L):
        """
        As above but aware of the energy cutoff implied by
        circular orbits (minimal possible energy given angular momentum L)
        """

        df_E_L = self._eval(E, L)

        return jnp.where(E < self.E_c(L, self.potential, self.scalefactor), 0, df_E_L)

    @abstractmethod
    def _logrho_from_df(self, logR, params):
        """
        Differentiable routine to compute the density from margenalisation over
        velocity space. This routine enters the optimization

        Arguments:
        logR (float) - Value of radial coordinate on log grid
        params (pytree) - Parameters of the internal model

        Returns:
        log(rho(log(R))) (float) - Value of the real space density obtained via
        marginization
        """
        pass

    def rho(self, R):
        """
        Real-space density of the optmized DF at radius R.

        Arguments:
        R (jnp.array) - Values of the radial coordinare

        Returns:
        rho(R) (jnp.array) - Values of the real space density obtained via
        marginization of the optimized DF
        """
        return jnp.exp(self._logrho_from_df_batched(jnp.log(R), self.df_params))


class ConstantAnisotropyDistribution(CylindricalDistributionFunction):
    def __init__(self, beta, *args, **kwargs):
        if beta >= 1:
            raise ValueError

        self.beta = beta
        self.init_params = init_mlp_params([1, 32, 32, 32, 1])
        super().__init__(*args, **kwargs)

        self.name = self.compute_name(beta, *args, **kwargs)

    @classmethod
    def compute_name(cls, beta, *args, **kwargs):
        combined = hashlib.sha256()
        combined.update(hashlib.md5(jnp.array(beta)).digest())
        combined.update(hashlib.md5(super().compute_name(*args, **kwargs)).digest())
        return combined.hexdigest()

    def _logrho_from_df(self, logR, params):
        # Transformed integrand to deal with sqrt divergence
        def integrand_small(t, R):
            return (1.0 / (1 - self.beta / 2)) * jnp.exp(
                self._logg_beta(
                    jnp.log(t ** (1.0 / (1 - self.beta / 2)) + self.potential(R)),
                    params,
                )
            )

        # Transformed integrand to deal with infinite upper bound
        def integrand_large(t, R):
            return t ** (self.beta / 2 - 2) * jnp.exp(
                self._logg_beta(jnp.log(1.0 / t + self.potential(R)), params)
            )

        R = jnp.exp(logR)

        # Split integral at twice the lower bound, i.e. V(R)
        Delta_V = 2 * self.potential(R)
        I1_lower = I2_lower = 0.0
        I1_upper = Delta_V ** (1 - self.beta / 2)
        I2_upper = 1.0 / Delta_V

        I1 = quad(lambda t: integrand_small(t, R), I1_lower, I1_upper)
        I2 = quad(lambda t: integrand_large(t, R), I2_lower, I2_upper)

        return (
            jnp.log(
                2
                * self.scalefactor ** (2 - self.beta)
                * jnp.sqrt(jnp.pi)
                * 2 ** (-self.beta / 2)
            )
            + gammaln((1 - self.beta) / 2)
            - gammaln((2 - self.beta) / 2)
            - self.beta * logR
            + jnp.log(I1 + I2)
        )

    @partial(vmap, in_axes=(None, 0, None))
    def _logg_beta(self, logE, params):
        return evaluate_mlp(logE, params)

    def _eval(self, E, L):
        return jnp.abs(L) ** (-self.beta) * jnp.exp(
            self._logg_beta(jnp.log(E), self.df_params)
        )
