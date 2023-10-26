import jax
import jax.numpy as jnp
from jax.scipy.special import betainc

from fuzzylli.utils import quad

# flat, radiation free cosmological background (1 = om + ol)
# (taken from axionCAMB output)
h = 0.6731
om = 0.314649


def E(a):
    omega_m0 = om
    return jnp.sqrt(omega_m0 / (a * a * a) + (1 - omega_m0))


def omega_m(a):
    omega_m0 = om
    return omega_m0 / (a * a * a * E(a) ** 2)


def linear_delta_sc_collapse_at(a_coll):
    return 3 / 5 * (3 * jnp.pi / 2) ** (2 / 3) * omega_m(a_coll) ** (0.0055)


def nonlinear_delta_sc_virial_at(a_vir):
    x = omega_m(a_vir) - 1
    return 18 * jnp.pi**2 + 82 * x - 39 * x**2


def D(a):
    """
    Proportional to growth factor. NORMALIZE IT
    """
    omega_m0 = om
    omega_l0 = 1 - om
    x = omega_l0 / E(a) ** 2
    return (
        5.0
        / 6
        * (omega_m0 / omega_l0) ** (1.0 / 3)
        * jnp.sqrt(1 + omega_m0 / (omega_l0 * a * a * a))
        * betainc(5.0 / 6, 2.0 / 3, x)
    )


def T_FDM_Hu2000(k, m22):
    k_Jeq = 9 * jnp.sqrt(m22) / h
    x = 1.61 * m22 ** (1 / 18) * (k / k_Jeq)
    return jnp.cos(x**3) / (1 + x**8)


def top_hat_x_W(kR):
    return 3.0 * (jnp.sin(kR) - kR * jnp.cos(kR)) / (kR * kR * kR)


top_hat_x_W.M = lambda R, rho_m: 4.0 / 3 * jnp.pi * rho_m * R**3
top_hat_x_W.R = lambda M, rho_m: (3 * M / (4.0 * jnp.pi * rho_m)) ** (1 / 3)


def sharp_k_W(kR):
    # return jax.nn.sigmoid(100 * (1 - kR))
    return jnp.heaviside(1 - kR, 0)


sharp_k_W.M = lambda R, rho_m: 2.5 * top_hat_x_W.M(R, rho_m)
sharp_k_W.R = lambda M, rho_m: top_hat_x_W.R(M, rho_m) * 2.5 ** (-1 / 3)


def smooth_k_W(kR):
    return 1.0 / (1 + (kR) ** 9.10049)


smooth_k_W.M = lambda R, rho_m: 2.1594 * top_hat_x_W.M(R, rho_m)
smooth_k_W.R = lambda M, rho_m: top_hat_x_W.R(M, rho_m) * 2.1594 ** (-1 / 3)


def sigma_j(j, R, P_k, W_kR):
    """
    R in Mpc/h
    """

    @jax.jit
    @jax.vmap
    def integrand(logk):
        k = jnp.exp(logk)
        return k ** (2 * j + 3) / (2 * jnp.pi**2) * P_k(k) * W_kR(R * k) ** 2

    # logk_bounds = jnp.log(
    #     10 ** jnp.linspace(-4, jnp.min(jnp.array([4, jnp.log10(1.0 / R)])), 200)
    # )
    logk_bounds = jnp.log(10 ** jnp.linspace(-4, 4, 200))
    quads = jax.vmap(quad, in_axes=(None, 0, 0))
    res = jnp.sum(quads(integrand, logk_bounds[:-1], logk_bounds[1:]))
    return jnp.sqrt(res)


def sigma(R, P_k, W_kR):
    return sigma_j(0, R, P_k, W_kR)
