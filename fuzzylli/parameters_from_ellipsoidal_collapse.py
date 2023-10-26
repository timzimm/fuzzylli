import jax
import jax.numpy as jnp

"""
Fit results of ellipsoidal collapse runs. These functions should be treated as
black box.
"""


def cylinder_length_physical_Mpc(Msun):
    loga, b = -10.41712413, 0.34399827
    loga, b = -6.38615268, 0.21784641
    return jnp.exp(loga) * Msun**b


def cylinder_scale_radius_physical_Mpc(Msun, beta):
    """
    Physical scale radius r0 set as geometric mean of the two collapsed axes at
    z=4 (r) and rescaled so that at within r mass_fraction % of the total mass
    of the cylinder is contained.
    """
    # loga, b = -11.91067261, 0.33317267
    loga, b = -12.02585532, 0.33718792
    mass_fraction = 0.9
    return (1.0 / mass_fraction - 1) ** (1 / (2 - beta)) * jnp.exp(loga) * Msun**b


def cylinder_sqrt_v_dispersion_physical_kms(Msun, beta):
    # loga, b = -4.77023028, 0.32800087
    loga, b = -6.78571601, 0.39107679
    return jnp.sqrt(2.0 / (2 - beta)) * jnp.exp(loga) * Msun**b


def cylinder_line_mass_physical_Msun_Mpc(Msun):
    # loga, b = 10.2652066, 0.66325106
    loga, b = 6.38615269, 0.78215359
    return jnp.exp(loga) * Msun**b


def cylinder_dndM_Msun_inv_Msun_Mpc3(Msun):
    # loga, b = 15.96909906, -1.76285939
    loga, b = 26.35112415, -2.19942357
    return jnp.exp(loga) * Msun**b


def sample_mass_from_powerlaw_dn_dM(N, Mmin, seed=42):
    """
    Inverse transform sampling from power law fit to full filament mass function
    for M>2e9 Msun
    """
    # b = 1.76285939
    b = -2.19942357

    u = jax.random.uniform(jax.random.PRNGKey(seed), shape=(N,))
    return Mmin * (1 - u) ** (1 / b)
