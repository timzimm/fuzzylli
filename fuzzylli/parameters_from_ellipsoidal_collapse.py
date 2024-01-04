import jax
import jax.numpy as jnp

"""
Fit results of ellipsoidal collapse runs. These functions should be treated as
black box.
"""


def cylinder_length_physical_Mpc(Msun):
    loga, b = -7.63270221, 0.27634879
    return jnp.exp(loga) * Msun**b


def cylinder_scale_radius_physical_Mpc(Msun, beta):
    """
    Physical scale radius r0 set as geometric mean of the two collapsed axes at
    z=4 (r) and rescaled so that at within r mass_fraction % of the total mass
    of the cylinder is contained.
    """
    # loga, b = -11.91067261, 0.33317267
    loga, b = -12.00153483, 0.33619234
    mass_fraction = 0.9
    return (1.0 / mass_fraction - 1) ** (1 / (2 - beta)) * jnp.exp(loga) * Msun**b


def cylinder_sqrt_v_dispersion_physical_kms(Msun, beta):
    # loga, b = -4.77023028, 0.32800087
    loga, b = -6.16244123, 0.3618256
    return jnp.sqrt(2.0 / (2 - beta)) * jnp.exp(loga) * Msun**b


def cylinder_line_mass_physical_Msun_Mpc(Msun):
    # loga, b = 10.2652066, 0.66325106
    loga, b = 7.63270222, 0.72365121
    return jnp.exp(loga) * Msun**b


def cylinder_dndM_Msun_inv_Msun_Mpc3(Msun):
    loga, b = 26.19080834, -2.19428402
    return jnp.exp(loga) * Msun**b


def sample_mass_from_powerlaw_dn_dM(N, Mmin, seed=42):
    """
    Inverse transform sampling from power law fit to full filament mass function
    for M>3e9 Msun
    """
    # b = 1.76285939
    b = -2.19428402

    u = jax.random.uniform(jax.random.PRNGKey(seed), shape=(N,))
    return Mmin * (1 - u) ** (1 / b)
