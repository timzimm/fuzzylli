import jax
import jax.numpy as jnp
from diffrax import diffeqsolve, ODETerm, Dopri8, DirectAdjoint

from fuzzylli.cosmology import (
    E as Ea,
    nonlinear_delta_sc_virial_at,
    D,
    om,
)
from fuzzylli.special import rd


def delta(a, a_123):
    """
    Nonlinear density contrast of ellisoid with repsect to rho = rho_m0 a^3
    """
    return a**3 / jnp.prod(a_123, axis=0) - 1


def linear_external_shear(a, a_123, lambda0, a0):
    """
    Linear approximation of external shear
    """
    return D(a) / D(a0) * (lambda0 - jnp.sum(lambda0) / 3)


def nonlinear_external_shear(a, a_123, lambda0, a0):
    """
    Linear approximation of external shear
    """
    return 5 / 4 * b(a_123)


def b(a_123):
    """
    Internal shear
    """
    a1, a2, a3 = a_123

    return (
        2
        / 3
        * jnp.prod(a_123)
        * jnp.stack(
            [
                rd(a2**2, a3**2, a1**2),
                rd(a3**2, a1**2, a2**2),
                rd(a1**2, a2**2, a3**2),
            ]
        )
        - 2 / 3
    )


def C(a, a_123, lambda0, a0):
    """
    Sum of all forces
    """
    return (
        (1 + delta(a, a_123)) / 3
        + b(a_123) / 2 * delta(a, a_123)
        + linear_external_shear(a, a_123, lambda0, a0)
    )


def virial_term(a, a_i, lambda0, a0):
    ol = 1 - om
    return (
        1.0
        / (a**2 * Ea(a) ** 2)
        * (3 * om / (2 * a**3) * C(a, a_i, lambda0, a0) - ol)
    )


dvirial_term = jax.jacfwd(virial_term)


def dai_da_freeze_out(a, y, args):
    """
    Vector field of d/dt (a_i, d/dt a_i) of the ellipsoidal collapse.
    Scale factor used as time.
    Once frozen_axis[i] turns Truethy (e.g. a non-zero float),
    we freeze out to the axis value frozen_axis[i]
    """
    (lambda0, a0, frozen_axis) = args
    a_i = y[:3]
    A_i = y[3:]

    return jnp.hstack(
        [
            jnp.where(frozen_axis, 0.0, A_i),
            jnp.where(
                frozen_axis,
                0.0,
                -(1.0 / a + jax.grad(Ea)(a) / Ea(a)) * A_i
                - virial_term(a, a_i, lambda0, a0) * a_i,
            ),
        ]
    )


def dai_da_constant_density(a, y, args):
    """
    Vector field of d/dt (a_i, d/dt a_i) of the ellipsoidal collapse.
    Scale factor used as time.
    Once frozen_axis[i] turns Truethy (e.g. a non-zero float),
    we freeze out to the axis value frozen_axis[i]*a
    """
    (lambda0, a0, frozen_axis) = args
    a_i = y[:3]
    A_i = y[3:]

    return jnp.hstack(
        [
            jnp.where(frozen_axis, frozen_axis, A_i),
            jnp.where(
                frozen_axis,
                0.0,
                -(1.0 / a + jax.grad(Ea)(a) / Ea(a)) * A_i
                - virial_term(a, a_i, lambda0, a0) * a_i,
            ),
        ]
    )


def ic(e, p, a0, delta0):
    """
    Initial conditions for a1,a2,a3 paramterized on the ellipticity and prolatness
    of initial shear field
    """
    lambda0 = delta0 / 3 * jnp.stack([1 + 3 * e + p, 1 - 2 * p, 1 - 3 * e + p])
    y0 = jnp.hstack([a0 * (1 - lambda0), 1 - 2 * lambda0])
    return y0, lambda0


# Freeze-out conditions
def above_nonlinear_density(a, y, lambda0, a0, signs, frozen_axis):
    return signs, jnp.repeat(delta(a, y[:3]) > 1e5, 3)


above_nonlinear_density.frozen_axis_0 = jnp.asarray(3 * [True])


def axis_i_virialized(a, y, lambda0, a0, signs, frozen_axis):
    A_i = y[3:]
    a_i = y[:3]
    signs_next = ~jnp.signbit(
        (A_i / a_i) ** 2
        - 1.0
        / (a**2 * Ea(a) ** 2)
        * (3 * om / (2 * a**3) * C(a, a_i, lambda0, a0) - (1 - om))
    )
    has_turned_around = Ea(a) * a * A_i < 0
    has_virialized = jnp.logical_xor(signs, signs_next)
    return signs_next, jnp.where(
        jnp.logical_or(frozen_axis, jnp.logical_and(has_turned_around, has_virialized)),
        jnp.where(frozen_axis, frozen_axis, a_i / a),
        0.0,
    )


axis_i_virialized.frozen_axis_0 = jnp.asarray(3 * [0.0])


def bond_meyers_freeze_out(a, y, lambda0, a0, signs, frozen_axis):
    A_i = y[3:]
    a_i = y[:3]
    signs_next = ~jnp.signbit(y[:3] - nonlinear_delta_sc_virial_at(a) ** (-1.0 / 3) * a)
    has_turned_around = Ea(a) * a * A_i < 0
    has_passed_threshold = jnp.logical_xor(signs, signs_next)
    return signs_next, jnp.where(
        jnp.logical_or(
            frozen_axis, jnp.logical_and(has_turned_around, has_passed_threshold)
        ),
        jnp.where(frozen_axis, frozen_axis, a_i / a),
        0.0,
    )


bond_meyers_freeze_out.frozen_axis_0 = jnp.asarray(3 * [0.0])


def ellipsoidal_collapse_trajectories(
    dai_da, freeze_out_condition, y0, a0, aend, lambda0
):
    """
    Given the ellipsoidal collapse vector field dai_da and intial conditions
    y0=(a10,a20,a30,a1'0,a2'0,a3'0) at a=a0, integrate until aend
    """

    # frozen_axis = jnp.asarray(3 * [0.0])
    frozen_axis = freeze_out_condition.frozen_axis_0
    signs = jnp.asarray(3 * [True])

    da = 5e-6

    solver = Dopri8()
    f = ODETerm(dai_da)

    args0 = (a0, y0, signs, frozen_axis)
    a = jnp.linspace(a0, aend, int((aend - a0) / da + 1), endpoint=False)

    def body(args, anext):
        a, y, signs, frozen_axis = args

        signs_next, frozen_axis_next = freeze_out_condition(
            a, y, lambda0, a0, signs, frozen_axis
        )

        made_jump = jnp.any(jnp.logical_xor(frozen_axis, frozen_axis_next))
        sol = diffeqsolve(
            f,
            solver,
            a,
            anext,
            da,
            y,
            adjoint=DirectAdjoint(),
            args=(lambda0, a0, frozen_axis_next),
            made_jump=made_jump,
        )
        return (anext, sol.ys.flatten(), signs_next, frozen_axis_next), (
            sol.ys.flatten(),
            made_jump,
        )

    _, (ys, made_jump) = jax.lax.scan(body, args0, a[1:])
    return (
        a,
        jnp.hstack([y0[:, None], ys.T]),
        jnp.concatenate([jnp.array([False]), made_jump]),
    )
