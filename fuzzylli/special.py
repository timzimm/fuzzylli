from functools import partial

import jax
import jax.numpy as jnp
from jax.lax import while_loop, fori_loop, cond


def rd(x, y, z, r=1e-7):
    """
    Symmetric elliptic integral of the second kind.
    Implementation follows as simplified version of Carlson's algorithm
    (arxiv:9409227) only applicable for non-zero real arguments x,y,z stolen
    from mpmath
    """

    xyz0 = jnp.array([x, y, z])
    A0 = (x + y + z + 2 * z) / 5
    Q = (0.25 * r) ** (-1 / 6) * jnp.max(jnp.abs(A0 - xyz0))
    pow4 = 1.0
    S = 0

    def body(args):
        (xyzm, Am, pow4, S) = args
        sxyz = jnp.sqrt(xyzm)
        lm = sxyz[0] * sxyz[1] + sxyz[0] * sxyz[2] + sxyz[1] * sxyz[2]
        Am = (Am + lm) * 0.25
        xyzm = (xyzm + lm) * 0.25
        dm = (sxyz[2] + sxyz[0]) * (sxyz[2] + sxyz[1]) * (sxyz[2] + sxyz[2])
        T = 1.0 * pow4 / dm
        S += T
        pow4 *= 0.25
        return xyzm, Am, pow4, S

    def run(args):
        (_, Am, pow4, _) = args
        return pow4 * Q > jnp.abs(Am)

    _, Am, pow4, S = while_loop(run, body, (xyz0, A0, pow4, S))

    t = pow4 / Am
    X = (A0 - x) * t
    Y = (A0 - y) * t
    Z = (A0 - z) * t
    E2 = X * Y + X * Z + Y * Z - 3 * Z**2
    E3 = X * Y * Z + 2 * E2 * Z + 4 * Z**3
    E4 = (2 * X * Y * Z + E2 * Z + 3 * Z**3) * Z
    E5 = X * Y * Z * Z**2
    P = (
        24024
        - 5148 * E2
        + 2457 * E2**2
        + 4004 * E3
        - 4158 * E2 * E3
        - 3276 * E4
        + 2772 * E5
    )
    Q = 24024
    v1 = pow4 * Am ** (-1.5) * P / Q
    v2 = 6 * S
    return v1 + v2


def bessel_int_J0(x):
    """
    Approximation by Chebyshev polynomials to the integral Bessel function
    of first kind and order 0: Ji0 = int( q^(-1) (1 - J0) )
    Luke, Y. L: Mathematical functions and their approximations (1975) Table 9.3
    """
    gamma = 0.5772156649015329

    a = jnp.array(
        [
            1.35105091918187636388,
            0.83791030734868376979,
            -0.35047963978529462711,
            0.12777415867753198659,
            -0.02981035698255560990,
            0.00455219841169387328,
            -0.00048408621967185359,
            0.00003780202859916883,
            -0.00000225886908506771,
            0.00000010664609068423,
            -0.00000000408005443149,
            0.00000000012909996251,
            -0.00000000000343577839,
            0.00000000000007799552,
            -0.00000000000000152842,
            0.00000000000000002612,
            -0.00000000000000000039,
            0.00000000000000000001,
        ]
    )

    c = jnp.array(
        [
            0.95360150809738558095 - 0.13917925930200001236j,
            -0.05860838853872331670 - 0.12902065726135067062j,
            -0.01020283575659856676 + 0.01103004348109535741j,
            0.00196012704043622581 + 0.00051817180856880364j,
            -0.00009574977697756219 - 0.00030928210173975681j,
            -0.00003570479477043714 + 0.00004647098443047525j,
            0.00001169677960430223 - 0.00000008198845340928j,
            -0.00000164386246452682 - 0.00000191888381006925j,
            -0.00000007415845751760 + 0.00000057813667761104j,
            0.00000011434387527717 - 0.00000008448997773317j,
            -0.00000003600903214141 - 0.00000000525612161520j,
            0.00000000601257386446 + 0.00000000763257790924j,
            0.00000000019124656215 - 0.00000000268643963177j,
            -0.00000000054892028385 + 0.00000000054279949860j,
            0.00000000022740445656 - 0.00000000001744365343j,
            -0.00000000005671490865 - 0.00000000003975692920j,
            0.00000000000607510983 + 0.00000000002069683990j,
            0.00000000000252060520 - 0.00000000000639623674j,
            -0.00000000000191255246 + 0.00000000000116359235j,
            0.00000000000074056501 + 0.00000000000006759603j,
            -0.00000000000018950214 - 0.00000000000016557337j,
            0.00000000000002021389 + 0.00000000000008425597j,
            0.00000000000001103617 - 0.00000000000002824474j,
            -0.00000000000000889993 + 0.00000000000000607698j,
            0.00000000000000388558 - 0.00000000000000003171j,
            -0.00000000000000119200 - 0.00000000000000077237j,
            0.00000000000000021456 + 0.00000000000000048022j,
            0.00000000000000002915 - 0.00000000000000019502j,
            -0.00000000000000004877 + 0.00000000000000005671j,
            0.00000000000000002737 - 0.00000000000000000862j,
            -0.00000000000000001080 - 0.00000000000000000269j,
            0.00000000000000000308 + 0.00000000000000000309j,
            -0.00000000000000000042 - 0.00000000000000000167j,
            -0.00000000000000000020 + 0.00000000000000000066j,
            0.00000000000000000020 - 0.00000000000000000019j,
            -0.00000000000000000011 + 0.00000000000000000003j,
            0.00000000000000000004 + 0.00000000000000000001j,
            -0.00000000000000000001 - 0.00000000000000000001j,
            0.00000000000000000000 + 0.00000000000000000001j,
        ]
    )

    def small_x_approximation(x):
        def body(i, T_012_ans):
            T0, T1, T2, ans = T_012_ans

            T2 = 2 * 0.125 * x * T1 - T0
            ans += (1 - i % 2) * a[i // 2] * T2
            T0 = T1
            T1 = T2
            return T0, T1, T2, ans

        return fori_loop(2, 36, body, (1.0, 0.125 * x, 0.0, a[0]))[-1]

    def large_x_approximation(x):
        def body(i, T_012_sum):
            T0, T1, T2, s = T_012_sum
            T2 = 2 * (10.0 / x - 1) * T1 - T0
            T0 = T1
            T1 = T2
            s += c[i] * T2
            return T0, T1, T2, s

        s = fori_loop(
            2, 39, body, (1.0, 10.0 / x - 1, 0.0, c[0] + c[1] * (10.0 / x - 1))
        )[-1]
        fac = jnp.cos(x + 0.25 * jnp.pi) * s.real - jnp.sin(x + 0.25 * jnp.pi) * s.imag
        return jnp.sqrt(2.0 / (jnp.pi * x)) / x * fac + gamma + jnp.log(0.5 * x)

    return cond(x < 8, small_x_approximation, large_x_approximation, x)


@partial(jax.custom_jvp, nondiff_argnums=(1, 2))
def lambertw(z: jnp.ndarray, tol: float = 1e-6, max_iter: int = 100) -> jnp.ndarray:
    """
    Lambert W function. Taken from
    https://github.com/ott-jax/ott/blob/main/src/ott/math/utils.py#L240
    """

    def initial_iacono(x: jnp.ndarray) -> jnp.ndarray:
        y = jnp.sqrt(1.0 + jnp.e * x)
        num = 1.0 + 1.14956131 * y
        denom = 1.0 + 0.45495740 * jnp.log1p(y)
        return -1.0 + 2.036 * jnp.log(num / denom)

    def _initial_winitzki(z: jnp.ndarray) -> jnp.ndarray:
        log1p_z = jnp.log1p(z)
        return log1p_z * (1.0 - jnp.log1p(log1p_z) / (2.0 - log1p_z))

    def cond_fun(cont):
        it, converged, _ = cont
        return jnp.logical_and(jnp.any(~converged), it < max_iter)

    def hailley_iteration(cont):
        it, _, w = cont

        f = w - z * jnp.exp(-w)
        delta = f / (w + 1.0 - 0.5 * (w + 2.0) * f / (w + 1.0))
        w_next = w - delta

        not_converged = jnp.abs(delta) <= tol * jnp.abs(w_next)
        return it + 1, not_converged, w_next

    w0 = initial_iacono(z)
    converged = jnp.zeros_like(w0, dtype=bool)

    _, _, w = while_loop(
        cond_fun=cond_fun, body_fun=hailley_iteration, init_val=(0, converged, w0)
    )
    return w


@lambertw.defjvp
def lambertw_jvp(tol: float, max_iter: int, primals, tangents):
    (z,) = primals
    (dz,) = tangents
    w = lambertw(z, tol=tol, max_iter=max_iter)
    pz = jnp.where(z == 0, 1.0, w / ((1.0 + w) * z))
    return w, pz * dz
