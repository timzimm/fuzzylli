from collections import namedtuple
import numpy as np
import jax.numpy as jnp
from jax.lax import fori_loop
from scipy.linalg import toeplitz

cheb_params = namedtuple("cheb_params", ["coeffs", "X_min", "X_max"])


def chebyshev_pts(N):
    x = jnp.sin(jnp.pi * ((N - 1) - 2 * jnp.linspace(N - 1, 0, N)) / (2 * (N - 1)))
    return x[::-1]


def chebyshev_dx(N):
    n1 = np.floor(N / 2).astype(int)
    n2 = np.ceil(N / 2).astype(int)
    k = np.arange(N)  # compute theta vector
    th = k * np.pi / (N - 1)

    x = chebyshev_pts(N)

    # Assemble the differentiation matrices
    T = np.tile(th / 2, (N, 1))
    DX = 2 * np.sin(T.T + T) * np.sin(T.T - T)  # trigonometric identity
    DX[n1:, :] = -np.flipud(np.fliplr(DX[0:n2, :]))  # flipping trick
    DX[range(N), range(N)] = 1.0  # diagonals of D
    DX = DX.T

    C = toeplitz((-1.0) ** k)
    C[0, :] *= 2
    C[-1, :] *= 2
    C[:, 0] *= 0.5
    C[:, -1] *= 0.5

    Z = 1.0 / DX  # Z contains entries 1/(x(k)-x(j))
    Z[range(N), range(N)] = 0.0  # with zeros on the diagonal.

    D = np.eye(N)

    D = Z * (C * np.tile(np.diag(D), (N, 1)).T - D)
    D[range(N), range(N)] = -np.sum(D, axis=1)

    return jnp.asarray(x), jnp.asarray(D)


def chebyshev_d2x(N):
    n1 = np.floor(N / 2).astype(int)
    n2 = np.ceil(N / 2).astype(int)
    k = np.arange(N)  # compute theta vector
    th = k * np.pi / (N - 1)

    x = chebyshev_pts(N)

    # Assemble the differentiation matrices
    T = np.tile(th / 2, (N, 1))
    DX = 2 * np.sin(T.T + T) * np.sin(T.T - T)  # trigonometric identity
    DX[n1:, :] = -np.flipud(np.fliplr(DX[0:n2, :]))  # flipping trick
    DX[range(N), range(N)] = 1.0  # diagonals of D
    DX = DX.T

    C = toeplitz((-1.0) ** k)
    C[0, :] *= 2
    C[-1, :] *= 2
    C[:, 0] *= 0.5
    C[:, -1] *= 0.5

    Z = 1.0 / DX  # Z contains entries 1/(x(k)-x(j))
    Z[range(N), range(N)] = 0.0  # with zeros on the diagonal.

    D = np.eye(N)

    D = Z * (C * np.tile(np.diag(D), (N, 1)).T - D)
    D[range(N), range(N)] = -np.sum(D, axis=1)
    D = 2 * Z * (C * np.tile(np.diag(D), (N, 1)).T - D)
    D[range(N), range(N)] = -np.sum(D, axis=1)

    return jnp.asarray(x), jnp.asarray(D)


def clenshaw_curtis_weights(n):
    """
    https://people.math.sc.edu/Burkardt/py_src/quadrule/clenshaw_curtis_compute.py
    """
    i = np.arange(n)
    theta = (n - 1 - i) * np.pi / (n - 1)

    w = np.zeros(n)

    for i in range(0, n):
        w[i] = 1.0

        jhi = (n - 1) // 2

        for j in range(0, jhi):
            if 2 * (j + 1) == (n - 1):
                b = 1.0
            else:
                b = 2.0

            w[i] = w[i] - b * np.cos(2.0 * float(j + 1) * theta[i]) / float(
                4 * j * (j + 2) + 3
            )

    w[0] = w[0] / float(n - 1)
    for i in range(1, n - 1):
        w[i] = 2.0 * w[i] / float(n - 1)
    w[n - 1] = w[n - 1] / float(n - 1)

    return w[::-1]


def barycentric_interpolation(x, x_j, f_j):
    """
    Barycentric interpolation at Chebyshev points
    """
    N = x_j.shape[0]
    C = (-1.0) ** jnp.arange(N)
    C = C.at[0].divide(2.0)
    C = C.at[N - 1].divide(2.0)
    w_div_dx = C / (x - x_j)
    num = jnp.dot(w_div_dx, f_j)
    denom = jnp.sum(w_div_dx)
    return num / denom


def _dct(data):
    """
    Compute DCT using FFT
    """
    N = len(data) // 2
    fftdata = jnp.fft.fft(data, axis=0)[: N + 1]
    fftdata /= N
    fftdata = fftdata.at[0].divide(2.0)
    fftdata = fftdata.at[-1].divide(2.0)
    if jnp.isrealobj(data):
        data = np.real(fftdata)
    else:
        data = fftdata
    return data


def cutoff_idx(cheb_params):
    scale = jnp.max(jnp.abs(cheb_params.coeffs))
    threshold = 128 * jnp.finfo(float).eps * scale
    inds = jnp.nonzero(
        abs(cheb_params.coeffs) >= threshold, size=cheb_params.coeffs.shape[0]
    )[0]
    return jnp.max(inds) + 1


def init_chebyshev_params(f_j, X_min, X_max):
    """
    Compute Chebyshev coefficients for values located on Chebyshev points.
    """
    f_j_evened = jnp.concatenate(
        [f_j, f_j[-2:0:-1]],
    )
    coeffs = _dct(f_j_evened)
    return cheb_params(coeffs=coeffs, X_min=X_min, X_max=X_max)


def eval_chebyshev_polynomial(x, cheb_params):
    def clenshaw_iteration(i, c0_c1):
        c0, c1 = c0_c1
        c0, c1 = cheb_params.coeffs[-i] - c1, c0 + c1 * (2 * t)
        return c0, c1

    x = jnp.clip(x, cheb_params.X_min, cheb_params.X_max)
    t = -1 + 2 * (x - cheb_params.X_min) / (cheb_params.X_max - cheb_params.X_min)
    c0, c1 = fori_loop(
        3,
        cheb_params.coeffs.shape[0] + 1,
        clenshaw_iteration,
        (cheb_params.coeffs[-2], cheb_params.coeffs[-1]),
    )

    return c0 + c1 * t
