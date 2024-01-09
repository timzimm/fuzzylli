import numpy as np
import jax.numpy as jnp
from scipy.linalg import toeplitz


def chebyshev_pts(N):
    x = jnp.sin(jnp.pi * ((N - 1) - 2 * jnp.linspace(N - 1, 0, N)) / (2 * (N - 1)))
    return x[::-1]


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
