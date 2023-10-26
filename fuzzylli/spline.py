import jax.numpy as np
from collections import namedtuple

spline_params = namedtuple("spline_params", ["x_i", "y_i", "coefs"])


def init_spline_params(x, y):
    """
    Initializes a third order (cubic) spline parameter dictionary assuming
    "not-a-knot" boundary conditions.
    """
    n_data = len(x)

    # Difference vectors
    h = np.diff(x)
    p = np.diff(y)

    # Special values for the first and last equations
    A00 = np.array([h[1]])
    A01 = np.array([-(h[0] + h[1])])
    A02 = np.array([h[0]])
    ANN = np.array([h[-2]])
    AN1 = np.array([-(h[-2] + h[-1])])
    AN2 = np.array([h[-1]])

    # Construct the tri-diagonal matrix A
    A = np.diag(np.concatenate((A00, 2 * (h[:-1] + h[1:]), ANN)))
    upper_diag1 = np.diag(np.concatenate((A01, h[1:])), k=1)
    upper_diag2 = np.diag(np.concatenate((A02, np.zeros(n_data - 3))), k=2)
    lower_diag1 = np.diag(np.concatenate((h[:-1], AN1)), k=-1)
    lower_diag2 = np.diag(np.concatenate((np.zeros(n_data - 3), AN2)), k=-2)
    A += upper_diag1 + upper_diag2 + lower_diag1 + lower_diag2
    # Construct RHS vector s
    center = 3 * (p[1:] / h[1:] - p[:-1] / h[:-1])
    s = np.pad(center, 1)

    # Compute spline coefficients by solving the system
    coefs = np.linalg.solve(A, s)

    return spline_params(x_i=x, y_i=y, coefs=coefs)


def init_periodic_spline_params(x, y):
    """
    Initializes a third order (cubic) spline parameter dictionary assuming
    periodic boundary conditions
    """
    L = x[-1]
    x = x[:-1]
    y = y[:-1]
    n_data = len(x)

    # Difference vectors
    h = np.diff(x)
    p = np.diff(y)

    A00 = np.array([2 * (L - x[-1] + h[0])])
    A01 = np.array([h[0]])
    A02 = np.array([0])
    ANN = np.array([2 * (L - x[-1] + h[-1])])
    AN1 = np.array([h[-1]])
    AN2 = np.array([0])

    # Construct the tri-diagonal matrix A
    A = np.diag(np.concatenate((A00, 2 * (h[:-1] + h[1:]), ANN)))
    upper_diag1 = np.diag(np.concatenate((A01, h[1:])), k=1)
    upper_diag2 = np.diag(np.concatenate((A02, np.zeros(n_data - 3))), k=2)
    lower_diag1 = np.diag(np.concatenate((h[:-1], AN1)), k=-1)
    lower_diag2 = np.diag(np.concatenate((np.zeros(n_data - 3), AN2)), k=-2)
    A += upper_diag1 + upper_diag2 + lower_diag1 + lower_diag2
    # Construct RHS vector s
    center = 3 * (p[1:] / h[1:] - p[:-1] / h[:-1])
    s = np.pad(center, 1)

    A = A.at[0, -1].set(L - x[-1])
    A = A.at[-1, 0].set(L - x[-1])
    s = s.at[0].set(3 * (p[0] / h[0] - (y[0] - y[-1]) / (L - x[-1])))
    s = s.at[-1].set(3 * ((y[0] - y[-1]) / (L - x[-1]) - p[-1] / h[-1]))

    # Compute spline coefficients by solving the system
    coefs = np.linalg.solve(A, s)
    coefs = np.concatenate([coefs, np.array([coefs[0]])])
    x = np.concatenate([x, np.array([L])])
    y = np.concatenate([y, np.array([y[0]])])

    return spline_params(x_i=x, y_i=y, coefs=coefs)


def evaluate_spline(x, spline_params):
    """
    Evaluates a third order (cubic) spline model defined by params
    """
    knots = spline_params.x_i
    y = spline_params.y_i
    coefs = spline_params.coefs

    # Determine the interval that x lies in
    ind = np.digitize(x, knots) - 1
    # Include the right endpoint in spline piece C[m-1]
    ind = np.clip(ind, 0, len(knots) - 2)
    t = x - knots[ind]
    h = np.diff(knots)[ind]
    c = coefs[ind]
    c1 = coefs[ind + 1]
    a = y[ind]
    a1 = y[ind + 1]
    b = (a1 - a) / h - (2 * c + c1) * h / 3.0
    d = (c1 - c) / (3 * h)
    return a + b * t + c * t**2 + d * t**3
