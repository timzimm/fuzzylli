import jax.numpy as jnp
from jax import vmap


class UniformHypercube:
    def __init__(self, N, extends=jnp.array([[0.0, 1.0]])):
        self._extends = jnp.atleast_2d(jnp.asarray(extends))
        self._dim = self._extends.shape[0]
        self.N = N
        self.dtype = self._extends.dtype
        self._points = jnp.array([])

    @property
    def dim(self):
        return self._dim

    @property
    def extends(self):
        return self._extends

    @property
    def N(self):
        return self._N

    @property
    def dxi(self):
        return self._dxi

    @property
    def dki(self):
        return 2 * jnp.pi / self.extends[:, 1]

    @property
    def L(self):
        return self._extends[:, 1] - self._extends[:, 0]

    @property
    def volumetric_center(self):
        return self._extends[:, 0] + (self.L) / 2

    @property
    def dV(self):
        return jnp.prod(self._dxi)

    @property
    def V(self):
        return jnp.prod(self._extends[:, 1] - self._extends[:, 0])

    @property
    def cell_centers(self):
        centers = [
            self.dxi[i] * jnp.arange(self._N[i], dtype=self.dtype)
            + self.dxi[i] / 2
            + self.extends[i][0]
            for i in range(self._extends.shape[0])
        ]
        return centers

    @property
    def cell_interfaces(self):
        interfaces = [
            self.dxi[i] * jnp.arange(self._N[i] - 1, dtype=self.dtype)
            + self.dxi[i]
            + self.extends[i][0]
            for i in range(self._extends.shape[0])
        ]
        return interfaces

    @staticmethod
    def __cartesian_product(*vecs):
        def cartesian_product(a, b):
            return vmap(
                lambda a, b: jnp.c_[a * jnp.ones(b.shape[0]), b], in_axes=(0, None)
            )(a, b).reshape(-1, a.ndim + b.ndim)

        axb = cartesian_product(vecs[-2], vecs[-1])
        for vec in reversed(vecs[:-2]):
            axb = cartesian_product(vec, axb)
        return axb

    @property
    def cell_centers_cartesian(self):
        if self.dim == 1:
            return self.cell_centers[0]
        return UniformHypercube.__cartesian_product(*self.cell_centers)

    @property
    def cell_interfaces_cartesian(self):
        if self.dim == 1:
            return self.cell_interfaces[0]
        return UniformHypercube.__cartesian_product(*self.cell_interfaces)

    @property
    def cell_centers_meshgrid(self):
        cartesian_grid = UniformHypercube.__cartesian_product(*self.cell_centers)
        return [cartesian_grid[:, i].reshape(*self.N) for i in range(self.dim)]

    @property
    def cell_interfaces_meshgrid(self):
        cartesian_grid = UniformHypercube.__cartesian_product(*self.cell_interfaces)
        return [cartesian_grid[i].reshape(self.N) for i in range(self.dim)]

    @property
    def points(self):
        return self._points

    @N.setter
    def N(self, N):
        assert len(N) == self.dim
        self._N = jnp.asarray(N)
        self._dxi = (self._extends[:, 1] - self._extends[:, 0]) / self._N

    @points.setter
    def points(self, points):
        mask = jnp.all(
            jnp.logical_and(self.extends[:, 0] <= points, points < self.extends[:, 1]),
            axis=-1,
        )
        self._points = points[mask]

    def translate(self, vector):
        assert vector.ndim == 1
        translated_cube = UniformHypercube(
            self.N, vector[:, jnp.newaxis] + self._extends
        )
        if self.points.shape[0] != 0:
            translated_cube.points = self.points + vector
        return translated_cube
