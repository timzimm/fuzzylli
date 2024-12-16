from collections import namedtuple
import hppfcl as fcl

import numpy as np
import jax.numpy as jnp
from jax.random import normal, PRNGKey, split
from jax import vmap
from jax.scipy.special import erf

from fuzzylli.domain import UniformHypercube
from fuzzylli.ray import init_ray_params, closest_point_on_ray
from fuzzylli.poisson_disk import PeriodicPoissonDisk

# A straight filament spine is defined by a ray (holding origin and direction)
# a finite length and its azimuthal orientation
finite_straight_filament_spine_params = namedtuple(
    "finite_straight_filament_spine_params", ["ray_params", "length", "orientation"]
)


def sample_from_poisson_disk(N, d, domain, seed):
    sampler = PeriodicPoissonDisk(*domain.L, d)
    return jnp.asarray(sampler.sample(N))


def sample_from_S(N, d, key):
    """
    Sample uniformly from S^d
    """
    xyz = normal(key, shape=(N, d))
    directions = xyz / jnp.linalg.norm(xyz)
    return directions


def R_from(a, b):
    """
    Construct rotation matrxi between unit vectors a and b, i.e. a gets
    rotated onto b
    See:
    https://math.stackexchange.com/questions/180418/calculate-rotation-matrix-to-align-vector-a-to-vector-b-in-3d
    """
    v = jnp.cross(a, b)
    s = jnp.linalg.norm(v)
    c = jnp.dot(a, b)
    M = jnp.cross(v, jnp.eye(3) * -1)
    return jnp.eye(3) + M + (1 - c) / s**2 * M @ M


def init_finite_straight_filament_spine_gas(N, r, l, seed, domain, lengths):
    """
    Initializes a random cylinder gas consisting of (at most) N cylinders.
    Cylinder locations are drawn from a periodic Poisson disk with point distance
    set to 2*L/N (magic number).
    Directions follow from uniformly sampling S^3 and orientations
    from a draw from S^2 located in the plane perpendicular to directions
    Cylinders are not allowed to collide.
    """
    init_rays_params = vmap(init_ray_params)
    origins = []
    directions = []
    cylinders = []
    translations = UniformHypercube(
        [3, 3, 3], jnp.array([[-1.5, 1.5], [-1.5, 1.5], [-1.5, 1.5]])
    )

    key = PRNGKey(seed)

    i = 0
    new_cylinder = fcl.Cylinder(float(r), float(l))
    sampler = PeriodicPoissonDisk(*domain.L, 2 * jnp.max(domain.L) / N, seed=seed)
    while len(origins) < N and i < 1000:
        key_dir, key_orient, key = split(key, 3)
        origin = sampler.sample(1)
        direction = sample_from_S(1, domain.dim, key_dir).squeeze()
        R = np.asarray(R_from(jnp.array([0, 0, 1]), direction))
        collide = False
        print(origin)
        for c in cylinders:
            for v in domain.L * translations.cell_centers_cartesian:
                new_cylinder_t = fcl.Transform3f(R, np.asarray(origin + v))

                new_coll_obj = fcl.CollisionObject(new_cylinder, new_cylinder_t)

                request = fcl.CollisionRequest()
                result = fcl.CollisionResult()

                collide = fcl.collide(c, new_coll_obj, request, result)
            if collide:
                break

        if not collide:
            new_cylinder_t = fcl.Transform3f(R, np.asarray(origin))
            new_coll_obj = fcl.CollisionObject(new_cylinder, new_cylinder_t)
            origins.append(origin)
            directions.append(direction)
            cylinders.append(new_coll_obj)
        i = i + 1
    origins = jnp.asarray(origins)
    directions = jnp.asarray(directions)

    N = origins.shape[0]

    # Orientation of cylinders
    v = sample_from_S(N, 2, key)
    v1 = jnp.cross(directions, origins)
    v2 = jnp.cross(directions, v1)

    # Basis vectors in plane perpendicular to directions
    v1 = v1 / jnp.linalg.norm(v1)
    v2 = v2 / jnp.linalg.norm(v2)
    orientations = v[:, 0, jnp.newaxis] * v1 + v[:, 1, jnp.newaxis] * v2
    orientations = orientations / jnp.linalg.norm(orientations)

    return [
        finite_straight_filament_spine_params(
            ray_params=init_rays_params(origins[i, :], directions[i, :]),
            length=jnp.asarray(lengths[i]),
            orientation=jnp.asarray(orientations[i]),
        )
        for i in range(N)
    ]


def Rphiz_from_xyz(x, finite_straight_filament_spine_params):
    ray_params = finite_straight_filament_spine_params.ray_params
    orientation = finite_straight_filament_spine_params.orientation

    p = closest_point_on_ray(x, ray_params)
    # Vector in plane orthorgonal to cylinder axis (ray_params.direction)
    x_m_p = x - p

    R = jnp.linalg.norm(x_m_p, axis=-1)

    # Polar coordinates of vector in orthogonal plane.
    # Cylinder orientation defines anti-clockwise direction
    x_2d = jnp.dot(x_m_p, orientation)
    y_2d = jnp.dot(jnp.cross(x_m_p, orientation), ray_params.direction)
    phi = jnp.arctan2(y_2d, x_2d)

    return R, phi, jnp.linalg.norm(p - ray_params.origin, axis=-1)


def hard_cylinder_cutoff_along_z(rho, norm_z, length):
    return jnp.where(norm_z < length / 2, rho, 0.0).squeeze()


def soft_cylinder_cutoff_along_z(rho, norm_z, length):
    eps = 10 / length
    return (1 / 2 * (erf(eps * (length / 2 - norm_z)) + 1) * rho).squeeze()


def generic_finite_cylinder_density(
    eval_density, x, finite_straight_filament_spine_params, density_params
):
    """
    Density profile extruded along finite straight spine
    """
    length = finite_straight_filament_spine_params.length
    R, phi, norm_z = Rphiz_from_xyz(x, finite_straight_filament_spine_params)

    return soft_cylinder_cutoff_along_z(
        eval_density(R, phi, density_params), norm_z, length
    )
