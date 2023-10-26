from collections import namedtuple
import jax.numpy as jnp

# A ray in R^d is defined by a origin vector and a direction vector
ray_params = namedtuple("ray_params", ["origin", "direction"])


def init_ray_params(origin, direction):
    """
    Initializes a ray (this is trivial an djust exists for consistency)
    """
    return ray_params(origin=origin, direction=direction)


def eval_ray(l, ray_params):
    """
    Evaluates a ray at parameter l
    """
    return ray_params.origin + l * ray_params.direction


def closest_point_on_ray(point, ray_param):
    """
    Trivial analytical solution to finding the closest point on a ray to a
    given point not necessarily on the ray. Used for radius computations
    """
    origin = ray_param.origin
    direction = ray_param.direction
    l = (point - origin) @ direction / jnp.dot(direction, direction)
    return eval_ray(l, ray_param)
