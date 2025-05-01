from functools import partial
import jax
import jax.numpy as jnp
from jax import jit


@partial(jit, static_argnums=(1, 2))
def compute_b_spline_matrix(x: jax.Array, degree: int, num_points: int) -> jax.Array:
    """
    Compute the B-spline basis matrix for a given degree and knot vector.

    Parameters:
        x (list or jax.Array): The knot vector.
        degree (int): The degree of the B-spline basis.
        num_points (int): The number of points to sample in the parameter domain.

    Returns:
        jax.Array: A matrix where each row represents a parameter value and each column corresponds to a basis function.
    """
    t_values = jnp.linspace(x[1], x[-2], num_points + 4)[2:-2]  # Exclude the first and last knot values
    b = jnp.where((x[:-1] <= t_values[:, None]) & (t_values[:, None] <= x[1:]), 1.0, 0.0)

    for d in range(1, degree + 1):
        left_d1, left_d2 = x[d:-1], x[:-d - 1]  # t_{i+d} - t_i
        b_left = jnp.where(left_d1 > left_d2, ((t_values[:, None] - left_d2) / (left_d1 - left_d2)) * b[:, :-1], 0.0)
        right_d1, right_d2 = x[d + 1:], x[1:-d]  # t_{i+d+1} - t_{i+1}
        b_right = jnp.where(right_d1 > right_d2, ((right_d1 - t_values[:, None]) / (right_d1 - right_d2)) * b[:, 1:], 0.0)
        b = b_left + b_right

    return b


@partial(jit, static_argnums=(1, 2, 3))
def sample_with_replacement(rng: jax.Array, M: int, N: int, num_samples: int) -> jax.Array:
    return jax.random.randint(rng, (num_samples, M), 0, N)
