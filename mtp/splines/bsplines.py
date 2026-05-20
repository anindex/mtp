"""B-spline basis matrix computation for MTP control trajectories.

Provides efficient computation of the B-spline basis matrix used for
smooth interpolation of tensor-sampled control waypoints.
"""

import jax
import jax.numpy as jnp


def compute_b_spline_matrix(
    x: jax.Array, degree: int, num_points: int
) -> jax.Array:
    """Compute the B-spline basis matrix for a given degree and knot vector.

    Constructs the basis function matrix using the Cox-de Boor recursion,
    evaluated at uniformly-spaced parameter values inside the *valid*
    parameter domain ``[x[degree], x[-degree-1]]`` so the partition of
    unity holds at every evaluation point. The resulting matrix can be
    used for efficient batch interpolation via einsum.

    Args:
        x: The knot vector, shape (M + degree + 1,).
        degree: The degree of the B-spline basis (typically 2 or 3).
        num_points: The number of evaluation points in the parameter domain.

    Returns:
        Basis matrix of shape (num_points, M) where each row corresponds
        to a parameter value and each column to a basis function. Each
        row sums to 1 (partition of unity).
    """
    # Evaluate strictly inside the valid B-spline parameter domain
    # [x[degree], x[-degree-1]] so rows of the returned basis matrix
    # satisfy partition-of-unity at every evaluation point.
    t_start = x[degree]
    t_end = x[-degree - 1]
    # Slight inward shrink avoids the right-open `<= x[i+1]` boundary case
    # at t == t_end (which would otherwise activate two intervals at once).
    eps = (t_end - t_start) * 1e-7
    t_values = jnp.linspace(t_start, t_end - eps, num_points)

    b = jnp.where(
        (x[:-1] <= t_values[:, None]) & (t_values[:, None] < x[1:]),
        1.0,
        0.0,
    )

    for d in range(1, degree + 1):
        left_d1, left_d2 = x[d:-1], x[:-d - 1]
        denom_left = jnp.where(left_d1 > left_d2, left_d1 - left_d2, 1.0)
        b_left = jnp.where(
            left_d1 > left_d2,
            ((t_values[:, None] - left_d2) / denom_left) * b[:, :-1],
            0.0,
        )

        right_d1, right_d2 = x[d + 1:], x[1:-d]
        denom_right = jnp.where(
            right_d1 > right_d2, right_d1 - right_d2, 1.0
        )
        b_right = jnp.where(
            right_d1 > right_d2,
            ((right_d1 - t_values[:, None]) / denom_right) * b[:, 1:],
            0.0,
        )

        b = b_left + b_right

    return b
