"""Akima spline interpolation for MTP control trajectories.

Implements the modified Akima spline algorithm for computing smooth
piecewise-polynomial interpolants through tensor-sampled control waypoints.
Optimized for JAX JIT compilation with vectorized evaluation.
"""

from functools import partial

import jax
import jax.numpy as jnp
from jax import jit, vmap


@jit
def poly_akima(x: jax.Array, C: jax.Array) -> jax.Array:
    """Compute Akima spline coefficients for control waypoints.

    Computes the piecewise cubic polynomial coefficients using the
    modified Akima interpolation method.

    Args:
        x: Knot positions, shape (M,). Must be uniformly spaced.
        C: Control waypoints, shape (M, D) where D is the control
           dimension.

    Returns:
        Polynomial coefficients A of shape (M-1, 4, D), where
        A[i] = [d_i, c_i, b_i, a_i] for the i-th segment.
    """
    dx = jnp.diff(x)
    M, dim = C.shape[0], C.shape[1]
    n_seg = M - 1  # number of segments

    # Compute slopes between breakpoints
    safe_dx = jnp.where(dx == 0, 1.0, dx)
    dxr = jnp.where(dx == 0, 0.0, 1.0 / safe_dx)[:, None]  # (n_seg, 1)

    y_l, y_r = C[:-1], C[1:]
    slopes = (y_r - y_l) * dxr  # (n_seg, D)

    # Special-case M==2: with only one real slope the Akima padded-slope
    # cascade reads uninitialised entries and produces wildly wrong
    # boundary derivatives. Fall back to a linear segment with both
    # endpoint derivatives equal to the single segment slope.
    if n_seg == 1:
        dydx_l = slopes * dxr  # (1, D)
        dydx_r = slopes * dxr
        ai = y_l
        bi = dydx_l
        ci = 3.0 * slopes - 2.0 * dydx_l - dydx_r
        di = -2.0 * slopes + dydx_l + dydx_r
        A = jnp.stack([di, ci, bi, ai], axis=-1)  # (1, D, 4)
        scale = dxr[:, None] ** jnp.arange(4)[::-1]  # (1, 1, 4)
        A = A / scale
        return jnp.moveaxis(A, -1, 1)  # (1, 4, D)

    # Build padded slope array: 2 extra on each side for Akima boundary
    # m[0], m[1] = boundary extrapolations (left)
    # m[2 : 2+n_seg] = actual slopes
    # m[2+n_seg], m[3+n_seg] = boundary extrapolations (right)
    m = jnp.zeros((n_seg + 4, dim))
    m = m.at[2:2 + n_seg].set(slopes)

    # Akima boundary extrapolation
    m = m.at[1].set(2.0 * m[2] - m[3])
    m = m.at[0].set(2.0 * m[1] - m[2])
    m = m.at[-2].set(2.0 * m[-3] - m[-4])
    m = m.at[-1].set(2.0 * m[-2] - m[-3])

    # Compute Akima derivative weights
    dm = jnp.abs(jnp.diff(m, axis=0))   # (n_seg+3, D)
    pm = jnp.abs(m[1:] + m[:-1])        # (n_seg+3, D)
    f1 = dm[2:] + 0.5 * pm[2:]          # (M, D)
    f2 = dm[:-2] + 0.5 * pm[:-2]        # (M, D)
    m2 = m[1:-2]                         # (M, D)
    m3 = m[2:-1]                         # (M, D)
    f12 = f1 + f2
    nonzero = f12 > 1e-9 * jnp.max(jnp.abs(f12), initial=0.0)
    dydx = jnp.where(
        nonzero,
        (f1 * m2 + f2 * m3) / jnp.where(nonzero, f12, 1.0),
        0.5 * (m2 + m3),
    )

    # Compute cubic polynomial coefficients per segment
    dydx_l = dydx[:-1] * dxr   # (n_seg, D)
    dydx_r = dydx[1:] * dxr    # (n_seg, D)
    ai = y_l                    # (n_seg, D)
    bi = dydx_l                 # (n_seg, D)
    ci = 3.0 * slopes - 2.0 * dydx_l - dydx_r
    di = -2.0 * slopes + dydx_l + dydx_r
    A = jnp.stack([di, ci, bi, ai], axis=-1)  # (n_seg, D, 4)

    # Scale coefficients for actual spacing (dx != 1)
    scale = dxr[:, None] ** jnp.arange(4)[::-1]  # (n_seg, 1, 4)
    A = A / scale
    A = jnp.moveaxis(A, -1, 1)  # (n_seg, 4, D)
    return A


@partial(jit, static_argnames=("num_points",))
def poly_interpolation(A: jax.Array, num_points: int = 5) -> jax.Array:
    """Evaluate Akima spline segments at uniformly-spaced query points.

    Uses direct matrix multiplication instead of jnp.vectorize for
    efficient XLA-fuseable evaluation.

    Args:
        A: Polynomial coefficients of shape (B, M-1, 4, D) where B is
           the batch dimension, M-1 is the number of segments, 4 is the
           polynomial degree+1, and D is the control dimension.
        num_points: Number of evaluation points per segment.

    Returns:
        Interpolated values of shape (B, (M-1)*num_points, D).
    """
    B, n_segments, _, dim = A.shape

    # Query points within each segment [0, 1)
    t = jnp.linspace(0, 1, num_points + 1)[:-1]  # (num_points,)

    # Vandermonde-like matrix for polynomial evaluation: t^3, t^2, t^1, t^0
    powers = jnp.arange(3, -1, -1)               # [3, 2, 1, 0]
    t_powers = t[:, None] ** powers[None, :]      # (num_points, 4)

    # Evaluate all segments in parallel: (B, n_segments, num_points, D)
    # A: (B, n_segments, 4, D), t_powers: (num_points, 4)
    # einsum: contract over polynomial degree axis
    spline = jnp.einsum("bspd,tp->bstd", A, t_powers)

    # Reshape to (B, n_segments * num_points, D)
    return spline.reshape(B, -1, dim)
