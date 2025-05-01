from functools import partial
import jax
import jax.numpy as jnp
from jax import jit, vmap


@jit
def poly_akima(x: jax.Array, C: jax.Array) -> jax.Array:
    # NOTE: this assume uniform spacing on the knots e.g., [1, 2, 3, ...]
    dx = jnp.diff(x)
    M, dim = C.shape[0], C.shape[1]

    # determine slopes between breakpoints
    mask = dx == 0
    dx = jnp.where(mask, 1, dx)
    dxr = jnp.where(mask, 0.0, 1 / dx)[:, None]
    m = jnp.empty((M + 3, dim))

    y_l, y_r = C[:-1], C[1:]
    diff = y_r - y_l  # (M - 1, D)
    diff = diff / dxr  # (M - 1, D)
    m = m.at[2:-2].set(diff)

    # add two additional points on the left and on the right
    m = m.at[1].set(2.0 * m[2] - m[3])
    m = m.at[0].set(2.0 * m[1] - m[2])
    m = m.at[-2].set(2.0 * m[-3] - m[-4])
    m = m.at[-1].set(2.0 * m[-2] - m[-3])

    # df = derivative of f at x (spline derivatives)
    dm = jnp.abs(m[1:] - m[:-1])  # (M + 2, D)
    pm = jnp.abs(m[1:] + m[:-1])  # (M + 2, D)
    f1 = dm[2:] + 0.5 * pm[2:]  # (M, D)
    f2 = dm[:-2] + 0.5 * pm[:-2]  # (M, D)
    m2 = m[1:-2]  # (M, D)
    m3 = m[2:-1]  # (M, D)
    f12 = f1 + f2
    mask = f12 > 1e-9 * jnp.max(f12, initial=-jnp.inf)
    dydx = (f1 * m2 + f2 * m3) / jnp.where(mask, f12, 1.0)
    dydx = jnp.where(mask, dydx, 0.5 * (m[3:] + m[:-3]))

    # compute akima spline coefficients
    dydx_l, dydx_r = dydx[:-1] * dxr, dydx[1:] * dxr # (M - 1, D)
    ai = y_l  # (M - 1, D)
    bi = dydx_l  # (M - 1, D)
    ci = 3 * diff - 2 * dydx_l - dydx_r  # (M - 1, D)
    di = -2 * diff + dydx_l + dydx_r  # (M - 1, D)
    A = jnp.stack([di, ci, bi, ai], axis=-1)  # (M - 1, D, 4)
    # handle non-uniform spacing
    A = A / (dxr[:, None] ** jnp.arange(4)[::-1])
    A = jnp.moveaxis(A, -1, 1)  # (M - 1, 4, D)
    return A  # (M - 1, 4, D)


@partial(jit, static_argnames=("num_points", "nu"))
def poly_interpolation(A: jax.Array, num_points: int = 5, nu: int = 0) -> jax.Array:
    """Get the spline for a given path.

    Parameters
    ----------
    A : array_like (B, M, N)
    num_points : int, optional
        Number of points to evaluate the spline at. Default is 20.
    nu : int, optional
        Order of derivative to evaluate. Default is 0.

    Returns
    -------
    spline : array_like
        Spline values for the given path.
    """
    x = jnp.linspace(0, 1, num_points + 1)[:-1]
    B, dim = A.shape[0], A.shape[-1]

    def get_segment(a: jax.Array) -> jax.Array:
        a = a[:, None].repeat(num_points, axis=1)
        a = jnp.vectorize(lambda x: jnp.polyder(x, nu), signature="(n)->(m)")(a.T)
        y = jnp.vectorize(jnp.polyval, signature="(n),()->()")(a, x).T
        return y

    spline = vmap(vmap(get_segment))(A).reshape(B, -1, dim)  # (B, num_points, N, dim)
    return spline
