"""Visualize MTP spline tensor structures.

Plots the multipartite graph and resulting spline trajectories for
linear, B-spline (p=2, p=3), and Akima interpolation methods.
"""

import numpy as np
import jax
import jax.numpy as jnp
from jax import vmap

import matplotlib
import matplotlib.pyplot as plt
from mtp.splines.akima import poly_akima, poly_interpolation
from mtp.splines.bsplines import compute_b_spline_matrix

matplotlib.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica"],
    "axes.labelsize": 9,
    "font.size": 9,
    "legend.fontsize": 9,
    "xtick.labelsize": 6,
    "ytick.labelsize": 6,
})


def interpolate_path(path, num_points):
    start, goal = path[:-1], path[1:]
    linspace = lambda x, y, n: jnp.linspace(x, y, n)
    return jax.vmap(linspace, in_axes=(0, 0, None))(
        start, goal, num_points
    ).reshape(-1, path.shape[-1])


M = 3
N = 9
num_points = 10

layer_indices = jnp.repeat(jnp.arange(N, dtype=jnp.int32)[None, :], M, axis=0)
grid_ids = jnp.meshgrid(*layer_indices)
pairwise_idx = jnp.stack([X.ravel() for X in grid_ids], axis=-1)

p = np.linspace(-1, 1, 3)
X, Y = np.meshgrid(p, p)
points = np.stack([X.flatten(), Y.flatten()], axis=-1)
G = jnp.stack([points, points, points], axis=0)
x = np.arange(0, M)


def get_path(path_id):
    return G[jnp.arange(M), path_id]


C = vmap(get_path)(pairwise_idx)

# Akima
A = vmap(poly_akima, in_axes=(None, 0))(x, C)
Aspline = poly_interpolation(A, num_points=num_points)

# B-spline p=2
knots2 = jnp.arange(1, M + 2 + 2)
B2 = compute_b_spline_matrix(knots2, 2, num_points * (M - 1))
Bspline2 = jnp.einsum("bmd,hm->bhd", C, B2)

# B-spline p=3
knots3 = jnp.arange(1, M + 3 + 2)
B3 = compute_b_spline_matrix(knots3, 3, num_points * (M - 1))
Bspline3 = jnp.einsum("bmd,hm->bhd", C, B3)

# Linear
linear_path = vmap(interpolate_path, in_axes=(0, None))(C, num_points)

# Plot
xs = jnp.linspace(1, M, num=(num_points * (M - 1)))
fig = plt.figure(figsize=(18, 4))

titles = [r"Linear", r"B-spline $p=2$", r"B-spline $p=3$", r"Akima"]
data = [linear_path, Bspline2, Bspline3, Aspline]
colors = ["black", "#d55e00", "#d55e00", "#029e73"]

for idx, (title, d, color) in enumerate(zip(titles, data, colors)):
    ax = fig.add_subplot(1, 4, idx + 1, projection="3d")
    for i in range(M):
        x_i = np.ones_like(G[i, :, 0]) * (i + 1)
        ax.plot(x_i, G[i, :, 0], G[i, :, 1], "o", c="black", ms=5)
    lw = 0.5 if idx == 0 else 1
    alpha = 0.1 if idx == 0 else 0.3
    for i in range(d.shape[0]):
        ax.plot(xs, d[i, :, 0], d[i, :, 1], c=color, linewidth=lw, alpha=alpha)
    ax.set_aspect("equal")
    ax.set_axis_off()
    ax.grid(False)
    ax.set_title(title)

fig.tight_layout(pad=0.1)
plt.show()
