import numpy as np
import jax
import jax.numpy as jnp
from jax import vmap

import matplotlib
import matplotlib.pyplot as plt
from mtp.splines.akima import poly_akima, poly_interpolation
from mtp.splines.bsplines import compute_b_spline_matrix

# Enable LaTeX-style fonts
matplotlib.rcParams.update({
    "text.usetex": True,           # Use LaTeX for all text rendering
    "font.family": "sans-serif",        # Use a serif font
    "font.sans-serif": ["Helvetica"],   # Set specific sans-serif font (e.g., Helvetica)
    "axes.labelsize": 9,          # Set axis label font size
    "font.size": 9,               # Set default font size
    "legend.fontsize": 9,         # Set legend font size
    "xtick.labelsize": 6,         # Set x-tick font size
    "ytick.labelsize": 6          # Set y-tick font size
})


def interpolate_path(path: jax.Array, num_points: int) -> jax.Array:
    start, goal = path[:-1], path[1:]
    linspace = lambda x, y, n: jnp.linspace(x, y, n)
    return jax.vmap(linspace, in_axes=(0, 0, None))(start, goal, num_points).reshape(-1, path.shape[-1])

M = 3
N = 9
num_points = 10
layer_indices = jnp.repeat(jnp.arange(N, dtype=jnp.int32)[None, :], M, axis=0)
grid_ids = jnp.meshgrid(*layer_indices)
pairwise_idx = jnp.stack([X.ravel() for X in grid_ids], axis=-1)
x = np.arange(0, M)
p = np.linspace(-1, 1, 3)
X, Y = np.meshgrid(p, p)
points = np.stack([X.flatten(), Y.flatten()], axis=-1)
G = jnp.stack([points, points, points], axis=0)
def get_path(path_id):
    return G[jnp.arange(M), path_id]
C = vmap(get_path)(pairwise_idx)

# Akima interpolation
A = vmap(poly_akima, in_axes=(None, 0))(x, C) # (B, M - 1, 4, 2)
Aspline = poly_interpolation(A, num_points=num_points)

# B-spline interpolation d = 2
p = 2
knots = jnp.arange(1, M + p + 2)
T = num_points * (M - 1)
B = compute_b_spline_matrix(knots, p, T)
Bspline2 = jnp.einsum("bmd,hm->bhd", C, B)

# B-spline interpolation d = 3
p = 3
knots = jnp.arange(1, M + p + 2)
T = num_points * (M - 1)
B = compute_b_spline_matrix(knots, p, T)
Bspline3 = jnp.einsum("bmd,hm->bhd", C, B)

# Linear interpolation
linear_path = vmap(interpolate_path, in_axes=(0, None))(C, num_points)

# 3d plot
fig = plt.figure(figsize=(18, 4))
ax = fig.add_subplot(1, 4, 1, projection="3d")
xs = jnp.linspace(1, M, num=(num_points * (M - 1)))

for i in range(M):
    x_i = np.ones_like(G[i, :, 0]) * (i + 1)
    ax.plot(x_i, G[i, :, 0], G[i, :, 1], "o", c='black', ms=5)

for i in range(linear_path.shape[0]):
    ax.plot(xs, linear_path[i, :, 0], linear_path[i, :, 1], c='black', linewidth=0.5, alpha=0.1, label="Linear interpolation")

ax.set_aspect('equal')
ax.set_axis_off()
ax.grid(False)
ax.set_title(r"Linear")

ax = fig.add_subplot(1, 4, 2, projection="3d")

for i in range(M):
    x_i = np.ones_like(G[i, :, 0]) * (i + 1)
    ax.plot(x_i, G[i, :, 0], G[i, :, 1], "o", c='black', ms=5)

for i in range(Bspline2.shape[0]):
    ax.plot(xs, Bspline2[i, :, 0], Bspline2[i, :, 1], c='#d55e00', linewidth=1, alpha=0.3, label="B-spline")

ax.set_aspect('equal')
ax.set_axis_off()
ax.grid(False)
ax.set_title(r"B-spline p=2")

ax = fig.add_subplot(1, 4, 3, projection="3d")
for i in range(M):
    x_i = np.ones_like(G[i, :, 0]) * (i + 1)
    ax.plot(x_i, G[i, :, 0], G[i, :, 1], "o", c='black', ms=5)

for i in range(Bspline3.shape[0]):
    ax.plot(xs, Bspline3[i, :, 0], Bspline3[i, :, 1], c='#d55e00', linewidth=1, alpha=0.3, label="B-spline")

ax.set_aspect('equal')
ax.set_axis_off()
ax.grid(False)
ax.set_title(r"B-spline p=3")

ax = fig.add_subplot(1, 4, 4, projection="3d")
for i in range(M):
    x_i = np.ones_like(G[i, :, 0]) * (i + 1)
    ax.plot(x_i, G[i, :, 0], G[i, :, 1], "o", c='black', ms=5)


for i in range(Aspline.shape[0]):
    ax.plot(xs, Aspline[i, :, 0], Aspline[i, :, 1], c='#029e73', linewidth=1, alpha=0.3, label="Akima-spline")

ax.set_aspect('equal')
ax.set_axis_off()
ax.grid(False)
ax.set_title(r"Akima-spline")

fig.tight_layout(pad=0.1)
plt.show()
