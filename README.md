# Model Tensor Planning (MTP)

[![arXiv](https://img.shields.io/badge/arXiv-2502.08378-brown)](https://arxiv.org/abs/2505.01059)
[![](https://img.shields.io/badge/Website-%F0%9F%9A%80-yellow)](https://sites.google.com/view/tensor-sampling/)
[![License: MIT](https://img.shields.io/badge/License-MIT-purple.svg)]()

This repository contains the official implementation of **Model Tensor Planning (MTP)**, a sampling-based model predictive control (MPC) framework that performs high-entropy control generation using structured tensor sampling.  See [project website](https://sites.google.com/view/tensor-sampling/).

MTP is implemented entirely in **JAX** and supports real-time control in high-dimensional systems with GPU acceleration via JIT and MuJoCo XLA.

<p float="middle">
  <img src="demos/pusht_mtp-akima.gif" width="32%" />
  <img src="demos/cube_mtp-akima.gif" width="32%" /> 
  <img src="demos/crane_mtp-bspline.gif" width="32%" />
</p>

## Key Features

- **Tensor Sampling**: Generates globally diverse trajectory candidates via sampling over randomized multipartite graphs.
- **Spline Grid Interpolation**: Smoothes sampled controls using B-spline and Akima splines for dynamically feasible execution.
- **β-Mixing Strategy**: Blends global (exploratory) and local (exploitative) samples at each planning iteration.

**NOTE:** `MTP-Bspline` and `MTP-Akima` depends on [hydrax fork](https://github.com/anindex/hydrax) that separates original [hydrax](https://github.com/vincekurtz/hydrax) since [spline support PR](https://github.com/vincekurtz/hydrax/pull/40). To match the newest commit, I implemented `MTP-Cubic` (untuned), a version that samples both global and local splines using `interpax`, matching the new API design of the original `hydrax`. To play around with `MTP-Cubic`, please checkout the branch `experimental` of both `mtp` and [hydrax fork](https://github.com/anindex/hydrax).

## Installation

```bash
# Create environment (Python ≥ 3.12 recommended)
conda create -n mtp python=3.12
conda activate mtp

# Install hydrax dependency
cd hydrax
pip install -e .

# Install MTP
cd ..
pip install -e .
```

## Running Examples

All examples are configured to run with MTP by default. To switch between planners (e.g., `cem`, `mppi`, `ps`, `oes`, `de`), replace the last argument.

```bash
python examples/navigation.py mtp
python examples/double_cart_pole.py mtp
python examples/pusht.py mtp
python examples/crane.py mtp
python examples/walker.py mtp
python examples/cube.py mtp
python examples/pendulum.py mtp
python examples/g1_standup.py mtp
python examples/g1_mocap.py mtp
```

## Tuning Tips

To achieve the best performance across tasks, here are recommended tuning guidelines:

| Symbol | Description                | Typical Range         |
|--------|----------------------------|------------------------|
| `M`    | Number of control waypoints (graph depth) | 2–3 (depending on horizon T)                  |
| `N`    | Number of control candidates per waypoint (graph width) | 30–100               |
| `β`    | Mixing rate (exploration vs. exploitation) | 0.01–0.6 (lower = more stable less exploration) |
| `E`    | Number of elites           | 5–100 (depends on task complexity) |
| `σ_min` | Minimum noise std for CEM sampling | 0.05–0.2              |
| `σ_max` | Maximum noise std for CEM sampling | 0.3–0.5              |
| `interpolation`    | Interpolation Types |'linear', 'bspline', 'akima'             |
| `α`    | CEM smoothing weight (optional) | 0.0–0.5              |


### Interpolation Strategy

- **B-Spline (default for stable tasks):** Good for underactuated systems or where smoothness is critical.
- **Akima Spline (use for aggressive control):** Works well in contact-rich environments (e.g., dexterous manipulation).

Run `scripts/plot_splines.py` to see spline tensors.


## Acknowledgments

This codebase builds upon [HydraX](https://github.com/vincekurtz/hydrax), [MuJoCo XLA](https://github.com/deepmind/mujoco). Special thanks to [Vince Kurtz](https://github.com/vincekurtz) and other contributors.

## Citation

If you found this repository useful, please consider citing these references:

```azure
@misc{le2025mtp,
      title={Model Tensor Planning}, 
      author={An T. Le and Khai Nguyen and Minh Nhat Vu and João Carvalho and Jan Peters},
      year={2025},
      eprint={2505.01059},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2505.01059}, 
}

@misc{kurtz2024hydrax,
  title={Hydrax: Sampling-based model predictive control on GPU with JAX and MuJoCo MJX},
  author={Kurtz, Vince},
  year={2024},
  note={https://github.com/vincekurtz/hydrax}
}
```
