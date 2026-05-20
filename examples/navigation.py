"""Interactive simulation of the BugTrap navigation task with MTP.

Uses the upstream ``hydrax.tasks.bugtrap.BugTrap`` task whose MJCF
includes U-shaped inner walls creating a local minimum between the
start and goal positions.  The straight-line path from start to goal
goes through a wall, which traps local samplers (PS, MPPI, CEM) -
MTP's tensor-graph sampling explores both branches around the U and
consistently finds the goal.

Double-click the green target, then drag it with [Ctrl + right-click].
Press [Tab] inside the viewer to hide the left/right side panels for a
cleaner, full-window view.
"""

import argparse
import functools

import mujoco
from mujoco import mjx

from mtp.mtp import MTP
from hydrax.algs import CEM, MPPI, PredictiveSampling
from hydrax.simulation.deterministic import run_interactive
from hydrax.tasks.bugtrap import BugTrap

# Parse command-line arguments
parser = argparse.ArgumentParser(
    description="Run an interactive simulation of the BugTrap navigation task."
)
parser.add_argument(
    "--warp",
    action="store_true",
    help="Whether to use the (experimental) MjWarp backend. (default: False)",
    required=False,
)
subparsers = parser.add_subparsers(
    dest="algorithm", help="Sampling algorithm (choose one)"
)
subparsers.add_parser("ps", help="Predictive Sampling")
subparsers.add_parser("mppi", help="Model Predictive Path Integral Control")
subparsers.add_parser("cem", help="Cross-Entropy Method")
subparsers.add_parser("mtp", help="Model Tensor Planning")
args = parser.parse_args()

task = BugTrap(impl="warp" if args.warp else "jax")

# Fix broadphase overflow for MjWarp batch rollouts.
# _original_make_data = task.make_data

# @functools.wraps(_original_make_data)
# def _make_data_large_buffers(**kwargs):
#     # Only set naconmax (nconmax is deprecated since mujoco-mjx >= 3.5).
#     kwargs.setdefault("naconmax", 2048)
#     return mjx.make_data(task.mj_model, impl=task.model.impl, **kwargs)

# task.make_data = _make_data_large_buffers

# Set the controller based on command-line arguments.
# Settings match upstream hydrax bugtrap.py for a fair comparison.
common_kwargs = dict(
    plan_horizon=1.0,
    spline_type="zero",
    num_knots=11,
)

if args.algorithm == "ps" or args.algorithm is None:
    print("Running Predictive Sampling")
    ctrl = PredictiveSampling(
        task,
        num_samples=32,
        noise_level=1.0,
        **common_kwargs,
    )
elif args.algorithm == "mppi":
    print("Running MPPI")
    ctrl = MPPI(
        task,
        num_samples=32,
        noise_level=1.0,
        temperature=0.01,
        **common_kwargs,
    )
elif args.algorithm == "cem":
    print("Running CEM")
    ctrl = CEM(
        task,
        num_samples=32,
        num_elites=1,
        sigma_start=1.0,
        sigma_min=0.5,
        explore_fraction=0.5,
        **common_kwargs,
    )
elif args.algorithm == "mtp":
    print("Running MTP")
    ctrl = MTP(
        task,
        num_samples=32,
        m_pts=5,
        n_per_layer=50,
        num_elites=1,
        sigma_start=0.7,
        sigma_min=0.5,
        sigma_max=1.0,
        beta=1.0,
        alpha=0.1,
        mtp_interpolation="akima",
        **common_kwargs,
    )
else:
    parser.error("Invalid algorithm")

# Define the model used for simulation
mj_model = task.mj_model
mj_data = mujoco.MjData(mj_model)
mj_data.qpos[:2] = [-0.15, 0.0]
mj_data.mocap_pos[0] = [0.25, 0.0, 0.01]

run_interactive(
    ctrl,
    mj_model,
    mj_data,
    frequency=50,
    show_traces=True,
    max_traces=3,
    trace_width=2.0,
)

