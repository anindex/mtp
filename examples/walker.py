"""Interactive simulation of the Walker locomotion task.

Sim settings (timestep 0.005, 50 Newton iterations) and the PS / MPPI
configurations mirror the upstream hydrax demo
(github.com/vincekurtz/hydrax @ 33ec819/examples/walker.py). CEM and MTP
share the same per-step planner budget for a fair head-to-head.
"""

import argparse
from copy import deepcopy

import mujoco

from mtp.mtp import MTP
from hydrax.algs import CEM, MPPI, PredictiveSampling
from hydrax.simulation.deterministic import run_interactive
from hydrax.tasks.walker import Walker

parser = argparse.ArgumentParser(description="Walker locomotion task.")
parser.add_argument(
    "--warp",
    action="store_true",
    help="Whether to use the (experimental) MjWarp backend. (default: False)",
    required=False,
)
subparsers = parser.add_subparsers(dest="algorithm", help="Sampling algorithm")
subparsers.add_parser("ps", help="Predictive Sampling")
subparsers.add_parser("mppi", help="Model Predictive Path Integral Control")
subparsers.add_parser("cem", help="Cross-Entropy Method")
subparsers.add_parser("mtp", help="Model Tensor Planning")
args = parser.parse_args()
if args.algorithm is None:
    args.algorithm = "ps"

task = Walker(impl="warp" if args.warp else "jax")

# Shared planner budget (matches upstream PS/MPPI in walker.py)
NUM_SAMPLES = 128
PLAN_HORIZON = 0.6
NUM_KNOTS = 5
SPLINE = "zero"

if args.algorithm == "ps":
    print("Running Predictive Sampling")
    ctrl = PredictiveSampling(
        task, num_samples=NUM_SAMPLES, noise_level=0.5,
        plan_horizon=PLAN_HORIZON, spline_type=SPLINE, num_knots=NUM_KNOTS,
    )
elif args.algorithm == "mppi":
    print("Running MPPI")
    ctrl = MPPI(
        task, num_samples=NUM_SAMPLES, noise_level=0.5, temperature=0.1,
        plan_horizon=PLAN_HORIZON, spline_type=SPLINE, num_knots=NUM_KNOTS,
    )
elif args.algorithm == "cem":
    print("Running CEM")
    ctrl = CEM(
        task, num_samples=NUM_SAMPLES, num_elites=20,
        sigma_min=0.5, sigma_start=0.5,
        plan_horizon=PLAN_HORIZON, spline_type=SPLINE, num_knots=NUM_KNOTS,
    )
elif args.algorithm == "mtp":
    print("Running MTP")
    # Tuned at upstream sim fidelity (timestep=0.005, iter=50, freq=50,
    # plan_horizon=0.6, num_knots=5). MPPI-style core (large num_elites,
    # soft temperature) plus a small tensor-graph budget (beta=0.05).
    ctrl = MTP(
        task, num_samples=NUM_SAMPLES, m_pts=3, n_per_layer=8,
        sigma_min=0.3, sigma_max=0.5, sigma_start=0.5,
        num_elites=20, temperature=0.1, beta=0.05, alpha=0.0,
        mtp_interpolation="bspline",
        plan_horizon=PLAN_HORIZON, spline_type=SPLINE, num_knots=NUM_KNOTS,
    )
else:
    parser.error("Invalid algorithm")

# Sim model (matches upstream)
mj_model = deepcopy(task.mj_model)
mj_model.opt.timestep = 0.005
mj_model.opt.iterations = 50
mj_data = mujoco.MjData(mj_model)

run_interactive(
    ctrl, mj_model, mj_data,
    frequency=50, fixed_camera_id=0, show_traces=False, max_traces=1,
)
