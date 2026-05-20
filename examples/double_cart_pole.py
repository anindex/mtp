"""Interactive simulation of the double cart-pole task.

Sim/baseline settings mirror upstream hydrax
(github.com/vincekurtz/hydrax @ 33ec819/examples/double_cart_pole.py):
freq=50 Hz, plan_horizon=1.0 s, **cubic** spline with num_knots=4.
Upstream only ships PS; MPPI / CEM / MTP share the same shared budget.
"""

import argparse

import mujoco

from mtp.mtp import MTP
from hydrax.algs import CEM, MPPI, PredictiveSampling
from hydrax.simulation.deterministic import run_interactive
from hydrax.tasks.double_cart_pole import DoubleCartPole

parser = argparse.ArgumentParser(description="Double cart-pole task.")
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

task = DoubleCartPole(impl="warp" if args.warp else "jax")

# Shared planner budget (upstream double_cart_pole.py)
NUM_SAMPLES = 1024
PLAN_HORIZON = 1.0
NUM_KNOTS = 4
SPLINE = "cubic"

if args.algorithm == "ps":
    print("Running Predictive Sampling")
    ctrl = PredictiveSampling(
        task, num_samples=NUM_SAMPLES, noise_level=0.3,
        plan_horizon=PLAN_HORIZON, spline_type=SPLINE, num_knots=NUM_KNOTS,
    )
elif args.algorithm == "mppi":
    print("Running MPPI")
    ctrl = MPPI(
        task, num_samples=NUM_SAMPLES, noise_level=0.3, temperature=0.1,
        plan_horizon=PLAN_HORIZON, spline_type=SPLINE, num_knots=NUM_KNOTS,
    )
elif args.algorithm == "cem":
    print("Running CEM")
    ctrl = CEM(
        task, num_samples=NUM_SAMPLES, num_elites=50,
        sigma_min=0.15, sigma_start=0.5,
        plan_horizon=PLAN_HORIZON, spline_type=SPLINE, num_knots=NUM_KNOTS,
    )
elif args.algorithm == "mtp":
    print("Running MTP")
    # PS-style core (greedy, num_elites=1, sharp t=1.0) at the upstream
    # cubic/4-knot setting. Small n_per_layer keeps the cubic interpolation
    # cheap; beta=0.05 supplies a small tensor-graph budget for non-local
    # jumps.
    ctrl = MTP(
        task, num_samples=NUM_SAMPLES, m_pts=3, n_per_layer=8,
        sigma_min=0.15, sigma_max=0.5, sigma_start=0.5,
        num_elites=1, temperature=1.0, beta=0.05, alpha=0.0,
        mtp_interpolation="bspline",
        plan_horizon=PLAN_HORIZON, spline_type=SPLINE, num_knots=NUM_KNOTS,
    )
else:
    parser.error("Invalid algorithm")

mj_model = task.mj_model
mj_data = mujoco.MjData(mj_model)

run_interactive(
    ctrl, mj_model, mj_data,
    frequency=50, fixed_camera_id=0, show_traces=False, max_traces=1,
)
