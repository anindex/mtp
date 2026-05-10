"""Interactive simulation of pendulum swing-up.

Sim/baseline settings mirror upstream hydrax
(github.com/vincekurtz/hydrax @ a3976e0/examples/pendulum.py):
freq=50 Hz, plan_horizon=1.0 s, zero-order spline with num_knots=11.
"""

import argparse

import mujoco
import numpy as np

from mtp.mtp import MTP
from hydrax.algs import CEM, MPPI, PredictiveSampling
from hydrax.simulation.deterministic import run_interactive
from hydrax.tasks.pendulum import Pendulum

task = Pendulum()

parser = argparse.ArgumentParser(description="Pendulum swing-up task.")
subparsers = parser.add_subparsers(dest="algorithm", help="Sampling algorithm")
subparsers.add_parser("ps", help="Predictive Sampling")
subparsers.add_parser("mppi", help="Model Predictive Path Integral Control")
subparsers.add_parser("cem", help="Cross-Entropy Method")
subparsers.add_parser("mtp", help="Model Tensor Planning")
args = parser.parse_args()
if args.algorithm is None:
    args.algorithm = "ps"

# Shared planner budget (upstream pendulum.py)
NUM_SAMPLES = 32
PLAN_HORIZON = 1.0
NUM_KNOTS = 11
SPLINE = "zero"

if args.algorithm == "ps":
    print("Running Predictive Sampling")
    ctrl = PredictiveSampling(
        task, num_samples=NUM_SAMPLES, noise_level=0.1,
        plan_horizon=PLAN_HORIZON, spline_type=SPLINE, num_knots=NUM_KNOTS,
    )
elif args.algorithm == "mppi":
    print("Running MPPI")
    ctrl = MPPI(
        task, num_samples=NUM_SAMPLES, noise_level=0.2, temperature=0.1,
        plan_horizon=PLAN_HORIZON, spline_type=SPLINE, num_knots=NUM_KNOTS,
    )
elif args.algorithm == "cem":
    print("Running CEM")
    ctrl = CEM(
        task, num_samples=NUM_SAMPLES, num_elites=10,
        sigma_min=0.1, sigma_start=0.5,
        plan_horizon=PLAN_HORIZON, spline_type=SPLINE, num_knots=NUM_KNOTS,
    )
elif args.algorithm == "mtp":
    print("Running MTP")
    # Weighted-CEM core (E=10, sharp t=0.03) at upstream's freq=50 / h=1.0
    # / num_knots=11. Tight sigma band keeps the elites informative around
    # upright; small tensor budget (beta=0.10) handles non-local recoveries.
    ctrl = MTP(
        task, num_samples=NUM_SAMPLES, M=2, N=8,
        sigma_min=0.10, sigma_max=0.5, sigma_start=0.5,
        num_elites=10, temperature=0.03, beta=0.10, alpha=0.0,
        # Akima beats bspline by ~1% on stationary swing-up tracking.
        mtp_interpolation="akima",
        plan_horizon=PLAN_HORIZON, spline_type=SPLINE, num_knots=NUM_KNOTS,
    )
else:
    parser.error("Invalid algorithm")

mj_model = task.mj_model
mj_data = mujoco.MjData(mj_model)
mj_data.qpos[:] = np.array([0.0])
mj_data.qvel[:] = np.array([0.0])

run_interactive(
    ctrl, mj_model, mj_data,
    frequency=50, fixed_camera_id=0, show_traces=False, max_traces=1,
)
