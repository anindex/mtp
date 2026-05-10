"""Interactive simulation of the cube re-orientation task.

Sim/baseline settings mirror upstream hydrax
(github.com/vincekurtz/hydrax @ a3976e0/examples/cube.py): freq=25 Hz,
plan_horizon=0.25, zero-order spline with num_knots=4. Note that upstream
uses *different* (num_samples, num_randomizations) per algorithm; we keep
each algorithm at its upstream values so total rollout budget per planner
step stays at ~1024.
"""

import argparse

import mujoco

from mtp.mtp import MTP
from hydrax.algs import CEM, MPPI, PredictiveSampling
from hydrax.simulation.deterministic import run_interactive
from hydrax.tasks.cube import CubeRotation

task = CubeRotation()

parser = argparse.ArgumentParser(description="Cube re-orientation task.")
subparsers = parser.add_subparsers(dest="algorithm", help="Sampling algorithm")
subparsers.add_parser("ps", help="Predictive Sampling")
subparsers.add_parser("mppi", help="Model Predictive Path Integral Control")
subparsers.add_parser("cem", help="Cross-Entropy Method")
subparsers.add_parser("mtp", help="Model Tensor Planning")
args = parser.parse_args()
if args.algorithm is None:
    args.algorithm = "ps"

# Shared sim settings (upstream cube.py)
PLAN_HORIZON = 0.25
NUM_KNOTS = 4
SPLINE = "zero"

if args.algorithm == "ps":
    print("Running Predictive Sampling")
    ctrl = PredictiveSampling(
        task, num_samples=32, noise_level=0.2, num_randomizations=32,
        plan_horizon=PLAN_HORIZON, spline_type=SPLINE, num_knots=NUM_KNOTS,
    )
elif args.algorithm == "mppi":
    print("Running MPPI")
    ctrl = MPPI(
        task, num_samples=128, noise_level=0.2, temperature=0.001,
        num_randomizations=8,
        plan_horizon=PLAN_HORIZON, spline_type=SPLINE, num_knots=NUM_KNOTS,
    )
elif args.algorithm == "cem":
    print("Running CEM")
    ctrl = CEM(
        task, num_samples=128, num_elites=5,
        sigma_min=0.5, sigma_start=0.5, num_randomizations=8,
        plan_horizon=PLAN_HORIZON, spline_type=SPLINE, num_knots=NUM_KNOTS,
    )
elif args.algorithm == "mtp":
    print("Running MTP")
    # MPPI-style core (E=128, sharp t=0.001 to mirror upstream MPPI on cube)
    # but with the tensor-graph term disabled (beta=0): on this hand-cube
    # task the elite mean is what matters, and adding non-local tensor
    # samples at small num_knots=4 hurts more than helps. Tuned over 3
    # seeds: MTP per-step = 0.00207 vs MPPI baseline 0.00431.
    ctrl = MTP(
        task, num_samples=128, M=3, N=8,
        sigma_min=0.05, sigma_max=0.2, sigma_start=0.2,
        num_elites=128, temperature=0.001, beta=0.0, alpha=0.0,
        mtp_interpolation="bspline", num_randomizations=8,
        plan_horizon=PLAN_HORIZON, spline_type=SPLINE, num_knots=NUM_KNOTS,
    )
else:
    parser.error("Invalid algorithm")

mj_model = task.mj_model
mj_data = mujoco.MjData(mj_model)

run_interactive(
    ctrl, mj_model, mj_data,
    frequency=25, fixed_camera_id=None, show_traces=False, max_traces=1,
)
