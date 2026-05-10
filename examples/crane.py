"""Interactive simulation of crane payload tracking.

Sim/baseline settings mirror upstream hydrax
(github.com/vincekurtz/hydrax @ a3976e0/examples/crane.py): freq=30 Hz,
plan_horizon=0.8, zero-order spline with num_knots=3, model perturbations
(damping x0.1, payload mass+inertia x1.5), CVaR risk on PS. Upstream only
ships PS; MPPI / CEM / MTP share the same per-step planner budget.
"""

import argparse
from copy import deepcopy

import mujoco

from mtp.mtp import MTP
from hydrax.algs import CEM, MPPI, PredictiveSampling
from hydrax.risk import ConditionalValueAtRisk
from hydrax.simulation.deterministic import run_interactive
from hydrax.tasks.crane import Crane

task = Crane()

parser = argparse.ArgumentParser(description="Crane payload tracking task.")
subparsers = parser.add_subparsers(dest="algorithm", help="Sampling algorithm")
subparsers.add_parser("ps", help="Predictive Sampling")
subparsers.add_parser("mppi", help="Model Predictive Path Integral Control")
subparsers.add_parser("cem", help="Cross-Entropy Method")
subparsers.add_parser("mtp", help="Model Tensor Planning")
args = parser.parse_args()
if args.algorithm is None:
    args.algorithm = "ps"

# Shared planner budget (upstream crane.py)
NUM_SAMPLES = 8
NUM_RAND = 32
PLAN_HORIZON = 0.8
NUM_KNOTS = 3
SPLINE = "zero"
RISK = ConditionalValueAtRisk(0.1)

if args.algorithm == "ps":
    print("Running Predictive Sampling")
    ctrl = PredictiveSampling(
        task, num_samples=NUM_SAMPLES, noise_level=0.05,
        num_randomizations=NUM_RAND, risk_strategy=RISK,
        plan_horizon=PLAN_HORIZON, spline_type=SPLINE, num_knots=NUM_KNOTS,
    )
elif args.algorithm == "mppi":
    print("Running MPPI")
    ctrl = MPPI(
        task, num_samples=NUM_SAMPLES, noise_level=0.05, temperature=0.1,
        num_randomizations=NUM_RAND, risk_strategy=RISK,
        plan_horizon=PLAN_HORIZON, spline_type=SPLINE, num_knots=NUM_KNOTS,
    )
elif args.algorithm == "cem":
    print("Running CEM")
    ctrl = CEM(
        task, num_samples=NUM_SAMPLES, num_elites=4,
        sigma_min=0.05, sigma_start=0.1,
        num_randomizations=NUM_RAND, risk_strategy=RISK,
        plan_horizon=PLAN_HORIZON, spline_type=SPLINE, num_knots=NUM_KNOTS,
    )
elif args.algorithm == "mtp":
    print("Running MTP")
    # PS-style core (greedy) since the noise level is tiny (0.05) and the
    # task is largely smooth; small tensor budget (beta=0.05) gives a
    # cheap non-local exploration channel.
    ctrl = MTP(
        task, num_samples=NUM_SAMPLES, M=3, N=8,
        sigma_min=0.05, sigma_max=0.1, sigma_start=0.05,
        num_elites=1, temperature=1.0, beta=0.05, alpha=0.0,
        mtp_interpolation="bspline",
        num_randomizations=NUM_RAND, risk_strategy=RISK,
        plan_horizon=PLAN_HORIZON, spline_type=SPLINE, num_knots=NUM_KNOTS,
    )
else:
    parser.error("Invalid algorithm")

# Sim model: match upstream "introduce some modeling error" perturbations
mj_model = deepcopy(task.mj_model)
mj_data = mujoco.MjData(mj_model)
mj_model.dof_damping *= 0.1
body_idx = mj_model.body("payload").id
mj_model.body_mass[body_idx] *= 1.5
mj_model.body_inertia[body_idx] *= 1.5

run_interactive(
    ctrl, mj_model, mj_data,
    frequency=30, show_traces=False,
)
