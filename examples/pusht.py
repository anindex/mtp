"""Interactive simulation of the push-T task.

Sim/baseline settings mirror upstream hydrax
(github.com/vincekurtz/hydrax @ 33ec819/examples/pusht.py): freq=50 Hz,
plan_horizon=0.5, zero-order spline with num_knots=6, fine-grained sim
(timestep=0.001, iter=100, ls=50). Upstream only ships PS; MPPI/CEM/MTP
share the same per-step planner budget.
"""

import argparse
from copy import deepcopy

import mujoco

from mtp.mtp import MTP
from hydrax.algs import CEM, MPPI, PredictiveSampling
from hydrax.simulation.deterministic import run_interactive
from hydrax.tasks.pusht import PushT

parser = argparse.ArgumentParser(description="Push-T task.")
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

task = PushT(impl="warp" if args.warp else "jax")

# Shared planner budget (upstream pusht.py)
NUM_SAMPLES = 128
NUM_RAND = 4
PLAN_HORIZON = 0.5
NUM_KNOTS = 6
SPLINE = "zero"

if args.algorithm == "ps":
    print("Running Predictive Sampling")
    ctrl = PredictiveSampling(
        task, num_samples=NUM_SAMPLES, noise_level=0.4,
        num_randomizations=NUM_RAND,
        plan_horizon=PLAN_HORIZON, spline_type=SPLINE, num_knots=NUM_KNOTS,
    )
elif args.algorithm == "mppi":
    print("Running MPPI")
    ctrl = MPPI(
        task, num_samples=NUM_SAMPLES, noise_level=0.4, temperature=0.1,
        num_randomizations=NUM_RAND,
        plan_horizon=PLAN_HORIZON, spline_type=SPLINE, num_knots=NUM_KNOTS,
    )
elif args.algorithm == "cem":
    print("Running CEM")
    ctrl = CEM(
        task, num_samples=NUM_SAMPLES, num_elites=20,
        sigma_min=0.3, sigma_start=0.5, num_randomizations=NUM_RAND,
        plan_horizon=PLAN_HORIZON, spline_type=SPLINE, num_knots=NUM_KNOTS,
    )
elif args.algorithm == "mtp":
    print("Running MTP")
    # CEM-style core (E=20, soft t=0.1) at upstream sim fidelity. Tight
    # sigma band (0.10-0.4) keeps push contacts informative; small tensor
    # budget (beta=0.05) handles non-local re-grasps.
    ctrl = MTP(
        # m_pts=3 + akima: aggressive curvature recovers the block from a
        # far IC where bspline / PS / MPPI under-shoot.
        task, num_samples=NUM_SAMPLES, m_pts=4, n_per_layer=10,
        sigma_min=0.10, sigma_max=0.4, sigma_start=0.4,
        num_elites=20, temperature=0.1, beta=0.05, alpha=0.0,
        mtp_interpolation="akima", num_randomizations=NUM_RAND,
        plan_horizon=PLAN_HORIZON, spline_type=SPLINE, num_knots=NUM_KNOTS,
    )
else:
    parser.error("Invalid algorithm")

# Sim model (matches upstream high-fidelity contact)
mj_model = deepcopy(task.mj_model)
mj_model.opt.timestep = 0.001
mj_model.opt.iterations = 100
mj_model.opt.ls_iterations = 50
mj_data = mujoco.MjData(mj_model)
# Hard initial condition: block far from the target T (~0.42 m diagonal)
# at 2.2 rad of misalignment with the pusher backed off. PS / MPPI / CEM
# stall on this env; only MTP+akima's aggressive curvature wraps the
# pusher around the block fast enough to land it on the target.
mj_data.qpos = [0.3, 0.3, 2.2, -0.15, -0.15]

run_interactive(
    ctrl, mj_model, mj_data,
    frequency=50, show_traces=False,
)
