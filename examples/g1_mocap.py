"""Interactive simulation of humanoid (G1) motion-capture tracking.

Sim/baseline settings mirror upstream hydrax
(github.com/vincekurtz/hydrax @ 33ec819/examples/humanoid_mocap.py):
freq=50 Hz, plan_horizon=0.8 s, zero-order spline with num_knots=4,
high-fidelity contact (timestep=0.01, iter=10, ls=50, override solimp),
IC = first reference frame, initial knots seeded from the reference
joint trajectory. Upstream only ships CEM; PS / MPPI / MTP share the
same per-step planner budget.
"""

import argparse
from copy import deepcopy

import mujoco

from mtp.mtp import MTP
from hydrax.algs import CEM, MPPI, PredictiveSampling
from hydrax.risk import AverageCost
from hydrax.simulation.deterministic import run_interactive
from hydrax.tasks.humanoid_mocap import HumanoidMocap, HumanoidMocapOptions

parser = argparse.ArgumentParser(description="Humanoid (G1) mocap tracking task.")
parser.add_argument(
    "--warp",
    action="store_true",
    help="Whether to use the (experimental) MjWarp backend. (default: False)",
    required=False,
)
parser.add_argument(
    "--reference_filename",
    type=str,
    default="Lafan1/mocap/UnitreeG1/walk1_subject1.npz",
    help="Reference mocap file from huggingface loco-mujoco-datasets.",
)
parser.add_argument(
    "--show_reference", action="store_true",
    help="Show the reference trajectory as a 'ghost' in the simulation.",
)
subparsers = parser.add_subparsers(dest="algorithm", help="Sampling algorithm")
subparsers.add_parser("ps", help="Predictive Sampling")
subparsers.add_parser("mppi", help="Model Predictive Path Integral Control")
subparsers.add_parser("cem", help="Cross-Entropy Method")
subparsers.add_parser("mtp", help="Model Tensor Planning")
args = parser.parse_args()
if args.algorithm is None:
    args.algorithm = "cem"

task = HumanoidMocap(
    reference_filename=args.reference_filename,
    options=HumanoidMocapOptions(),
    impl="warp" if args.warp else "jax",
)

# Shared planner budget (upstream humanoid_mocap.py uses CEM)
PLAN_HORIZON = 0.8
NUM_KNOTS = 4
SPLINE = "zero"
SEED = 1111

if args.algorithm == "ps":
    print("Running Predictive Sampling")
    ctrl = PredictiveSampling(
        task, num_samples=128, noise_level=0.1, num_randomizations=1,
        risk_strategy=AverageCost(),
        plan_horizon=PLAN_HORIZON, spline_type=SPLINE, num_knots=NUM_KNOTS,
        seed=SEED,
    )
elif args.algorithm == "mppi":
    print("Running MPPI")
    ctrl = MPPI(
        task, num_samples=128, noise_level=0.1, temperature=0.1,
        num_randomizations=1, risk_strategy=AverageCost(),
        plan_horizon=PLAN_HORIZON, spline_type=SPLINE, num_knots=NUM_KNOTS,
        seed=SEED,
    )
elif args.algorithm == "cem":
    print("Running CEM")
    # Upstream config (unchanged from hydrax).
    ctrl = CEM(
        task, num_samples=1024, num_elites=10,
        sigma_start=0.2, sigma_min=0.05, explore_fraction=0.5,
        num_randomizations=1, risk_strategy=AverageCost(),
        plan_horizon=PLAN_HORIZON, spline_type=SPLINE, num_knots=NUM_KNOTS,
        seed=SEED,
    )
elif args.algorithm == "mtp":
    print("Running MTP")
    # Tuning notes (28-DoF humanoid mocap tracking):
    #   * n_per_layer<=8 keeps the JAX/XLA kernel within ptxas register
    #     budget.
    #   * Tight CEM-style core (E=10, sharp t=0.03) mirrors upstream CEM
    #     which dominates this task; sigma decays from 0.2 -> 0.05 like
    #     upstream's CEM (sigma_start, sigma_min).
    #   * beta=0.05 supplies a small tensor-graph budget for non-local
    #     posture correction without exploding the rollout count.
    ctrl = MTP(
        # bspline + m_pts=3: smoother per-knot blending tracks the mocap
        # reference more conservatively than akima on this env.
        task, num_samples=1024, m_pts=3, n_per_layer=8,
        sigma_min=0.05, sigma_max=0.2, sigma_start=0.2,
        num_elites=10, temperature=0.03, beta=0.05, alpha=0.0,
        mtp_interpolation="bspline", num_randomizations=1,
        risk_strategy=AverageCost(),
        plan_horizon=PLAN_HORIZON, spline_type=SPLINE, num_knots=NUM_KNOTS,
        seed=SEED,
    )
else:
    parser.error("Invalid algorithm")

# Sim model (matches upstream high-fidelity contact)
mj_model = deepcopy(task.mj_model)
mj_model.opt.timestep = 0.01
mj_model.opt.iterations = 10
mj_model.opt.ls_iterations = 50
mj_model.opt.o_solimp = [0.9, 0.95, 0.001, 0.5, 2]
mj_model.opt.enableflags = mujoco.mjtEnableBit.mjENBL_OVERRIDE

mj_data = mujoco.MjData(mj_model)
mj_data.qpos[:] = task.reference_qpos[0]
initial_knots = task.reference_qpos[: ctrl.num_knots, 7:]

reference = task.reference_qpos if args.show_reference else None

run_interactive(
    ctrl, mj_model, mj_data,
    frequency=50, show_traces=False,
    reference=reference, reference_fps=task.reference_fps,
    initial_knots=initial_knots,
)
