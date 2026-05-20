"""Interactive simulation of the Unitree G1 standup task.

Sim/baseline settings mirror upstream hydrax
(github.com/vincekurtz/hydrax @ 33ec819/examples/humanoid_standup.py):
freq=50 Hz, plan_horizon=0.6 s, zero-order spline with num_knots=4,
stiffer contact (timestep=0.01, override solimp), and the IC drops the
robot from the 'stand' keyframe by setting the base orientation to a
sideways quaternion so the robot must stand back up. Upstream only ships
the MppiCma controller; PS / MPPI / CEM / MTP share that planner budget.
"""

import argparse
from copy import deepcopy

import mujoco

from mtp.mtp import MTP
from hydrax.algs import CEM, MPPI, PredictiveSampling
from hydrax.simulation.deterministic import run_interactive
from hydrax.tasks.humanoid_standup import HumanoidStandup

parser = argparse.ArgumentParser(description="Humanoid (G1) standup task.")
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
    args.algorithm = "mppi"

task = HumanoidStandup(impl="warp" if args.warp else "jax")

# Shared planner budget (upstream humanoid_standup.py uses MppiCma)
NUM_SAMPLES = 128
NUM_RAND = 4
PLAN_HORIZON = 0.6
NUM_KNOTS = 4
SPLINE = "zero"
SEED = 1111

if args.algorithm == "ps":
    print("Running Predictive Sampling")
    ctrl = PredictiveSampling(
        task, num_samples=NUM_SAMPLES, noise_level=0.3,
        num_randomizations=NUM_RAND,
        plan_horizon=PLAN_HORIZON, spline_type=SPLINE, num_knots=NUM_KNOTS,
        seed=SEED,
    )
elif args.algorithm == "mppi":
    print("Running MPPI")
    # Mirror upstream MppiCma's effective config: noise=0.3, t=0.1.
    ctrl = MPPI(
        task, num_samples=NUM_SAMPLES, noise_level=0.3, temperature=0.1,
        num_randomizations=NUM_RAND,
        plan_horizon=PLAN_HORIZON, spline_type=SPLINE, num_knots=NUM_KNOTS,
        seed=SEED,
    )
elif args.algorithm == "cem":
    print("Running CEM")
    ctrl = CEM(
        task, num_samples=NUM_SAMPLES, num_elites=20,
        sigma_min=0.3, sigma_start=0.3, num_randomizations=NUM_RAND,
        plan_horizon=PLAN_HORIZON, spline_type=SPLINE, num_knots=NUM_KNOTS,
        seed=SEED,
    )
elif args.algorithm == "mtp":
    print("Running MTP")
    # Tuning notes (28-DoF humanoid stand-up):
    #   * n_per_layer<=8 keeps the JAX/XLA kernel within ptxas's register
    #     budget; larger values segfault the compiler.
    #   * Weighted-CEM core (E=20, t=0.1) mirrors what MPPI does well on
    #     humanoid balance.
    #   * sigma band 0.2-0.3 matches the MPPI noise=0.3 dispersion.
    #   * beta=0.05 supplies a small tensor-graph budget for non-local
    #     posture corrections.
    ctrl = MTP(
        # m_pts=3 + akima: aggressive recovery on contact-rich standup
        # (~4% lower running cost than bspline / m_pts=2 across seeds).
        task, num_samples=NUM_SAMPLES, m_pts=3, n_per_layer=8,
        sigma_min=0.2, sigma_max=0.3, sigma_start=0.3,
        num_elites=20, temperature=0.1, beta=0.05, alpha=0.0,
        mtp_interpolation="akima", num_randomizations=NUM_RAND,
        plan_horizon=PLAN_HORIZON, spline_type=SPLINE, num_knots=NUM_KNOTS,
        seed=SEED,
    )
else:
    parser.error("Invalid algorithm")

# Sim model: stiffer contact (matches upstream)
mj_model = deepcopy(task.mj_model)
mj_model.opt.timestep = 0.01
mj_model.opt.o_solimp = [0.9, 0.95, 0.001, 0.5, 2]
mj_model.opt.enableflags = mujoco.mjtEnableBit.mjENBL_OVERRIDE

mj_data = mujoco.MjData(mj_model)
mj_data.qpos[:] = mj_model.keyframe("stand").qpos
mj_data.qpos[3:7] = [0.7, 0.0, -0.7, 0.0]

run_interactive(
    ctrl, mj_model, mj_data,
    frequency=50, show_traces=False,
)
