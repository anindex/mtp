import argparse
from evosax.algorithms import (
    DiffusionEvolution,
    Open_ES,
)

from mtp.mtp import MTP
from hydrax.algs import MPPI, PredictiveSampling, CEM, Evosax
from hydrax.simulation.deterministic import run_interactive
from hydrax.tasks.pendulum import Pendulum

"""
Run an interactive simulation of the pendulum swingup task.
"""

# Define the task (cost and dynamics)
task = Pendulum()
# Parse command-line arguments
parser = argparse.ArgumentParser(
    description="Run an interactive simulation of the walker task."
)
subparsers = parser.add_subparsers(
    dest="algorithm", help="Sampling algorithm (choose one)"
)
subparsers.add_parser("ps", help="Predictive Sampling")
subparsers.add_parser("mppi", help="Model Predictive Path Integral Control")
subparsers.add_parser("cem", help="Cross-Entropy Method")
subparsers.add_parser("mtp", help="MTP")
subparsers.add_parser("oes", help="OpenES")
subparsers.add_parser("de", help="Diffusion Evolution")
args = parser.parse_args()

seed = 111

# Set the controller based on command-line arguments
if args.algorithm == "mtp" or args.algorithm is None:
    print("Running predictive sampling")
    ctrl = PredictiveSampling(
        task,
        num_samples=32,
        noise_level=0.1,
        plan_horizon=0.5,
        spline_type="cubic",
        num_knots=11,
        seed=seed,
    )
elif args.algorithm == "mppi":
    print("Running MPPI")
    ctrl = MPPI(
        task,
        num_samples=32,
        noise_level=0.1,
        plan_horizon=0.5,
        spline_type="cubic",
        num_knots=11,
        temperature=0.1,
        seed=seed,
    )
elif args.algorithm == "cem":
    print("Running CEM")
    ctrl = CEM(
        task,
        num_samples=32,
        sigma_min=0.1,
        sigma_start=0.2,
        sigma_max=0.2,
        plan_horizon=0.5,
        spline_type="cubic",
        num_knots=11,
        num_elites=10,
        seed=seed,
    )
elif args.algorithm == "mtp":
    print("Running MTP")
    ctrl = MTP(
        task,
        num_samples=32,
        N=50,
        sigma_min=0.1,
        sigma_max=0.1,
        num_elites=5,
        beta=0.5,
        alpha=0.1,
        plan_horizon=0.5,
        spline_type="cubic",
        num_knots=11,
        num_randomizations=4,
        seed=seed,
    )
elif args.algorithm == "oes":
    print("Running OpenES")
    ctrl = Evosax(
        task,
        Open_ES,
        num_samples=32,
        plan_horizon=0.5,
        spline_type="cubic",
        num_knots=11,
        num_randomizations=4,
        seed=seed,
    )

elif args.algorithm == "de":
    print("Running Diffusion Evolution (DE)")
    ctrl = Evosax(
        task,
        DiffusionEvolution,
        num_samples=32,
        plan_horizon=0.5,
        spline_type="cubic",
        num_knots=11,
        num_randomizations=4,
        seed=seed,
    )
else:
    parser.error("Invalid algorithm")

# Define the model used for simulation
mj_model, mj_data = task.reset(seed=seed)

# Run the interactive simulation
run_interactive(
    ctrl,
    mj_model,
    mj_data,
    frequency=50,
    fixed_camera_id=0,
    show_traces=True,
    max_traces=1,
    show_ui=False,
    record_video=False,
    seed=seed,
)
