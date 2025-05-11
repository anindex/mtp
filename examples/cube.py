import argparse

from evosax.algorithms import (
    Open_ES,
    DiffusionEvolution,
)
import mujoco
from mtp.mtp import MTP
from hydrax.algs import CEM, MPPI, Evosax, PredictiveSampling
from hydrax.simulation.deterministic import run_interactive
from hydrax.tasks.cube import CubeRotation
from hydrax.simulation.deterministic import run_interactive

"""
Run an interactive simulation of the cube rotation task.

Double click on the floating target cube, then change the goal orientation with
[ctrl + left click].
"""

# Define the task (cost and dynamics)
task = CubeRotation()

# Parse command-line arguments
parser = argparse.ArgumentParser(
    description="Run an interactive simulation of the cube rotation task."
)
subparsers = parser.add_subparsers(
    dest="algorithm", help="Sampling algorithm (choose one)"
)
subparsers.add_parser("ps", help="Predictive Sampling")
subparsers.add_parser("mppi", help="Model Predictive Path Integral Control")
subparsers.add_parser("cem", help="Cross-Entropy Method")
subparsers.add_parser("oes", help="OpenAIES")
subparsers.add_parser("de", help="Diffusion Evolution")
subparsers.add_parser("mtp", help="MTP")
args = parser.parse_args()

seed = 111

# Set the controller based on command-line arguments
if args.algorithm == "ps" or args.algorithm is None:
    print("Running predictive sampling")
    ctrl = PredictiveSampling(
        task, 
        num_samples=128, 
        noise_level=0.15,
        plan_horizon=0.25,
        spline_type="cubic",
        num_knots=4,
        num_randomizations=8,
        seed=seed
    )
elif args.algorithm == "mppi":
    print("Running MPPI")
    ctrl = MPPI(
        task,
        num_samples=128,
        noise_level=0.15,
        temperature=0.1,
        plan_horizon=0.25,
        spline_type="cubic",
        num_knots=4,
        num_randomizations=8,
        seed=seed,
    )
elif args.algorithm == "cem":
    print("Running CEM")
    ctrl = CEM(
        task,
        num_samples=128,
        num_elites=5,
        sigma_min=0.15,
        sigma_start=0.3,
        plan_horizon=0.25,
        spline_type="cubic",
        num_knots=4,
        num_randomizations=8,
        seed=seed,
    )
elif args.algorithm == "mtp":
    print("Running MTP")
    ctrl = MTP(
        task,
        num_samples=128,
        N=50,
        sigma_min=0.15,
        sigma_max=0.15,
        num_elites=5,
        beta=0.5,
        alpha=0.1,
        plan_horizon=0.25,
        spline_type="cubic",
        num_knots=4,
        num_randomizations=8,
        seed=seed,
    )
elif args.algorithm == "oes":
    print("Running OpenES")
    ctrl = Evosax(
        task, 
        Open_ES, 
        num_samples=128,
        plan_horizon=0.25,
        spline_type="cubic",
        num_knots=4,
        num_randomizations=8, 
        seed=seed
    )

elif args.algorithm == "de":
    print("Running Diffusion Evolution (DE)")
    ctrl = Evosax(
        task,
        DiffusionEvolution,
        num_samples=128,
        plan_horizon=0.25,
        spline_type="cubic",
        num_knots=4,
        num_randomizations=8,
        seed=seed,
    )
else:
    parser.error("Invalid algorithm")

# Define the model used for simulation
mj_model = task.mj_model
mj_data = mujoco.MjData(mj_model)

# Run the interactive simulation
run_interactive(
    ctrl,
    mj_model,
    mj_data,
    frequency=50,
    show_traces=False,
    fixed_camera_id=0,
    show_ui=False,
    record_video=False,
    seed=seed,
)
