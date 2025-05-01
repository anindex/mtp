import argparse

from evosax.algorithms import (
    Open_ES,
    DiffusionEvolution,
)
from mtp.mtp import MTP
from hydrax.algs import CEM, MPPI, Evosax, PredictiveSampling
from hydrax.simulation.deterministic import run_interactive
from hydrax.tasks.humanoid_mocap import HumanoidMocap

"""
Run an interactive simulation of the humanoid motion capture tracking task.
"""

# Parse command-line arguments
parser = argparse.ArgumentParser(
    description="Run an interactive simulation of humanoid (G1) standup."
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

# Define the task (cost and dynamics)
task = HumanoidMocap(
    planning_horizon=4,
)

seed = 1111
frequency = 100

# Set the controller based on command-line arguments
if args.algorithm == "ps" or args.algorithm is None:
    print("Running predictive sampling")
    ctrl = PredictiveSampling(
        task, num_samples=128, noise_level=0.1, num_randomizations=4, seed=seed
    )
elif args.algorithm == "mppi":
    print("Running MPPI")
    ctrl = MPPI(
        task,
        num_samples=128,
        noise_level=0.1,
        temperature=0.1,
        num_randomizations=4,
        seed=seed,
    )
elif args.algorithm == "cem":
    print("Running CEM")
    ctrl = CEM(
        task,
        num_samples=128,
        num_elites=100,
        sigma_min=0.1,
        sigma_start=0.1,
        num_randomizations=4,
        seed=seed,
    )
elif args.algorithm == "mtp":
    ctrl = MTP(
            task,
            num_samples=128,
            M=2,
            N=100,
            sigma_min=0.1,
            sigma_max=0.2,
            num_elites=100,
            beta=0.02,
            alpha=0.,
            interpolation='akima',
            num_randomizations=4,
            seed=seed,
        )
elif args.algorithm == "oes":
    print("Running OpenAIES")
    ctrl = Evosax(
        task,
        Open_ES,
        num_samples=128,
        num_randomizations=4,
        seed=seed,
    )
elif args.algorithm == "de":
    print("Running Diffusion Evolution")
    ctrl = Evosax(
        task,
        DiffusionEvolution,
        num_samples=128,
        num_randomizations=4,
        seed=seed,
    )

# Define the model used for simulation
mj_model, mj_data = task.reset(seed=seed)

print("Running deterministic simulation")
run_interactive(
    ctrl,
    mj_model,
    mj_data,
    frequency=frequency,
    # max_step=1500,
    show_traces=False,
    fixed_camera_id=1,
    show_ui=False,
    record_video=False,
    seed=seed,
)
