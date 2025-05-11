import argparse

from evosax.algorithms import (
    Sep_CMA_ES,
    SAMR_GA,
    NoiseReuseES,
    DiffusionEvolution,
    Open_ES,
    SimpleGA,
    GradientlessDescent,
)
from mtp.mtp import MTP
from hydrax.algs import CEM, MPPI, Evosax, PredictiveSampling
from hydrax.simulation.deterministic import run_interactive
from hydrax.tasks.particle import Particle

"""
Run an interactive simulation of the particle tracking task.

Double click on the green target, then drag it around with [ctrl + right-click].
"""

# Define the task (cost and dynamics)
task = Particle()

# Parse command-line arguments
parser = argparse.ArgumentParser(
    description="Run an interactive simulation of the particle tracking task."
)
subparsers = parser.add_subparsers(
    dest="algorithm", help="Sampling algorithm (choose one)"
)
subparsers.add_parser("ps", help="Predictive Sampling")
subparsers.add_parser("mppi", help="Model Predictive Path Integral Control")
subparsers.add_parser("cem", help="Cross-Entropy Method")
subparsers.add_parser("cmaes", help="CMA-ES")
subparsers.add_parser("nes", help="NoiseReuseES")
subparsers.add_parser("oes", help="OpenES")
subparsers.add_parser(
    "samr", help="Genetic Algorithm with Self-Adaptation Mutation Rate (SAMR)"
)
subparsers.add_parser("de", help="Differential Evolution")
subparsers.add_parser("gld", help="Gradient-Less Descent")
subparsers.add_parser("rs", help="Uniform Random Search")
subparsers.add_parser("sga", help="Simple Genetic Algorithm")
subparsers.add_parser("mtp", help="MTP")
subparsers.add_parser("mstp", help="MSTP")
args = parser.parse_args()

seed = 111

# Set the controller based on command-line arguments
if args.algorithm == "ps" or args.algorithm is None:
    print("Running predictive sampling")
    ctrl = PredictiveSampling(
        task,
        num_samples=256,
        noise_level=0.1,
        plan_horizon=0.5,
        spline_type="cubic",
        num_knots=5,
        num_randomizations=8,
        seed=seed,
    )

elif args.algorithm == "mppi":
    print("Running MPPI")
    ctrl = MPPI(
        task,
        num_samples=256,
        noise_level=0.1,
        plan_horizon=0.5,
        temperature=0.1,
        spline_type="cubic",
        num_knots=5,
        num_randomizations=8,
        seed=seed,
    )
elif args.algorithm == "cem":
    print("Running CEM")
    ctrl = CEM(
        task,
        num_samples=256,
        num_elites=50,
        sigma_min=0.2,
        sigma_start=0.5,
        plan_horizon=0.5,
        spline_type="cubic",
        num_knots=5,
        num_randomizations=8,
        seed=seed,
    )

elif args.algorithm == "cmaes":
    print("Running CMA-ES")
    ctrl = Evosax(
        task,
        Sep_CMA_ES,
        num_samples=256,
        plan_horizon=0.5,
        spline_type="cubic",
        num_knots=5,
        num_randomizations=8,
        seed=seed,
    )

elif args.algorithm == "samr":
    print("Running genetic algorithm with Self-Adaptation Mutation Rate (SAMR)")
    ctrl = Evosax(
        task,
        SAMR_GA,
        num_samples=256,
        plan_horizon=0.5,
        spline_type="cubic",
        num_knots=5,
        num_randomizations=8,
        seed=seed,
    )

elif args.algorithm == "nes":
    print("Running NoiseReuseES")
    ctrl = Evosax(
        task,
        NoiseReuseES,
        num_samples=256,
        plan_horizon=0.5,
        spline_type="cubic",
        num_knots=5,
        num_randomizations=8,
        seed=seed,
    )

elif args.algorithm == "sga":
    print("Running Simple Genetic Algorithm (SGA)")
    ctrl = Evosax(
        task,
        SimpleGA,
        num_samples=256,
        plan_horizon=0.5,
        spline_type="cubic",
        num_knots=5,
        num_randomizations=8,
        seed=seed,
    )

elif args.algorithm == "oes":
    print("Running OpenES")
    ctrl = Evosax(
        task,
        Open_ES,
        num_samples=256,
        plan_horizon=0.5,
        spline_type="cubic",
        num_knots=5,
        num_randomizations=8,
        seed=seed,
    )

elif args.algorithm == "de":
    print("Running Diffusion Evolution (DE)")
    ctrl = Evosax(
        task,
        DiffusionEvolution,
        num_samples=256,
        plan_horizon=0.5,
        spline_type="cubic",
        num_knots=5,
        num_randomizations=8,
        seed=seed,
    )

elif args.algorithm == "gld":
    print("Running Gradient-Less Descent (GLD)")
    ctrl = Evosax(
        task,
        GradientlessDescent,
        num_samples=256,
        plan_horizon=0.5,
        spline_type="cubic",
        num_knots=5,
        num_randomizations=8,
        seed=seed,
    )

elif args.algorithm == "mtp":
    print("Running MTP")
    ctrl = MTP(
        task,
        num_samples=256,
        N=50,
        beta=1.0,
        plan_horizon=0.5,
        spline_type="cubic",
        num_knots=5,
        num_randomizations=8,
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
    show_traces=True,
    max_traces=5,
    # fixed_camera_id=0,
    record_video=False,
    show_ui=False,
    seed=seed,
)
