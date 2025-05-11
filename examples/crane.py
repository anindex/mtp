import argparse
from evosax.algorithms import (
    DiffusionEvolution,
    Open_ES,
)
from mtp.mtp import MTP
from hydrax.algs import CEM, MPPI, Evosax, PredictiveSampling
from hydrax.simulation.deterministic import run_interactive
from hydrax.tasks.crane import Crane

"""
Run an interactive simulation of crane payload tracking
"""
# Parse command-line arguments
parser = argparse.ArgumentParser(
    description="Run an interactive simulation of mocap tracking with the G1."
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
task = Crane()
seed = 111

if args.algorithm == "ps" or args.algorithm is None:
    print("Running predictive sampling")
    ctrl = PredictiveSampling(
        task, 
        num_samples=16, 
        noise_level=0.05,
        plan_horizon=0.8,
        spline_type="cubic",
        num_knots=3, 
        num_randomizations=32, 
        seed=seed
    )
elif args.algorithm == "mppi":
    print("Running MPPI")
    ctrl = MPPI(
        task,
        num_samples=16,
        noise_level=0.05,
        plan_horizon=0.8,
        spline_type="cubic",
        num_knots=3,
        temperature=0.1,
        num_randomizations=32,
        seed=seed,
    )
elif args.algorithm == "cem":
    print("Running CEM")
    ctrl = CEM(
        task,
        num_samples=16,
        num_elites=8,
        sigma_min=0.05,
        sigma_start=0.3,
        plan_horizon=0.8,
        spline_type="cubic",
        num_knots=3,
        num_randomizations=32,
        seed=seed,
    )
elif args.algorithm == "mtp":
    ctrl = MTP(
            task,
            num_samples=16,
            N=30,
            sigma_start=0.05,
            sigma_min=0.05,
            num_elites=8,
            beta=0.5,
            alpha=0.,
            plan_horizon=0.8,
            spline_type="cubic",
            num_knots=3,
            num_randomizations=32,
            seed=seed,
        )
elif args.algorithm == "oes":
    print("Running OpenAI-ES")
    ctrl = Evosax(
        task,
        Open_ES,
        plan_horizon=0.8,
        spline_type="cubic",
        num_knots=3,
        num_samples=16,
        num_randomizations=32,
        seed=seed,
    )
elif args.algorithm == "de":
    print("Running Diffusion Evolution")
    ctrl = Evosax(
        task,
        DiffusionEvolution,
        plan_horizon=0.8,
        spline_type="cubic",
        num_knots=3,
        num_samples=16,
        num_randomizations=32,
        seed=seed,
    )

# Define the model used for simulation
mj_model, mj_data = task.reset(seed=seed)

# Run the interactive simulation
run_interactive(
    ctrl,
    mj_model,
    mj_data,
    frequency=30,
    show_traces=False,
    fixed_camera_id=0,
    show_ui=False,
    record_video=False,
    seed=seed,
)
