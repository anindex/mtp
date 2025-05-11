import argparse
from evosax.algorithms import (
    Open_ES,
    DiffusionEvolution,
)
from mtp.mtp import MTP
from hydrax.algs import CEM, MPPI, Evosax, PredictiveSampling
# from mtp.algs.mtp_spline import MTP
from hydrax.algs import PredictiveSampling
from hydrax.simulation.deterministic import run_interactive
from hydrax.tasks.double_cart_pole import DoubleCartPole

"""
Run an interactive simulation of a double pendulum on a cart. Only the cart
is actuated, and the goal is to swing up the pendulum and balance it upright.
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
subparsers.add_parser("cmaes", help="CMA-ES")
subparsers.add_parser("oes", help="OpenAIES")
subparsers.add_parser("de", help="Diffusion Evolution")
subparsers.add_parser("mtp", help="MTP")
args = parser.parse_args()

# Define the task (cost and dynamics)
task = DoubleCartPole()
seed = 111

if args.algorithm == "ps" or args.algorithm is None:
    print("Running predictive sampling")
    ctrl = PredictiveSampling(
        task, 
        num_samples=512, 
        noise_level=0.3, 
        plan_horizon=0.5,
        spline_type="cubic",
        num_knots=4,
        num_randomizations=1, 
        seed=seed
    )
elif args.algorithm == "mppi":
    print("Running MPPI")
    ctrl = MPPI(
        task,
        num_samples=512, 
        noise_level=0.3, 
        plan_horizon=0.5,
        temperature=0.1,
        spline_type="cubic",
        num_knots=4,
        num_randomizations=1, 
        seed=seed,
    )
elif args.algorithm == "cem":
    print("Running CEM")
    ctrl = CEM(
        task,
        num_samples=512,
        num_elites=50,
        sigma_min=0.2,
        sigma_start=0.5,
        plan_horizon=0.5,
        spline_type="cubic",
        num_knots=4,
        num_randomizations=1, 
        seed=seed,
    )
elif args.algorithm == "mtp":
    ctrl = MTP(
            task,
            num_samples=512,
            N=50,
            sigma_max=0.2,
            sigma_min=0.2,
            num_elites=5,
            beta=0.01,
            alpha=0.005,
            plan_horizon=0.5,
            spline_type="cubic",
            num_knots=4,
            num_randomizations=1, 
            seed=seed,
        )
elif args.algorithm == "oes":
    print("Running OpenAIES")
    ctrl = Evosax(
        task,
        Open_ES,
        plan_horizon=0.5,
        spline_type="cubic",
        num_knots=4,
        num_samples=512,
        num_randomizations=1,
        seed=seed,
    )
elif args.algorithm == "de":
    print("Running Diffusion Evolution")
    ctrl = Evosax(
        task,
        DiffusionEvolution,
        plan_horizon=0.5,
        spline_type="cubic",
        num_knots=4,
        num_samples=512,
        num_randomizations=1,
        seed=seed,
    )

# Define the model used for simulation
mj_model, mj_data = task.reset(seed=seed)
# Run the interactive simulation
run_interactive(
    ctrl,
    mj_model,
    mj_data,
    frequency=50,
    fixed_camera_id=0,
    show_traces=False,
    show_ui=False,
    record_video=False,
    max_traces=5,
    seed=seed,
)
