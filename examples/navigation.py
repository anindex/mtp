"""Interactive simulation of the particle tracking task with MTP.

Uses the locally-vendored ``NavigationParticle`` task whose MJCF includes
the U-shaped inner walls. The straight-line path from start to goal goes
through a wall, which creates a local minimum that local samplers (PS,
MPPI, CEM) struggle to escape - MTP's tensor-graph sampling explores
both branches around the U and consistently finds the goal.

Double-click the green target, then drag it with [Ctrl + right-click].
Press [Tab] inside the viewer to hide the left/right side panels for a
cleaner, full-window view.
"""

import argparse

from mtp.mtp import MTP
from mtp.tasks.navigation import NavigationParticle
from hydrax.algs import CEM, MPPI, PredictiveSampling
from hydrax.simulation.deterministic import run_interactive

# Define the task
task = NavigationParticle()

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
subparsers.add_parser("mtp", help="Model Tensor Planning")
args = parser.parse_args()

# Set the controller based on command-line arguments.
#
# Note on rendering smoothness: ``hydrax.simulation.deterministic.run_interactive``
# is synchronous - the viewer waits for each replan, so realtime rate ≈
# ``min(frequency, 1 / plan_time)``. We keep the per-step compute budget
# modest (128 samples x 4 randomizations = 512 rollouts/replan @ 25 Hz)
# and cap rollout traces to keep the Python ``mjv_connector`` overhead low.
common_kwargs = dict(
    plan_horizon=1.0,
    spline_type="zero",
    num_knots=20,
    num_randomizations=4,
)

if args.algorithm == "ps" or args.algorithm is None:
    print("Running Predictive Sampling")
    ctrl = PredictiveSampling(
        task,
        num_samples=128,
        noise_level=0.1,
        **common_kwargs,
    )
elif args.algorithm == "mppi":
    print("Running MPPI")
    ctrl = MPPI(
        task,
        num_samples=128,
        noise_level=1.0,
        temperature=0.01,
        **common_kwargs,
    )
elif args.algorithm == "cem":
    print("Running CEM")
    ctrl = CEM(
        task,
        num_samples=128,
        num_elites=20,
        sigma_min=1.0,
        sigma_start=1.0,
        **common_kwargs,
    )
elif args.algorithm == "mtp":
    print("Running MTP")
    ctrl = MTP(
        task,
        num_samples=128,
        M=3,
        N=20,
        beta=1.0,
        mtp_interpolation="akima",
        **common_kwargs,
    )
else:
    parser.error("Invalid algorithm")

# Define the model used for simulation
mj_model = task.mj_model
mj_data = task.make_initial_data(seed=0)

# Run the interactive simulation. Lower the planner frequency to 25 Hz to
# leave a comfortable 40 ms compute budget per replan, and cap the number
# of rendered trajectory traces (each trace is a polyline of ``ctrl_steps``
# line segments redrawn every replan from Python).
run_interactive(
    ctrl,
    mj_model,
    mj_data,
    frequency=25,
    show_traces=True,
    max_traces=3,
    trace_width=2.0,
)
