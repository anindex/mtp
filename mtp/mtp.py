from typing import Literal, Tuple

import jax
import jax.numpy as jnp
from flax.struct import dataclass
from functools import partial
from hydrax.alg_base import SamplingBasedController, Trajectory, SamplingParams
from hydrax.risk import RiskStrategy
from hydrax.task_base import Task
from mtp.splines.akima import poly_akima, poly_interpolation
from mtp.splines.bsplines import compute_b_spline_matrix


@dataclass
class MTPParams(SamplingParams):
    """Policy parameters for model-predictive path integral control.
    """
    cov: jax.Array = None


class MTP(SamplingBasedController):
    """Model Tensor Planning using interpax, MTP-Interpax."""

    def __init__(
        self,
        task: Task,
        num_samples: int,
        N: int = 50,
        degree: int = 2,
        num_elites: int = 5,
        sigma_start: float = 0.5,
        sigma_min: float = 0.1,
        sigma_max: float = 1.0,
        temperature: float = 0.1,
        num_randomizations: int = 1,
        beta: float = 0.1,
        alpha: float = 0.5,
        risk_strategy: RiskStrategy = None,
        plan_horizon: float = 1.0,
        spline_type: Literal["zero", "linear", "cubic"] = "zero",
        num_knots: int = 4,
        iterations: int = 1,
        seed: int = 0,
    ):
        """Initialize the controller.

        Args:
            task: The dynamics and cost for the system we want to control.
            num_samples: The number of control sequences to sample.
            temperature: The temperature parameter λ. Higher values take a more
                         even average over the samples.
            num_randomizations: The number of domain randomizations to use.
            risk_strategy: How to combining costs from different randomizations.
                           Defaults to average cost.
            seed: The random seed for domain randomization.
        """
        super().__init__(
            task,
            num_randomizations=num_randomizations,
            risk_strategy=risk_strategy,
            seed=seed,
            plan_horizon=plan_horizon,
            spline_type=spline_type,
            num_knots=num_knots,
            iterations=iterations,
        )
        assert degree >= 2, "degree must be at least 2."
        self.degree = degree
        self.N = N
        self.M = num_knots
        self.beta = beta
        self.num_samples = num_samples
        self.num_elites = num_elites
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_start = sigma_start
        self.temperature = temperature
        self.alpha = alpha

    def init_params(self, initial_knots: jax.Array = None, seed: int = 0) -> MTPParams:
        """Initialize the policy parameters."""
        _params = super().init_params(initial_knots, seed)
        cov = jnp.full_like(_params.mean, self.sigma_start)
        return MTPParams(rng=_params.rng, tk=_params.tk, mean=_params.mean, cov=cov)

    def sample_knots(
        self, params: MTPParams
    ) -> Tuple[jax.Array, MTPParams]:
        """Sample a control sequence."""
        rng = params.rng
        mtp_samples = int(self.num_samples * self.beta)
        # The previous knots is included as a sample
        controls = params.mean[None, ...]
        if mtp_samples > 1:
            # Sample the control points for the MTP
            rng, sample_rng = jax.random.split(rng)
            control_points = jax.random.uniform(
                sample_rng,
                (
                    self.M,
                    self.N,
                    self.task.model.nu,
                ),
                minval=self.task.u_min,
                maxval=self.task.u_max,
            )
            # sample points from the graph
            rng, sample_rng = jax.random.split(rng)
            layer_indices = jax.random.randint(rng, (mtp_samples, self.M), 0, self.N - 1)
            def get_path(path_id: jax.Array) -> jax.Array:
                return control_points[jnp.arange(self.M), path_id]
            mtp_controls = jax.vmap(get_path)(layer_indices)
            controls = jnp.concatenate([controls, mtp_controls], axis=0)

        mppi_samples = self.num_samples - mtp_samples - 1
        if mppi_samples > 0:
            # Sample mppi_samples control sequences
            rng, sample_rng = jax.random.split(rng)
            noise = jax.random.normal(
                sample_rng,
                (
                    mppi_samples,
                    self.num_knots,
                    self.task.model.nu,
                ),
            )
            mppi_controls = params.mean + params.cov * noise
            controls = jnp.concatenate([controls, mppi_controls], axis=0)

        return controls, params.replace(rng=rng)

    def update_params(
        self, params: MTPParams, rollouts: Trajectory
    ) -> MTPParams:
        """Update the mean with an exponentially weighted average."""
        costs = jnp.sum(rollouts.costs, axis=1)  # sum over time steps
        elite_indices = jnp.argsort(costs)[:self.num_elites]
        elite_controls = rollouts.knots[elite_indices]
        weights = jnp.nan_to_num(jax.nn.softmax(-costs[elite_indices] / self.temperature, axis=0))
        # The new proposal distribution is a Gaussian fit to the elites.
        weighted_controls = weights[:, None, None] * elite_controls
        mean = jnp.sum(weighted_controls, axis=0)
        cov = jnp.sqrt(jnp.sum(weights[:, None, None] * (elite_controls - mean) ** 2, axis=0))
        cov = jnp.clip(cov, self.sigma_min, self.sigma_max)
        mean = mean + self.alpha * (params.mean - mean)
        cov = cov + self.alpha * (params.cov - cov)

        # different from original MTP, this sends the mean not the best knots
        return params.replace(mean=mean, cov=cov)
