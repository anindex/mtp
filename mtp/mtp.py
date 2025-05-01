from typing import Tuple

import jax
import jax.numpy as jnp
from flax.struct import dataclass
from functools import partial
from hydrax.alg_base import SamplingBasedController, Trajectory
from hydrax.risk import RiskStrategy
from hydrax.task_base import Task
from mtp.splines.akima import poly_akima, poly_interpolation
from mtp.splines.bsplines import compute_b_spline_matrix


@dataclass
class MTPParams:
    """Policy parameters for model-predictive path integral control.
    """
    rng: jax.Array
    mean: jax.Array = None
    cov: jax.Array = None
    spline: jax.Array = None


@partial(jax.jit, static_argnums=1)
def interpolate_path(path: jax.Array, num_points: int) -> jax.Array:
    start, goal = path[:-1], path[1:]
    linspace = lambda x, y, n: jnp.linspace(x, y, n + 1)[:-1]
    return jax.vmap(linspace, in_axes=(0, 0, None))(start, goal, num_points).reshape(-1, path.shape[-1])


class MTP(SamplingBasedController):
    """Model Tensor Planning."""

    def __init__(
        self,
        task: Task,
        num_samples: int,
        M: int = 3,
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
        interpolation: str = 'akima',
        risk_strategy: RiskStrategy = None,
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
        super().__init__(task, num_randomizations, risk_strategy, seed)
        assert degree >= 2, "degree must be at least 2."
        self.degree = degree
        self.N = N
        self.M = M
        self.beta = beta
        self.num_samples = num_samples
        self.num_elites = num_elites
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_start = sigma_start
        self.aknots = jnp.linspace(1, self.M, self.M)
        self.bknots = jnp.arange(self.M + self.degree + 1)
        self.bmat = compute_b_spline_matrix(self.bknots, self.degree, self.task.planning_horizon)
        self.temperature = temperature
        self.interpolation = interpolation
        self.alpha = alpha

    def init_params(self, seed: int = 0) -> MTPParams:
        """Initialize the policy parameters."""
        rng = jax.random.key(seed)
        spline = jnp.zeros((self.task.planning_horizon, self.task.model.nu))
        mean = jnp.zeros((self.task.planning_horizon, self.task.model.nu))
        cov = jnp.full_like(mean, self.sigma_start)
        return MTPParams(rng=rng, spline=spline, mean=mean, cov=cov)

    def sample_controls(
        self, params: MTPParams
    ) -> Tuple[jax.Array, MTPParams]:
        """Sample a control sequence."""
        rng = params.rng
        mtp_samples = int(self.num_samples * self.beta)
        # The previous spline is included as a sample
        controls = params.spline[None, ...]
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
            control_points = jax.vmap(get_path)(layer_indices)
            # interpolate the control points
            if self.interpolation == 'akima':
                A = jax.vmap(poly_akima, in_axes=(None, 0))(self.aknots, control_points) # (batch, M - 1, 4, nu)
                num_interp = self.task.planning_horizon // (self.M - 1)
                remain = self.task.planning_horizon - num_interp * (self.M - 1)
                mtp_controls = poly_interpolation(A, num_interp)
                remain_controls = jnp.repeat(control_points[:, -1, None], remain, axis=1)
                mtp_controls = jnp.concatenate([mtp_controls, remain_controls], axis=1)
            elif self.interpolation == 'bspline':
                mtp_controls = jnp.einsum("bmd,hm->bhd", control_points, self.bmat)
            elif self.interpolation == 'linear':
                num_interp = self.task.planning_horizon // (self.M - 1)
                remain = self.task.planning_horizon - num_interp * (self.M - 1)
                mtp_controls = jax.vmap(interpolate_path, in_axes=(0, None))(control_points, num_interp)
                remain_controls = jnp.repeat(control_points[:, -1, None], remain, axis=1)
                mtp_controls = jnp.concatenate([mtp_controls, remain_controls], axis=1)
            else:
                raise ValueError(f"Invalid sampling strategy: {self.interpolation}")
            controls = jnp.concatenate([controls, mtp_controls], axis=0)

        mppi_samples = self.num_samples - mtp_samples - 1
        if mppi_samples > 0:
            # Sample mppi_samples control sequences
            rng, sample_rng = jax.random.split(rng)
            noise = jax.random.normal(
                sample_rng,
                (
                    mppi_samples,
                    self.task.planning_horizon,
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
        elite_controls = rollouts.controls[elite_indices]
        weights = jnp.nan_to_num(jax.nn.softmax(-costs[elite_indices] / self.temperature, axis=0))
        # The new proposal distribution is a Gaussian fit to the elites.
        weighted_controls = weights[:, None, None] * elite_controls
        mean = jnp.sum(weighted_controls, axis=0)
        cov = jnp.sqrt(jnp.sum(weights[:, None, None] * (elite_controls - mean) ** 2, axis=0))
        cov = jnp.clip(cov, self.sigma_min, self.sigma_max)
        mean = mean + self.alpha * (params.mean - mean)
        cov = cov + self.alpha * (params.cov - cov)
        spline = rollouts.controls[elite_indices[0]]
        params = params.replace(mean=mean, cov=cov)

        return params.replace(spline=spline)

    def get_action(self, params: MTPParams, t: float) -> jax.Array:
        """Get the control action for the current time step, zero order hold."""
        idx_float = t / self.task.dt  # zero order hold
        idx = jnp.floor(idx_float).astype(jnp.int32)
        action = params.spline[idx]
        return action
