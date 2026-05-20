"""Model Tensor Planning (MTP) - Core Algorithm.

Implements the MTP sampling-based MPC framework that generates globally
diverse trajectory candidates via structured tensor sampling over
randomized multipartite graphs, with spline-based control interpolation.

This module is designed to work with upstream hydrax (>=0.0.2) and its
spline-native API.

Reference
---------
Le et al., "Model Tensor Planning", TMLR 2025.
https://arxiv.org/abs/2505.01059

The parameters ``m_pts`` and ``n_per_layer`` correspond to *M* and *N*
in the paper (Section 3, Algorithm 1).
"""

from typing import Literal, Optional, Tuple

import jax
import jax.numpy as jnp
from flax.struct import dataclass

from hydrax.alg_base import SamplingBasedController, SamplingParams, Trajectory
from hydrax.risk import RiskStrategy
from hydrax.task_base import Task

from mtp.splines.akima import poly_akima, poly_interpolation
from mtp.splines.bsplines import compute_b_spline_matrix

MTPInterpolationType = Literal["akima", "bspline", "linear"]


@dataclass
class MTPParams(SamplingParams):
    """Policy parameters for Model Tensor Planning.

    Extends the base SamplingParams with MTP-specific fields for the
    CEM-style local distribution and the best spline from the prior
    optimization step.

    Attributes:
        tk: Knot times of the control spline (inherited).
        mean: Mean of the local CEM distribution (inherited).
        rng: PRNG key (inherited).
        cov: Diagonal standard deviation of the local CEM distribution.
        best_knots: Best spline knots from the previous optimization step.
    """

    cov: jax.Array
    best_knots: jax.Array


class MTP(SamplingBasedController):
    """Model Tensor Planning controller.

    MTP generates control trajectories by:
    1. Constructing a randomized M-partite graph with N candidates per layer
    2. Sampling paths through the graph via random index selection
    3. Interpolating each path with Akima/B-spline/linear interpolation
    4. Mixing these global (exploratory) samples with local CEM samples
    5. Selecting elites and updating the local distribution

    The β parameter controls the mixing ratio: β=1.0 means all tensor
    samples, β=0.0 means all local CEM samples.

    Notation (paper ↔ code):
        M  ↔  ``m_pts``            - graph depth (waypoint layers)
        N  ↔  ``n_per_layer``      - graph width (candidates per layer)
        β  ↔  ``beta``             - tensor / CEM mix ratio
        K  ↔  ``num_elites``       - elite count
        α  ↔  ``alpha``            - CEM variance smoothing
        λ  ↔  ``temperature``      - softmax temperature

    Args:
        task: The dynamics and cost for the system to control.
        num_samples: Total number of control trajectories to evaluate.
        m_pts: Number of control waypoints / graph depth (*M* in the paper).
        n_per_layer: Number of candidates per waypoint / graph width
            (*N* in the paper).
        degree: B-spline degree (must be >= 2). Only used when
                mtp_interpolation='bspline'.
        num_elites: Number of elite samples for CEM update.
        sigma_start: Initial standard deviation for CEM distribution.
        sigma_min: Minimum standard deviation (prevents collapse).
        sigma_max: Maximum standard deviation (prevents explosion).
        temperature: Softmax temperature for elite weighting.
        num_randomizations: Number of domain randomizations.
        beta: Mixing rate (fraction of tensor samples vs CEM samples).
        alpha: CEM smoothing weight (0 = full update, 1 = keep old).
        mtp_interpolation: Spline type for tensor path interpolation
                           ('akima', 'bspline', or 'linear').
        risk_strategy: Strategy for combining costs across randomizations.
        seed: Random seed.
        plan_horizon: Planning horizon in seconds.
        spline_type: Hydrax spline type for control interpolation
                     ('zero', 'linear', 'cubic').
        num_knots: Number of hydrax spline knots.
        iterations: Number of optimization iterations per planning step.
    """

    def __init__(
        self,
        task: Task,
        num_samples: int,
        m_pts: int = 3,
        n_per_layer: int = 50,
        degree: int = 2,
        num_elites: int = 5,
        sigma_start: float = 0.5,
        sigma_min: float = 0.1,
        sigma_max: float = 1.0,
        temperature: float = 0.1,
        num_randomizations: int = 1,
        beta: float = 0.1,
        alpha: float = 0.5,
        mtp_interpolation: MTPInterpolationType = "akima",
        risk_strategy: Optional[RiskStrategy] = None,
        seed: int = 0,
        plan_horizon: float = 1.0,
        spline_type: Literal["zero", "linear", "cubic"] = "zero",
        num_knots: int = 4,
        iterations: int = 1,
    ) -> None:
        assert degree >= 2, "B-spline degree must be at least 2."
        if mtp_interpolation == "bspline":
            assert m_pts >= degree + 1, (
                f"B-spline interpolation requires m_pts >= degree + 1 "
                f"(got m_pts={m_pts}, degree={degree}). With m_pts < degree+1 "
                f"the valid B-spline parameter domain collapses to a point "
                f"and the tensor branch produces flat (constant) control "
                f"samples. Use 'akima' or 'linear' for small m_pts, or "
                f"increase m_pts."
            )
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
        self.degree = degree
        self.n_per_layer = n_per_layer
        self.m_pts = m_pts
        self.beta = beta
        self.num_samples = num_samples
        self.num_elites = num_elites
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_start = sigma_start
        self.temperature = temperature
        self.mtp_interpolation = mtp_interpolation
        self.alpha = alpha

        # Precompute spline structures
        self.aknots = jnp.linspace(1, self.m_pts, self.m_pts)
        self.bknots = jnp.arange(self.m_pts + self.degree + 1)
        self.bmat = compute_b_spline_matrix(
            self.bknots, self.degree, self.num_knots
        )

        # Precompute linear-interpolation matrix so the linear branch is a
        # single einsum. For each query point t in [0, m_pts-1] expressed as
        # t = i + s with i ∈ {0, …, m_pts-2} and s ∈ [0, 1), the value is
        # (1 - s) * paths[i] + s * paths[i+1]. We bake that into a fixed
        # (num_interp*(m_pts-1), m_pts) matrix and pre-truncate to num_knots.
        if self.m_pts > 1:
            n_interp_lin = -(-self.num_knots // (self.m_pts - 1))  # ceil-div
            n_total = n_interp_lin * (self.m_pts - 1)
            t_lin = jnp.linspace(0.0, self.m_pts - 1, n_total + 1)[:-1]
            i_idx = jnp.clip(
                jnp.floor(t_lin).astype(jnp.int32), 0, self.m_pts - 2
            )
            s = t_lin - i_idx
            cols = jnp.arange(self.m_pts)
            # (n_total, m_pts): row k has weight (1-s_k) at i_k and s_k at
            # i_k+1
            linmat_full = (
                jnp.where(
                    cols[None, :] == i_idx[:, None], 1.0 - s[:, None], 0.0
                )
                + jnp.where(
                    cols[None, :] == i_idx[:, None] + 1, s[:, None], 0.0
                )
            )
            # Pre-truncate to num_knots so the JIT compiler sees a static
            # shape and we avoid a dynamic slice on every sample_knots call.
            self.linmat = linmat_full[: self.num_knots]
        else:
            self.linmat = jnp.ones((self.num_knots, 1))

    def init_params(
        self, initial_knots: jax.Array = None, seed: int = 0
    ) -> MTPParams:
        """Initialize the policy parameters.

        Args:
            initial_knots: Optional initial knot values. If None, zeros.
            seed: Random seed for PRNG initialization.

        Returns:
            Initialized MTPParams with zero mean, uniform cov, and
            zero best_knots.
        """
        _params = super().init_params(initial_knots, seed)
        cov = jnp.full_like(_params.mean, self.sigma_start)
        best_knots = jnp.zeros_like(_params.mean)
        return MTPParams(
            tk=_params.tk,
            mean=_params.mean,
            rng=_params.rng,
            cov=cov,
            best_knots=best_knots,
        )

    def sample_knots(
        self, params: MTPParams
    ) -> Tuple[jax.Array, MTPParams]:
        """Sample control spline knots using tensor + CEM mixing.

        Generates a batch of control knot sequences by:
        1. Including the previous best as one sample
        2. Sampling β*num_samples tensor paths through the M-partite graph
        3. Sampling (1-β)*num_samples local CEM perturbations

        Args:
            params: Current policy parameters.

        Returns:
            Tuple of (knots, updated_params) where knots has shape
            (num_samples, num_knots, nu).
        """
        rng = params.rng
        nu = self.task.model.nu

        # Always reserve one slot for `best`; the remaining (num_samples - 1)
        # slots are split between tensor paths (mtp_samples) and local CEM
        # perturbations (cem_samples). Capping at `num_samples - 1` keeps
        # the batch size exactly equal to num_samples even when beta == 1.
        mtp_samples = int(round(self.num_samples * self.beta))
        mtp_samples = min(max(mtp_samples, 0), self.num_samples - 1)

        # Always include the previous best as one sample
        best = params.best_knots[None, ...]  # (1, num_knots, nu)

        if mtp_samples >= 1:
            # --- Tensor sampling: sample paths through the M-partite graph ---

            # Single split for both control points and path indices.
            rng, cp_rng, idx_rng = jax.random.split(rng, 3)

            # Clamp infinite actuator bounds (hydrax uses ±inf for unlimited
            # actuators); fall back to a finite range derived from the current
            # CEM mean ± k·sigma_max so jax.random.uniform never returns NaN
            # and the tensor branch stays finite.
            k = 3.0
            mean_lo = jnp.min(params.mean, axis=0) - k * self.sigma_max
            mean_hi = jnp.max(params.mean, axis=0) + k * self.sigma_max
            u_min = jnp.where(
                jnp.isfinite(self.task.u_min), self.task.u_min, mean_lo
            )
            u_max = jnp.where(
                jnp.isfinite(self.task.u_max), self.task.u_max, mean_hi
            )
            control_points = jax.random.uniform(
                cp_rng,
                (self.m_pts, self.n_per_layer, nu),
                minval=u_min,
                maxval=u_max,
            )

            # Sample random paths through the M-partite graph.
            layer_indices = jax.random.randint(
                idx_rng, (mtp_samples, self.m_pts), 0, self.n_per_layer
            )

            # Direct broadcast gather avoids vmap closure overhead.
            paths = control_points[
                jnp.arange(self.m_pts)[None, :], layer_indices
            ]  # (mtp_samples, m_pts, nu)

            # Interpolate paths to num_knots points. When num_knots is not
            # divisible by (m_pts-1) we evaluate the spline on
            # ceil(num_knots/(m_pts-1)) points-per-segment and then truncate
            # to exactly num_knots samples; padding with a flat plateau
            # would inject a step-discontinuity and bias the elite
            # distribution towards trailing-zero controls.
            if self.m_pts > 1:
                num_interp = -(
                    -self.num_knots // (self.m_pts - 1)
                )  # ceil-div
            else:
                num_interp = self.num_knots

            if self.mtp_interpolation == "akima":
                A = jax.vmap(poly_akima, in_axes=(None, 0))(
                    self.aknots, paths
                )  # (mtp_samples, m_pts-1, 4, nu)
                mtp_knots = poly_interpolation(A, num_interp)[
                    :, : self.num_knots
                ]

            elif self.mtp_interpolation == "bspline":
                mtp_knots = jnp.einsum("bmd,hm->bhd", paths, self.bmat)

            elif self.mtp_interpolation == "linear":
                # Single einsum, fully XLA-fuseable, matching the
                # Vandermonde-style matmul used by the Akima branch.
                mtp_knots = jnp.einsum(
                    "bmd,hm->bhd", paths, self.linmat
                )

            else:
                raise ValueError(
                    f"Invalid MTP interpolation: {self.mtp_interpolation}. "
                    "Expected one of ['akima', 'bspline', 'linear']."
                )

            all_knots = jnp.concatenate([best, mtp_knots], axis=0)
        else:
            all_knots = best

        # --- Local CEM samples ---
        cem_samples = self.num_samples - all_knots.shape[0]
        if cem_samples > 0:
            rng, sample_rng = jax.random.split(rng)
            noise = jax.random.normal(
                sample_rng,
                (cem_samples, self.num_knots, nu),
            )
            cem_knots = params.mean + params.cov * noise
            all_knots = jnp.concatenate([all_knots, cem_knots], axis=0)

        return all_knots, params.replace(rng=rng)

    def update_params(
        self, params: MTPParams, rollouts: Trajectory
    ) -> MTPParams:
        """Update the policy parameters using weighted elite fitting.

        Selects the top-K trajectories by cost, fits a Gaussian to the
        elite knots with softmax weighting, and applies momentum
        smoothing controlled by α.

        Uses jax.lax.top_k for O(N + K log K) elite selection instead
        of a full O(N log N) sort.

        Args:
            params: Current policy parameters.
            rollouts: Trajectory data from the current iteration.

        Returns:
            Updated MTPParams with new mean, cov, and best_knots.
        """
        costs = jnp.sum(rollouts.costs, axis=1)  # (num_samples,)

        # Use top_k for O(N + K log K) elite selection.
        neg_costs = -costs
        _, elite_indices = jax.lax.top_k(neg_costs, self.num_elites)

        elite_knots = rollouts.knots[elite_indices]
        elite_costs = costs[elite_indices]

        # Subtract the min cost as a baseline before softmax. Mathematically
        # a no-op for softmax, but it keeps the exponent in a numerically
        # safe range when `temperature` is small (cf. MPPI-generic, Eq. 8).
        baseline = jnp.min(elite_costs)
        weights = jnp.nan_to_num(
            jax.nn.softmax(
                -(elite_costs - baseline) / self.temperature, axis=0
            )
        )

        # Weighted mean and (Bessel-corrected) std. The plain weighted
        # population std is biased low for small `num_elites`; the standard
        # CMA / weighted-CEM correction divides the variance by
        # (1 - Σ w_i^2), which equals (E-1)/E for uniform weights and
        # recovers the classical Bessel correction.
        weighted_knots = weights[:, None, None] * elite_knots
        mean = jnp.sum(weighted_knots, axis=0)
        var = jnp.sum(
            weights[:, None, None] * (elite_knots - mean) ** 2, axis=0
        )
        bessel = 1.0 / jnp.maximum(1.0 - jnp.sum(weights**2), 1e-6)
        cov = jnp.sqrt(var * bessel)
        cov = jnp.clip(cov, self.sigma_min, self.sigma_max)

        # Momentum smoothing on μ (arithmetic) and σ² (convex blend on the
        # variance - standard CMA/CEM convention). α=0: full update.
        mean = mean + self.alpha * (params.mean - mean)
        new_var = cov**2
        old_var = params.cov**2
        cov = jnp.sqrt(new_var + self.alpha * (old_var - new_var))

        # Best knots for warm-starting next iteration
        best_idx = elite_indices[0]
        best_knots = rollouts.knots[best_idx]

        return params.replace(mean=mean, cov=cov, best_knots=best_knots)
