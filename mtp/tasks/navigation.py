"""Velocity-controlled particle in a U-shaped maze.

Vendored from the legacy `anindex/hydrax @ feff4ce` particle task. The
upstream `hydrax.tasks.particle.Particle` no longer ships the inner walls
that create a local minimum between the start ``(-0.2, 0)`` and goal
``(0.25, 0)``; that local minimum is precisely what makes the task a
showcase for MTP's exploratory tensor sampling. We keep the asset and
cost local to the MTP package so the demo is reproducible regardless of
hydrax churn.

The class is adapted to the current `hydrax.task_base.Task` API
(only ``mj_model``, ``trace_sites``, ``impl`` are accepted).
"""

from pathlib import Path
from typing import Dict

import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx

from hydrax.task_base import Task

_MODEL_DIR = Path(__file__).resolve().parent.parent / "models" / "particle"


class NavigationParticle(Task):
    """Planar point mass that must navigate around U-shaped inner walls."""

    def __init__(self, impl: str = "jax") -> None:
        """Load the maze MJCF and set task parameters."""
        mj_model = mujoco.MjModel.from_xml_path(
            (_MODEL_DIR / "scene.xml").as_posix()
        )
        super().__init__(
            mj_model,
            trace_sites=["pointmass"],
            impl=impl,
        )

        # Initial pointmass configuration: behind the U-opening so the
        # straight-line path to the goal goes through a wall.
        self._initial_qpos = jnp.array([-0.2, 0.0])

        # Pre-extract the inner-wall geometry for the SDF cost. These three
        # walls (`wall_ix`, `wall_iy`, `wall_neg_iy`) form the U-shape that
        # creates the local minimum.
        self.wall_pos = jnp.array(
            [
                mj_model.geom("wall_ix").pos[:2],
                mj_model.geom("wall_iy").pos[:2],
                mj_model.geom("wall_neg_iy").pos[:2],
            ]
        )
        self.wall_size = jnp.array(
            [
                mj_model.geom("wall_ix").size[:2],
                mj_model.geom("wall_iy").size[:2],
                mj_model.geom("wall_neg_iy").size[:2],
            ]
        )

        self.pointmass_id = mj_model.site("pointmass").id

    # ---------------------------------------------------------------------
    # Convenience helpers (mirrors of the legacy `reset()` API).
    # ---------------------------------------------------------------------
    def make_initial_data(self, seed: int = 0) -> mujoco.MjData:
        """Return an `mujoco.MjData` initialised at the start configuration."""
        rng = jax.random.PRNGKey(seed)
        jitter = 0.02 * jax.random.normal(rng, (2,))
        mj_data = mujoco.MjData(self.mj_model)
        mj_data.qpos[:2] = jnp.asarray(self._initial_qpos + jitter)
        return mj_data

    # ---------------------------------------------------------------------
    # Costs.
    # ---------------------------------------------------------------------
    def running_cost(self, state: mjx.Data, control: jax.Array) -> jax.Array:
        """ℓ(xₜ, uₜ): wall-SDF penalty + terminal pose + control."""
        # Box-SDF distance to each inner wall (signed, axis-aligned).
        wall_dist = (
            jnp.abs(state.site_xpos[self.pointmass_id][None, :2] - self.wall_pos)
            - self.wall_size
        )
        outside_dist = jnp.maximum(wall_dist, 1e-12)
        inside_dist = jnp.minimum(jnp.max(wall_dist, axis=-1), 0.0)
        dist = (jnp.linalg.norm(outside_dist, axis=-1) + inside_dist).min(axis=-1)
        wall_cost = 5.0 * jnp.exp(-50.0 * dist)
        control_cost = jnp.sum(jnp.square(control))
        return wall_cost + self.terminal_cost(state) + 0.1 * control_cost

    def terminal_cost(self, state: mjx.Data) -> jax.Array:
        """ϕ(x_T): position tracking + small velocity regularisation."""
        position_cost = jnp.sum(
            jnp.square(state.site_xpos[self.pointmass_id] - state.mocap_pos[0])
        )
        velocity_cost = jnp.sum(jnp.square(state.qvel))
        return 5.0 * position_cost + 0.1 * velocity_cost

    # ---------------------------------------------------------------------
    # Domain randomisation (kept compatible with the upstream `Particle`).
    # ---------------------------------------------------------------------
    def domain_randomize_model(self, rng: jax.Array) -> Dict[str, jax.Array]:
        """Perturb actuator gains by ±10%."""
        multiplier = jax.random.uniform(
            rng,
            self.model.actuator_gainprm[:, 0].shape,
            minval=0.9,
            maxval=1.1,
        )
        new_gains = self.model.actuator_gainprm[:, 0] * multiplier
        new_gains = self.model.actuator_gainprm.at[:, 0].set(new_gains)
        return {"actuator_gainprm": new_gains}

    def domain_randomize_data(
        self, data: mjx.Data, rng: jax.Array
    ) -> Dict[str, jax.Array]:
        """Apply a small position offset to simulate state-estimation noise."""
        shift = jax.random.uniform(rng, (2,), minval=-0.01, maxval=0.01)
        return {"qpos": data.qpos + shift}
