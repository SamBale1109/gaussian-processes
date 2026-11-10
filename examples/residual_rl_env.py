"""Gymnasium environment for residual-policy obstacle avoidance.

The agent learns a *residual* velocity only.  Each step the environment

1. computes the base-policy command ``v_nom`` (pure goal attraction,
   see :class:`gaussian_processes.controllers.NominalPolicy`),
2. executes ``v = v_nom + d_v`` with ``d_v = v_max * action``,
   saturated to the maximum speed,
3. integrates the robot and the (possibly moving) obstacles.

The reward rewards progress towards the goal, penalizes small
clearance and large residuals, and ends the episode with a large
penalty on collision (obstacle or wall).  The ``w_res`` term keeps
the residual near zero in free space, which is what makes the
learned policy transfer to a different base policy (e.g. an LfD
policy) later.

Worlds (start, goal, obstacles) are resampled on every reset; pass
``options={"scenario": path}`` to :meth:`ResidualAvoidanceEnv.reset`
to evaluate on a fixed JSON scenario instead (see
``create_environment.py`` and ``reactive_avoidance_testbed.py``).

Observation layout (all lengths divided by the world size)::

    [ goal - pose           (2)
      wall distances x,y    (4)
      v_nom                 (2)
      per K nearest obs:    (5 * K)
          signed distance,
          SDF gradient (2),
          obstacle velocity (2) ]
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import gymnasium as gym
import numpy as np
from reactive_avoidance_testbed import (
    BoxObstacle,
    CircleObstacle,
    Obstacle,
    ObstacleEnvironment,
    load_scenario,
)

from gaussian_processes.controllers import NominalPolicy


@dataclass
class ResidualConfig:
    """Parameters of the residual-avoidance MDP and its reward."""

    # -- world / dynamics --------------------------------------------------
    world_size: float = 10.0
    """Side length of the square world."""

    v_max: float = 1.0
    """Maximum speed (saturation of ``v_nom + d_v``)."""

    dt: float = 0.05
    """Integration time step [s]."""

    max_steps: int = 400
    """Episode length limit (truncation)."""

    arrive_radius: float = 0.15
    """Goal arrival radius (also used by the base policy)."""

    # -- observation ---------------------------------------------------------
    k_obstacles: int = 5
    """Number of nearest obstacles included in the observation."""

    # -- world sampling -------------------------------------------------------
    n_obstacles: Tuple[int, int] = (2, 5)
    """Min / max number of obstacles per world."""

    radius_range: Tuple[float, float] = (0.4, 1.1)
    """Radius range for circle obstacles."""

    box_prob: float = 0.3
    """Probability that a sampled obstacle is a box."""

    box_extent_range: Tuple[float, float] = (0.4, 1.0)
    """Half-extent range for box obstacles."""

    moving_prob: float = 0.25
    """Probability that one obstacle moves (random direction)."""

    max_move_speed: float = 0.3
    """Maximum per-axis speed of a moving obstacle."""

    wall_margin: float = 0.3
    """Minimum clearance of obstacle centers to the walls."""

    point_margin: float = 0.6
    """Extra clearance kept between start/goal and obstacles."""

    pair_margin: float = 0.4
    """Extra clearance kept between pairs of obstacles."""

    # -- reward ---------------------------------------------------------------
    w_progress: float = 5.0
    """Reward per unit distance closed to the goal."""

    w_time: float = 0.2
    """Time cost per second (hurrying bonus)."""

    w_clearance: float = 2.0
    """Penalty per unit clearance deficit below ``d_safe``."""

    d_safe: float = 0.8
    """Preferred minimum clearance to obstacles / walls."""

    w_res: float = 0.05
    """Penalty ``w_res * ||d_v||^2`` (keeps the residual small)."""

    r_goal: float = 10.0
    """Terminal bonus for reaching the goal."""

    r_collision: float = -100.0
    """Terminal penalty on collision (obstacle or wall)."""

    r_timeout: float = -5.0
    """Terminal penalty on truncation (time limit)."""


# ---------------------------------------------------------------------------
# Shared observation builder (env + testbed)
# ---------------------------------------------------------------------------
def obstacle_gradient(ob: Obstacle, pose: np.ndarray) -> np.ndarray:
    """Analytic SDF gradient of a single obstacle at ``pose``."""
    p = pose
    if isinstance(ob, CircleObstacle):
        q = p - ob.center
        n = float(np.linalg.norm(q))
        return q / n if n > 1e-8 else np.zeros(2)
    q = np.abs(p - ob.center) - ob.half_extents
    sgn = np.sign(p - ob.center)
    w = np.maximum(q, 0.0)
    d_out = float(np.linalg.norm(w))
    if d_out > 1e-8:  # outside: toward the closest corner
        v = w * sgn
        return v / d_out
    # inside: toward the nearest exit face
    axis = int(np.argmax(q))
    g = np.zeros(2)
    g[axis] = sgn[axis] if sgn[axis] != 0.0 else 1.0
    return g


def build_observation(
    env: ObstacleEnvironment,
    pose: np.ndarray,
    goal: np.ndarray,
    lo: Sequence[float],
    hi: Sequence[float],
    scale: float,
    v_nom: np.ndarray,
    cfg: Optional[ResidualConfig] = None,
) -> np.ndarray:
    """Build the residual-policy observation.

    Single source of truth for the observation layout: both
    :meth:`ResidualAvoidanceEnv._obs` (training) and the testbed's
    ``--controller residual_rl`` (evaluation) call this function, so
    the policy sees identical features in both settings.  ``scale``
    must be the world side length (or the larger side for
    rectangular bounds); ``v_nom`` is the base-policy velocity at
    ``pose``.
    """
    cfg = cfg if cfg is not None else ResidualConfig()
    lo = np.asarray(lo, dtype=float)
    hi = np.asarray(hi, dtype=float)
    parts: List[np.ndarray] = [
        (goal - pose) / scale,
        np.r_[pose - lo, hi - pose] / scale,
        v_nom / cfg.v_max,
    ]
    feats: List[np.ndarray] = []
    if env is not None and env.obstacles:
        p = pose.reshape(1, 2)
        dists = np.array(
            [float(ob.signed_distance(p)[0])
             for ob in env.obstacles]
        )
        order = np.argsort(dists)
        for idx in order[: cfg.k_obstacles]:
            ob = env.obstacles[int(idx)]
            vel = (
                ob.velocity
                if ob.velocity is not None
                else np.zeros(2)
            )
            feats.append(
                np.r_[
                    float(dists[int(idx)]) / scale,
                    obstacle_gradient(ob, pose),
                    vel / cfg.v_max,
                ]
            )
    parts.extend(feats)
    pad = (cfg.k_obstacles - len(feats)) * 5
    if pad:
        parts.append(np.zeros(pad))
    return np.concatenate(parts)


class ResidualAvoidanceEnv(gym.Env):
    """Residual-RL MDP on top of the 2D obstacle-avoidance testbed."""

    def __init__(
        self, config: Optional[ResidualConfig] = None
    ) -> None:
        self.cfg = config if config is not None else ResidualConfig()
        obs_dim = 8 + 5 * self.cfg.k_obstacles
        self.observation_space = gym.spaces.Box(
            -np.inf, np.inf, (obs_dim,), dtype=np.float64
        )
        self.action_space = gym.spaces.Box(
            -1.0, 1.0, (2,), dtype=np.float64
        )
        self.pose = np.zeros(2)
        self.start = np.zeros(2)
        self.goal = np.zeros(2)
        self.lo: Tuple[float, float] = (0.0, 0.0)
        self.hi: Tuple[float, float] = (0.0, 0.0)
        self.scale = float(self.cfg.world_size)
        self.env: Optional[ObstacleEnvironment] = None
        self.base: Optional[NominalPolicy] = None
        self._rng: Optional[np.random.Generator] = None
        self._step_i = 0
        self._min_clear = float("inf")

    # -- Gymnasium API ------------------------------------------------------
    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        self._rng = self.np_random
        if options is not None and "scenario" in options:
            self._load_scenario(str(options["scenario"]))
        else:
            self._sample_world()
        self.pose = self.start.copy()
        self._step_i = 0
        self._min_clear = float("inf")
        self.base = NominalPolicy(
            self.goal, self.cfg.v_max, self.cfg.arrive_radius
        )
        return self._obs(), {}

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, dict]:
        cfg = self.cfg
        a = np.clip(np.asarray(action, dtype=float), -1.0, 1.0)
        d_v = cfg.v_max * a
        v_nom = self.base.compute_velocity(self.pose)
        v = v_nom + d_v
        n = float(np.linalg.norm(v))
        if n > cfg.v_max:
            v = v * (cfg.v_max / n)

        d_goal_prev = float(np.linalg.norm(self.pose - self.goal))
        self.pose = self.pose + v * cfg.dt
        self.env.update_environment(cfg.dt)
        self._bounce_moving()
        self._step_i += 1

        # Clearance: obstacles (testbed semantics) and walls.
        if self.env.obstacles:
            d_obs = float(
                self.env.signed_distance_field(
                    self.pose.reshape(1, 2)
                )[0]
            )
        else:
            d_obs = float("inf")
        d_wall = float(
            np.min(
                np.r_[
                    self.pose - np.asarray(self.lo, dtype=float),
                    np.asarray(self.hi, dtype=float) - self.pose,
                ]
            )
        )
        d_clear = min(d_obs, d_wall)
        self._min_clear = min(self._min_clear, d_obs)
        d_goal_next = float(np.linalg.norm(self.pose - self.goal))

        reward = cfg.w_progress * max(
            0.0, d_goal_prev - d_goal_next
        )
        reward -= cfg.w_time * cfg.dt
        if d_clear < cfg.d_safe:
            reward -= cfg.w_clearance * (cfg.d_safe - d_clear)
        reward -= cfg.w_res * float(np.dot(d_v, d_v))

        info = {
            "steps": self._step_i,
            "min_clearance": self._min_clear,
            "clearance": d_clear,
            "v": v,
        }
        if d_clear < 0.0:
            info["collided"] = True
            reward += cfg.r_collision
            return self._obs(), float(reward), True, False, info
        if d_goal_next <= cfg.arrive_radius:
            info["reached_goal"] = True
            reward += cfg.r_goal
            return self._obs(), float(reward), True, False, info
        if self._step_i >= cfg.max_steps:
            reward += cfg.r_timeout
            return (
                self._obs(),
                float(reward),
                False,
                True,
                info,
            )
        return self._obs(), float(reward), False, False, info

    # -- World sampling -------------------------------------------------------
    def _load_scenario(self, path: str) -> None:
        """Load a fixed world from a scenario JSON file."""
        tb_env, start, goal = load_scenario(path)
        self.env = tb_env
        self.lo, self.hi = tb_env.lo, tb_env.hi
        self.start = np.asarray(start, dtype=float).copy()
        self.goal = np.asarray(goal, dtype=float).copy()
        self.scale = float(
            max(
                self.hi[0] - self.lo[0], self.hi[1] - self.lo[1]
            )
        )

    def _sample_world(self) -> None:
        """Sample start, goal and obstacles (resampled each reset)."""
        cfg = self.cfg
        rng = self._rng
        L = cfg.world_size
        lo = np.array([0.0, 0.0])
        hi = np.array([L, L])
        self.lo = (float(lo[0]), float(lo[1]))
        self.hi = (float(hi[0]), float(hi[1]))
        self.scale = float(L)

        # Start / goal with some separation.
        start = rng.uniform(1.0, L - 1.0, size=2)
        goal = start
        for _ in range(200):
            goal = rng.uniform(1.0, L - 1.0, size=2)
            if float(np.linalg.norm(goal - start)) > 0.5 * L:
                break
        self.start = start
        self.goal = goal

        env = ObstacleEnvironment((self.lo, self.hi))
        placed: List[Tuple[Obstacle, float]] = []
        n = int(rng.integers(cfg.n_obstacles[0],
                             cfg.n_obstacles[1] + 1))
        for _ in range(n):
            ob, r = self._new_obstacle(rng, lo, hi)
            for _ in range(200):
                if self._clears(ob, r, placed):
                    break
                ob, r = self._new_obstacle(rng, lo, hi)
            placed.append((ob, r))
            env.add_obstacle(ob)

        # Optionally give one obstacle a constant velocity.
        if placed and float(rng.random()) < cfg.moving_prob:
            ob, _ = placed[int(rng.integers(len(placed)))]
            vel = rng.uniform(
                -cfg.max_move_speed, cfg.max_move_speed, 2
            )
            if float(np.linalg.norm(vel)) < 0.1 * cfg.v_max:
                vel[0] = cfg.max_move_speed
            ob.velocity = vel
        self.env = env

    def _new_obstacle(
        self, rng: np.random.Generator, lo: np.ndarray,
        hi: np.ndarray,
    ) -> Tuple[Obstacle, float]:
        cfg = self.cfg
        box = bool(rng.random() < cfg.box_prob)
        if box:
            h = rng.uniform(
                cfg.box_extent_range[0],
                cfg.box_extent_range[1],
                size=2,
            )
        else:
            radius = float(
                rng.uniform(cfg.radius_range[0], cfg.radius_range[1])
            )
            h = np.array([radius, radius], dtype=float)
        m = cfg.wall_margin
        center = np.empty(2)
        for axis in range(2):
            center[axis] = float(
                rng.uniform(lo[axis] + m + h[axis],
                            hi[axis] - m - h[axis])
            )
        if box:
            return BoxObstacle(center, h), float(np.max(h))
        return CircleObstacle(center, float(h[0])), float(h[0])

    def _clears(
        self,
        ob: Obstacle,
        r: float,
        placed: List[Tuple[Obstacle, float]],
    ) -> bool:
        cfg = self.cfg
        center = ob.center
        for point in (self.start, self.goal):
            if float(np.linalg.norm(center - point)) < (
                r + cfg.point_margin
            ):
                return False
        for other, r_other in placed:
            if float(np.linalg.norm(center - other.center)) < (
                r + r_other + cfg.pair_margin
            ):
                return False
        return True

    def _bounce_moving(self) -> None:
        """Reflect moving obstacles at the walls (keeps them in-world)."""
        lo = np.asarray(self.lo, dtype=float)
        hi = np.asarray(self.hi, dtype=float)
        for ob in self.env.obstacles:
            if not ob.is_moving:
                continue
            half = (
                np.full(2, ob.radius)
                if isinstance(ob, CircleObstacle)
                else np.asarray(ob.half_extents, dtype=float)
            )
            for axis in range(2):
                if ob.center[axis] < lo[axis] + half[axis]:
                    ob.center[axis] = lo[axis] + half[axis]
                    ob.velocity[axis] = abs(ob.velocity[axis])
                elif ob.center[axis] > hi[axis] - half[axis]:
                    ob.center[axis] = hi[axis] - half[axis]
                    ob.velocity[axis] = -abs(ob.velocity[axis])

    # -- Observation ----------------------------------------------------------
    def _obs(self) -> np.ndarray:
        v_nom = self.base.compute_velocity(self.pose)
        return build_observation(
            self.env, self.pose, self.goal, self.lo, self.hi,
            self.scale, v_nom, self.cfg,
        )

    def _obstacle_grad(self, ob: Obstacle) -> np.ndarray:
        """Analytic SDF gradient at the robot pose."""
        return obstacle_gradient(ob, self.pose)


def make_env(
    config: Optional[ResidualConfig] = None,
) -> ResidualAvoidanceEnv:
    """Factory for (freshly resampled) environments, e.g. for VecEnv."""
    return ResidualAvoidanceEnv(config)
