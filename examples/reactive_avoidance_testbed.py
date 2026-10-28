"""2D testbed for evaluating reactive obstacle avoidance methods.

This module provides a compact, self-contained framework to benchmark
reactive obstacle-avoidance strategies built on distance / gradient
fields (e.g. Gaussian Process distance fields, IDMP).  The field
estimation algorithm itself is *not* part of this testbed: you plug in
your own estimator by subclassing :class:`DistanceGradientFieldBase`.
A ground-truth dummy (:class:`GroundTruthField`) is provided so the
framework is runnable out of the box.

Components
----------
- :class:`ObstacleEnvironment`    2D world with static/dynamic obstacles
- :class:`SensorSimulator`        2D LiDAR (raycasting) point clouds
- :class:`NominalPolicy`          obstacle-free target-seeking policy
- :class:`DistanceGradientFieldBase`  field estimator interface (ABC)
- :class:`GroundTruthField`       dummy field using the exact distances
- :class:`ReactiveController`     velocity blending / modulation
- :class:`SimulationEngine`       Euler integration loop + logging
- :func:`run_demo` / ``__main__`` ready-to-run demo scenario

Dependencies: numpy, matplotlib.

Usage::

    python examples/reactive_avoidance_testbed.py            # live anim
    python examples/reactive_avoidance_testbed.py --no-anim  # static plot
    python examples/reactive_avoidance_testbed.py --save-plot out.png
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle as CirclePatch
from matplotlib.patches import Rectangle as RectanglePatch
from gaussian_processes.distance_gradient_fields import DistanceGradientFieldBase, GPDistanceGradientField, FusedGPDistanceGradientField
from gaussian_processes.controllers import NominalPolicy, ReactiveController, ReactiveCirculationController, ReactiveCirculationHysteresisController

__all__ = [
    "Obstacle",
    "CircleObstacle",
    "MovingCircleObstacle",
    "BoxObstacle",
    "MovingBoxObstacle",
    "ObstacleEnvironment",
    "SensorSimulator",
    "NominalPolicy",
    "GroundTruthField",
    "ReactiveController",
    "SimulationEngine",
    "SimulationResult",
    "run_demo",
]

# Type aliases.
Bounds = Tuple[Tuple[float, float], Tuple[float, float]]
"""((x_min, x_max), (y_min, y_max)) of the world."""


# ---------------------------------------------------------------------------
# Obstacles
# ---------------------------------------------------------------------------
class Obstacle(ABC):
    """Base class for 2D obstacles.

    Subclasses must implement the vectorized :meth:`signed_distance`
    and the ray intersection :meth:`ray_hit`.  Obstacles with a
    non-zero ``velocity`` are moved by :meth:`update` each time step.
    """

    def __init__(self, velocity: Optional[Sequence[float]] = None) -> None:
        self.velocity: Optional[np.ndarray] = (
            np.asarray(velocity, dtype=float) if velocity is not None else None
        )

    @property
    def is_moving(self) -> bool:
        """True if the obstacle has a non-zero velocity."""
        return self.velocity is not None and np.any(self.velocity != 0.0)

    @abstractmethod
    def signed_distance(self, points: np.ndarray) -> np.ndarray:
        """Signed distance of (N, 2) points to the obstacle surface.

        Negative values denote points inside the obstacle.
        """

    @abstractmethod
    def ray_hit(self, origin: np.ndarray, direction: np.ndarray) -> float:
        """First intersection distance ``t >= 0`` of a unit ray.

        Returns ``np.inf`` if the ray does not intersect the obstacle.
        """

    def update(self, dt: float) -> None:
        """Advance the obstacle by its (possibly zero) velocity."""
        del dt  # overridden by subclasses with motion

    def make_patch(self) -> matplotlib.patches.Patch:
        """Create an unfilled matplotlib patch for visualization."""
        raise NotImplementedError


class CircleObstacle(Obstacle):
    """Circular obstacle, optionally moving with constant velocity."""

    def __init__(
        self,
        center: Sequence[float],
        radius: float,
        velocity: Optional[Sequence[float]] = None,
    ) -> None:
        super().__init__(velocity)
        self.center = np.asarray(center, dtype=float).copy()
        self.radius = float(radius)

    def update(self, dt: float) -> None:
        if self.is_moving and self.velocity is not None:
            self.center = self.center + self.velocity * dt

    def signed_distance(self, points: np.ndarray) -> np.ndarray:
        p = np.atleast_2d(np.asarray(points, dtype=float))
        return np.linalg.norm(p - self.center, axis=-1) - self.radius

    def ray_hit(self, origin: np.ndarray, direction: np.ndarray) -> float:
        o = origin - self.center
        b = 2.0 * float(np.dot(o, direction))
        c = float(np.dot(o, o)) - self.radius**2
        disc = b * b - 4.0 * c
        if disc < 0.0:
            return np.inf
        s = math.sqrt(disc)
        t = (-b - s) / 2.0
        if t < 0.0:
            t = (-b + s) / 2.0
        if t < 0.0:
            return np.inf  # Beide Schnittpunkte liegen hinter dem Strahl!
        return t

    def make_patch(self) -> matplotlib.patches.Patch:
        return CirclePatch(tuple(self.center), self.radius, fill=False)


class MovingCircleObstacle(CircleObstacle):
    """Convenience subclass: a circle with a mandatory velocity."""

    def __init__(
        self,
        initial_center: Sequence[float],
        velocity: Sequence[float],
        radius: float,
    ) -> None:
        super().__init__(
            center=initial_center, radius=radius, velocity=velocity
        )


class BoxObstacle(Obstacle):
    """Axis-aligned rectangular obstacle (axis-parallel box).

    Parameters
    ----------
    center:
        Center of the box.
    half_extents:
        ``[half_width, half_height]`` of the box.
    """

    def __init__(
        self,
        center: Sequence[float],
        half_extents: Sequence[float],
        velocity: Optional[Sequence[float]] = None,
    ) -> None:
        super().__init__(velocity)
        self.center = np.asarray(center, dtype=float).copy()
        self.half_extents = np.asarray(half_extents, dtype=float).copy()

    @property
    def bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """Lower and upper corner of the box."""
        return (
            self.center - self.half_extents,
            self.center + self.half_extents,
        )

    def update(self, dt: float) -> None:
        if self.is_moving and self.velocity is not None:
            self.center = self.center + self.velocity * dt

    def signed_distance(self, points: np.ndarray) -> np.ndarray:
        p = np.atleast_2d(np.asarray(points, dtype=float))
        q = np.abs(p - self.center) - self.half_extents
        d_outside = np.linalg.norm(np.maximum(q, 0.0), axis=-1)
        d_inside = np.minimum(np.max(q, axis=-1), 0.0)
        return d_outside + d_inside

    def ray_hit(self, origin: np.ndarray, direction: np.ndarray) -> float:
        lo, hi = self.bounds
        tmin, tmax = 0.0, np.inf
        for a in range(2):
            if abs(direction[a]) < 1e-12:
                if origin[a] < lo[a] or origin[a] > hi[a]:
                    return np.inf
            else:
                t1 = (lo[a] - origin[a]) / direction[a]
                t2 = (hi[a] - origin[a]) / direction[a]
                tmin = max(tmin, min(t1, t2))
                tmax = min(tmax, max(t1, t2))
        if tmax < tmin:
            return np.inf
        return max(tmin, 0.0)

    def make_patch(self) -> matplotlib.patches.Patch:
        lo, _ = self.bounds
        return RectanglePatch(
            (float(lo[0]), float(lo[1])),
            float(2.0 * self.half_extents[0]),
            float(2.0 * self.half_extents[1]),
            fill=False,
        )


class MovingBoxObstacle(BoxObstacle):
    """Convenience subclass: an axis-aligned box with constant velocity."""

    def __init__(
        self,
        initial_center: Sequence[float],
        half_extents: Sequence[float],
        velocity: Sequence[float],
    ) -> None:
        super().__init__(
            center=initial_center,
            half_extents=half_extents,
            velocity=velocity,
        )


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
class ObstacleEnvironment:
    """2D world with static and dynamic obstacles.

    The world spans the axis-aligned box given by ``bounds``; by default
    this is the unit-square scaled to ``[0, 10] x [0, 10]``.
    """

    def __init__(self, bounds: Bounds = ((0.0, 10.0), (0.0, 10.0))) -> None:
        self.bounds = bounds
        self.obstacles: List[Obstacle] = []
        self.time = 0.0

    @property
    def lo(self) -> Tuple[float, float]:
        return (self.bounds[0][0], self.bounds[1][0])

    @property
    def hi(self) -> Tuple[float, float]:
        return (self.bounds[0][1], self.bounds[1][1])

    def add_obstacle(self, obstacle: Obstacle) -> None:
        """Register an obstacle in the environment."""
        self.obstacles.append(obstacle)

    def update_environment(self, dt: float) -> None:
        """Advance all (moving) obstacles by ``dt`` seconds."""
        for obstacle in self.obstacles:
            obstacle.update(dt)
        self.time += dt

    def signed_distance_field(self, points: np.ndarray) -> np.ndarray:
        """Minimum signed distance of (N, 2) points to all obstacles.

        Points inside any obstacle yield a negative distance.  Returns
        an array of shape (N,).  With no obstacles the distance is
        ``+inf`` everywhere.
        """
        p = np.atleast_2d(np.asarray(points, dtype=float))
        if not self.obstacles:
            return np.full(len(p), np.inf)
        distances = [ob.signed_distance(p) for ob in self.obstacles]
        return np.min(np.stack(distances, axis=0), axis=0)

    def get_ground_truth_distance_and_gradient(
        self, x: Sequence[float]
    ) -> Tuple[float, np.ndarray]:
        """Exact euklidian distance ``d(x)`` and gradient ``grad d(x)``.

        The gradient is computed with central finite differences of the
        (piecewise smooth) signed distance field.  Useful as benchmark
        reference for field estimators.

        Returns
        -------
        (d, grad) with ``d`` a float and ``grad`` an array of shape (2,).
        """
        x = np.asarray(x, dtype=float)
        eps = 1e-4
        d = float(self.signed_distance_field(x)[0])
        grad = np.zeros(2, dtype=float)
        for a in range(2):
            xp = x.copy()
            xm = x.copy()
            xp[a] += eps
            xm[a] -= eps
            grad[a] = (
                self.signed_distance_field(xp)[0]
                - self.signed_distance_field(xm)[0]
            ) / (2.0 * eps)
        return d, grad


# ---------------------------------------------------------------------------
# Sensor
# ---------------------------------------------------------------------------
class SensorSimulator:
    """Simulated 2D LiDAR sensor based on raycasting.

    For each of ``num_rays`` uniformly spaced rays (within ``fov``) the
    first intersection with any obstacle is computed analytically
    (circle: quadratic, box: slab method).  Ranges beyond ``max_range``
    are discarded; optional Gaussian noise (``noise_std``) is added to
    the hit points.
    """

    def __init__(
        self,
        max_range: float = 4.0,
        num_rays: int = 72,
        noise_std: float = 0.0,
        fov: float = 2.0 * math.pi,
        seed: Optional[int] = None,
    ) -> None:
        self.max_range = float(max_range)
        self.num_rays = int(num_rays)
        self.noise_std = float(noise_std)
        self.fov = float(fov)
        self._rng = np.random.default_rng(seed)

    def get_pointcloud(
        self,
        robot_pose: Sequence[float],
        env: ObstacleEnvironment,
        *,
        max_range: Optional[float] = None,
        num_rays: Optional[int] = None,
        noise_std: Optional[float] = None,
    ) -> np.ndarray:
        """Raycast the environment from ``robot_pose``.

        Returns an array of shape (N, 2) with the 2D measurement points
        (possibly empty).  ``max_range`` / ``num_rays`` / ``noise_std``
        optionally override the values from the constructor.
        """
        origin = np.asarray(robot_pose, dtype=float)
        rng = self.max_range if max_range is None else float(max_range)
        n_rays = self.num_rays if num_rays is None else int(num_rays)
        sigma = self.noise_std if noise_std is None else float(noise_std)
        angles = np.linspace(0.0, self.fov, n_rays, endpoint=False)
        hits: List[np.ndarray] = []
        for angle in angles:
            direction = np.array(
                [math.cos(angle), math.sin(angle)], dtype=float
            )
            t_hit = min(
                (ob.ray_hit(origin, direction) for ob in env.obstacles),
                default=np.inf,
            )
            if t_hit < rng:
                hits.append(origin + t_hit * direction)
        if not hits:
            return np.empty((0, 2))
        pointcloud = np.asarray(hits, dtype=float)
        if sigma > 0.0:
            pointcloud = pointcloud + sigma * self._rng.normal(
                size=pointcloud.shape
            )
        return pointcloud




# ---------------------------------------------------------------------------
# Dummy distance & gradient field
# ---------------------------------------------------------------------------
class GroundTruthField(DistanceGradientFieldBase):
    """Dummy field that returns the *exact* environment distances.

    Use this to validate the framework; replace it with your GP-based
    estimator for actual benchmarks.
    """

    def __init__(self, env: ObstacleEnvironment) -> None:
        self.env = env

    def update(self, pointcloud: np.ndarray, robot_pose: np.ndarray) -> None:
        # Ground truth needs no learning: the measurement is ignored.
        del pointcloud, robot_pose

    def query(self, x: Sequence[float]) -> Tuple[float, np.ndarray]:
        return self.env.get_ground_truth_distance_and_gradient(x)


# ---------------------------------------------------------------------------
# Simulation engine
# ---------------------------------------------------------------------------
@dataclass
class SimulationResult:
    """Logged data of one simulation run (all poses of shape (T, 2))."""

    times: np.ndarray
    poses: np.ndarray
    velocities: np.ndarray
    true_distances: np.ndarray
    estimated_distances: np.ndarray
    pointclouds: List[np.ndarray]
    obstacle_states: List[List[Dict[str, object]]]
    start_pose: np.ndarray
    goal: np.ndarray
    reached_goal: bool
    collided: bool

    @property
    def final_pose(self) -> np.ndarray:
        return self.poses[-1]

    @property
    def min_clearance(self) -> float:
        """Smallest signed distance to an obstacle along the run."""
        return float(np.min(self.true_distances)) if len(self.times) else 0.0


def _grid_field(
    field_model: DistanceGradientFieldBase,
    env: ObstacleEnvironment,
    resolution: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate the field model on a grid.

    Returns ``(X, Y, D, G)`` with ``G`` of shape
    ``(resolution, resolution, 2)``.
    """
    xs = np.linspace(env.lo[0], env.hi[0], resolution)
    ys = np.linspace(env.lo[1], env.hi[1], resolution)
    X, Y = np.meshgrid(xs, ys)
    P = np.column_stack([X.ravel(), Y.ravel()])
    D = np.empty(len(P), dtype=float)
    G = np.empty((len(P), 2), dtype=float)
    for i, p in enumerate(P):
        d, g = field_model.query(p)
        D[i] = d
        G[i] = g
    return X, Y, D.reshape(X.shape), G.reshape(X.shape + (2,))


def _obstacle_state(env: ObstacleEnvironment) -> List[Dict[str, object]]:
    """Compact per-frame copy of the obstacle configuration."""
    state: List[Dict[str, object]] = []
    for ob in env.obstacles:
        if isinstance(ob, CircleObstacle):
            state.append(
                {
                    "type": "circle",
                    "center": ob.center.copy(),
                    "radius": ob.radius,
                }
            )
        elif isinstance(ob, BoxObstacle):
            state.append(
                {
                    "type": "box",
                    "center": ob.center.copy(),
                    "half_extents": ob.half_extents.copy(),
                }
            )
    return state


def _set_patch(
    patch: matplotlib.patches.Patch, state: Dict[str, object]
) -> None:
    """Move/resize a matplotlib patch to a recorded obstacle state."""
    center = state["center"]
    if state["type"] == "circle":
        patch.center = (float(center[0]), float(center[1]))
        patch.radius = float(state["radius"])
    else:
        half = state["half_extents"]
        patch.set_xy((float(center[0] - half[0]), float(center[1] - half[1])))
        patch.set_width(float(2.0 * half[0]))
        patch.set_height(float(2.0 * half[1]))


class SimulationEngine:
    """Euler-integration loop over the full perception-to-control stack.

    The step sequence mirrors the reference skeleton::

        env.update_environment(dt)
        pointcloud = sensor.get_pointcloud(robot_pose, env)
        field_model.update(pointcloud, robot_pose)
        v_cmd = controller.compute_velocity(robot_pose)
        robot_pose = robot_pose + v_cmd * dt
        # logging / visualization

    Termination: collision (true distance < 0) or arrival within the
    nominal policy's ``arrive_radius``.
    """

    def __init__(
        self,
        env: ObstacleEnvironment,
        sensor: SensorSimulator,
        field_model: DistanceGradientFieldBase,
        controller: ReactiveController,
        start_pose: Sequence[float],
        dt: float = 0.05,
        num_steps: int = 500,
    ) -> None:
        self.env = env
        self.sensor = sensor
        self.field_model = field_model
        self.controller = controller
        self.pose = np.asarray(start_pose, dtype=float).copy()
        self.start_pose = self.pose.copy()
        self.dt = float(dt)
        self.num_steps = int(num_steps)
        self._frames: Dict[str, list] = {
            "times": [],
            "poses": [],
            "velocities": [],
            "true_distances": [],
            "estimated_distances": [],
            "pointclouds": [],
            "obstacle_states": [],
        }

    def _log_frame(self, v_cmd: Optional[np.ndarray] = None) -> None:
        d_true, _ = self.env.get_ground_truth_distance_and_gradient(self.pose)
        d_est, _ = self.field_model.query(self.pose)
        f = self._frames
        f["times"].append(self.env.time)
        f["poses"].append(self.pose.copy())
        f["velocities"].append(v_cmd if v_cmd is not None else np.zeros(2))
        f["true_distances"].append(d_true)
        f["estimated_distances"].append(d_est)
        f["pointclouds"].append(self._last_pointcloud)
        f["obstacle_states"].append(_obstacle_state(self.env))

    def run(self) -> SimulationResult:
        """Execute the simulation loop; returns the logged result."""
        self._frames = {
            "times": [],
            "poses": [],
            "velocities": [],
            "true_distances": [],
            "estimated_distances": [],
            "pointclouds": [],
            "obstacle_states": [],
        }
        self.pose = self.start_pose.copy()
        self._last_pointcloud = np.empty((0, 2))
        self._log_frame()  # initial state
        for _ in range(self.num_steps):
            # 1. Update environment (moving obstacles).
            self.env.update_environment(self.dt)
            # 2. Sensor measurement.
            self._last_pointcloud = self.sensor.get_pointcloud(
                self.pose, self.env
            )
            # 3. Update field model (your GP model plugs in here).
            self.field_model.update(self._last_pointcloud, self.pose)
            # 4. Reactive velocity command.
            v_cmd = self.controller.compute_velocity(self.pose)
            # 5. Integrate the robot (forward Euler).
            self.pose = self.pose + v_cmd * self.dt
            # 6. Logging / termination checks.
            self._log_frame(v_cmd)
            if self.true_distance_at_pose() < 0.0:
                break
            if self._at_goal():
                break
        return self._build_result()

    def true_distance_at_pose(self) -> float:
        d, _ = self.env.get_ground_truth_distance_and_gradient(self.pose)
        return d

    def _at_goal(self) -> bool:
        err = self.pose - self.controller.policy.target
        return (
            float(np.linalg.norm(err)) <= self.controller.policy.arrive_radius
        )

    def _build_result(self) -> SimulationResult:
        f = self._frames
        return SimulationResult(
            times=np.asarray(f["times"]),
            poses=np.asarray(f["poses"]),
            velocities=np.asarray(f["velocities"]),
            true_distances=np.asarray(f["true_distances"]),
            estimated_distances=np.asarray(f["estimated_distances"]),
            pointclouds=f["pointclouds"],
            obstacle_states=f["obstacle_states"],
            start_pose=self.start_pose,
            goal=self.controller.policy.target,
            reached_goal=self._at_goal(),
            collided=bool(np.min(f["true_distances"]) < 0.0),
        )

    # -- Visualization ------------------------------------------------------
    def animate(
        self,
        result: SimulationResult,
        field_resolution: int = 15,
        field_refresh: int = 5,
        interval_ms: int = 40,
    ) -> FuncAnimation:
        """Play back a recorded run as a matplotlib animation.

        Shows trajectory, moving obstacles, LiDAR point cloud, robot
        state and the (replayed) field model as distance background +
        quiver plot.  The field is re-evaluated every
        ``field_refresh`` frames.
        """
        # Fields that maintain internal state (e.g. the fused global map)
        # are reset so the replay builds the map from scratch again.
        reset = getattr(self.field_model, "reset", None)
        if callable(reset):
            reset()

        fig, ax = plt.subplots(figsize=(7.5, 7.5))
        ax.set_xlim(*self.env.bounds[0])
        ax.set_ylim(*self.env.bounds[1])
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)
        ax.set_title("Reactive obstacle avoidance (2D testbed)")

        extent = (
            self.env.lo[0],
            self.env.hi[0],
            self.env.lo[1],
            self.env.hi[1],
        )
        im = ax.imshow(
            np.zeros((1, 1)),
            origin="lower",
            extent=extent,
            cmap="viridis",
            alpha=0.35,
            vmin=0.0,
            vmax=self.sensor.max_range,
            aspect="auto",
            zorder=0,
        )
        quivers: List[matplotlib.pyplot.Quiver] = []
        patches: List[matplotlib.patches.Patch] = []
        for state in result.obstacle_states[-1]:
            patch = (
                CirclePatch((0, 0), 1.0, fill=False)
                if state["type"] == "circle"
                else RectanglePatch((0, 0), 2.0, 2.0, fill=False)
            )
            patch.set_edgecolor(
                "tab:red" if state["type"] == "circle" else "tab:purple"
            )
            patch.set_linewidth(1.5)
            ax.add_patch(patch)
            patches.append(patch)
        pc_artist = ax.scatter(
            [], [], s=10, c="tab:orange", zorder=3, label="LiDAR points"
        )
        map_artist = ax.scatter(
            [],
            [],
            s=6,
            c="tab:cyan",
            alpha=0.8,
            zorder=2.5,
            label="global map",
        )
        (traj,) = ax.plot([], [], "b-", lw=1.5, zorder=2, label="trajectory")
        (robot,) = ax.plot([], [], "k^", ms=10, zorder=5, label="robot")
        v_arrow = ax.quiver(
            [],
            [],
            [],
            [],
            color="red",
            angles="xy",
            scale_units="xy",
            scale=1.0,
            zorder=4,
        )
        ax.plot(
            [result.start_pose[0]],
            [result.start_pose[1]],
            "gs",
            ms=9,
            zorder=4,
            label="start",
        )
        ax.plot(
            [result.goal[0]],
            [result.goal[1]],
            "r*",
            ms=14,
            zorder=4,
            label="goal",
        )
        info = ax.text(
            0.02,
            0.02,
            "",
            transform=ax.transAxes,
            family="monospace",
            fontsize=9,
            bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"),
        )
        ax.legend(loc="upper left", fontsize=8)

        def frame(i: int) -> Tuple[matplotlib.artist.Artist, ...]:
            nonlocal v_arrow
            pose = result.poses[i]
            v_cmd = result.velocities[i]
            pc = result.pointclouds[i]
            for patch, state in zip(patches, result.obstacle_states[i]):
                _set_patch(patch, state)
            pc_artist.set_offsets(pc)
            traj.set_data(result.poses[: i + 1, 0], result.poses[: i + 1, 1])
            robot.set_data([pose[0]], [pose[1]])
            v_arrow.remove()
            v_arrow = ax.quiver(
                [pose[0]],
                [pose[1]],
                [v_cmd[0]],
                [v_cmd[1]],
                color="red",
                angles="xy",
                scale_units="xy",
                scale=1.0,
                zorder=4,
            )
            if i % field_refresh == 0:
                self.field_model.update(pc, pose)
                X, Y, D, G = _grid_field(
                    self.field_model, self.env, field_resolution
                )
                im.set_array(D)
                for q in quivers:
                    q.remove()
                quivers.clear()
                quivers.append(
                    ax.quiver(
                        X,
                        Y,
                        G[..., 0],
                        G[..., 1],
                        color="k",
                        alpha=0.6,
                        angles="xy",
                        scale_units="xy",
                        scale=10.0,
                        zorder=1,
                    )
                )
            # Show the accumulated map *after* the update, so the points
            # always match the field background of this frame.
            map_pts = getattr(self.field_model, "X", None)
            map_artist.set_offsets(
                np.asarray(map_pts, dtype=float)
                if map_pts is not None and len(map_pts) > 0
                else np.empty((0, 2))
            )
            info.set_text(
                f"t = {result.times[i]:6.2f} s   "
                f"d_true = {result.true_distances[i]:+6.3f}   "
                f"d_est  = {result.estimated_distances[i]:+6.3f}"
            )
            return (
                *quivers,
                *patches,
                pc_artist,
                map_artist,
                traj,
                robot,
                v_arrow,
                im,
                info,
            )

        anim = FuncAnimation(
            fig,
            frame,
            frames=len(result.poses),
            interval=interval_ms,
            blit=False,
        )
        plt.show()
        return anim

    def plot_post_simulation(
        self,
        result: SimulationResult,
        field_resolution: int = 25,
        save_path: Optional[str] = None,
    ) -> matplotlib.figure.Figure:
        """Static 2x2 figure: trajectory, field quiver, distances, error."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        ax_traj, ax_field, ax_dist, ax_err = axes.ravel()

        # (1) Trajectory with final obstacle configuration.
        ax_traj.set_title("Trajectory")
        self._draw_obstacles(ax_traj, result.obstacle_states[-1])
        ax_traj.plot(result.poses[:, 0], result.poses[:, 1], "b-", lw=1.5)
        final_pc = result.pointclouds[-1]
        if len(final_pc):
            ax_traj.scatter(
                final_pc[:, 0],
                final_pc[:, 1],
                s=8,
                c="tab:orange",
                label="LiDAR",
            )
        ax_traj.plot(
            [result.start_pose[0]], [result.start_pose[1]], "gs", ms=9
        )
        ax_traj.plot([result.goal[0]], [result.goal[1]], "r*", ms=13)
        self._style_axes(ax_traj)

        # (2) Distance field + gradient quiver (final field state).
        ax_field.set_title("Distance field and gradient (quiver)")
        X, Y, D, G = _grid_field(self.field_model, self.env, field_resolution)
        ax_field.imshow(
            D,
            origin="lower",
            extent=(
                self.env.lo[0],
                self.env.hi[0],
                self.env.lo[1],
                self.env.hi[1],
            ),
            cmap="viridis",
            alpha=0.5,
            vmin=0.0,
            vmax=self.sensor.max_range,
        )
        ax_field.quiver(
            X,
            Y,
            G[..., 0],
            G[..., 1],
            color="k",
            alpha=0.7,
            angles="xy",
            scale_units="xy",
            scale=10.0,
        )
        self._draw_obstacles(ax_field, result.obstacle_states[-1])
        map_pts = getattr(self.field_model, "X", None)
        if map_pts is not None and len(map_pts) > 0:
            ax_field.scatter(
                map_pts[:, 0],
                map_pts[:, 1],
                s=6,
                c="tab:cyan",
                zorder=5,
                label="global map",
            )
            ax_field.legend(loc="upper right", fontsize=8)
        ax_field.plot(
            [result.start_pose[0]], [result.start_pose[1]], "gs", ms=9
        )
        ax_field.plot([result.goal[0]], [result.goal[1]], "r*", ms=13)
        ax_field.set_aspect("equal")

        # (3) True vs. estimated distance over time.
        ax_dist.set_title("Distance to nearest obstacle")
        ax_dist.plot(result.times, result.true_distances, "k-", label="true")
        ax_dist.plot(
            result.times, result.estimated_distances, "r--", label="estimated"
        )
        ax_dist.set_xlabel("time [s]")
        ax_dist.set_ylabel("distance")
        ax_dist.legend()
        ax_dist.grid(alpha=0.3)

        # (4) Estimation error over time.
        ax_err.set_title("|d_est - d_true|")
        error = np.abs(result.estimated_distances - result.true_distances)
        ax_err.plot(result.times, error, "m-")
        ax_err.set_xlabel("time [s]")
        ax_err.set_ylabel("abs. error")
        ax_err.grid(alpha=0.3)

        fig.suptitle(
            f"Result: "
            f"{'COLLISION' if result.collided else 'OK'} | "
            f"{'reached goal' if result.reached_goal else 'goal not reached'} "
            f"| min clearance {result.min_clearance:+.3f}",
            fontsize=11,
        )
        fig.tight_layout(rect=[0, 0, 1, 0.97])
        if save_path is not None:
            fig.savefig(save_path, dpi=150)
        return fig

    def _draw_obstacles(
        self,
        ax: matplotlib.axes.Axes,
        states: List[Dict[str, object]],
    ) -> None:
        for state in states:
            patch = (
                CirclePatch(
                    (float(state["center"][0]), float(state["center"][1])),
                    float(state["radius"]),
                    fill=False,
                )
                if state["type"] == "circle"
                else self._box_patch(state)
            )
            patch.set_edgecolor(
                "tab:red" if state["type"] == "circle" else "tab:purple"
            )
            patch.set_linewidth(1.5)
            ax.add_patch(patch)

    @staticmethod
    def _box_patch(state: Dict[str, object]) -> matplotlib.patches.Patch:
        center = state["center"]
        half = state["half_extents"]
        return RectanglePatch(
            (float(center[0] - half[0]), float(center[1] - half[1])),
            float(2.0 * half[0]),
            float(2.0 * half[1]),
            fill=False,
        )

    def _style_axes(self, ax: matplotlib.axes.Axes) -> None:
        ax.set_xlim(*self.env.bounds[0])
        ax.set_ylim(*self.env.bounds[1])
        ax.set_aspect("equal")
        ax.grid(alpha=0.3)


# ---------------------------------------------------------------------------
# Demo scenario / CLI
# ---------------------------------------------------------------------------
def build_demo_scenario() -> Tuple[
    ObstacleEnvironment,
    SensorSimulator,
    NominalPolicy,
    GroundTruthField,
    ReactiveControllerBase,
]:
    """Build the reference scenario (matches the skeleton in the spec).

    Note: the static circle is offset from the exact [5, 5] start-goal
    symmetry line.  With the simple repulsive-vector placeholder, a
    perfectly collinear start/obstacle/goal configuration is a local
    minimum where the repulsive and attractive terms cancel exactly
    and the robot stalls; a small offset lets the demo navigate
    around the circle.  Your own modulation layer can be expected to
    handle the exact-collision case.
    """
    env = ObstacleEnvironment()
    env.add_obstacle(CircleObstacle(center=[4.8, 5.2], radius=1.0))
    env.add_obstacle(
        MovingCircleObstacle(
            initial_center=[3.0, 7.0], velocity=[0.1, -0.05], radius=0.8
        )
    )
    env.add_obstacle(BoxObstacle(center=[7.5, 3.5], half_extents=[0.7, 0.4]))
    sensor = SensorSimulator(max_range=8.0, num_rays=72, seed=42)
    policy = NominalPolicy(target=[9.0, 9.0])
    # Replace with your GP-based DistanceGradientFieldBase for real runs.
    # field_model = GroundTruthField(env)
    # field_model = GPDistanceGradientField(length_scale=0.8, variance=1.0)
    field_model = FusedGPDistanceGradientField(
    length_scale=0.5,
    variance=1.0,
    voxel_size=0.15,       # 15 cm Voxel-Raster für die Kartierungsdichte
    max_sensor_range=6.0,
    max_points=300
)
    controller = ReactiveCirculationHysteresisController(policy, field_model,d_min=0.2,d_max=0.5)
    return env, sensor, policy, field_model, controller


def _interactive_backend() -> bool:
    """True if matplotlib can open an interactive window here."""
    if matplotlib.get_backend().lower() == "agg":
        return False
    return bool(
        os.environ.get("DISPLAY")
        or os.environ.get("WAYLAND_DISPLAY")
        or sys.platform in ("darwin", "win32")
    )


def run_demo(
    num_steps: int = 500,
    dt: float = 0.05,
    animate: bool = True,
    save_path: Optional[str] = None,
) -> SimulationResult:
    """Set up the demo scenario, run the simulation, visualize."""
    env, sensor, policy, field_model, controller = build_demo_scenario()
    engine = SimulationEngine(
        env,
        sensor,
        field_model,
        controller,
        start_pose=[1.0, 1.0],
        dt=dt,
        num_steps=num_steps,
    )
    result = engine.run()
    print(f"steps executed : {len(result.times) - 1}")
    print(f"final pose     : {result.final_pose.round(3)}")
    print(
        f"dist. to goal  : "
        f"{float(np.linalg.norm(result.final_pose - result.goal)):.3f}"
    )
    print(f"min clearance  : {result.min_clearance:+.3f}")
    print(f"collided       : {result.collided}")
    print(f"reached goal   : {result.reached_goal}")
    if animate and _interactive_backend():
        engine.animate(result, field_resolution=50, field_refresh=5)
    else:
        print("Plotting post-simulation figure ...")
        fig = engine.plot_post_simulation(result)
        if save_path is not None:
            fig.savefig(save_path, dpi=150)
            print(f"Plot saved to    : {save_path}")
        else:
            plt.show()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="2D reactive obstacle avoidance testbed"
    )
    parser.add_argument(
        "--steps", type=int, default=500, help="number of simulation steps"
    )
    parser.add_argument(
        "--dt", type=float, default=0.05, help="integration time step [s]"
    )
    parser.add_argument(
        "--no-anim", action="store_true", help="skip the live animation"
    )
    parser.add_argument(
        "--save-plot",
        default=None,
        metavar="PNG",
        help="save the post-simulation figure to a file",
    )
    args = parser.parse_args()
    run_demo(
        num_steps=args.steps,
        dt=args.dt,
        animate=not args.no_anim,
        save_path=args.save_plot,
    )


if __name__ == "__main__":
    main()
