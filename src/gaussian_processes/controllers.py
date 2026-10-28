from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np
from typing import Dict, List, Optional, Sequence, Tuple
from gaussian_processes.distance_gradient_fields import DistanceGradientFieldBase


# ---------------------------------------------------------------------------
# Nominal policy
# ---------------------------------------------------------------------------
class NominalPolicy:
    """Obstacle-free velocity controller that drives toward a target.

    Proportional attractor with saturation::

        v_nom = -v_max * (x - x_goal) / max(||x - x_goal||, r_arr)

    The velocity is zero once the robot is within ``arrive_radius`` of
    the target.
    """

    def __init__(
        self,
        target: Sequence[float],
        v_max: float = 1.0,
        arrive_radius: float = 0.15,
    ) -> None:
        self.target = np.asarray(target, dtype=float)
        self.v_max = float(v_max)
        self.arrive_radius = float(arrive_radius)

    def compute_velocity(self, pose: Sequence[float]) -> np.ndarray:
        """Nominal velocity at ``pose`` (saturated to ``v_max``)."""
        p = np.asarray(pose, dtype=float)
        err = self.target - p
        dist = float(np.linalg.norm(err))
        if dist <= self.arrive_radius:
            return np.zeros(2, dtype=float)
        return self.v_max * err / max(dist, self.arrive_radius)

class Controllerbase(ABC):
    """Abstract interface for reactive controllers.

    Subclass this to integrate your own controller (e.g. projection /
    CBF-based).  The simulation engine calls :meth:`compute_velocity`
    once per time step with the current robot pose.
    """

    @abstractmethod
    def compute_velocity(self, pose: Sequence[float]) -> np.ndarray:
        """Compute a velocity command at ``pose`` (saturated to ``v_max``)."""

# ---------------------------------------------------------------------------
# Reactive controller
# ---------------------------------------------------------------------------
class ReactiveController(Controllerbase):
    """Blends the nominal velocity with a repulsive field term.

    Placeholder modulation model::

        v_res = w(d) * v_max * grad d / ||grad d|| + (1 - w(d)) * v_nom

    where ``w(d)`` is a linear distance weight that is 1 for
    ``d <= d_min`` (fully repulsive) and 0 for ``d >= d_max`` (fully
    nominal).  Replace :meth:`compute_velocity` with your own
    modulation law (e.g. projection / CBF-based) for real experiments.
    """

    def __init__(
        self,
        policy: NominalPolicy,
        field_model: DistanceGradientFieldBase,
        d_min: float = 0.4,
        d_max: float = 1.2,
    ) -> None:
        self.policy = policy
        self.field_model = field_model
        self.d_min = float(d_min)
        self.d_max = float(d_max)

    def _weight(self, d: float) -> float:
        """Distance-based blending weight ``w(d) in [0, 1]``."""
        if d <= self.d_min:
            return 1.0
        if d >= self.d_max:
            return 0.0
        return (self.d_max - d) / (self.d_max - self.d_min)

    def compute_velocity(self, pose: Sequence[float]) -> np.ndarray:
        """Reactive velocity command at ``pose`` (saturated to ``v_max``)."""
        v_nom = self.policy.compute_velocity(pose)
        d, grad = self.field_model.query(pose)
        w = self._weight(d)
        if w == 0.0:
            v = v_nom
        else:
            g_norm = float(np.linalg.norm(grad))
            grad_dir = grad / g_norm if g_norm > 1e-12 else np.zeros(2)
            # grad d is a (unit) repulsive normal; scale to v_max so both
            # terms have comparable magnitude.
            v_rep = self.policy.v_max * grad_dir
            v = w * v_rep + (1.0 - w) * v_nom
        speed = float(np.linalg.norm(v))
        if speed > self.policy.v_max:
            v = v * (self.policy.v_max / speed)
        return v


# ---------------------------------------------------------------------------
# Reactive controller
# ---------------------------------------------------------------------------
class ReactiveCirculationController(Controllerbase):
    """Blends the nominal velocity with a repulsive field term.

    Placeholder modulation model::

        v_res = w(d) * v_max * grad d / ||grad d|| + (1 - w(d)) * v_nom

    where ``w(d)`` is a linear distance weight that is 1 for
    ``d <= d_min`` (fully repulsive) and 0 for ``d >= d_max`` (fully
    nominal).  Replace :meth:`compute_velocity` with your own
    modulation law (e.g. projection / CBF-based) for real experiments.
    """

    def __init__(
        self,
        policy: NominalPolicy,
        field_model: DistanceGradientFieldBase,
        d_min: float = 0.4,
        d_max: float = 1.2,
    ) -> None:
        self.policy = policy
        self.field_model = field_model
        self.d_min = float(d_min)
        self.d_max = float(d_max)

    def _weight(self, d: float) -> float:
        """Distance-based blending weight ``w(d) in [0, 1]``."""
        if d <= self.d_min:
            return 1.0
        if d >= self.d_max:
            return 0.0
        return (self.d_max - d) / (self.d_max - self.d_min)

    def compute_velocity(self, pose: Sequence[float]) -> np.ndarray:
        """Reactive velocity command at ``pose`` (saturated to ``v_max``)."""
        v_nom = self.policy.compute_velocity(pose)
        d, grad = self.field_model.query(pose)
        w = self._weight(d)
        if w == 0.0:
            v = v_nom
        else:
            g_norm = float(np.linalg.norm(grad))
            grad_dir = grad / g_norm if g_norm > 1e-12 else np.zeros(2)

            # tangential direction left:
            tan_dir_left = np.array([-grad_dir[1], grad_dir[0]])

            # choose left or right based on the angle to the nominal velocity
            if np.dot(tan_dir_left, v_nom) >= 0:
                tan_dir = tan_dir_left
            else:
                tan_dir = -tan_dir_left

            # grad d is a (unit) repulsive normal; scale to v_max so both
            # terms have comparable magnitude.
            v_rep = self.policy.v_max * grad_dir
            v_tan = self.policy.v_max * tan_dir
            # Nahe am Hindernis dominiert die Zirkulation (z. B. 70% t + 30% n)
            v_avoid = 0.3 * v_rep + 0.7 * v_tan
            # Endgültige Geschwindigkeitsmischung mit der Ziel-Policy
            v = (1.0 - w) * v_nom + w * v_avoid
        speed = float(np.linalg.norm(v))
        if speed > self.policy.v_max:
            v = v * (self.policy.v_max / speed)
        return v


class ReactiveCirculationHysteresisController:
    def __init__(
        self,
        policy,
        field_model,
        d_min: float = 0.2,
        d_max: float = 1.0,
    ) -> None:
        self.policy = policy
        self.field_model = field_model
        self.d_min = float(d_min)
        self.d_max = float(d_max)

        # Hysterese-Zustand: None = freie Fahrt, +1 = linksherum, -1 = rechtsherum
        self.circ_direction = None

    def _weight(self, d: float) -> float:
        if d <= self.d_min:
            return 1.0
        if d >= self.d_max:
            return 0.0
        return (self.d_max - d) / (self.d_max - self.d_min)

    def compute_velocity(self, pose: Sequence[float]) -> np.ndarray:
        v_nom = self.policy.compute_velocity(pose)
        d, grad = self.field_model.query(pose)
        g_norm = float(np.linalg.norm(grad))

        # 1. Wenn der Abstand wieder groß genug ist, Zustand zurücksetzen
        if g_norm < 1e-12 or d >= self.d_max:
            self.circ_direction = None  # Speicher löschen für das nächste Hindernis
            return v_nom

        # Normalenvektor n (vom Hindernis weg)
        n = grad / g_norm

        # Basis-Tangentialvektor t_left (90° nach links gedreht)
        t_left = np.array([-n[1], n[0]])

        # 2. Richtungsentscheidung NUR TREFFEN, WENN DIESES HINDERNIS NEU IST
        if self.circ_direction is None:
            if np.dot(v_nom, t_left) >= 0:
                self.circ_direction = 1.0   # Links vorbeisteuern
                print("set circulation direction to left")
            else:
                self.circ_direction = -1.0  # Rechts vorbeisteuern
                print("set circulation direction to right")

        # 3. Den einmal gewählten Tangentialvektor beibehalten
        t = t_left * self.circ_direction

        # 4. Vektormischung durchführen
        w = self._weight(d)
        v_rep = self.policy.v_max * n
        v_circ = self.policy.v_max * t

        v_avoid = 0.3 * v_rep + 0.7 * v_circ
        v_res = (1.0 - w) * v_nom + w * v_avoid

        # Sättigung auf v_max
        speed = float(np.linalg.norm(v_res))
        if speed > self.policy.v_max:
            v_res = v_res * (self.policy.v_max / speed)

        return v_res