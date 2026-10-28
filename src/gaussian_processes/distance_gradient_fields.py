import numpy as np
from typing import Sequence, Tuple
from abc import ABC, abstractmethod

from gaussian_processes.kernels import rbf_kernel



# ---------------------------------------------------------------------------
# Distance / gradient field interface
# ---------------------------------------------------------------------------
class DistanceGradientFieldBase(ABC):
    """Abstract interface for distance field estimators.

    Subclass this to integrate your own estimator (GP distance field,
    IDMP, ...).  The simulation engine calls :meth:`update` once per
    time step with the newest LiDAR measurement, and then queries the
    field via :meth:`query`.
    """

    @abstractmethod
    def update(self, pointcloud: np.ndarray, robot_pose: np.ndarray) -> None:
        """Ingest a new LiDAR point cloud (shape (N, 2)).

        Parameters
        ----------
        pointcloud:
            Current 2D measurement points.
        robot_pose:
            Robot position when the measurement was taken, (2,).
        """

    @abstractmethod
    def query(self, x: Sequence[float]) -> Tuple[float, np.ndarray]:
        """Return ``(d(x), grad d(x))`` at a single 2D position ``x``."""


class GPDistanceGradientField(DistanceGradientFieldBase):
    """Gaussian Process Distance and Gradient Field implementation

    Based on 'Interactive Distance Field Mapping and Planning' (Ali et al., 2024).
    Uses a Reverting Log-GP model with an RBF kernel and analytical derivatives.
    """

    def __init__(
        self,
        length_scale: float = 0.8,
        variance: float = 1.0,
        noise_variance: float = 1e-3,
        max_points: int = 200,
    ) -> None:
        """
        Parameters
        ----------
        length_scale : l (Kernel Längenskala)
        variance : sigma^2 (Kernel Signalvarianz)
        noise_variance : sigma_o^2 (Beobachtungsrauschen)
        max_points : Maximale Anzahl Punkte für Online-Performance (z. B. Subsampling/Downsampling)
        """
        self.l = float(length_scale)
        self.sigma2 = float(variance)
        self.sigma2_o = float(noise_variance)
        self.max_points = int(max_points)

        self.X: np.ndarray = np.empty((0, 2))  # Trainingspunkte (Punktwolke)
        self.alpha: np.ndarray = np.empty((0,)) # Gewichtsvektor (K_XX + sigma_o^2 I)^(-1) * 1

    def update(self, pointcloud: np.ndarray, robot_pose: np.ndarray) -> None:
        """Nimmt eine neue Punktwolke auf und berechnet den GP-Gewichtsvektor alpha."""
        del robot_pose  # Bei einfachem Reverting-GP werden die Punkte direkt verwendet
        
        if len(pointcloud) == 0:
            self.X = np.empty((0, 2))
            self.alpha = np.empty((0,))
            return

        # Optional: Subsampling, falls zu viele LiDAR-Punkte vorhanden sind (Performance)
        if len(pointcloud) > self.max_points:
            indices = np.random.choice(len(pointcloud), self.max_points, replace=False)
            self.X = pointcloud[indices].copy()
        else:
            self.X = pointcloud.copy()

        N = len(self.X)
        
        # 1. RBF Covariance Matrix K_XX berechnen
        K_XX = rbf_kernel(self.X,self.X,length_scale=self.l,variance=self.sigma2)


        # 2. alpha = (K_XX + sigma_o^2 * I)^(-1) * 1
        A = K_XX + self.sigma2_o * np.eye(N)
        ones = np.ones(N)
        
        # Cholesky-Zerlegung für numerische Stabilität und Schnelligkeit
        try:
            L = np.linalg.cholesky(A)
            self.alpha = np.linalg.solve(L.T, np.linalg.solve(L, ones))
        except np.linalg.LinAlgError:
            self.alpha = np.linalg.solve(A, ones)

    def query(self, x: Sequence[float]) -> Tuple[float, np.ndarray]:
        """Berechnet (d, grad_d) an Position x = [x, y]."""
        if len(self.X) == 0:
            # Kein Hindernis im Sichtfeld -> große Distanz, Null-Gradient
            return 10.0, np.zeros(2)

        x_query = np.asarray(x, dtype=float)  # Shape: (2,)

        # Distanzen zu allen Trainingspunkten: (x_* - x_j)
        diff = x_query - self.X               # Shape: (N, 2)

        # Kernel-Vektor k(x_*, X)
        k_xX = rbf_kernel(x_query[None, :],self.X,length_scale=self.l,variance=self.sigma2)[0]

        # 1. Occupancy-Wert o_hat(x_*)
        o_hat = float(np.dot(k_xX, self.alpha))

        # 2. Reverting Function für Distanz d_hat
        # Numerischer Schutz: o_hat muss im Bereich (0, sigma2] liegen
        o_clipped = np.clip(o_hat, 1e-8, self.sigma2)
        d_hat = float(np.sqrt(-2.0 * (self.l**2) * np.log(o_clipped / self.sigma2)))

        # 3. Analytischer Gradient der Occupancy Nabla o_hat(x_*)
        # dk/dx_* = -(x_* - x_j) / l^2 * k(x_*, x_j)
        grad_k = -(diff / (self.l**2)) * k_xX[:, None]  # Shape: (N, 2)
        grad_o = np.dot(self.alpha, grad_k)             # Shape: (2,)

        # Gradient der Distanz grad_d zeige vom Hindernis WEG (repulsiv)
        # Da o_hat nahe am Hindernis hoch ist, zeigt grad_o ZUM Hindernis.
        # Daher kehren wir die Richtung um:
        grad_o_norm = np.linalg.norm(grad_o)
        if grad_o_norm > 1e-12:
            grad_d = -grad_o / grad_o_norm  # Normierter Vektor vom Hindernis weg
        else:
            grad_d = np.zeros(2)

        return d_hat, grad_d




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
