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

    def reset(self) -> None:
        """Leert das Modell (z. B. vor dem Animation-Playback)."""
        self.X = np.empty((0, 2))
        self.alpha = np.empty((0,))

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


class FusedGPDistanceGradientField(DistanceGradientFieldBase):
    """Inkrementelles Fused GP Distance and Gradient Field nach IDMP (Ali et al., 2024).

    Baut kontinuierlich eine globale Karte aus LiDAR-Punktwolken auf,
    verdichtet sie über ein Voxel-Gitter und bereinigt bewegte/entfernte Hindernisse
    über Free-Space / Ray Clearing.
    """

    def __init__(
        self,
        length_scale: float = 0.5,
        variance: float = 1.0,
        noise_variance: float = 1e-3,
        voxel_size: float = 0.15,      # Räumliche Voxel-Auflösung der Karte (15 cm)
        max_sensor_range: float = 6.0,  # Maximale Reichweite des LiDARs
        max_points: int = 300,          # Bounded Size für den GP
    ) -> None:
        self.l = float(length_scale)
        self.sigma2 = float(variance)
        self.sigma2_o = float(noise_variance)
        self.voxel_size = float(voxel_size)
        self.max_sensor_range = float(max_sensor_range)
        self.max_points = int(max_points)

        # Globale Trainingspunkte des Fused Fields und Gewichtsvektor alpha
        self.X: np.ndarray = np.empty((0, 2))
        self.alpha: np.ndarray = np.empty((0,))

    def reset(self) -> None:
        """Leert die globale Karte und den GP-Zustand (z. B. vor dem Animation-Playback)."""
        self.X = np.empty((0, 2))
        self.alpha = np.empty((0,))

    def _voxel_downsample(self, points: np.ndarray) -> np.ndarray:
        """Konsolidiert nahe beieinander liegende Punkte in einem Voxel-Raster."""
        if len(points) == 0:
            return points

        # Diskrete Voxel-Koordinaten berechnen
        grid_coords = np.floor(points / self.voxel_size).astype(int)
        unique_voxels, inverse_indices = np.unique(grid_coords, axis=0, return_inverse=True)

        downsampled = np.zeros((len(unique_voxels), 2))
        counts = np.zeros(len(unique_voxels))

        for idx, point in zip(inverse_indices, points):
            downsampled[idx] += point
            counts[idx] += 1

        return downsampled / counts[:, None]

    def update(self, pointcloud: np.ndarray, robot_pose: np.ndarray) -> None:
        """Inkrementelles Fused Field Update:
        1. Ray-Clearing: Entfernt alte Kartenpunkte im Sichtbereich, wo jetzt freie Sicht ist.
        2. Fusion: Zusammenführung neuer LiDAR-Messungen mit der bestehenden Karte.
        3. Voxel-Downsampling & Berechnung des GP-Gewichtsvektors alpha.
        """
        robot_pose = np.asarray(robot_pose, dtype=float)

        # 1. Free-Space / Ray Clearing (für dynamische Hindernisse)
        if len(self.X) > 0 and len(pointcloud) > 0:
            vecs_to_map = self.X - robot_pose
            dists_to_map = np.linalg.norm(vecs_to_map, axis=1)

            # Nur Punkte im aktuellen Sensor-Sichtbereich betrachten
            in_frustum_mask = dists_to_map <= (self.max_sensor_range - 0.2)
            keep_mask = np.ones(len(self.X), dtype=bool)

            pc_vecs = pointcloud - robot_pose
            pc_dists = np.linalg.norm(pc_vecs, axis=1)
            pc_dirs = pc_vecs / (pc_dists[:, None] + 1e-9)

            for i, map_pt in enumerate(self.X):
                if not in_frustum_mask[i]:
                    continue

                map_dist = dists_to_map[i]
                map_dir = vecs_to_map[i] / (map_dist + 1e-9)

                # Prüfen, ob ein aktueller LiDAR-Strahl in gleicher Richtung weiter fliegt als der alte Punkt
                cos_sims = np.dot(pc_dirs, map_dir)
                best_match_idx = np.argmax(cos_sims)

                if cos_sims[best_match_idx] > 0.98 and pc_dists[best_match_idx] > map_dist + 0.3:
                    keep_mask[i] = False  # Alter Punkt wird gelöscht (Hindernis existiert nicht mehr)

            self.X = self.X[keep_mask]

        # 2. Fusion: Neue Punkte zur Karte hinzufügen
        if len(pointcloud) > 0:
            if len(self.X) == 0:
                combined_pts = pointcloud
            else:
                combined_pts = np.vstack([self.X, pointcloud])

            # Voxel Downsampling durchführen
            self.X = self._voxel_downsample(combined_pts)

        # 3. Maximalanzahl Punkte begrenzen (Punkte nahe am Roboter priorisieren)
        if len(self.X) > self.max_points:
            dists = np.linalg.norm(self.X - robot_pose, axis=1)
            sorted_indices = np.argsort(dists)
            self.X = self.X[sorted_indices[:self.max_points]]

        if len(self.X) == 0:
            self.alpha = np.empty((0,))
            return

        # 4. GP-Gewichtsvektor alpha für das akkumulierte Fused Field berechnen
        N = len(self.X)
        K_XX = rbf_kernel(self.X, self.X, length_scale=self.l, variance=self.sigma2)
        A = K_XX + self.sigma2_o * np.eye(N)
        ones = np.ones(N)

        try:
            L = np.linalg.cholesky(A)
            self.alpha = np.linalg.solve(L.T, np.linalg.solve(L, ones))
        except np.linalg.LinAlgError:
            self.alpha = np.linalg.solve(A, ones)

    def query(self, x: Sequence[float]) -> Tuple[float, np.ndarray]:
        """Liest Distanz und Gradienten aus dem globalen Fused Field aus."""
        if len(self.X) == 0:
            return 10.0, np.zeros(2)

        x_query = np.asarray(x, dtype=float)
        diff = x_query - self.X

        k_xX = rbf_kernel(x_query[None, :], self.X, length_scale=self.l, variance=self.sigma2)[0]
        o_hat = float(np.dot(k_xX, self.alpha))

        o_clipped = np.clip(o_hat, 1e-8, self.sigma2)
        d_hat = float(np.sqrt(-2.0 * (self.l**2) * np.log(o_clipped / self.sigma2)))

        grad_k = -(diff / (self.l**2)) * k_xX[:, None]
        grad_o = np.dot(self.alpha, grad_k)

        grad_o_norm = np.linalg.norm(grad_o)
        if grad_o_norm > 1e-12:
            grad_d = -grad_o / grad_o_norm
        else:
            grad_d = np.zeros(2)

        return d_hat, grad_d
