from abc import ABC, abstractmethod
import numpy as np
from typing import Optional, Sequence
from gaussian_processes.distance_gradient_fields import DistanceGradientFieldBase
from gaussian_processes.qp import solve_qp


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


class ReactiveCirculationHysteresisController(Controllerbase):
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


# ---------------------------------------------------------------------------
# Shared circulation guidance (used by the CBF and MPC controllers)
# ---------------------------------------------------------------------------
class CirculationGuidance:
    """Adds a tangential "swirl" around obstacles to a reference velocity.

    Safety filters (CBF) and constraint-based MPC only restrict the
    *normal* velocity component near an obstacle, so a reference that
    also has a tangential component slides around it instead of
    stalling::

        v_ref = v_nom + w(d) * gain * v_max * s * t_left

    where ``t_left`` is the obstacle normal rotated by +90 deg and
    ``s = +-1`` is the circulation direction.  ``s`` is chosen once, when
    an obstacle comes within ``d_max`` (sign of ``v_nom . t_left``), and
    is kept until the robot is farther than ``d_max`` again
    (hysteresis, as in ``ReactiveCirculationHysteresisController``).
    The weight ``w`` is 1 for ``d <= d_min`` and 0 for ``d >= d_max``.
    ``gain`` should be about 1 so the swirl beats the lateral pull of
    the nominal policy back towards the goal line.
    """

    def __init__(
        self, d_min: float = 0.2, d_max: float = 0.8, gain: float = 1.0
    ) -> None:
        self.d_min = float(d_min)
        self.d_max = float(d_max)
        self.gain = float(gain)
        self.direction: Optional[float] = None

    def reset(self) -> None:
        self.direction = None

    def _weight(self, d: float) -> float:
        if d <= self.d_min:
            return 1.0
        if d >= self.d_max:
            return 0.0
        return (self.d_max - d) / (self.d_max - self.d_min)

    def reference(self, v_nom, d, grad, v_max, direction=None):
        """Pure function: ``(v_ref, direction_used)``; no state is changed.

        ``direction_used`` is None when no obstacle is in range.
        """
        g_norm = float(np.linalg.norm(grad))
        if g_norm < 1e-9 or d >= self.d_max:
            return v_nom, None
        n = np.asarray(grad, dtype=float) / g_norm
        t_left = np.array([-n[1], n[0]])
        if direction is None:
            direction = 1.0 if float(np.dot(v_nom, t_left)) >= 0.0 else -1.0
        w = self._weight(float(d))
        return v_nom + w * self.gain * v_max * direction * t_left, direction

    def step(self, v_nom, d, grad, v_max) -> np.ndarray:
        """Latching version: decide / keep / clear the direction."""
        v_ref, used = self.reference(v_nom, d, grad, v_max, self.direction)
        self.direction = used
        return v_ref


# ---------------------------------------------------------------------------
# CBF + QP safety-filter controller
# ---------------------------------------------------------------------------
class CBFQPController(Controllerbase):
    """Safety filter that solves a control-barrier-function QP.

    The robot dynamics in the testbed are ``pose_dot = u`` (velocity
    command applied directly), i.e. affine with ``f = 0`` and
    ``g = I``.  With the safety function ``h(x) = d(x) - d_min``
    (``d`` the estimated distance to the nearest obstacle) the CBF
    condition for the linear gain ``alpha(h) = gamma * h`` is

        h_dot + gamma * h = grad_d^T u + gamma * (d - d_min) >= 0,

    which is linear in the input ``u``.  The controller therefore
    solves the minimum-modification QP (safety filter)

        min_u  ||u - u_nom||^2
        s.t.   grad_d^T u + gamma * (d - d_min) >= 0   (CBF)
               -v_max <= u_i <= v_max                   (box input limits)

    i.e. it tracks the nominal (goal-seeking) velocity as closely as
    possible while keeping the set ``{d >= d_min}`` forward invariant.
    Far from obstacles the CBF row is inactive and the nominal velocity
    is returned unchanged.  The QP is solved with OSQP via
    :mod:`qpsolvers` (see :mod:`gaussian_processes.qp`); if the
    problem is infeasible (e.g. deep inside an obstacle with a flat
    field) the nominal velocity is returned.
    """

    def __init__(
        self,
        policy: NominalPolicy,
        field_model: DistanceGradientFieldBase,
        d_min: float = 0.2,
        gamma: float = 1.0,
        circulation: Optional["CirculationGuidance"] = None,
    ) -> None:
        self.policy = policy
        self.field_model = field_model
        self.d_min = float(d_min)
        self.gamma = float(gamma)
        self.v_max = float(policy.v_max)
        # Optional tangential swirl added to the nominal reference, so the
        # filter slides around obstacles instead of stalling in front.
        self.circulation = circulation

    def compute_velocity(self, pose: Sequence[float]) -> np.ndarray:
        """CBF-QP velocity command at ``pose`` (bounded by ``v_max``)."""
        v_nom = self.policy.compute_velocity(pose)
        d, grad = self.field_model.query(pose)
        g_norm = float(np.linalg.norm(grad))
        v_ref = v_nom
        if self.circulation is not None:
            v_ref = self.circulation.step(v_nom, d, grad, self.v_max)
        if g_norm < 1e-9:
            # Flat field (no nearby obstacle): no useful barrier,
            # so just follow the nominal policy.
            return v_nom

        # Objective: ||u - v_ref||^2 = 0.5 u^T (2 I) u - (2 v_ref)^T u.
        G = 2.0 * np.eye(2)
        q = -2.0 * v_ref

        # Constraints A u <= b.
        # CBF: grad_d^T u + gamma*(d - d_min) >= 0
        #      -> -grad_d^T u <= gamma*(d - d_min)
        cbf_row = -grad
        cbf_rhs = self.gamma * (d - self.d_min)
        v_max = self.v_max
        A = np.vstack([cbf_row, np.eye(2), -np.eye(2)])
        b = np.array(
            [cbf_rhs, v_max, v_max, v_max, v_max], dtype=float
        )

        v = solve_qp(G, q, A, b)
        if v is None or not np.all(np.isfinite(v)):
            # Infeasible or solver failure: fall back to the nominal.
            return v_nom

        # Final speed saturation, consistent with the other controllers.
        speed = float(np.linalg.norm(v))
        if speed > v_max:
            v = v * (v_max / speed)
        return v


# ---------------------------------------------------------------------------
# Model predictive controller (sequential QP)
# ---------------------------------------------------------------------------
class MPCController(Controllerbase):
    """Receding-horizon controller with linearised obstacle constraints.

    The robot is a single integrator, ``x_{k+1} = x_k + dt * u_k``, so
    the predicted positions are affine in the stacked input
    ``u = [u_0, ..., u_{N-1}]``::

        x_k = x_0 + dt * (u_0 + ... + u_{k-1})

    Every control step solves

        min_{u, s}  sum_{k=1..N} w_k ||x_k - x_goal||^2    (goal tracking,
                                                            w_N = w_terminal)
                  + w_u  sum_j ||u_j||^2                    (control effort)
                  + w_s  sum_j ||u_j - u_{j-1}||^2          (smoothness,
                                                            u_{-1} = last command)
                  + w_slack * ||s||^2                       (constraint slack)
        s.t.   d(xbar_k) + grad_d(xbar_k)^T (x_k - xbar_k) - d_min + s_k >= 0
               -v_max <= u_j,i <= v_max,    s_k >= 0

    The distance field is only available through point queries, so the
    obstacle constraint is linearised around a reference trajectory
    ``xbar``.  The reference starts as the (shifted) previous solution
    and is refined for ``sqp_iters`` iterations (sequential QP).  The
    slack ``s`` keeps the QP feasible when the robot starts inside the
    safety margin; its large weight makes violations very expensive.
    Only the first input ``u_0`` is applied, the rest warm-starts the
    next call.

    Notes
    -----
    * The field is assumed static over the horizon (moving obstacles are
      only handled through re-planning every step).
    * Like the CBF filter, this is a local method: a goal exactly behind a
      symmetric obstacle can still produce a stall.
    * Cost per call: ``horizon * sqp_iters`` field queries plus
      ``sqp_iters`` QP solves with ``2 * horizon + (#active rows)``
      variables.
    """

    def __init__(
        self,
        policy: NominalPolicy,
        field_model: DistanceGradientFieldBase,
        dt: float = 0.05,
        horizon: int = 15,
        d_min: float = 0.2,
        w_goal: float = 1.0,
        w_terminal: float = 10.0,
        w_control: float = 0.01,
        w_smooth: float = 0.05,
        w_slack: float = 1.0e5,
        sqp_iters: int = 2,
        circulation: Optional["CirculationGuidance"] = None,
        w_vel: float = 5.0,
    ) -> None:
        if horizon < 1:
            raise ValueError("horizon must be >= 1")
        if dt <= 0.0:
            raise ValueError("dt must be > 0")
        self.policy = policy
        self.field_model = field_model
        self.dt = float(dt)
        self.horizon = int(horizon)
        self.d_min = float(d_min)
        self.w_goal = float(w_goal)
        self.w_terminal = float(w_terminal)
        self.w_control = float(w_control)
        self.w_smooth = float(w_smooth)
        self.w_slack = float(w_slack)
        self.sqp_iters = max(1, int(sqp_iters))
        self.v_max = float(policy.v_max)
        # Optional swirl: adds  w_vel * sum_j ||u_j - v_ref_j||^2  to the
        # cost, with v_ref_j the circulation-augmented nominal velocity
        # evaluated along the reference trajectory.
        self.circulation = circulation
        self.w_vel = float(w_vel)

        # Warm start / smoothness state.
        self._u_warm: Optional[np.ndarray] = None  # (N, 2)
        self._u_last = np.zeros(2, dtype=float)

        # Constant matrices: P_k @ u = sum_{j<k} u_j (without dt), k = 1..N.
        n = self.horizon
        self._P = []
        for k in range(1, n + 1):
            P = np.zeros((2, 2 * n))
            for j in range(k):
                P[:, 2 * j : 2 * j + 2] = np.eye(2)
            self._P.append(self.dt * P)
        # Difference operator: (D u)_j = u_j - u_{j-1}, j = 1..N-1,
        # and (D u)_0 = u_0 (the last command is subtracted separately).
        D = np.eye(2 * n)
        for j in range(1, n):
            D[2 * j : 2 * j + 2, 2 * (j - 1) : 2 * (j - 1) + 2] = -np.eye(2)
        self._D = D

    # -- helpers -------------------------------------------------------------
    def reset(self) -> None:
        """Forget the warm start (e.g. when a new run begins)."""
        self._u_warm = None
        self._u_last = np.zeros(2, dtype=float)

    def _rollout(self, x0: np.ndarray, U: np.ndarray) -> np.ndarray:
        """Predicted positions x_1..x_N (shape (N, 2)) for inputs ``U``."""
        return x0 + self.dt * np.cumsum(U, axis=0)

    def _initial_guess(self, x0: np.ndarray) -> np.ndarray:
        """Shifted previous solution, or a nominal-policy rollout."""
        if self._u_warm is not None:
            return np.vstack([self._u_warm[1:], self._u_warm[-1:]])
        U = np.zeros((self.horizon, 2))
        x = x0.copy()
        for k in range(self.horizon):
            U[k] = self.policy.compute_velocity(x)
            x = x + self.dt * U[k]
        return U

    # -- control law -----------------------------------------------------------
    def compute_velocity(self, pose: Sequence[float]) -> np.ndarray:
        """First input of the optimised sequence (bounded by ``v_max``)."""
        x0 = np.asarray(pose, dtype=float)
        v_nom = self.policy.compute_velocity(x0)
        if not np.any(v_nom):
            # Arrived at the goal: stop and drop the stale plan.
            self.reset()
            return v_nom

        n = self.horizon
        target = self.policy.target
        U_bar = self._initial_guess(x0)

        # --- cost (independent of the linearisation) --------------------------
        H = 2.0 * self.w_control * np.eye(2 * n)
        q = np.zeros(2 * n)
        for k, P in enumerate(self._P):
            w = self.w_terminal if k == n - 1 else self.w_goal
            H += 2.0 * w * P.T @ P
            q += 2.0 * w * P.T @ (x0 - target)
        c = np.zeros(2 * n)
        c[:2] = self._u_last
        H += 2.0 * self.w_smooth * self._D.T @ self._D
        q += -2.0 * self.w_smooth * self._D.T @ c

        # Input box: -v_max <= u <= v_max.
        box_A = np.vstack([np.eye(2 * n), -np.eye(2 * n)])
        box_b = np.full(4 * n, self.v_max)

        U_sol = None
        for _ in range(self.sqp_iters):
            X_bar = self._rollout(x0, U_bar)

            # Swirl reference: v_ref_0 at x0 (latching), v_ref_j at x_j.
            H_it, q_it = H, q
            if self.circulation is not None:
                H_it = H + 2.0 * self.w_vel * np.eye(2 * n)
                q_it = q.copy()
                d0, g0 = self.field_model.query(x0)
                v_ref = self.circulation.step(v_nom, float(d0), g0, self.v_max)
                q_it[:2] += -2.0 * self.w_vel * v_ref
                dir_h = self.circulation.direction

            rows, rhs = [], []
            for k in range(n):
                d, grad = self.field_model.query(X_bar[k])
                d = float(d)
                grad = np.asarray(grad, dtype=float)
                if self.circulation is not None and k < n - 1:
                    v_k, used = self.circulation.reference(
                        self.policy.compute_velocity(X_bar[k]),
                        d, grad, self.v_max, dir_h,
                    )
                    if dir_h is None:
                        dir_h = used
                    q_it[2 * (k + 1) : 2 * (k + 2)] += -2.0 * self.w_vel * v_k
                if float(np.linalg.norm(grad)) < 1e-9:
                    continue  # flat field: no obstacle information
                # The plan can move x_k by at most 2 v_max dt (k+1) from
                # xbar_k; farther obstacles cannot become active.
                if d - self.d_min > 2.0 * self.v_max * self.dt * (k + 1) + 0.5:
                    continue
                rows.append(-grad @ self._P[k])
                rhs.append(d - self.d_min + grad @ (x0 - X_bar[k]))

            m = len(rows)
            # Decision vector z = [u (2N); s (m)].
            Hz = np.zeros((2 * n + m, 2 * n + m))
            Hz[: 2 * n, : 2 * n] = H_it
            Hz[2 * n :, 2 * n :] = 2.0 * self.w_slack * np.eye(m)
            qz = np.concatenate([q_it, np.zeros(m)])

            A = np.zeros((4 * n + 2 * m, 2 * n + m))
            b = np.zeros(4 * n + 2 * m)
            A[: 4 * n, : 2 * n] = box_A
            b[: 4 * n] = box_b
            if m:
                A[4 * n : 4 * n + m, : 2 * n] = np.asarray(rows)
                A[4 * n : 4 * n + m, 2 * n :] = -np.eye(m)
                b[4 * n : 4 * n + m] = np.asarray(rhs)
                A[4 * n + m :, 2 * n :] = -np.eye(m)  # s >= 0 (b = 0)

            z = solve_qp(Hz, qz, A, b)
            if z is None or not np.all(np.isfinite(z)):
                break
            U_sol = np.asarray(z[: 2 * n], dtype=float).reshape(n, 2)
            U_bar = U_sol

        if U_sol is None:
            # Solver failure: fall back to the nominal command.
            self.reset()
            return v_nom

        self._u_warm = U_sol
        v = U_sol[0].copy()
        speed = float(np.linalg.norm(v))
        if speed > self.v_max:
            v = v * (self.v_max / speed)
        self._u_last = v.copy()
        return v


# ---------------------------------------------------------------------------
# Riemannian Motion Policy (RMP) controller
# ---------------------------------------------------------------------------
class RMPController(Controllerbase):
    """Reactive controller in the Riemannian Motion Policy framework.

    In the RMP framework each task ("leaf") is a pair ``(F, M)`` -- a
    generalized force and a positive-definite Riemannian metric
    (generalized inertia) -- that induces the motion ``a = M^-1 F``.
    RMPs compose by summing forces and metrics, so the composed command
    is

        v = (sum_i M_i) ** -1 (sum_i F_i)

    This controller composes two (optionally three) isotropic leaves in
    the 2D plane:

    * goal attraction    ``F_g = m_g * v_nom``,    ``M_g = m_g * I``
    * obstacle avoidance ``F_a = m_a * w(d) * v_max * n``,
                         ``M_a = m_a * w(d) * I``
    * circulation (opt.) ``F_c = m_c * w(d) * v_max * s * t``,
                         ``M_c = m_c * w(d) * I``

    where ``n = grad d / ||grad d||`` points away from the nearest
    obstacle, ``t`` is ``n`` rotated by +90 deg, ``s = +-1`` is a
    latched circulation direction, and ``w(d)`` is a smooth proximity
    weight (1 for ``d <= d_min``, 0 for ``d >= d_max``).

    Notes
    -----
    * The avoidance (and circulation) *metric* is gated by the same
      weight as its force, so an inactive task contributes neither
      force nor inertia: in free space the command reduces exactly to
      the nominal policy.
    * With isotropic metrics the composition is a weighted average of
      the leaf motions, weighted by their metrics; the ratio
      ``m_a / m_g`` sets how strongly avoidance competes with
      goal-seeking (equal masses give a plain average of the two unit
      directions at close range).  With anisotropic tasks the same
      code is the exact RMP composition with
      ``np.linalg.solve(M_total, F_total)``.
    * Without circulation the controller can still stall when the goal
      lies exactly behind an obstacle (the attractive and repulsive
      forces cancel), the same local limitation as the CBF filter;
      ``circulation=True`` adds the tangential swirl leaf to break the
      symmetry, latched while the obstacle is within ``d_max``.
    """

    def __init__(
        self,
        policy: NominalPolicy,
        field_model: DistanceGradientFieldBase,
        d_min: float = 0.2,
        d_max: float = 1.0,
        m_goal: float = 1.0,
        m_avoid: float = 1.0,
        m_circ: float = 1.0,
        circulation: bool = False,
    ) -> None:
        if m_goal <= 0.0 or m_avoid <= 0.0 or m_circ <= 0.0:
            raise ValueError("metrics (masses) must be > 0")
        if d_min < 0.0 or d_max <= d_min:
            raise ValueError("require 0 <= d_min < d_max")
        self.policy = policy
        self.field_model = field_model
        self.d_min = float(d_min)
        self.d_max = float(d_max)
        self.m_goal = float(m_goal)
        self.m_avoid = float(m_avoid)
        self.m_circ = float(m_circ)
        self.circulation = bool(circulation)
        self.v_max = float(policy.v_max)
        # Latched circulation direction: None / +1 (left) / -1 (right).
        self._circ_dir: Optional[float] = None

    def reset(self) -> None:
        """Forget the latched circulation direction."""
        self._circ_dir = None

    def _weight(self, d: float) -> float:
        """Smooth proximity weight: 1 for ``d <= d_min``, 0 for
        ``d >= d_max`` (quadratic ramp in between, C^1 continuous)."""
        if d <= self.d_min:
            return 1.0
        if d >= self.d_max:
            return 0.0
        w = (self.d_max - d) / (self.d_max - self.d_min)
        return w * w

    def compute_velocity(self, pose: Sequence[float]) -> np.ndarray:
        """Composed RMP velocity at ``pose`` (saturated to ``v_max``)."""
        v_nom = self.policy.compute_velocity(pose)
        d, grad = self.field_model.query(pose)
        g_norm = float(np.linalg.norm(grad))
        w = self._weight(float(d))
        if w == 0.0 or g_norm < 1e-12:
            # No obstacle in range (or a flat field): the avoidance and
            # circulation leaves vanish, so the composed command is
            # exactly the nominal velocity.
            self._circ_dir = None
            return v_nom

        n = np.asarray(grad, dtype=float) / g_norm
        # Compose the leaves: sum forces, sum metrics.
        F = self.m_goal * v_nom
        M = self.m_goal
        F = F + self.m_avoid * w * self.v_max * n
        M += self.m_avoid * w
        if self.circulation:
            t_left = np.array([-n[1], n[0]])
            if self._circ_dir is None:
                # Latch the direction once, like the hysteresis
                # controller: left if the nominal heading has a
                # component along the left tangent, right otherwise.
                self._circ_dir = (
                    1.0 if float(np.dot(v_nom, t_left)) >= 0.0 else -1.0
                )
            F = F + self.m_circ * w * self.v_max * self._circ_dir * t_left
            M += self.m_circ * w

        # Isotropic metrics: M_total = M * I, so M_total^-1 F = F / M.
        # (With matrix metrics this would be np.linalg.solve(M, F).)
        v = F / M
        speed = float(np.linalg.norm(v))
        if speed > self.v_max:
            v = v * (self.v_max / speed)
        return v
