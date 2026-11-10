"""Unit tests for the Riemannian Motion Policy (RMP) controller."""

import numpy as np
import pytest

from gaussian_processes.controllers import NominalPolicy, RMPController
from gaussian_processes.distance_gradient_fields import (
    DistanceGradientFieldBase,
)


class StubField(DistanceGradientFieldBase):
    """Field with a fixed (distance, gradient) answer."""

    def __init__(self, d: float, grad: np.ndarray) -> None:
        self.d = float(d)
        self.grad = np.asarray(grad, dtype=float)

    def update(self, pointcloud: np.ndarray, robot_pose: np.ndarray) -> None:
        del pointcloud, robot_pose

    def query(self, x) -> tuple[float, np.ndarray]:
        del x
        return self.d, self.grad


def make_controller(field, circulation=False, **kwargs) -> RMPController:
    policy = NominalPolicy(target=[5.0, 1.0], v_max=1.0)
    return RMPController(
        policy, field, circulation=circulation, **kwargs
    )


def test_free_space_returns_nominal() -> None:
    """Far from obstacles the command is exactly the nominal velocity."""
    ctrl = make_controller(StubField(d=5.0, grad=np.zeros(2)))
    v = ctrl.compute_velocity([1.0, 1.0])
    assert np.allclose(v, [1.0, 0.0])


def test_composition_matches_hand_computed() -> None:
    """v = (F_g + F_a) / (m_g + m_a w) with the quadratic weight."""
    # pose [1,1], goal [5,1]: v_nom = [1,0].  Obstacle to the left of
    # the robot: d = 0.6, n = [-1,0].  w = ((1-0.6)/(1-0.2))^2 = 0.25.
    ctrl = make_controller(StubField(d=0.6, grad=np.array([-1.6, 0.0])))
    v = ctrl.compute_velocity([1.0, 1.0])
    expected = (
        np.array([1.0, 0.0]) + 0.25 * np.array([-1.0, 0.0])
    ) / (1.0 + 0.25)
    assert np.allclose(v, expected)  # [0.6, 0]


def test_metric_ratio_shifts_towards_repulsion() -> None:
    """A larger avoidance metric pulls the command towards the
    repulsive direction (weighted average with m_a as weight)."""
    ctrl = make_controller(
        StubField(d=0.6, grad=np.array([-1.6, 0.0])), m_avoid=3.0
    )
    v = ctrl.compute_velocity([1.0, 1.0])
    expected = (
        np.array([1.0, 0.0]) + 3.0 * 0.25 * np.array([-1.0, 0.0])
    ) / (1.0 + 0.75)
    assert np.allclose(v, expected)  # [1/7, 0]


def test_symmetric_stall_without_circulation() -> None:
    """Goal directly behind the obstacle: attraction and repulsion
    cancel exactly, so the plain RMP stalls (local limitation)."""
    ctrl = make_controller(StubField(d=0.2, grad=np.array([-1.2, 0.0])))
    v = ctrl.compute_velocity([1.0, 1.0])
    assert float(np.linalg.norm(v)) < 1e-12


def test_circulation_breaks_stall() -> None:
    """The swirl leaf supplies a tangential force that moves the robot.

    v_nom = [1,0], n = [-1,0] -> t_left = [0,-1]; v_nom . t_left = 0
    >= 0 so the latched direction is +1:
    v = ([1,0] + [-1,0] + [0,-1]) / 3 = [0, -1/3].
    """
    ctrl = make_controller(
        StubField(d=0.2, grad=np.array([-1.2, 0.0])), circulation=True
    )
    v = ctrl.compute_velocity([1.0, 1.0])
    assert np.allclose(v, [0.0, -1.0 / 3.0])


def test_circulation_direction_latches_and_releases() -> None:
    field = StubField(d=0.2, grad=np.array([-1.2, 0.0]))
    ctrl = make_controller(field, circulation=True)
    ctrl.compute_velocity([1.0, 1.0])
    assert ctrl._circ_dir == 1.0
    # Same obstacle (now approaching from a side view) keeps the
    # latched direction instead of re-deciding.
    ctrl.compute_velocity([1.0, 1.0])
    assert ctrl._circ_dir == 1.0
    # Obstacle out of range: direction is released.
    field.d = 2.0
    ctrl.compute_velocity([1.0, 1.0])
    assert ctrl._circ_dir is None
    # reset() clears the latched direction as well.
    field.d = 0.2
    ctrl.compute_velocity([1.0, 1.0])
    ctrl.reset()
    assert ctrl._circ_dir is None


def test_speed_never_exceeds_v_max() -> None:
    """The composition is a weighted average of leaf motions, each
    bounded by v_max, so the result never exceeds v_max."""
    for d in (0.0, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0):
        for angle in np.linspace(0.0, 2.0 * np.pi, 9, endpoint=False):
            n = np.array([np.cos(angle), np.sin(angle)])
            ctrl = make_controller(StubField(d=d, grad=2.0 * n))
            v = ctrl.compute_velocity([1.0, 1.0])
            assert float(np.linalg.norm(v)) <= 1.0 + 1e-12


def test_constructor_validation() -> None:
    field = StubField(d=1.0, grad=np.zeros(2))
    policy = NominalPolicy(target=[5.0, 1.0])
    with pytest.raises(ValueError):
        RMPController(policy, field, m_goal=0.0)
    with pytest.raises(ValueError):
        RMPController(policy, field, d_max=0.1)
