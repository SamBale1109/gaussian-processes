"""Smoke tests for the residual RL environment.

Requires the optional ``rl`` dependency group (gymnasium); skipped
otherwise.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("gymnasium")

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"
sys.path.insert(0, str(EXAMPLES))

from residual_rl_env import (  # noqa: E402
    ResidualAvoidanceEnv,
    ResidualConfig,
    build_observation,
)


def test_reset_shapes() -> None:
    env = ResidualAvoidanceEnv()
    obs, _ = env.reset(seed=0)
    assert obs.shape == env.observation_space.shape
    assert env.observation_space.contains(obs)
    assert env.env is not None and env.env.obstacles


def test_episode_terminates() -> None:
    env = ResidualAvoidanceEnv()
    obs, _ = env.reset(seed=1)
    steps = 0
    terminated = truncated = False
    while not (terminated or truncated) and steps < 1000:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, _ = env.step(action)
        steps += 1
        assert obs.shape == env.observation_space.shape
        assert isinstance(reward, float)
    assert terminated or truncated
    assert steps <= env.cfg.max_steps


def test_zero_residual_equals_base_policy() -> None:
    """A zero action must reproduce the base policy exactly (1 step)."""
    cfg = ResidualConfig()
    env = ResidualAvoidanceEnv(cfg)
    env.reset(seed=2)
    pose0 = env.pose.copy()
    env.step(np.zeros(2))
    v_nom = env.base.compute_velocity(pose0)
    expected = pose0 + v_nom * cfg.dt
    assert np.allclose(env.pose, expected, atol=1e-10)


def test_scenario_option(tmp_path: Path) -> None:
    scenario = tmp_path / "scenario.json"
    data = {
        "bounds": [[0.0, 8.0], [0.0, 8.0]],
        "start": [1.0, 1.0],
        "goal": [7.0, 7.0],
        "obstacles": [
            {
                "type": "circle",
                "center": [4.0, 4.0],
                "radius": 0.8,
                "velocity": [0.2, -0.1],
            }
        ],
    }
    scenario.write_text(json.dumps(data))
    env = ResidualAvoidanceEnv(ResidualConfig())
    obs, _ = env.reset(seed=3, options={"scenario": str(scenario)})
    assert obs.shape == env.observation_space.shape
    assert env.lo == (0.0, 0.0) and env.hi == (8.0, 8.0)
    assert env.scale == 8.0
    assert len(env.env.obstacles) == 1
    assert env.env.obstacles[0].is_moving


def test_build_observation_matches_env_obs() -> None:
    """The shared builder must reproduce the env's observation."""
    env = ResidualAvoidanceEnv()
    env.reset(seed=7)
    v_nom = env.base.compute_velocity(env.pose)
    direct = build_observation(
        env.env, env.pose, env.goal, env.lo, env.hi,
        env.scale, v_nom, env.cfg,
    )
    assert np.allclose(direct, env._obs())


def test_testbed_residual_rl_controller(tmp_path: Path) -> None:
    """A saved model runs through the testbed engine end-to-end."""
    pytest.importorskip("stable_baselines3")
    import reactive_avoidance_testbed as tb
    from residual_rl_env import make_env
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv

    cfg = ResidualConfig()
    vec = DummyVecEnv([lambda: make_env(cfg)])
    model = PPO(
        "MlpPolicy",
        vec,
        n_steps=32,
        batch_size=32,
        n_epochs=1,
        policy_kwargs=dict(net_arch=dict(pi=[16, 16], vf=[16, 16])),
    )
    model.learn(64)
    model_path = str(tmp_path / "tiny.zip")
    model.save(model_path)

    env, sensor, policy, field, controller = tb.build_demo_scenario(
        controller_name="residual_rl", rl_model=model_path,
    )
    assert isinstance(controller, tb.ResidualRLController)
    engine = tb.SimulationEngine(
        env,
        sensor,
        field,
        controller,
        start_pose=[1.0, 1.0],
        dt=0.05,
        num_steps=20,
    )
    result = engine.run()
    assert result.poses.shape == (len(result.times), 2)
    assert len(result.times) >= 20
    assert isinstance(result.collided, bool)
