"""Train and evaluate a residual RL policy for obstacle avoidance.

The agent learns a residual velocity on top of a pure goal-attraction
base policy (see ``residual_rl_env.py``).  After training the script
evaluates the base policy *alone* and the residual policy on fresh
random layouts (and optionally a custom JSON scenario) and prints a
comparison table.

Usage
-----
    # install the optional RL dependencies (gymnasium, stable-baselines3)
    uv sync --group rl

    # train + evaluate (a few minutes on CPU / GPU)
    uv run --group rl python examples/train_residual_rl.py

    # larger run
    uv run --group rl python examples/train_residual_rl.py \
        --total-steps 1000000 --n-envs 32 --out runs/ppo_big

    # evaluate an existing model only
    uv run --group rl python examples/train_residual_rl.py \
        --eval-only runs/ppo_big/model.zip

    # also evaluate on a custom scenario (create_environment.py)
    uv run --group rl python examples/train_residual_rl.py \
        --scenario my_scenario.json
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
import time
from typing import Callable, Dict, List, Optional

import numpy as np
from residual_rl_env import (
    ResidualAvoidanceEnv,
    ResidualConfig,
    make_env,
)

POLICY = Callable[[np.ndarray], np.ndarray]


def run_episode(
    env: ResidualAvoidanceEnv,
    policy: POLICY,
    seed: Optional[int] = None,
    options: Optional[dict] = None,
) -> Dict[str, float]:
    """Roll one episode out; returns summary stats."""
    obs, _ = env.reset(seed=seed, options=options)
    steps = 0
    reached = False
    collided = False
    min_clear = float("inf")
    while True:
        action = policy(obs)
        obs, _reward, terminated, truncated, info = env.step(action)
        steps += 1
        min_clear = min(min_clear, float(info["min_clearance"]))
        if terminated or truncated:
            reached = bool(info.get("reached_goal", False))
            collided = bool(info.get("collided", False))
            break
    return {
        "reached": float(reached),
        "collided": float(collided),
        "steps": float(steps),
        "min_clearance": min_clear,
    }


def summarize(results: List[Dict[str, float]]) -> Dict[str, float]:
    """Aggregate per-episode stats into percentages / medians."""
    n = len(results)
    clearance = [
        min(r["min_clearance"], 10.0) for r in results
    ]
    return {
        "episodes": n,
        "success_pct": 100.0 * sum(r["reached"] for r in results) / n,
        "collision_pct": 100.0 * sum(r["collided"] for r in results) / n,
        "timeout_pct": 100.0
        * sum(
            1.0 - r["reached"] - r["collided"] for r in results
        )
        / n,
        "mean_steps": float(np.mean([r["steps"] for r in results])),
        "median_clearance": float(np.median(clearance)),
    }


def print_table(rows: Dict[str, Dict[str, float]]) -> None:
    header = (
        f"{'policy':<24}{'success':>9}{'collision':>10}"
        f"{'timeouts':>10}{'steps':>9}{'clearance':>11}"
    )
    print(header)
    print("-" * len(header))
    for name, s in rows.items():
        print(
            f"{name:<24}"
            f"{s['success_pct']:>8.1f}%"
            f"{s['collision_pct']:>9.1f}%"
            f"{s['timeout_pct']:>9.1f}%"
            f"{s['mean_steps']:>9.0f}"
            f"{s['median_clearance']:>11.2f}"
        )


def evaluate(
    policies: Dict[str, POLICY],
    config: ResidualConfig,
    n: int,
    seed0: int,
    scenario: Optional[str] = None,
) -> Dict[str, Dict[str, float]]:
    """Evaluate all policies on fresh random layouts (+ scenario)."""
    table: Dict[str, Dict[str, float]] = {}
    for name, policy in policies.items():
        env = ResidualAvoidanceEnv(config)
        results = [
            run_episode(env, policy, seed=seed0 + i)
            for i in range(n)
        ]
        table[name] = summarize(results)
        print(f"  {name}: {n} episodes done")
    if scenario is not None:
        for name, policy in policies.items():
            env = ResidualAvoidanceEnv(config)
            result = run_episode(
                env,
                policy,
                options={"scenario": scenario},
            )
            table[f"{name} [scenario]"] = summarize([result])
    return table


def apply_config_overrides(
    args: argparse.Namespace, config: ResidualConfig
) -> ResidualConfig:
    """Apply optional CLI overrides to the world sampling config."""
    if args.min_radius is not None:
        config.radius_range = (args.min_radius,
                               config.radius_range[1])
    if args.max_radius is not None:
        config.radius_range = (config.radius_range[0],
                               args.max_radius)
    if args.max_obstacles is not None:
        config.n_obstacles = (config.n_obstacles[0],
                              args.max_obstacles)
    if args.box_prob is not None:
        config.box_prob = args.box_prob
    if args.moving_prob is not None:
        config.moving_prob = args.moving_prob
    if args.pair_margin is not None:
        config.pair_margin = args.pair_margin
    # reward overrides
    if args.w_progress is not None:
        config.w_progress = args.w_progress
    if args.w_time is not None:
        config.w_time = args.w_time
    if args.w_clearance is not None:
        config.w_clearance = args.w_clearance
    if args.w_res is not None:
        config.w_res = args.w_res
    return config


def train(args: argparse.Namespace) -> None:
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv

    config = apply_config_overrides(
        args, ResidualConfig()
    )
    vec = DummyVecEnv(
        [lambda: make_env(config) for _ in range(args.n_envs)]
    )
    vec.seed(args.seed)
    model = PPO(
        "MlpPolicy",
        vec,
        learning_rate=args.lr,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.epochs,
        gamma=args.gamma,
        seed=args.seed,
        verbose=1,
        policy_kwargs=dict(
            net_arch=[
                dict(pi=[args.hidden, args.hidden],
                     vf=[args.hidden, args.hidden])
            ]
        ),
    )
    start = time.time()
    model.learn(args.total_steps)
    print(f"training took {time.time() - start:.0f} s")

    os.makedirs(args.out, exist_ok=True)
    model_path = os.path.join(args.out, "model.zip")
    model.save(model_path)
    print(f"model saved to    : {model_path}")
    with open(os.path.join(args.out, "config.json"), "w") as fh:
        json.dump(
            {
                "env": dataclasses.asdict(config),
                "ppo": {
                    "lr": args.lr,
                    "n_steps": args.n_steps,
                    "batch_size": args.batch_size,
                    "epochs": args.epochs,
                    "gamma": args.gamma,
                    "hidden": args.hidden,
                    "seed": args.seed,
                    "total_steps": args.total_steps,
                },
            },
            fh,
            indent=2,
        )

    print(
        f"\nevaluation on {args.eval_n} fresh random "
        f"layouts (seed {args.eval_seed}+)"
    )
    zero_res = lambda obs: np.zeros(2)  # noqa: E731
    table = evaluate(
        {
            "base only (no RL)": zero_res,
            "residual RL (PPO)": lambda obs: model.predict(
                obs, deterministic=True
            )[0],
        },
        config,
        n=args.eval_n,
        seed0=args.eval_seed,
        scenario=args.scenario,
    )
    print()
    print_table(table)
    summary = {"random_layouts": table}
    with open(
        os.path.join(args.out, "eval_summary.json"), "w"
    ) as fh:
        json.dump(summary, fh, indent=2)


def evaluate_only(args: argparse.Namespace) -> None:
    from stable_baselines3 import PPO

    config = apply_config_overrides(
        args, ResidualConfig()
    )
    model = PPO.load(args.eval_only)
    print(f"\nevaluation on {args.eval_n} fresh random "
          f"layouts (seed {args.eval_seed}+)")
    zero_res = lambda obs: np.zeros(2)  # noqa: E731
    table = evaluate(
        {
            "base only (no RL)": zero_res,
            "residual RL (saved)": lambda obs: model.predict(
                obs, deterministic=True
            )[0],
        },
        config,
        n=args.eval_n,
        seed0=args.eval_seed,
        scenario=args.scenario,
    )
    print()
    print_table(table)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train / evaluate a residual RL policy for "
        "obstacle avoidance (PPO)."
    )
    # training
    parser.add_argument("--n-envs", type=int, default=16,
                        help="parallel environments (default 16)")
    parser.add_argument("--total-steps", type=int, default=500000,
                        help="total environment steps (default 5e5)")
    parser.add_argument("--hidden", type=int, default=64,
                        help="MLP hidden width (default 64)")
    parser.add_argument("--n-steps", type=int, default=512,
                        help="rollout length per env (default 512)")
    parser.add_argument("--batch-size", type=int, default=256,
                        help="PPO minibatch size (default 256)")
    parser.add_argument("--epochs", type=int, default=10,
                        help="PPO epochs per update (default 10)")
    parser.add_argument("--lr", type=float, default=3e-4,
                        help="learning rate (default 3e-4)")
    parser.add_argument("--gamma", type=float, default=0.99,
                        help="discount (default 0.99)")
    parser.add_argument("--seed", type=int, default=0,
                        help="random seed (default 0)")
    parser.add_argument("--out", default=None,
                        help="output dir for model / summaries "
                             "(default runs/residual_rl_<timestamp>)")
    # evaluation
    parser.add_argument("--eval-n", type=int, default=30,
                        help="random layouts per policy (default 30)")
    parser.add_argument("--eval-seed", type=int, default=1000,
                        help="seed of the eval layouts (default 1000)")
    parser.add_argument("--scenario", default=None, metavar="JSON",
                        help="also evaluate on this scenario file")
    parser.add_argument("--eval-only", default=None, metavar="ZIP",
                        help="skip training; evaluate this model")
    # world sampling overrides (default: see ResidualConfig)
    parser.add_argument("--min-radius", type=float, default=None,
                        help="min obstacle radius (default 0.4)")
    parser.add_argument("--max-radius", type=float, default=None,
                        help="max obstacle radius (default 1.1)")
    parser.add_argument("--max-obstacles", type=int, default=None,
                        help="max obstacles per world (default 5)")
    parser.add_argument("--box-prob", type=float, default=None,
                        help="probability of a box obstacle "
                             "(default 0.3)")
    parser.add_argument("--moving-prob", type=float, default=None,
                        help="probability of one moving obstacle "
                             "(default 0.25)")
    parser.add_argument("--pair-margin", type=float, default=None,
                        help="min edge clearance between obstacles "
                             "(default 0.4; negative values allow "
                             "overlap up to |value|)")
    # reward overrides (default: see ResidualConfig)
    parser.add_argument("--w-progress", type=float, default=None,
                        help="reward per unit distance to goal "
                             "(default 5.0)")
    parser.add_argument("--w-time", type=float, default=None,
                        help="time cost per second (default 0.2)")
    parser.add_argument("--w-clearance", type=float, default=None,
                        help="clearance deficit penalty (default 2.0)")
    parser.add_argument("--w-res", type=float, default=None,
                        help="residual magnitude penalty (default 0.05)")
    args = parser.parse_args()
    if args.out is None:
        args.out = time.strftime(
            "runs/residual_rl_%Y%m%d_%H%M%S"
        )
    if args.eval_only is not None:
        evaluate_only(args)
    else:
        train(args)


if __name__ == "__main__":
    sys.exit(main())
