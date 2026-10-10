"""Run a policy for N episodes and summarise the scores."""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from rlgames.envs import make_env


def run_episodes(
    policy,
    env_name: str,
    episodes: int = 10,
    seed: int = 10_000,
    env_kwargs: dict[str, Any] | None = None,
    max_steps: int | None = None,
) -> dict[str, Any]:
    """Seeds are disjoint from training seeds (training uses 0..num_envs)."""
    env = make_env(env_name, **(env_kwargs or {}))
    returns, lengths = [], []
    t0 = time.time()
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        done, ret, length = False, 0.0, 0
        while not done:
            obs, reward, terminated, truncated, _ = env.step(policy.act(obs))
            ret += float(reward)
            length += 1
            done = terminated or truncated or (max_steps is not None and length >= max_steps)
        returns.append(ret)
        lengths.append(length)
    env.close()
    return summarise(returns, lengths, time.time() - t0)


def summarise(returns: list[float], lengths: list[int], seconds: float = 0.0) -> dict[str, Any]:
    r = np.asarray(returns, dtype=np.float64)
    return {
        "episodes": len(returns),
        "mean_return": float(r.mean()),
        "std_return": float(r.std()),
        "median_return": float(np.median(r)),
        "min_return": float(r.min()),
        "max_return": float(r.max()),
        "mean_length": float(np.mean(lengths)),
        "returns": [float(x) for x in returns],
        "seconds": round(seconds, 1),
    }
