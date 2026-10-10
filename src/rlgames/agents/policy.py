"""Checkpoints and the inference-time ``Policy`` interface used by watch/bench tools.

A checkpoint is a ``torch.save``-d dict::

    {"algo": "dqn" | "ppo", "env": "space_invaders", "env_kwargs": {...},
     "obs_shape": [...], "n_actions": int, "net_kwargs": {...},
     "state_dict": ..., "step": int, "config": {...}}
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

import numpy as np
import torch

from rlgames.agents.networks import ActorCritic, QNetwork


class Policy(Protocol):
    name: str
    env: str
    env_kwargs: dict[str, Any]

    def act(self, obs: np.ndarray) -> int: ...


def save_checkpoint(path: str | Path, **payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)  # atomic, so a crash never leaves a half-written checkpoint


def build_network(algo: str, obs_shape, n_actions: int, net_kwargs: dict[str, Any]) -> torch.nn.Module:
    if algo == "dqn":
        return QNetwork(obs_shape, n_actions, **net_kwargs)
    if algo == "ppo":
        return ActorCritic(obs_shape, n_actions, **net_kwargs)
    raise ValueError(f"Unknown algo {algo!r}")


class NetworkPolicy:
    """Greedy (DQN, with optional epsilon) or sampled/argmax (PPO) action selection."""

    def __init__(self, ckpt: dict[str, Any], epsilon: float | None = None, deterministic: bool | None = None, seed: int = 0):
        self.algo = ckpt["algo"]
        self.env = ckpt["env"]
        self.env_kwargs = ckpt.get("env_kwargs", {})
        self.n_actions = ckpt["n_actions"]
        self.step = ckpt.get("step", 0)
        self.net = build_network(self.algo, ckpt["obs_shape"], self.n_actions, ckpt.get("net_kwargs", {}))
        self.net.load_state_dict(ckpt["state_dict"])
        self.net.eval()
        eval_cfg = ckpt.get("eval", {})
        self.epsilon = eval_cfg.get("epsilon", 0.0) if epsilon is None else epsilon
        self.deterministic = eval_cfg.get("deterministic", False) if deterministic is None else deterministic
        self.rng = np.random.default_rng(seed)
        self.name = f"{self.algo}@{self.step}"

    @torch.no_grad()
    def act(self, obs: np.ndarray) -> int:
        if self.algo == "dqn" and self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.n_actions))
        x = torch.as_tensor(np.asarray(obs)[None])
        if self.algo == "dqn":
            return int(self.net(x).argmax(1).item())
        logits, _ = self.net(x)
        if self.deterministic:
            return int(logits.argmax(1).item())
        return int(torch.distributions.Categorical(logits=logits).sample().item())


class RandomPolicy:
    def __init__(self, env: str, n_actions: int, env_kwargs: dict[str, Any] | None = None, seed: int = 0):
        self.env = env
        self.env_kwargs = env_kwargs or {}
        self.n_actions = n_actions
        self.rng = np.random.default_rng(seed)
        self.name = "random"

    def act(self, obs: np.ndarray) -> int:
        return int(self.rng.integers(self.n_actions))


def init_checkpoint_path(path: str | Path) -> Path:
    """A checkpoint file, or a run directory's best.pt (falling back to latest.pt)."""
    path = Path(path)
    if path.is_dir():
        for candidate in ("best.pt", "latest.pt", "model.pt"):
            if (path / candidate).exists():
                return path / candidate
        raise FileNotFoundError(f"No checkpoint found in {path}")
    return path


def load_policy(path: str | Path, **kwargs: Any) -> NetworkPolicy:
    """Load a policy from a checkpoint file or a run directory (uses its best/latest checkpoint)."""
    path = init_checkpoint_path(path)
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    policy = NetworkPolicy(ckpt, **kwargs)
    policy.name = f"{path.parent.name}/{path.stem}" if path.parent.name else path.stem
    return policy
