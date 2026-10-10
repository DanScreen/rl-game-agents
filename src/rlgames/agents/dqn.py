"""Deep Q-Network (Mnih et al. 2015) with Double DQN and optional dueling heads.

Works for vector observations (CartPole) and stacked Atari frames (Space Invaders).
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from rlgames.agents.networks import QNetwork
from rlgames.agents.policy import NetworkPolicy, init_checkpoint_path, save_checkpoint
from rlgames.envs import make_vector_env
from rlgames.evaluate import run_episodes
from rlgames.logger import Logger, cpu_state_dict, resolve_device

DEFAULTS: dict[str, Any] = {
    "lr": 1e-4,
    "gamma": 0.99,
    "buffer_size": 100_000,
    "batch_size": 32,
    "learning_starts": 10_000,
    "train_freq": 4,  # env steps (summed over envs) between updates
    "gradient_steps": 1,  # gradient steps per update
    "target_update_interval": 1_000,  # env steps between hard target syncs (when tau == 1)
    "tau": 1.0,  # < 1 switches to soft target updates: target <- (1 - tau) * target + tau * online
    "eps_start": 1.0,
    "eps_end": 0.01,
    "eps_fraction": 0.1,  # fraction of total_steps over which epsilon decays
    "double": True,
    "dueling": False,
    "hidden": [128, 128],
    "clip_rewards": False,
    "max_grad_norm": 10.0,
    "loss": "huber",
    "anneal_lr": False,  # linearly decay the learning rate to 0 over training (stabilises the final policy)
}


class ReplayBuffer:
    """Uniform replay. For stacked frames (C, H, W) only the newest frame of next_obs is
    stored, because next_obs[:-1] == obs[1:] within an episode (we always store the true
    final observation, never the post-reset one). This cuts Atari memory use by ~45%."""

    def __init__(self, capacity: int, obs_shape: tuple[int, ...], obs_dtype: np.dtype):
        self.capacity = capacity
        self.frames = len(obs_shape) == 3
        self.obs = np.zeros((capacity, *obs_shape), dtype=obs_dtype)
        next_shape = (1, *obs_shape[1:]) if self.frames else obs_shape
        self.next_obs = np.zeros((capacity, *next_shape), dtype=obs_dtype)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        self.pos = 0
        self.size = 0

    def add(self, obs, action, reward, next_obs, terminated) -> None:
        for i in range(len(obs)):
            self.obs[self.pos] = obs[i]
            self.next_obs[self.pos] = next_obs[i][-1:] if self.frames else next_obs[i]
            self.actions[self.pos] = action[i]
            self.rewards[self.pos] = reward[i]
            self.dones[self.pos] = terminated[i]
            self.pos = (self.pos + 1) % self.capacity
            self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, rng: np.random.Generator) -> tuple[torch.Tensor, ...]:
        idx = rng.integers(0, self.size, size=batch_size)
        obs = self.obs[idx]
        next_obs = np.concatenate([obs[:, 1:], self.next_obs[idx]], axis=1) if self.frames else self.next_obs[idx]
        return (
            torch.as_tensor(obs),
            torch.as_tensor(self.actions[idx]),
            torch.as_tensor(self.rewards[idx]),
            torch.as_tensor(next_obs),
            torch.as_tensor(self.dones[idx]),
        )


def linear_schedule(start: float, end: float, duration: float, t: float) -> float:
    return end + (start - end) * max(0.0, 1.0 - t / max(duration, 1.0))


def train(cfg: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    hp = {**DEFAULTS, **cfg.get("algo", {})}
    env_name = cfg["env"]
    env_kwargs = cfg.get("env_kwargs", {})
    seed = cfg.get("seed", 0)
    total_steps = int(cfg["total_steps"])
    num_envs = int(cfg.get("num_envs", 1))
    eval_cfg = cfg.get("eval", {})
    eval_interval = int(eval_cfg.get("interval", max(total_steps // 20, 1)))
    eval_episodes = int(eval_cfg.get("episodes", 5))
    eval_epsilon = float(eval_cfg.get("epsilon", 0.0))
    log_interval = int(cfg.get("log_interval", 1000))

    rng = np.random.default_rng(seed)
    device = resolve_device(cfg.get("device", "auto"))
    print(f"Training on {device}", flush=True)
    envs = make_vector_env(env_name, num_envs, seed=seed, asynchronous=cfg.get("async_envs", False), **env_kwargs)
    obs_shape = envs.single_observation_space.shape
    n_actions = int(envs.single_action_space.n)
    net_kwargs = {"hidden": list(hp["hidden"]), "dueling": bool(hp["dueling"])}
    q = QNetwork(obs_shape, n_actions, **net_kwargs).to(device)
    q_target = QNetwork(obs_shape, n_actions, **net_kwargs).to(device)
    q_target.load_state_dict(q.state_dict())
    if cfg.get("init_from"):  # continue from an earlier run's checkpoint
        q.load_state_dict(torch.load(init_checkpoint_path(cfg["init_from"]), map_location=device, weights_only=False)["state_dict"])
        print(f"Initialised from {cfg['init_from']}", flush=True)
        q_target.load_state_dict(q.state_dict())
    opt = torch.optim.Adam(q.parameters(), lr=hp["lr"], eps=1.5e-4 if len(obs_shape) == 3 else 1e-8)
    buffer = ReplayBuffer(int(hp["buffer_size"]), obs_shape, envs.single_observation_space.dtype)
    logger = Logger(run_dir, use_tensorboard=cfg.get("tensorboard", True))

    def checkpoint(step: int) -> dict[str, Any]:
        return {
            "algo": "dqn", "env": env_name, "env_kwargs": env_kwargs, "obs_shape": list(obs_shape),
            "n_actions": n_actions, "net_kwargs": net_kwargs, "state_dict": cpu_state_dict(q),
            "step": step, "eval": {"epsilon": eval_epsilon}, "config": cfg,
        }

    obs, _ = envs.reset(seed=seed)
    ep_returns = np.zeros(num_envs)
    ep_lengths = np.zeros(num_envs, dtype=np.int64)
    recent_returns: list[float] = []
    best_eval = -np.inf
    losses: list[float] = []
    q_means: list[float] = []
    t_start = time.time()
    step = 0
    next_log, next_eval, next_target = log_interval, eval_interval, hp["target_update_interval"]
    grad_debt = 0.0
    history = []

    while step < total_steps:
        eps = linear_schedule(hp["eps_start"], hp["eps_end"], hp["eps_fraction"] * total_steps, step)
        rand = rng.random(num_envs) < eps  # per-env epsilon-greedy
        if rand.all():
            actions = rng.integers(n_actions, size=num_envs)
        else:
            with torch.no_grad():
                actions = q(torch.as_tensor(obs, device=device)).argmax(1).cpu().numpy()
            actions[rand] = rng.integers(n_actions, size=int(rand.sum()))

        next_obs, rewards, terminated, truncated, info = envs.step(actions)
        real_next = next_obs.copy()
        if "final_obs" in info:
            for i in np.flatnonzero(info["_final_obs"]):
                real_next[i] = info["final_obs"][i]
        train_rewards = np.sign(rewards) if hp["clip_rewards"] else rewards
        buffer.add(obs, actions, train_rewards, real_next, terminated)
        obs = next_obs

        ep_returns += rewards
        ep_lengths += 1
        for i in np.flatnonzero(terminated | truncated):
            recent_returns.append(float(ep_returns[i]))
            ep_returns[i] = 0.0
            ep_lengths[i] = 0
        step += num_envs

        if step >= hp["learning_starts"]:
            if hp["anneal_lr"]:
                opt.param_groups[0]["lr"] = hp["lr"] * max(0.0, 1.0 - step / total_steps)
            grad_debt += num_envs / hp["train_freq"]
            for _ in range(int(grad_debt) * int(hp["gradient_steps"])):
                b_obs, b_act, b_rew, b_next, b_done = (t.to(device) for t in buffer.sample(hp["batch_size"], rng))
                with torch.no_grad():
                    if hp["double"]:
                        next_a = q(b_next).argmax(1, keepdim=True)
                        next_q = q_target(b_next).gather(1, next_a).squeeze(1)
                    else:
                        next_q = q_target(b_next).max(1).values
                    target = b_rew + hp["gamma"] * (1.0 - b_done) * next_q
                q_pred = q(b_obs).gather(1, b_act[:, None]).squeeze(1)
                loss = F.smooth_l1_loss(q_pred, target) if hp["loss"] == "huber" else F.mse_loss(q_pred, target)
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(q.parameters(), hp["max_grad_norm"])
                opt.step()
                losses.append(loss.item())
                q_means.append(q_pred.mean().item())
                if hp["tau"] < 1.0:  # soft (Polyak) target update after every gradient step
                    with torch.no_grad():
                        for p, tp in zip(q.parameters(), q_target.parameters()):
                            tp.lerp_(p, hp["tau"])
            grad_debt -= int(grad_debt)
            if hp["tau"] >= 1.0 and step >= next_target:
                q_target.load_state_dict(q.state_dict())
                next_target += hp["target_update_interval"]

        if step >= next_log:
            next_log += log_interval
            sps = step / (time.time() - t_start)
            logger.log(step, {
                "train/episode_return": float(np.mean(recent_returns[-20:])) if recent_returns else float("nan"),
                "train/episodes": len(recent_returns),
                "train/epsilon": eps,
                "train/loss": float(np.mean(losses[-100:])) if losses else float("nan"),
                "train/q_mean": float(np.mean(q_means[-100:])) if q_means else float("nan"),
                "perf/steps_per_sec": sps,
            })

        if step >= next_eval or step >= total_steps:
            next_eval += eval_interval
            policy = NetworkPolicy(checkpoint(step), seed=seed)
            stats = run_episodes(policy, env_name, eval_episodes, env_kwargs=env_kwargs, seed=10_000 + step)
            history.append({"step": step, "mean_return": stats["mean_return"], "std_return": stats["std_return"]})
            logger.log(step, {"eval/mean_return": stats["mean_return"], "eval/std_return": stats["std_return"]})
            save_checkpoint(run_dir / "latest.pt", **checkpoint(step))
            if stats["mean_return"] >= best_eval:
                best_eval = stats["mean_return"]
                save_checkpoint(run_dir / "best.pt", **checkpoint(step))

    envs.close()
    summary = {
        "algo": "dqn", "env": env_name, "total_steps": step, "best_eval_return": best_eval,
        "final_eval_return": history[-1]["mean_return"] if history else None,
        "wall_clock_sec": round(time.time() - t_start, 1), "eval_history": history,
    }
    logger.write_summary(summary)
    logger.close()
    return summary
