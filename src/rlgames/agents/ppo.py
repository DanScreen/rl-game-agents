"""Proximal Policy Optimisation (Schulman et al. 2017), clipped objective + GAE.

Follows the "37 implementation details" recipe (Huang et al. 2022): vectorised envs,
orthogonal init, advantage normalisation, value clipping, global grad-norm clipping,
learning-rate annealing. Truncated episodes bootstrap from the value of the true
final observation.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from rlgames.agents.networks import ActorCritic
from rlgames.agents.policy import NetworkPolicy, init_checkpoint_path, save_checkpoint
from rlgames.envs import make_vector_env
from rlgames.evaluate import run_episodes
from rlgames.logger import Logger, cpu_state_dict, resolve_device

DEFAULTS: dict[str, Any] = {
    "lr": 2.5e-4,
    "anneal_lr": True,
    "num_steps": 128,  # rollout length per env
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "num_minibatches": 4,
    "update_epochs": 4,
    "clip_coef": 0.1,
    "clip_vloss": True,
    "ent_coef": 0.01,
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
    "hidden": [64, 64],
    "clip_rewards": False,
}


def train(cfg: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    hp = {**DEFAULTS, **cfg.get("algo", {})}
    env_name = cfg["env"]
    env_kwargs = cfg.get("env_kwargs", {})
    seed = cfg.get("seed", 0)
    total_steps = int(cfg["total_steps"])
    num_envs = int(cfg.get("num_envs", 8))
    eval_cfg = cfg.get("eval", {})
    eval_interval = int(eval_cfg.get("interval", max(total_steps // 20, 1)))
    eval_episodes = int(eval_cfg.get("episodes", 5))
    eval_deterministic = bool(eval_cfg.get("deterministic", False))

    device = resolve_device(cfg.get("device", "auto"))
    print(f"Training on {device}", flush=True)
    envs = make_vector_env(env_name, num_envs, seed=seed, asynchronous=cfg.get("async_envs", False), **env_kwargs)
    obs_shape = envs.single_observation_space.shape
    n_actions = int(envs.single_action_space.n)
    net_kwargs = {"hidden": list(hp["hidden"])}
    net = ActorCritic(obs_shape, n_actions, **net_kwargs).to(device)
    if cfg.get("init_from"):  # continue from an earlier run's checkpoint
        net.load_state_dict(torch.load(init_checkpoint_path(cfg["init_from"]), map_location=device, weights_only=False)["state_dict"])
        print(f"Initialised from {cfg['init_from']}", flush=True)
    opt = torch.optim.Adam(net.parameters(), lr=hp["lr"], eps=1e-5)
    logger = Logger(run_dir, use_tensorboard=cfg.get("tensorboard", True))

    T = int(hp["num_steps"])
    batch_size = T * num_envs
    minibatch_size = batch_size // int(hp["num_minibatches"])
    num_updates = max(total_steps // batch_size, 1)

    obs_buf = torch.zeros((T, num_envs, *obs_shape), dtype=torch.as_tensor(np.zeros(1, envs.single_observation_space.dtype)).dtype)
    act_buf = torch.zeros((T, num_envs), dtype=torch.long)
    logp_buf = torch.zeros((T, num_envs))
    rew_buf = torch.zeros((T, num_envs))
    done_buf = torch.zeros((T, num_envs))
    val_buf = torch.zeros((T, num_envs))

    def checkpoint(step: int) -> dict[str, Any]:
        return {
            "algo": "ppo", "env": env_name, "env_kwargs": env_kwargs, "obs_shape": list(obs_shape),
            "n_actions": n_actions, "net_kwargs": net_kwargs, "state_dict": cpu_state_dict(net),
            "step": step, "eval": {"deterministic": eval_deterministic}, "config": cfg,
        }

    obs_np, _ = envs.reset(seed=seed)
    obs = torch.as_tensor(obs_np)
    ep_returns = np.zeros(num_envs)
    recent_returns: list[float] = []
    best_eval = -np.inf
    history = []
    step = 0
    next_eval = eval_interval
    t_start = time.time()

    for update in range(1, num_updates + 1):
        if hp["anneal_lr"]:
            opt.param_groups[0]["lr"] = hp["lr"] * (1.0 - (update - 1.0) / num_updates)

        for t in range(T):
            obs_buf[t] = obs
            with torch.no_grad():
                logits, value = net(obs.to(device))
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample()
            act_buf[t] = action.cpu()
            logp_buf[t] = dist.log_prob(action).cpu()
            val_buf[t] = value.cpu()

            next_obs, rewards, terminated, truncated, info = envs.step(action.numpy())
            train_rewards = np.sign(rewards) if hp["clip_rewards"] else rewards.astype(np.float32)
            # Truncation is not a real terminal: bootstrap with V(final_obs).
            if truncated.any() and "final_obs" in info:
                trunc_idx = np.flatnonzero(truncated & ~terminated)
                if len(trunc_idx):
                    final = torch.as_tensor(np.stack([info["final_obs"][i] for i in trunc_idx]))
                    with torch.no_grad():
                        v_final = net(final.to(device))[1].cpu()
                    train_rewards = train_rewards.copy()
                    train_rewards[trunc_idx] += hp["gamma"] * v_final.numpy()
            rew_buf[t] = torch.as_tensor(train_rewards, dtype=torch.float32)
            done_buf[t] = torch.as_tensor(terminated | truncated, dtype=torch.float32)
            obs = torch.as_tensor(next_obs)

            ep_returns += rewards
            for i in np.flatnonzero(terminated | truncated):
                recent_returns.append(float(ep_returns[i]))
                ep_returns[i] = 0.0
        step += batch_size

        # Generalised advantage estimation.
        with torch.no_grad():
            next_value = net(obs.to(device))[1].cpu()
        adv = torch.zeros_like(rew_buf)
        last_gae = torch.zeros(num_envs)
        for t in reversed(range(T)):
            nv = next_value if t == T - 1 else val_buf[t + 1]
            not_done = 1.0 - done_buf[t]
            delta = rew_buf[t] + hp["gamma"] * nv * not_done - val_buf[t]
            last_gae = delta + hp["gamma"] * hp["gae_lambda"] * not_done * last_gae
            adv[t] = last_gae
        returns = adv + val_buf

        b_obs = obs_buf.reshape(batch_size, *obs_shape).to(device)
        b_act, b_logp = act_buf.reshape(-1).to(device), logp_buf.reshape(-1).to(device)
        b_adv, b_ret, b_val = adv.reshape(-1).to(device), returns.reshape(-1).to(device), val_buf.reshape(-1).to(device)

        clipfracs, pg_losses, v_losses, entropies, approx_kls = [], [], [], [], []
        for _epoch in range(int(hp["update_epochs"])):
            perm = torch.randperm(batch_size, device=device)
            for start in range(0, batch_size, minibatch_size):
                mb = perm[start:start + minibatch_size]
                logits, new_v = net(b_obs[mb])
                dist = torch.distributions.Categorical(logits=logits)
                new_logp = dist.log_prob(b_act[mb])
                entropy = dist.entropy().mean()
                log_ratio = new_logp - b_logp[mb]
                ratio = log_ratio.exp()
                with torch.no_grad():
                    approx_kls.append(((ratio - 1) - log_ratio).mean().item())
                    clipfracs.append(((ratio - 1.0).abs() > hp["clip_coef"]).float().mean().item())

                mb_adv = b_adv[mb]
                mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)
                pg_loss = torch.max(
                    -mb_adv * ratio, -mb_adv * ratio.clamp(1 - hp["clip_coef"], 1 + hp["clip_coef"])
                ).mean()
                if hp["clip_vloss"]:
                    v_clipped = b_val[mb] + (new_v - b_val[mb]).clamp(-hp["clip_coef"], hp["clip_coef"])
                    v_loss = 0.5 * torch.max((new_v - b_ret[mb]) ** 2, (v_clipped - b_ret[mb]) ** 2).mean()
                else:
                    v_loss = 0.5 * F.mse_loss(new_v, b_ret[mb])
                loss = pg_loss - hp["ent_coef"] * entropy + hp["vf_coef"] * v_loss
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), hp["max_grad_norm"])
                opt.step()
                pg_losses.append(pg_loss.item())
                v_losses.append(v_loss.item())
                entropies.append(entropy.item())

        logger.log(step, {
            "train/episode_return": float(np.mean(recent_returns[-20:])) if recent_returns else float("nan"),
            "train/episodes": len(recent_returns),
            "train/policy_loss": float(np.mean(pg_losses)),
            "train/value_loss": float(np.mean(v_losses)),
            "train/entropy": float(np.mean(entropies)),
            "train/approx_kl": float(np.mean(approx_kls)),
            "train/clipfrac": float(np.mean(clipfracs)),
            "train/lr": opt.param_groups[0]["lr"],
            "perf/steps_per_sec": step / (time.time() - t_start),
        }, echo=(update % max(1, num_updates // 200) == 0) or update == num_updates)

        if step >= next_eval or update == num_updates:
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
        "algo": "ppo", "env": env_name, "total_steps": step, "best_eval_return": best_eval,
        "final_eval_return": history[-1]["mean_return"] if history else None,
        "wall_clock_sec": round(time.time() - t_start, 1), "eval_history": history,
    }
    logger.write_summary(summary)
    logger.close()
    return summary
