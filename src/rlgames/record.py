"""Watch agents play: record MP4/GIF videos, or open a live window (``human`` mode)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw

from rlgames.envs import is_atari, make_env


def _annotate(frame: np.ndarray, text: str, scale: int) -> np.ndarray:
    game = Image.fromarray(frame)
    if scale != 1:
        game = game.resize((game.width * scale, game.height * scale), Image.NEAREST)
    img = Image.new("RGB", (game.width, game.height + 16), (0, 0, 0))  # caption strip above the game
    img.paste(game, (0, 16))
    draw = ImageDraw.Draw(img)
    draw.text((4, 2), text, fill=(255, 255, 255))
    out = np.asarray(img)
    # libx264 wants even dimensions
    h, w = out.shape[:2]
    return out[: h - h % 2, : w - w % 2]


def record_episodes(
    policy,
    env_name: str,
    out_path: str | Path,
    episodes: int = 1,
    seed: int = 0,
    env_kwargs: dict[str, Any] | None = None,
    fps: int | None = None,
    max_steps: int = 20_000,
) -> list[float]:
    """Play ``episodes`` games and write them to one video. Returns the episode scores."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")  # off-screen rendering needs no display
    env = make_env(env_name, render_mode="rgb_array", **(env_kwargs or {}))
    scale = 2 if is_atari(env_name) else 1
    fps = fps or (30 if is_atari(env_name) else 50)
    frames: list[np.ndarray] = []
    scores = []
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        score, steps, done = 0.0, 0, False
        frames.append(_annotate(env.render(), f"{policy.name}  ep {ep + 1}  score {score:g}", scale))
        while not done and steps < max_steps:
            obs, reward, terminated, truncated, _ = env.step(policy.act(obs))
            score += float(reward)
            steps += 1
            done = terminated or truncated
            frames.append(_annotate(env.render(), f"{policy.name}  ep {ep + 1}  score {score:g}", scale))
        frames.extend([frames[-1]] * fps)  # hold the final frame for a second
        scores.append(score)
    env.close()
    if out_path.suffix == ".gif":
        imageio.mimsave(out_path, frames[::2], duration=2000 / fps, loop=0)  # half the frames keeps GIFs small
    else:
        imageio.mimsave(out_path, frames, fps=fps, macro_block_size=1)
    return scores


def play_human(policy, env_name: str, episodes: int = 1, seed: int = 0, env_kwargs: dict[str, Any] | None = None) -> list[float]:
    """Render in a desktop window (needs a display and pygame)."""
    env = make_env(env_name, render_mode="human", **(env_kwargs or {}))
    scores = []
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        done, score = False, 0.0
        while not done:
            obs, reward, terminated, truncated, _ = env.step(policy.act(obs))
            score += float(reward)
            done = terminated or truncated
        print(f"episode {ep + 1}: score {score:g}")
        scores.append(score)
    env.close()
    return scores
