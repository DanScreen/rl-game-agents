"""Atari (Arcade Learning Environment) setup following the DQN/PPO literature."""

from __future__ import annotations

import ale_py
import gymnasium as gym
from gymnasium.wrappers import AtariPreprocessing, FrameStackObservation

gym.register_envs(ale_py)


def make_atari_env(
    env_id: str,
    render_mode: str | None = None,
    frame_stack: int = 4,
    noop_max: int = 30,
    sticky_actions: float = 0.25,
    max_episode_frames: int = 108_000,
) -> gym.Env:
    """Observation: uint8 array of shape (frame_stack, 84, 84).

    ``sticky_actions=0.25`` is the ALE v5 default (Machado et al. 2018) and makes the
    game stochastic so agents cannot memorise a fixed action sequence.
    """
    env = gym.make(
        env_id,
        frameskip=1,  # AtariPreprocessing does the frame skipping (with max-pooling)
        repeat_action_probability=sticky_actions,
        max_num_frames_per_episode=max_episode_frames,
        render_mode=render_mode,
    )
    env = AtariPreprocessing(env, noop_max=noop_max, frame_skip=4, screen_size=84, grayscale_obs=True)
    env = FrameStackObservation(env, stack_size=frame_stack)
    return env
