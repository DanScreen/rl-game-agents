"""Environment factory.

Short names map to tuned setups:

* ``cartpole``        -> ``CartPole-v1`` (the "very simple game"; solved at return 500)
* ``space_invaders``  -> ``ALE/SpaceInvaders-v5`` with standard Atari preprocessing
  (84x84 grayscale, frame skip 4, 4-frame stack, 30 random no-ops at reset)

Any other string is passed straight to ``gymnasium.make`` (Atari ids starting with
``ALE/`` get the Atari preprocessing), so new games can be tried without code changes.

Environments always return the *raw* game reward so episode returns are real game
scores; reward clipping for Atari training happens inside the learners.
"""

from __future__ import annotations

from typing import Any, Callable

import gymnasium as gym
from gymnasium.vector import AutoresetMode

ALIASES = {
    "cartpole": "CartPole-v1",
    "space_invaders": "ALE/SpaceInvaders-v5",
    "breakout": "ALE/Breakout-v5",
    "pong": "ALE/Pong-v5",
}


def resolve_env_id(name: str) -> str:
    return ALIASES.get(name, name)


def is_atari(name: str) -> bool:
    return resolve_env_id(name).startswith("ALE/")


def make_env(name: str, render_mode: str | None = None, **kwargs: Any) -> gym.Env:
    env_id = resolve_env_id(name)
    if is_atari(env_id):
        from rlgames.envs.atari import make_atari_env

        return make_atari_env(env_id, render_mode=render_mode, **kwargs)
    return gym.make(env_id, render_mode=render_mode, **kwargs)


def make_vector_env(
    name: str, num_envs: int, seed: int = 0, asynchronous: bool = False, **kwargs: Any
) -> gym.vector.VectorEnv:
    """Vectorised env with SAME_STEP autoreset: on episode end the returned obs is
    already the reset obs and the true final obs is in ``info["final_obs"]``."""

    def thunk(i: int) -> Callable[[], gym.Env]:
        def _make() -> gym.Env:
            env = make_env(name, **kwargs)
            env.action_space.seed(seed + i)
            return env

        return _make

    fns = [thunk(i) for i in range(num_envs)]
    cls = gym.vector.AsyncVectorEnv if asynchronous and num_envs > 1 else gym.vector.SyncVectorEnv
    envs = cls(fns, autoreset_mode=AutoresetMode.SAME_STEP)
    return envs
