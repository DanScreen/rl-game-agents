import numpy as np
import pytest

from rlgames.agents.dqn import ReplayBuffer, linear_schedule
from rlgames.benchmark import record_result
from rlgames.config import apply_overrides
from rlgames.envs import make_env, make_vector_env


def test_overrides_parse_types_and_nesting():
    cfg = apply_overrides({"a": 1, "algo": {"lr": 0.1}}, ["a=5", "algo.lr=3e-4", "algo.hidden=[32,32]", "new.flag=true"])
    assert cfg["a"] == 5
    assert cfg["algo"]["lr"] == pytest.approx(3e-4)
    assert cfg["algo"]["hidden"] == [32, 32]
    assert cfg["new"]["flag"] is True


def test_override_requires_equals():
    with pytest.raises(ValueError):
        apply_overrides({}, ["oops"])


def test_linear_schedule():
    assert linear_schedule(1.0, 0.1, 100, 0) == pytest.approx(1.0)
    assert linear_schedule(1.0, 0.1, 100, 50) == pytest.approx(0.55)
    assert linear_schedule(1.0, 0.1, 100, 500) == pytest.approx(0.1)


def test_replay_buffer_reconstructs_stacked_next_obs():
    rng = np.random.default_rng(0)
    buf = ReplayBuffer(10, (4, 3, 3), np.uint8)
    frames = rng.integers(0, 255, size=(8, 3, 3), dtype=np.uint8)
    obs = np.stack([frames[i:i + 4] for i in range(4)])  # 4 consecutive stacked observations
    nxt = np.stack([frames[i + 1:i + 5] for i in range(4)])
    buf.add(obs, np.arange(4), np.ones(4), nxt, np.zeros(4))
    b_obs, b_act, _, b_next, _ = buf.sample(64, rng)
    for o, a, n in zip(b_obs.numpy(), b_act.numpy(), b_next.numpy()):
        np.testing.assert_array_equal(o, obs[a])
        np.testing.assert_array_equal(n, nxt[a])


def test_space_invaders_env_shape():
    env = make_env("space_invaders")
    obs, _ = env.reset(seed=0)
    assert obs.shape == (4, 84, 84) and obs.dtype == np.uint8
    assert env.action_space.n == 6
    env.close()


def test_vector_env_exposes_final_obs():
    envs = make_vector_env("cartpole", 2, seed=0)
    envs.reset(seed=0)
    for _ in range(500):
        _, _, term, trunc, info = envs.step(np.zeros(2, dtype=np.int64))  # always push left -> falls fast
        if (term | trunc).any():
            assert "final_obs" in info
            break
    else:
        pytest.fail("episode never ended")
    envs.close()


def test_leaderboard_written(tmp_path):
    results, board = tmp_path / "r.jsonl", tmp_path / "LB.md"
    stats = {"episodes": 3, "mean_return": 10.0, "std_return": 1.0, "median_return": 10.0, "min_return": 9.0, "max_return": 11.0}
    record_result({"game": "CartPole-v1", "agent": "a", **stats}, results, board)
    record_result({"game": "chess", "agent": "x", "opponent": "random", "games": 2, "wins": 2, "draws": 0,
                   "losses": 0, "score": 1.0, "elo_diff": 400.0}, results, board)
    text = board.read_text()
    assert "## CartPole-v1" in text and "## chess" in text and "| x | random |" in text
