"""End-to-end smoke tests: tiny training runs, checkpoint loading, video and HTML output."""

from pathlib import Path

import pytest

from rlgames.agents.policy import load_policy
from rlgames.cli import main
from rlgames.config import load_config
from rlgames.evaluate import run_episodes
from rlgames.record import record_episodes

CONFIGS = Path(__file__).resolve().parent.parent / "configs"


def _train(tmp_path, config, *overrides):
    main(["train", str(CONFIGS / config), *overrides, "tensorboard=false", "--runs-dir", str(tmp_path)])
    (run_dir,) = [p for p in tmp_path.iterdir() if p.is_dir()]
    return run_dir


@pytest.mark.parametrize("config,overrides", [
    ("cartpole_dqn.yaml", ["total_steps=1500", "algo.learning_starts=500", "eval.interval=1500", "eval.episodes=2"]),
    ("cartpole_ppo.yaml", ["total_steps=2048", "num_envs=2", "eval.interval=2048", "eval.episodes=2"]),
])
def test_cartpole_train_watch_bench(tmp_path, config, overrides):
    run_dir = _train(tmp_path, config, *overrides)
    for name in ("best.pt", "latest.pt", "metrics.csv", "summary.json", "config.yaml"):
        assert (run_dir / name).exists(), name
    policy = load_policy(run_dir)
    stats = run_episodes(policy, policy.env, episodes=2)
    assert stats["episodes"] == 2 and stats["mean_return"] > 0
    video = record_episodes(policy, policy.env, tmp_path / "v.gif", episodes=1, max_steps=50)
    assert (tmp_path / "v.gif").stat().st_size > 0 and len(video) == 1


def test_space_invaders_dqn_runs(tmp_path):
    run_dir = _train(tmp_path, "space_invaders_dqn.yaml", "total_steps=600", "num_envs=2", "algo.learning_starts=200",
                     "algo.buffer_size=1000", "eval.interval=600", "eval.episodes=1")
    policy = load_policy(run_dir / "latest.pt")
    assert policy.n_actions == 6


def test_space_invaders_ppo_runs(tmp_path):
    _train(tmp_path, "space_invaders_ppo.yaml", "total_steps=256", "num_envs=2", "async_envs=false",
           "algo.num_steps=64", "eval.interval=256", "eval.episodes=1")


def test_chess_td_train_and_bench(tmp_path):
    run_dir = _train(tmp_path, "chess_td.yaml", "total_games=8", "algo.games_per_iter=4", "algo.batch_size=64",
                     "algo.train_steps_per_iter=2", "algo.max_plies=40", "eval.interval_games=8", "eval.games=2",
                     "eval.opponents=[random]", "eval.max_plies=40")
    assert (run_dir / "best.pt").exists()
    out = tmp_path / "games"
    main(["chess-bench", str(run_dir), "--opponents", "random", "--games", "2", "--max-plies", "30",
          "--no-record", "--save-games", str(out)])
    assert list(out.glob("*.html"))
    main(["chess-watch", str(run_dir / "best.pt"), "greedy", "--max-plies", "20", "--out", str(tmp_path / "g.html"), "--gif"])
    assert (tmp_path / "g.html").exists() and (tmp_path / "g.gif").exists()


def test_configs_load():
    for path in CONFIGS.glob("*.yaml"):
        cfg = load_config(path)
        assert cfg["algo"]["name"] in ("dqn", "ppo", "chess_td"), path
