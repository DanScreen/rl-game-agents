# Pretrained agents

Trained overnight on a 4-core CPU (no GPU). Scores are from `rlg bench` / `rlg chess-bench`;
see [`../benchmarks/LEADERBOARD.md`](../benchmarks/LEADERBOARD.md) for the full tables.

| File | Game | How it was trained | Result |
|---|---|---|---|
| `cartpole_ppo.pt` | CartPole | `configs/cartpole_ppo.yaml`, 200k steps (~1 min) | 500 / 500 on all 30 episodes (random: 19.5) |
| `cartpole_dqn.pt` | CartPole | `configs/cartpole_dqn.yaml`, 50k steps (~3 min) | 500 / 500 on all 30 episodes |
| `space_invaders_ppo.pt` | Space Invaders | `configs/space_invaders_ppo.yaml` for 2.5M steps, then continued with `init_from=` for another 2.5M (`algo.lr=1.5e-4`); ~4.5 h total | **543** mean over 50 games (random: 142) |
| `space_invaders_dqn.pt` | Space Invaders | `configs/space_invaders_dqn.yaml`, 1M steps (~3.5 h) | **426** mean over 50 games |
| `chess_td.pt` | Chess | stage 1 `configs/chess_td.yaml` (30k games) + stage 2 `configs/chess_td_stage2.yaml` (~18k games) | depth 2: 99% vs greedy, **50% vs minimax-2** (+12 =16 -12) |
| `chess_td_stage1.pt` | Chess | stage 1 only | depth 1: 69% vs greedy, 10% vs minimax-2 |

Notes

* Each checkpoint stores the config it was trained with (`torch.load(path)["config"]`).
* The PPO Space Invaders checkpoint's `step` field counts only the continuation run (2.5M);
  it has seen 5M steps in total.
* For both Space Invaders runs the final checkpoint benchmarked better over 30 games than the
  "best" one chosen from 5-game evaluations during training, so the configs now evaluate on 10
  games.
* Atari scores are with sticky actions (p = 0.25), so the same agent scores differently from
  game to game. Expect roughly ±25 on the 50-game means.
