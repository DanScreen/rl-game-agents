# rl-game-agents

Training neural networks to play games with reinforcement learning, from scratch, on
three games of increasing difficulty:

| Game | Why | Algorithms | Observation |
|---|---|---|---|
| **CartPole** | the "very simple game": balance a pole; trains in under a minute, ideal for checking things work | DQN, PPO | 4 numbers |
| **Space Invaders** (Atari 2600) | classic deep-RL benchmark from the DQN paper | DQN (Double + Dueling), PPO | raw pixels (4 stacked 84x84 frames) |
| **Chess** | two-player, huge state space, sparse reward | TD(λ) self-play value network + search | board encoding |

Everything runs from one command-line tool, `rlg`, which can **train**, **watch** (record
videos / HTML replays), **benchmark** (leaderboard against baselines) and **play** against
the chess agent.

<!-- RESULTS -->

## Quick start

```bash
git clone https://github.com/DanScreen/rl-game-agents && cd rl-game-agents
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"          # PyTorch, Gymnasium, ALE (Atari ROMs included), python-chess
pytest -q                        # ~1 minute smoke test of everything
```

Python 3.10+. A GPU is optional: training picks CUDA or Apple-silicon MPS automatically
(`device: auto`; force one with `device=cpu` etc.). On CPU-only machines you can save
~2 GB of downloads with `pip install torch --index-url https://download.pytorch.org/whl/cpu`
before the install above.

### 1. Watch the pretrained agents

Pretrained checkpoints from the first training runs are in [`pretrained/`](pretrained/).

```bash
# record videos (MP4, or .gif with --out x.gif) -> videos/
rlg watch pretrained/cartpole_ppo.pt
rlg watch pretrained/space_invaders_ppo.pt --episodes 2
rlg watch random --env space_invaders               # for comparison

# watch live in a window instead (needs a display)
rlg watch pretrained/space_invaders_dqn.pt --human

# chess: writes an HTML replay viewer (open it in a browser) and a GIF
rlg chess-watch pretrained/chess_td.pt greedy --gif
rlg chess-watch pretrained/chess_td.pt pretrained/chess_td.pt --games 3   # self-play

# play chess against it yourself in the terminal
rlg chess-play pretrained/chess_td.pt --color white
```

### 2. Launch training runs

```bash
rlg train configs/cartpole_ppo.yaml                 # ~30 s on a laptop
rlg train configs/cartpole_dqn.yaml
rlg train configs/space_invaders_ppo.yaml           # hours on CPU; much faster on GPU
rlg train configs/space_invaders_dqn.yaml
rlg train configs/chess_td.yaml

# override any config value on the command line (dots for nested keys)
rlg train configs/space_invaders_ppo.yaml total_steps=10000000 num_envs=16 algo.lr=3e-4 --name si_ppo_big

# continue training from any checkpoint (fine-tune, extend a run, or chess stage 2)
rlg train configs/space_invaders_ppo.yaml init_from=pretrained/space_invaders_ppo.pt total_steps=5000000
rlg train configs/chess_td_stage2.yaml init_from=runs/chess_td-<timestamp>/best.pt

# or start several in the background
./scripts/train_all.sh                              # all five
./scripts/train_all.sh cartpole_ppo chess_td        # a subset
```

Each run writes to `runs/<name>-<timestamp>/`:

| File | Contents |
|---|---|
| `config.yaml` | the exact config used (after overrides) |
| `metrics.csv` | training and evaluation metrics over time |
| `tb/` | TensorBoard logs: `tensorboard --logdir runs` |
| `best.pt` / `latest.pt` | checkpoint with the best evaluation score / the most recent one |
| `summary.json` | final results and evaluation history |

During training the agent is evaluated periodically on held-out seeds (`eval.*` in the configs).

### 3. Benchmark

```bash
# Gym/Atari agents: N episodes each, mean/std/median/min/max score
rlg bench runs/space_invaders_ppo-*/ pretrained/space_invaders_dqn.pt --with-random --episodes 30

# chess: a match against each opponent (colours alternate, random openings so games differ)
rlg chess-bench pretrained/chess_td.pt --opponents random greedy minimax2 --games 40
rlg chess-bench pretrained/chess_td.pt --depth 2 --save-games videos/   # 2-ply search + HTML replays

rlg leaderboard                                     # print the leaderboard

# learning curves of one or more runs -> PNG
rlg plot runs/space_invaders_dqn-*/ runs/space_invaders_ppo-*/ --out videos/si_curves.png
rlg plot runs/chess_td-*/ -m eval/score_vs_random -m eval/score_vs_greedy -m eval/score_vs_minimax2
```

Results are appended to `benchmarks/results.jsonl`, and
[`benchmarks/LEADERBOARD.md`](benchmarks/LEADERBOARD.md) is regenerated each time.
Chess baselines: `random`, `greedy` (takes the most material, mates in one when it can),
`minimax2` / `minimax3` (alpha-beta search on material), and `stockfish0`–`stockfish20` if a
`stockfish` binary is installed (`brew install stockfish` / `apt install stockfish`).

## How it works

### CartPole and Space Invaders: DQN and PPO

* **DQN** ([`src/rlgames/agents/dqn.py`](src/rlgames/agents/dqn.py)): Q-network trained
  from a replay buffer against a target network (Mnih et al. 2015), with Double-DQN targets,
  optional dueling head, Huber loss, ε-greedy exploration and optional soft (Polyak) target
  updates. For Atari the buffer stores frame stacks compactly (the next observation shares 3
  of its 4 frames with the current one).
* **PPO** ([`src/rlgames/agents/ppo.py`](src/rlgames/agents/ppo.py)): clipped policy
  gradient with GAE and parallel environments, following the
  ["37 implementation details"](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/)
  recipe; truncated episodes bootstrap from the value of the true final state.
* **Atari setup** ([`src/rlgames/envs/atari.py`](src/rlgames/envs/atari.py)): the standard
  preprocessing (frame skip 4 with max-pooling, 84x84 grayscale, 4-frame stack, up to 30
  no-ops at reset, sticky actions with p = 0.25). Rewards are clipped to ±1 for learning only;
  all reported scores are raw game scores.

### Chess: TD(λ) self-play

[`src/rlgames/chess/`](src/rlgames/chess/) learns a value network V(position) ∈ [-1, 1]
(the expected result for the side to move) with no human games and no engine, in the
tradition of TD-Gammon, KnightCap and Giraffe:

1. **Play.** The agent chooses moves by searching one ply ahead with its own value network
   (softmax over move scores early in training for exploration, greedy later). Games mix
   self-play with games against the random and greedy baselines, which give a from-scratch
   agent some wins to learn from.
2. **Learn.** Each position's value is regressed toward its TD(λ) return, built from the values
   of later positions and the final result. Games that reach the ply limit are scored by
   material (`truncation_value: material`); set `bootstrap` for a purely self-taught signal.
3. **Encoding.** Positions are always shown from the mover's point of view (the board is
   mirrored when black is to move), as 12 piece-occupancy planes plus castling rights.

At play time the same network can search deeper: `--depth 2` (full-width, one batched network
call per move) or `--depth 3`+ (alpha-beta with the network at the leaves, ~0.2 s/move at
depth 3). Deeper search is noticeably stronger. `configs/chess_td_stage2.yaml` continues
training a stage-1 network against tougher opponents, including the 2-ply minimax baseline.

## Project layout

```
configs/                 YAML experiment configs (one per game x algorithm)
src/rlgames/
  cli.py                 the `rlg` command
  envs/                  environment factory + Atari preprocessing
  agents/                networks, DQN, PPO, checkpoint/policy loading
  chess/                 encoding, value net, TD training, baselines, arena, HTML/GIF rendering
  evaluate.py            episode runner used by training-time eval and `rlg bench`
  record.py              video recording / live viewing
  benchmark.py           results store + leaderboard generation
  plot.py                learning-curve plots
pretrained/              checkpoints from the first training runs
benchmarks/              results.jsonl + generated LEADERBOARD.md
scripts/train_all.sh     launch several runs in the background
tests/                   unit + end-to-end smoke tests (run in CI)
```

## Adding a new game

Any Gymnasium environment with discrete actions works without code changes; put its id in
a config (`env: Acrobot-v1`, `env: ALE/Breakout-v5`, ...). Atari ids (`ALE/...`) get the
Atari preprocessing automatically. Short aliases live in `src/rlgames/envs/__init__.py`.

## Ideas for next steps

* **Chess:** AlphaZero-style MCTS with a policy + value network; Stockfish benchmarks; more
  self-play compute; a residual CNN over 8x8 planes.
* **Atari:** longer runs on a GPU (10M+ steps), Rainbow components (prioritised replay,
  n-step returns, distributional RL), more games (Breakout and Pong are ready to go).
* **Tooling:** hyperparameter sweeps, multi-seed benchmarking with confidence intervals.
