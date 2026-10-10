"""Learn a chess value function by TD(lambda) from games the agent plays.

The idea goes back to TD-Gammon (Tesauro 1995) and KnightCap/Giraffe for chess:
the agent plays games choosing moves by a shallow search over its own value
network, then each position's value is regressed toward the lambda-return built
from later positions' values and the final result.

With negamax values (V = value for the side to move) the lambda-return is::

    G_T = exact result (or truncation value) for the side to move at the last position
    G_t = -[(1 - lambda) * V(s_{t+1}) + lambda * G_{t+1}]

Games are a mix of self-play and games against baseline opponents (random, greedy),
which gives the from-scratch agent early, easy wins to learn from. Games that hit
``max_plies`` are scored by ``truncation_value``: ``material`` (tanh of the material
balance; a dense, mildly hand-crafted signal that speeds early learning), ``bootstrap``
(the network's own estimate, pure RL) or ``draw``.
"""

from __future__ import annotations

import math
import random
import time
from pathlib import Path
from typing import Any

import chess
import numpy as np
import torch
import torch.nn.functional as F

from rlgames.chess.agents import ValueNetAgent, make_baseline
from rlgames.chess.arena import play_match
from rlgames.chess.encoding import NUM_FEATURES, encode_boards, material_balance, terminal_value
from rlgames.chess.model import ValueNet, save_value_net
from rlgames.logger import Logger

DEFAULTS: dict[str, Any] = {
    "lr": 1e-3,
    "weight_decay": 1e-5,
    "hidden": [512, 256, 64],
    "batch_size": 512,
    "buffer_size": 300_000,
    "games_per_iter": 16,
    "train_steps_per_iter": 64,
    "td_lambda": 0.7,
    "max_plies": 200,
    "truncation_value": "material",  # material | bootstrap | draw
    "material_scale": 5.0,
    "search_depth": 1,
    "temperature_start": 0.3,
    "temperature_end": 0.05,
    "temperature_decay_fraction": 0.5,
    "opening_plies_max": 6,
    "opponents": {"self": 0.6, "random": 0.2, "greedy": 0.2},
}


class PositionBuffer:
    def __init__(self, capacity: int):
        self.x = np.zeros((capacity, NUM_FEATURES), dtype=np.float32)
        self.y = np.zeros(capacity, dtype=np.float32)
        self.capacity, self.pos, self.size = capacity, 0, 0

    def add(self, x: np.ndarray, y: np.ndarray) -> None:
        for i in range(len(x)):
            self.x[self.pos], self.y[self.pos] = x[i], y[i]
            self.pos = (self.pos + 1) % self.capacity
            self.size = min(self.size + 1, self.capacity)

    def sample(self, n: int, rng: np.random.Generator) -> tuple[torch.Tensor, torch.Tensor]:
        idx = rng.integers(0, self.size, size=n)
        return torch.from_numpy(self.x[idx]), torch.from_numpy(self.y[idx])


def lambda_returns(values: np.ndarray, final_value: float, lam: float) -> np.ndarray:
    """values[t] = V(s_t) for t < T (side-to-move view); returns G_t for t < T."""
    T = len(values)
    G = np.zeros(T, dtype=np.float32)
    next_v, next_g = final_value, final_value
    for t in range(T - 1, -1, -1):
        G[t] = -((1.0 - lam) * next_v + lam * next_g)
        next_v, next_g = values[t], G[t]
    return G


def play_training_game(learner: ValueNetAgent, opponent, learner_white: bool, hp: dict[str, Any], rng: random.Random):
    board = chess.Board()
    for _ in range(rng.randint(0, hp["opening_plies_max"])):
        board.push(rng.choice(list(board.legal_moves)))
    positions = []
    while True:
        tv = terminal_value(board, claim_draw=True)
        if tv is not None:
            return positions, board, tv, "terminal"
        if board.ply() >= hp["max_plies"]:
            return positions, board, None, "truncated"
        positions.append(board.copy(stack=False))
        if opponent is None or (board.turn == chess.WHITE) == learner_white:
            move = learner.select_move(board)
        else:
            move = opponent.select_move(board)
        board.push(move)


def train(cfg: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    hp = {**DEFAULTS, **cfg.get("algo", {})}
    seed = cfg.get("seed", 0)
    total_games = int(cfg["total_games"])
    eval_cfg = cfg.get("eval", {})
    eval_interval = int(eval_cfg.get("interval_games", max(total_games // 20, 1)))
    eval_games = int(eval_cfg.get("games", 20))
    eval_opponents = list(eval_cfg.get("opponents", ["random", "greedy"]))
    eval_max_plies = int(eval_cfg.get("max_plies", 300))

    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)
    net = ValueNet(hp["hidden"])
    opt = torch.optim.AdamW(net.parameters(), lr=hp["lr"], weight_decay=hp["weight_decay"])
    buffer = PositionBuffer(int(hp["buffer_size"]))
    logger = Logger(run_dir, use_tensorboard=cfg.get("tensorboard", True))
    learner = ValueNetAgent(net, depth=hp["search_depth"], temperature=hp["temperature_start"], seed=seed, name="learner")
    opponent_names = list(hp["opponents"])
    opponent_weights = [float(hp["opponents"][k]) for k in opponent_names]
    baselines = {name: make_baseline(name, seed) for name in opponent_names if name != "self"}

    games = 0
    next_eval = eval_interval
    best_score = -math.inf
    history = []
    stats: dict[str, list[float]] = {"result": [], "plies": [], "truncated": [], "loss": []}
    t_start = time.time()

    def checkpoint_extra() -> dict[str, Any]:
        return {"games": games, "config": cfg, "search_depth": hp["search_depth"]}

    while games < total_games:
        frac = min(games / max(hp["temperature_decay_fraction"] * total_games, 1), 1.0)
        learner.temperature = hp["temperature_start"] + frac * (hp["temperature_end"] - hp["temperature_start"])
        net.eval()
        for _ in range(int(hp["games_per_iter"])):
            opp_name = rng.choices(opponent_names, opponent_weights)[0]
            opponent = None if opp_name == "self" else baselines[opp_name]
            learner_white = rng.random() < 0.5
            positions, final_board, final_value, how = play_training_game(learner, opponent, learner_white, hp, rng)
            if final_value is None:  # truncated
                if hp["truncation_value"] == "material":
                    final_value = math.tanh(material_balance(final_board) / hp["material_scale"])
                elif hp["truncation_value"] == "bootstrap":
                    final_value = float(net.evaluate([final_board])[0])
                else:
                    final_value = 0.0
            if positions:
                x = encode_boards(positions)
                with torch.no_grad():
                    values = net(torch.from_numpy(x)).numpy()
                buffer.add(x, lambda_returns(values, final_value, hp["td_lambda"]))
            # result from the learner's point of view (self-play counts as white)
            outcome = final_board.outcome(claim_draw=True)
            if outcome is None or outcome.winner is None:
                res = 0.0
            else:
                res = 1.0 if outcome.winner == (learner_white or opponent is None) else -1.0
            stats["result"].append(res if opponent is not None else float("nan"))
            stats["plies"].append(final_board.ply())
            stats["truncated"].append(float(how == "truncated"))
            games += 1

        net.train()
        if buffer.size >= hp["batch_size"]:
            for _ in range(int(hp["train_steps_per_iter"])):
                xb, yb = buffer.sample(int(hp["batch_size"]), np_rng)
                loss = F.mse_loss(net(xb), yb)
                opt.zero_grad()
                loss.backward()
                opt.step()
                stats["loss"].append(loss.item())
        net.eval()

        recent = [r for r in stats["result"][-200:] if not math.isnan(r)]
        logger.log(games, {
            "train/loss": float(np.mean(stats["loss"][-200:])) if stats["loss"] else float("nan"),
            "train/score_vs_baselines": (float(np.mean(recent)) + 1) / 2 if recent else float("nan"),
            "train/game_plies": float(np.mean(stats["plies"][-200:])),
            "train/truncated_frac": float(np.mean(stats["truncated"][-200:])),
            "train/temperature": learner.temperature,
            "train/buffer": buffer.size,
            "perf/games_per_min": games / (time.time() - t_start) * 60,
        }, echo=games % (int(hp["games_per_iter"]) * 10) == 0)

        if games >= next_eval or games >= total_games:
            next_eval += eval_interval
            agent = ValueNetAgent(net, depth=hp["search_depth"], temperature=0.0, seed=seed, name="learner")
            metrics, scores = {}, []
            for opp in eval_opponents:
                m = play_match(agent, make_baseline(opp, seed + 1), games=eval_games, max_plies=eval_max_plies,
                               opening_plies=4, seed=games)
                metrics[f"eval/score_vs_{opp}"] = m.score
                scores.append(m.score)
                print(f"  [eval @ {games} games] {m}", flush=True)
            mean_score = float(np.mean(scores))
            metrics["eval/mean_score"] = mean_score
            logger.log(games, metrics)
            history.append({"games": games, **metrics})
            save_value_net(run_dir / "latest.pt", net, **checkpoint_extra())
            if mean_score >= best_score:
                best_score = mean_score
                save_value_net(run_dir / "best.pt", net, **checkpoint_extra())

    summary = {"algo": "chess_td", "total_games": games, "best_eval_mean_score": best_score,
               "wall_clock_sec": round(time.time() - t_start, 1), "eval_history": history}
    logger.write_summary(summary)
    logger.close()
    return summary
