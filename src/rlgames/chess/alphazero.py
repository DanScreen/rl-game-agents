"""AlphaZero-style training: the network learns only from games against itself.

Loop:
  1. Self-play: ``parallel_games`` games run at once. Each move is chosen by MCTS
     (Dirichlet noise at the root; moves sampled from visit counts for the first
     ``temperature_plies`` plies, then the most-visited move).
  2. Every position is stored with the search's visit distribution (policy target)
     and, once the game ends, the result from that position's mover's view (value target).
  3. The network is trained on samples from a replay buffer of recent positions:
     loss = cross-entropy(policy, visits) + MSE(value, result), with weight decay.

Nothing chess-specific is given beyond the rules, unless ``truncation_value: material``
is set (games hitting ``max_plies`` are then scored by material instead of as draws).

The architecture comes from ``model.arch`` (see ``rlgames.chess.nets``), so different
networks can be trained with an identical procedure and compared.
"""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any

import chess
import numpy as np
import torch
import torch.nn.functional as F

from rlgames.agents.policy import save_checkpoint
from rlgames.chess.agents import make_baseline
from rlgames.chess.arena import play_match
from rlgames.chess.az_encoding import NUM_ACTIONS, NUM_PLANES, encode_planes
from rlgames.chess.encoding import material_balance
from rlgames.chess.mcts import AlphaZeroAgent, BatchedMCTS, Node, net_evaluator
from rlgames.chess.nets import build_net, count_params
from rlgames.logger import Logger, cpu_state_dict, resolve_device

DEFAULTS: dict[str, Any] = {
    "simulations": 64,
    "c_puct": 1.5,
    "fpu": 0.0,
    "dirichlet_alpha": 0.3,
    "dirichlet_frac": 0.25,
    "temperature_plies": 30,
    "max_plies": 200,
    "truncation_value": "draw",  # draw | material
    "material_scale": 5.0,
    "parallel_games": 32,
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "batch_size": 256,
    "buffer_size": 200_000,
    "train_steps_per_iter": 100,
    "games_per_iter": 32,
    "min_buffer": 2_000,
    # Start-position diversity (still pure self-play: nothing about piece values is given).
    "opening_random_plies": [0, 0],  # random moves played before self-play starts, [min, max]
    "curriculum_frac": 0.0,  # fraction of games starting with a random subset of pieces removed
    "curriculum_keep": [0.2, 0.7],  # fraction of non-king pieces kept in those games, [min, max]
}


class SelfPlayBuffer:
    """Positions with sparse policy targets (only legal moves get probability)."""

    def __init__(self, capacity: int):
        self.planes = np.zeros((capacity, NUM_PLANES, 8, 8), dtype=np.uint8)  # planes 18 is the only non-binary one
        self.halfmove = np.zeros(capacity, dtype=np.float32)
        self.actions: list[np.ndarray | None] = [None] * capacity
        self.probs: list[np.ndarray | None] = [None] * capacity
        self.values = np.zeros(capacity, dtype=np.float32)
        self.capacity, self.pos, self.size = capacity, 0, 0

    def add(self, planes: np.ndarray, actions: np.ndarray, probs: np.ndarray, value: float) -> None:
        i = self.pos
        self.planes[i] = planes > 0.5
        self.halfmove[i] = planes[18, 0, 0]
        self.actions[i], self.probs[i], self.values[i] = actions, probs.astype(np.float32), value
        self.pos = (i + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, n: int, rng: np.random.Generator) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        idx = rng.integers(0, self.size, size=n)
        x = self.planes[idx].astype(np.float32)
        x[:, 18] = self.halfmove[idx, None, None]
        pi = np.zeros((n, NUM_ACTIONS), dtype=np.float32)
        for row, i in enumerate(idx):
            pi[row, self.actions[i]] = self.probs[i]
        return torch.from_numpy(x), torch.from_numpy(pi), torch.from_numpy(self.values[idx])


class SelfPlayGame:
    def __init__(self, board: chess.Board | None = None) -> None:
        self.board = board or chess.Board()
        self.root = Node()
        self.history: list[tuple[np.ndarray, np.ndarray, np.ndarray, chess.Color]] = []


def start_position(rng: np.random.Generator, hp: dict[str, Any]) -> chess.Board:
    """The standard start, optionally after some random moves, or (curriculum) the start
    position with a random subset of non-king pieces removed. Removing material puts
    checkmates within reach of a weak search, so decisive results (and value signal) keep
    coming instead of self-play collapsing into repetition draws."""
    for _ in range(100):
        board = chess.Board()
        if rng.random() < hp["curriculum_frac"]:
            keep = rng.uniform(*hp["curriculum_keep"])
            for square, piece in list(board.piece_map().items()):
                if piece.piece_type != chess.KING and rng.random() > keep:
                    board.remove_piece_at(square)
            board.castling_rights = board.clean_castling_rights()
        lo, hi = hp["opening_random_plies"]
        for _ in range(int(rng.integers(lo, hi + 1))):
            moves = list(board.legal_moves)
            if not moves:
                break
            board.push(moves[int(rng.integers(len(moves)))])
        if board.is_valid() and not board.is_game_over(claim_draw=True):
            return board
    return chess.Board()


def game_result(board: chess.Board, hp: dict[str, Any]) -> tuple[float, str]:
    """Result for white (+1/0/-1, or a material score if truncated) and how the game ended."""
    outcome = board.outcome(claim_draw=True)
    if outcome is not None:
        return (0.0 if outcome.winner is None else (1.0 if outcome.winner == chess.WHITE else -1.0)), outcome.termination.name.lower()
    if hp["truncation_value"] == "material":
        mb = material_balance(board) * (1 if board.turn == chess.WHITE else -1)
        return math.tanh(mb / hp["material_scale"]), "truncated"
    return 0.0, "truncated"


def checkpoint_payload(net: torch.nn.Module, model_cfg: dict[str, Any], cfg: dict[str, Any], games: int) -> dict[str, Any]:
    return {"algo": "chess_az", "model": model_cfg, "state_dict": cpu_state_dict(net), "games": games,
            "params": count_params(net), "config": cfg}


def load_az_net(path: str | Path) -> tuple[torch.nn.Module, dict[str, Any]]:
    path = Path(path)
    if path.is_dir():
        path = path / "best.pt" if (path / "best.pt").exists() else path / "latest.pt"
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    net = build_net(ckpt["model"])
    net.load_state_dict(ckpt["state_dict"])
    net.eval()
    return net, ckpt


def train(cfg: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    hp = {**DEFAULTS, **cfg.get("algo", {})}
    model_cfg = dict(cfg.get("model", {"arch": "resnet"}))
    seed = int(cfg.get("seed", 0))
    total_games = int(cfg["total_games"])
    eval_cfg = cfg.get("eval", {})
    eval_interval = int(eval_cfg.get("interval_games", max(total_games // 10, 1)))
    eval_games = int(eval_cfg.get("games", 20))
    eval_opponents = list(eval_cfg.get("opponents", ["random", "greedy"]))
    eval_sims = int(eval_cfg.get("simulations", hp["simulations"]))
    eval_max_plies = int(eval_cfg.get("max_plies", 300))

    rng = np.random.default_rng(seed)
    device = resolve_device(cfg.get("device", "auto"))
    net = build_net(model_cfg).to(device)
    if cfg.get("init_from"):
        init_net, _ = load_az_net(cfg["init_from"])
        net.load_state_dict(init_net.state_dict())
        print(f"Initialised from {cfg['init_from']}", flush=True)
    n_params = count_params(net)
    print(f"Training {model_cfg.get('arch', 'resnet')} ({n_params:,} parameters) on {device}", flush=True)
    opt = torch.optim.AdamW(net.parameters(), lr=hp["lr"], weight_decay=hp["weight_decay"])
    buffer = SelfPlayBuffer(int(hp["buffer_size"]))
    logger = Logger(run_dir, use_tensorboard=cfg.get("tensorboard", True))
    mcts = BatchedMCTS(net_evaluator(net, device), c_puct=hp["c_puct"], fpu=hp["fpu"])

    def add_noise(node: Node) -> None:
        node.add_dirichlet_noise(hp["dirichlet_alpha"], hp["dirichlet_frac"], rng)

    games_done, next_train, next_eval = 0, int(hp["games_per_iter"]), eval_interval
    best_score = -math.inf
    history, results, plies, endings, losses = [], [], [], [], []
    t_start = time.time()
    selfplay_time = train_time = 0.0
    slots = [SelfPlayGame(start_position(rng, hp)) for _ in range(int(hp["parallel_games"]))]
    games_started = len(slots)

    while games_done < total_games:
        # ---- one move in every active game -------------------------------------------
        t0 = time.time()
        net.eval()
        active = [g for g in slots if g is not None]
        mcts.run([g.board for g in active], [g.root for g in active], int(hp["simulations"]), on_root_expanded=add_noise)
        for k, g in enumerate(slots):
            if g is None:
                continue
            root = g.root
            visits = root.N
            if len(g.history) < hp["temperature_plies"]:  # plies played in this game
                a = int(rng.choice(len(visits), p=visits / visits.sum()))
            else:
                a = int(np.argmax(visits))
            g.history.append((encode_planes(g.board), root.actions, root.visit_distribution(), g.board.turn))
            g.board.push(root.moves[a])
            child = root.children[a]
            g.root = child if child is not None else Node()  # reuse the searched subtree
            if g.root.expanded:
                add_noise(g.root)

            over = g.board.is_game_over(claim_draw=True) or len(g.history) >= hp["max_plies"]
            if over:
                z_white, how = game_result(g.board, hp)
                for planes, actions, probs, turn in g.history:
                    buffer.add(planes, actions, probs, z_white if turn == chess.WHITE else -z_white)
                results.append(z_white)
                plies.append(g.board.ply())
                endings.append(how)
                games_done += 1
                if games_started < total_games:
                    slots[k] = SelfPlayGame(start_position(rng, hp))
                    games_started += 1
                else:
                    slots[k] = None
        selfplay_time += time.time() - t0

        # ---- learn ------------------------------------------------------------------------
        if games_done >= next_train and buffer.size >= hp["min_buffer"]:
            next_train += int(hp["games_per_iter"])
            t0 = time.time()
            net.train()
            for _ in range(int(hp["train_steps_per_iter"])):
                x, pi, z = (t.to(device) for t in buffer.sample(int(hp["batch_size"]), rng))
                logits, v = net(x)
                policy_loss = -(pi * F.log_softmax(logits, dim=1)).sum(dim=1).mean()
                value_loss = F.mse_loss(v, z)
                loss = policy_loss + value_loss
                opt.zero_grad()
                loss.backward()
                opt.step()
                losses.append((policy_loss.item(), value_loss.item()))
            net.eval()
            train_time += time.time() - t0
            recent = endings[-200:]
            logger.log(games_done, {
                "train/policy_loss": float(np.mean([p for p, _ in losses[-200:]])),
                "train/value_loss": float(np.mean([v for _, v in losses[-200:]])),
                "selfplay/decisive_frac": float(np.mean([r != 0 for r in results[-200:]])),
                "selfplay/checkmate_frac": float(np.mean([e == "checkmate" for e in recent])),
                "selfplay/truncated_frac": float(np.mean([e == "truncated" for e in recent])),
                "selfplay/white_score": float(np.mean(results[-200:]) + 1) / 2,
                "selfplay/game_plies": float(np.mean(plies[-200:])),
                "buffer/positions": buffer.size,
                "perf/games_per_hour": games_done / (time.time() - t_start) * 3600,
                "perf/selfplay_frac": selfplay_time / max(selfplay_time + train_time, 1e-9),
            })
        elif games_done >= next_train:
            next_train += int(hp["games_per_iter"])  # buffer still warming up

        # ---- evaluate ---------------------------------------------------------------------
        if games_done >= next_eval or games_done >= total_games:
            next_eval += eval_interval
            agent = AlphaZeroAgent(net, simulations=eval_sims, c_puct=hp["c_puct"], name="learner", device=str(device))
            metrics, scores = {}, []
            for opp in eval_opponents:
                m = play_match(agent, make_baseline(opp, seed + 1), games=eval_games, max_plies=eval_max_plies,
                               opening_plies=4, seed=games_done)
                metrics[f"eval/score_vs_{opp}"] = m.score
                scores.append(m.score)
                print(f"  [eval @ {games_done} games] {m}", flush=True)
            mean_score = float(np.mean(scores))
            metrics["eval/mean_score"] = mean_score
            logger.log(games_done, metrics)
            history.append({"games": games_done, "elapsed_sec": round(time.time() - t_start), **metrics})
            payload = checkpoint_payload(net, model_cfg, cfg, games_done)
            save_checkpoint(run_dir / "latest.pt", **payload)
            if mean_score >= best_score:
                best_score = mean_score
                save_checkpoint(run_dir / "best.pt", **payload)

    summary = {
        "algo": "chess_az", "arch": model_cfg.get("arch", "resnet"), "model": model_cfg, "params": n_params,
        "total_games": games_done, "best_eval_mean_score": best_score,
        "wall_clock_sec": round(time.time() - t_start, 1), "selfplay_sec": round(selfplay_time, 1),
        "train_sec": round(train_time, 1), "eval_history": history,
    }
    logger.write_summary(summary)
    logger.close()
    return summary
