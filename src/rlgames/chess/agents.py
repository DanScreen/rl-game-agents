"""Chess players. Every agent implements ``select_move(board) -> chess.Move``.

Baselines (for benchmarking learned agents):

* ``RandomAgent``          uniformly random legal move
* ``GreedyMaterialAgent``  mates in one if possible, otherwise grabs the most material (1 ply)
* ``MinimaxAgent``         alpha-beta on material to a fixed depth (default 2 ply)
* ``StockfishAgent``       a UCI engine (optional; needs the ``stockfish`` binary)

Learned:

* ``ValueNetAgent``        1- or 2-ply search over a learned value network
"""

from __future__ import annotations

import random
import shutil

import chess
import numpy as np

from rlgames.chess.encoding import material_balance, terminal_value
from rlgames.chess.model import ValueNet

MATE = 10_000


class RandomAgent:
    name = "random"

    def __init__(self, seed: int | None = None):
        self.rng = random.Random(seed)

    def select_move(self, board: chess.Board) -> chess.Move:
        return self.rng.choice(list(board.legal_moves))


class GreedyMaterialAgent:
    name = "greedy-material"

    def __init__(self, seed: int | None = None):
        self.rng = random.Random(seed)

    def select_move(self, board: chess.Board) -> chess.Move:
        best, best_score = [], -np.inf
        for move in board.legal_moves:
            board.push(move)
            if board.is_checkmate():
                score = MATE
            else:
                score = -material_balance(board)  # board.turn is now the opponent
            board.pop()
            if score > best_score:
                best, best_score = [move], score
            elif score == best_score:
                best.append(move)
        return self.rng.choice(best)


class MinimaxAgent:
    """Negamax alpha-beta on material; random tie-breaking among equally good moves."""

    def __init__(self, depth: int = 2, seed: int | None = None):
        self.depth = depth
        self.rng = random.Random(seed)
        self.name = f"minimax-d{depth}"

    def _search(self, board: chess.Board, depth: int, alpha: float, beta: float, ply: int) -> float:
        if board.is_checkmate():
            return -MATE + ply  # prefer faster mates / slower losses
        if board.is_stalemate() or board.is_insufficient_material():
            return 0.0
        if depth == 0:
            return float(material_balance(board))
        moves = sorted(board.legal_moves, key=lambda m: not board.is_capture(m))  # captures first
        value = -np.inf
        for move in moves:
            board.push(move)
            value = max(value, -self._search(board, depth - 1, -beta, -alpha, ply + 1))
            board.pop()
            alpha = max(alpha, value)
            if alpha >= beta:
                break
        return value

    def select_move(self, board: chess.Board) -> chess.Move:
        moves = list(board.legal_moves)
        self.rng.shuffle(moves)
        best, best_score = [], -np.inf
        for move in moves:
            board.push(move)
            # full window at the root so ties are scored exactly (keeps play varied)
            score = -self._search(board, self.depth - 1, -np.inf, np.inf, 1)
            board.pop()
            if score > best_score:
                best, best_score = [move], score
            elif score == best_score:
                best.append(move)
        return self.rng.choice(best)


class StockfishAgent:
    """Wraps a UCI engine. ``skill`` 0-20 is Stockfish's 'Skill Level'."""

    def __init__(self, skill: int = 0, time_limit: float = 0.05, path: str | None = None):
        import chess.engine

        path = path or shutil.which("stockfish")
        if not path:
            raise RuntimeError("stockfish binary not found (install it, e.g. `apt install stockfish` / `brew install stockfish`)")
        self.engine = chess.engine.SimpleEngine.popen_uci(path)
        self.engine.configure({"Skill Level": skill})
        self.limit = chess.engine.Limit(time=time_limit)
        self.name = f"stockfish-skill{skill}"

    def select_move(self, board: chess.Board) -> chess.Move:
        return self.engine.play(board, self.limit).move

    def close(self) -> None:
        self.engine.quit()


class ValueNetAgent:
    """Picks the move whose resulting position the value net likes best.

    depth=1: score(m) = -V(child)            (child is from the opponent's view)
    depth=2: score(m) = min over replies r of V(grandchild)   (opponent minimises our value)
    depth>=3: negamax alpha-beta with the value net at the leaves (captures searched
              first; the children of each last-ply node are evaluated in one batch)

    ``temperature > 0`` samples moves from softmax(score / T) (used for exploration in
    training); ``temperature == 0`` plays the best move with random tie-breaking.
    """

    def __init__(self, net: ValueNet, depth: int = 1, temperature: float = 0.0, seed: int | None = None, name: str | None = None):
        self.net = net
        self.depth = depth
        self.temperature = temperature
        self.rng = np.random.default_rng(seed)
        self.name = name or f"valuenet-d{depth}"

    def move_scores(self, board: chess.Board) -> tuple[list[chess.Move], np.ndarray]:
        moves = list(board.legal_moves)
        scores = np.zeros(len(moves), dtype=np.float32)
        if self.depth <= 1:
            children, idx = [], []
            for i, move in enumerate(moves):
                board.push(move)
                tv = terminal_value(board)
                if tv is None:
                    children.append(board.copy(stack=False))
                    idx.append(i)
                else:
                    scores[i] = -tv
                board.pop()
            scores[idx] = -self.net.evaluate(children)
            return moves, scores

        # depth 2: batch every grandchild into one forward pass
        grandchildren, owner = [], []
        reply_values: list[list[float] | None] = []  # None = move ends the game
        for i, move in enumerate(moves):
            board.push(move)
            tv = terminal_value(board)
            if tv is not None:
                scores[i] = -tv
                reply_values.append(None)
            else:
                vals: list[float] = []
                for reply in board.legal_moves:
                    board.push(reply)
                    tv2 = terminal_value(board)
                    if tv2 is None:
                        grandchildren.append(board.copy(stack=False))
                        owner.append(i)
                    else:
                        vals.append(tv2)
                    board.pop()
                reply_values.append(vals)
            board.pop()
        for v, i in zip(self.net.evaluate(grandchildren), owner):
            reply_values[i].append(float(v))  # type: ignore[union-attr]
        for i, vals in enumerate(reply_values):
            if vals is not None:
                scores[i] = min(vals)  # a non-terminal position always has a legal reply
        return moves, scores

    def _one_ply_value(self, board: chess.Board) -> float:
        """Negamax value of ``board`` searched one ply deep (children batched)."""
        best, children = -np.inf, []
        for move in board.legal_moves:
            board.push(move)
            tv = terminal_value(board)
            if tv is None:
                children.append(board.copy(stack=False))
            else:
                best = max(best, -tv)
            board.pop()
        if children:
            best = max(best, float(-self.net.evaluate(children).min()))
        return best

    def _alphabeta(self, board: chess.Board, depth: int, alpha: float, beta: float) -> float:
        tv = terminal_value(board)
        if tv is not None:
            return tv
        if depth == 1:
            return self._one_ply_value(board)
        value = -np.inf
        for move in sorted(board.legal_moves, key=lambda m: not board.is_capture(m)):
            board.push(move)
            value = max(value, -self._alphabeta(board, depth - 1, -beta, -alpha))
            board.pop()
            alpha = max(alpha, value)
            if alpha >= beta:
                break
        return value

    def _alphabeta_root(self, board: chess.Board) -> chess.Move:
        moves, prior = ValueNetAgent(self.net, depth=1).move_scores(board)
        order = np.argsort(-(prior + self.rng.uniform(0, 1e-4, len(moves))))  # best-first, random ties
        best_move, alpha = moves[int(order[0])], -np.inf
        for i in order:
            board.push(moves[i])
            score = -self._alphabeta(board, self.depth - 1, -np.inf, -alpha)
            board.pop()
            if score > alpha:
                alpha, best_move = score, moves[i]
        return best_move

    def select_move(self, board: chess.Board) -> chess.Move:
        if self.depth >= 3 and self.temperature == 0:
            return self._alphabeta_root(board)
        moves, scores = self.move_scores(board)
        if self.temperature > 0:
            logits = (scores - scores.max()) / self.temperature
            probs = np.exp(logits)
            probs /= probs.sum()
            return moves[int(self.rng.choice(len(moves), p=probs))]
        best = np.flatnonzero(scores >= scores.max() - 1e-6)
        return moves[int(self.rng.choice(best))]


def make_baseline(name: str, seed: int | None = None):
    """Build a baseline opponent from a short name: random, greedy, minimax2, minimax3, stockfish0..20."""
    if name == "random":
        return RandomAgent(seed)
    if name in ("greedy", "greedy-material"):
        return GreedyMaterialAgent(seed)
    if name.startswith("minimax"):
        depth = int(name.removeprefix("minimax").removeprefix("-d") or 2)
        return MinimaxAgent(depth, seed)
    if name.startswith("stockfish"):
        skill = int(name.removeprefix("stockfish").removeprefix("-skill") or 0)
        return StockfishAgent(skill)
    raise ValueError(f"Unknown baseline {name!r}")
