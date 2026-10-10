"""Monte Carlo Tree Search with PUCT (Silver et al. 2017/2018), batched across games.

Each simulation walks from the root, picking the child that maximises

    Q(s, a) + c_puct * P(s, a) * sqrt(N(s)) / (1 + N(s, a))

until it reaches an unexpanded position, which the network evaluates (policy priors P
and value v). The value is backed up the path with alternating sign. After all
simulations, the root's visit counts are both the move choice and the policy training
target.

``BatchedMCTS`` advances many independent searches in lock-step so that one network
call evaluates a leaf from every search at once, which is what makes this usable
from Python on a CPU.
"""

from __future__ import annotations

import math
from typing import Callable

import chess
import numpy as np
import torch

from rlgames.chess.az_encoding import encode_planes, legal_actions, quick_terminal

# evaluate(planes (B, 20, 8, 8)) -> (logits (B, 4168), values (B,))
Evaluator = Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]]


def net_evaluator(net: torch.nn.Module, device: torch.device | str = "cpu") -> Evaluator:
    @torch.no_grad()
    def evaluate(planes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        logits, values = net(torch.from_numpy(planes).to(device))
        return logits.float().cpu().numpy(), values.float().cpu().numpy()

    return evaluate


class Node:
    __slots__ = ("moves", "actions", "P", "N", "W", "children", "terminal")

    def __init__(self) -> None:
        self.moves: list[chess.Move] | None = None  # None until expanded
        self.terminal: float | None = None  # exact value for the side to move, if game over

    @property
    def expanded(self) -> bool:
        return self.moves is not None

    def expand(self, moves: list[chess.Move], actions: np.ndarray, priors: np.ndarray) -> None:
        self.moves = moves
        self.actions = actions
        self.P = priors.astype(np.float64)
        self.N = np.zeros(len(moves), dtype=np.float64)
        self.W = np.zeros(len(moves), dtype=np.float64)
        self.children: list[Node | None] = [None] * len(moves)

    def select(self, c_puct: float, fpu: float) -> int:
        total = self.N.sum()
        q = np.where(self.N > 0, self.W / np.maximum(self.N, 1.0), fpu)
        u = c_puct * self.P * math.sqrt(max(total, 1.0)) / (1.0 + self.N)
        return int(np.argmax(q + u))

    def add_dirichlet_noise(self, alpha: float, frac: float, rng: np.random.Generator) -> None:
        noise = rng.dirichlet([alpha] * len(self.P))
        self.P = (1 - frac) * self.P + frac * noise

    def visit_distribution(self) -> np.ndarray:
        return self.N / max(self.N.sum(), 1.0)


def _softmax_over(logits: np.ndarray, actions: np.ndarray) -> np.ndarray:
    x = logits[actions].astype(np.float64)
    x = np.exp(x - x.max())
    return x / x.sum()


class BatchedMCTS:
    """Runs ``simulations`` PUCT simulations for each (board, root) pair, batching leaves."""

    def __init__(self, evaluate: Evaluator, c_puct: float = 1.5, fpu: float = 0.0):
        self.evaluate = evaluate
        self.c_puct = c_puct
        self.fpu = fpu  # value assumed for unvisited children (first-play urgency)

    def run(self, boards: list[chess.Board], roots: list[Node], simulations: int,
            on_root_expanded: Callable[[Node], None] | None = None) -> None:
        """``boards[i]`` must be the position at ``roots[i]``; boards are restored afterwards.

        ``on_root_expanded`` is called once per root right after it is first expanded
        (used to add exploration noise in self-play).
        """
        fresh = [i for i, r in enumerate(roots) if not r.expanded and r.terminal is None]
        if fresh:
            self._round([boards[i] for i in fresh], [roots[i] for i in fresh])  # one batched expansion
            if on_root_expanded is not None:
                for i in fresh:
                    if roots[i].expanded:
                        on_root_expanded(roots[i])
        for _ in range(simulations):
            active = [i for i, r in enumerate(roots) if r.terminal is None]
            if not active:
                break
            self._round([boards[i] for i in active], [roots[i] for i in active])

    def _round(self, boards: list[chess.Board], roots: list[Node]) -> None:
        pending = []  # (leaf node, path, moves, actions) awaiting the network
        planes = []
        for board, root in zip(boards, roots):
            node, path = root, []
            while node.expanded and node.terminal is None:
                a = node.select(self.c_puct, self.fpu)
                path.append((node, a))
                board.push(node.moves[a])
                child = node.children[a]
                if child is None:
                    child = node.children[a] = Node()
                node = child
            if node.terminal is None:
                moves = list(board.legal_moves)
                node.terminal = quick_terminal(board, moves)
                if node.terminal is None:
                    moves, actions = legal_actions(board, moves)
                    pending.append((node, path, moves, actions))
                    planes.append(encode_planes(board))
            for _ in path:
                board.pop()
            if node.terminal is not None:
                self._backup(path, node.terminal)
        if not pending:
            return
        logits, values = self.evaluate(np.stack(planes))
        for (node, path, moves, actions), lg, v in zip(pending, logits, values):
            node.expand(moves, actions, _softmax_over(lg, actions))
            self._backup(path, float(v))

    @staticmethod
    def _backup(path: list[tuple[Node, int]], value: float) -> None:
        # ``value`` is for the side to move at the leaf; each step up flips the perspective
        for node, a in reversed(path):
            value = -value
            node.N[a] += 1.0
            node.W[a] += value


class AlphaZeroAgent:
    """Plays the most-visited move after ``simulations`` MCTS simulations (no noise)."""

    def __init__(self, net: torch.nn.Module, simulations: int = 100, c_puct: float = 1.5,
                 name: str | None = None, device: str = "cpu"):
        net.eval()
        self.mcts = BatchedMCTS(net_evaluator(net, device), c_puct=c_puct)
        self.simulations = simulations
        self.name = name or f"alphazero-s{simulations}"

    def select_move(self, board: chess.Board) -> chess.Move:
        root = Node()
        work = board.copy()  # keeps the move stack for repetition detection
        self.mcts.run([work], [root], self.simulations)
        if not root.expanded:  # cannot happen for a legal, non-terminal position
            return next(iter(board.legal_moves))
        return root.moves[int(np.argmax(root.N))]
