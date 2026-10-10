"""Play chess games and matches between agents."""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import chess
import chess.pgn


@dataclass
class GameRecord:
    board: chess.Board  # final position with the full move stack
    result: str  # "1-0", "0-1", "1/2-1/2"
    reason: str
    white: str
    black: str
    opening_plies: int = 0

    def pgn(self) -> str:
        game = chess.pgn.Game.from_board(self.board)
        game.headers["White"] = self.white
        game.headers["Black"] = self.black
        game.headers["Result"] = self.result
        game.headers["Termination"] = self.reason
        if self.opening_plies:
            game.headers["Annotator"] = f"first {self.opening_plies} plies random"
        return str(game)


def play_random_opening(board: chess.Board, plies: int, rng: random.Random) -> None:
    """Push up to ``plies`` random moves, stopping early if the game ends (e.g. fool's mate)."""
    for _ in range(plies):
        if board.is_game_over():
            return
        board.push(rng.choice(list(board.legal_moves)))


def play_game(white, black, max_plies: int = 300, opening_plies: int = 0, rng: random.Random | None = None) -> GameRecord:
    """``opening_plies`` random moves are played first so repeated games differ."""
    rng = rng or random.Random()
    board = chess.Board()
    play_random_opening(board, opening_plies, rng)
    while True:
        outcome = board.outcome(claim_draw=True)
        if outcome is not None:
            return GameRecord(board, outcome.result(), outcome.termination.name.lower(), white.name, black.name, opening_plies)
        if board.ply() >= max_plies:
            return GameRecord(board, "1/2-1/2", f"adjudicated draw at {max_plies} plies", white.name, black.name, opening_plies)
        player = white if board.turn == chess.WHITE else black
        board.push(player.select_move(board))


def elo_diff(score: float, games: int) -> float:
    """Elo difference implied by a match score, clamped so 100% / 0% stay finite."""
    eps = 1.0 / (2 * max(games, 1))
    s = min(max(score, eps), 1 - eps)
    return -400.0 * math.log10(1.0 / s - 1.0)


@dataclass
class MatchResult:
    agent: str
    opponent: str
    wins: int = 0
    draws: int = 0
    losses: int = 0
    games: list[GameRecord] = field(default_factory=list)
    seconds: float = 0.0

    @property
    def n(self) -> int:
        return self.wins + self.draws + self.losses

    @property
    def score(self) -> float:
        return (self.wins + 0.5 * self.draws) / max(self.n, 1)

    def summary(self) -> dict[str, Any]:
        return {
            "agent": self.agent, "opponent": self.opponent, "games": self.n, "wins": self.wins,
            "draws": self.draws, "losses": self.losses, "score": self.score,
            "elo_diff": elo_diff(self.score, self.n), "seconds": round(self.seconds, 1),
        }

    def __str__(self) -> str:
        s = self.summary()
        return (f"{s['agent']} vs {s['opponent']}: +{s['wins']} ={s['draws']} -{s['losses']} "
                f"(score {s['score']:.1%}, Elo diff {s['elo_diff']:+.0f})")


def play_match(
    agent, opponent, games: int = 20, max_plies: int = 300, opening_plies: int = 4, seed: int = 0,
    on_game: Callable[[int, GameRecord], None] | None = None,
) -> MatchResult:
    """Colours alternate every game; game ``2k`` and ``2k+1`` share the same random opening."""
    result = MatchResult(agent.name, opponent.name)
    t0 = time.time()
    for g in range(games):
        rng = random.Random(seed * 100_003 + g // 2)
        agent_white = g % 2 == 0
        white, black = (agent, opponent) if agent_white else (opponent, agent)
        record = play_game(white, black, max_plies=max_plies, opening_plies=opening_plies, rng=rng)
        if record.result == "1/2-1/2":
            result.draws += 1
        elif (record.result == "1-0") == agent_white:
            result.wins += 1
        else:
            result.losses += 1
        result.games.append(record)
        if on_game:
            on_game(g, record)
    result.seconds = time.time() - t0
    return result
