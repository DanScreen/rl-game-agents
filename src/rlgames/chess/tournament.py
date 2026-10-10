"""Round-robin tournaments between chess agents, with maximum-likelihood Elo ratings."""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field

from rlgames.chess.arena import MatchResult, play_match


@dataclass
class TournamentResult:
    names: list[str]
    matches: dict[tuple[int, int], MatchResult] = field(default_factory=dict)  # (i, j) with i < j, from i's view
    ratings: list[float] = field(default_factory=list)

    def score(self, i: int, j: int) -> tuple[float, int]:
        """Points scored by agent i against agent j, and games played."""
        if (i, j) in self.matches:
            m = self.matches[(i, j)]
            return m.wins + 0.5 * m.draws, m.n
        m = self.matches[(j, i)]
        return m.losses + 0.5 * m.draws, m.n

    def table(self) -> str:
        order = sorted(range(len(self.names)), key=lambda i: -self.ratings[i])
        short = [f"#{k + 1}" for k in range(len(order))]
        lines = ["| # | Agent | Elo | Score | " + " | ".join(short) + " |",
                 "|---|---|---|---|" + "---|" * len(order)]
        for rank, i in enumerate(order):
            pts = games = 0.0
            cells = []
            for j in order:
                if i == j:
                    cells.append("-")
                    continue
                s, n = self.score(i, j)
                pts, games = pts + s, games + n
                cells.append(f"{s:g}/{n}")
            lines.append(f"| {rank + 1} | {self.names[i]} | {self.ratings[i]:+.0f} | {pts / max(games, 1):.0%} | "
                         + " | ".join(cells) + " |")
        return "\n".join(lines)


def fit_elo(names: list[str], score_fn, anchor: str | None = "random", iters: int = 3000, prior_draws: float = 1.0) -> list[float]:
    """Bradley-Terry / Elo maximum likelihood. Each pair also gets ``prior_draws`` virtual
    draws so a 100% score gives a large but finite gap. Ratings are shifted so ``anchor``
    is 0 (or the mean is 0 if it is not playing)."""
    k = len(names)
    r = [0.0] * k
    lr = 20.0
    for _ in range(iters):
        grad = [0.0] * k
        for i, j in itertools.combinations(range(k), 2):
            s, n = score_fn(i, j)
            if n == 0:
                continue
            s, n = s + 0.5 * prior_draws, n + prior_draws
            e = 1.0 / (1.0 + 10 ** ((r[j] - r[i]) / 400.0))
            grad[i] += s - n * e
            grad[j] -= s - n * e
        r = [ri + lr * g / max(1.0, k) for ri, g in zip(r, grad)]
    shift = r[names.index(anchor)] if anchor in names else sum(r) / k
    return [ri - shift for ri in r]


def run_tournament(agents: list, games_per_pair: int = 10, max_plies: int = 300, opening_plies: int = 4,
                   seed: int = 0, on_match=None) -> TournamentResult:
    result = TournamentResult([a.name for a in agents])
    for i, j in itertools.combinations(range(len(agents)), 2):
        m = play_match(agents[i], agents[j], games=games_per_pair, max_plies=max_plies,
                       opening_plies=opening_plies, seed=seed + 1000 * i + j)
        result.matches[(i, j)] = m
        if on_match:
            on_match(m)
    result.ratings = fit_elo(result.names, result.score)
    return result


