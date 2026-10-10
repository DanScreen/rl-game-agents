"""Value network for chess: V(position) in [-1, 1] for the side to move."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import chess
import numpy as np
import torch
from torch import nn

from rlgames.chess.encoding import NUM_FEATURES, encode_boards


class ValueNet(nn.Module):
    def __init__(self, hidden: Sequence[int] = (512, 256, 64)):
        super().__init__()
        layers: list[nn.Module] = []
        d = NUM_FEATURES
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU()]
            d = h
        layers += [nn.Linear(d, 1), nn.Tanh()]
        self.net = nn.Sequential(*layers)
        self.hidden = list(hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)

    @torch.no_grad()
    def evaluate(self, boards: list[chess.Board]) -> np.ndarray:
        if not boards:
            return np.zeros(0, dtype=np.float32)
        return self(torch.from_numpy(encode_boards(boards))).numpy()


def save_value_net(path: str | Path, net: ValueNet, **extra: Any) -> None:
    from rlgames.agents.policy import save_checkpoint

    save_checkpoint(path, algo="chess_td", hidden=net.hidden, state_dict=net.state_dict(), **extra)


def load_value_net(path: str | Path) -> tuple[ValueNet, dict[str, Any]]:
    path = Path(path)
    if path.is_dir():
        path = path / "best.pt" if (path / "best.pt").exists() else path / "latest.pt"
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    net = ValueNet(ckpt["hidden"])
    net.load_state_dict(ckpt["state_dict"])
    net.eval()
    return net, ckpt
