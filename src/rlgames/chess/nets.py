"""Policy + value network architectures for chess, all behind one interface.

Every network maps planes (B, 20, 8, 8) to ``(policy_logits (B, 4168), value (B,))``,
with value in [-1, 1] for the side to move, so any of them can be dropped into the
same AlphaZero training loop. Pick one in a config::

    model:
      arch: resnet        # mlp | cnn | resnet | transformer
      blocks: 6
      channels: 64

To add an architecture, write an ``nn.Module`` with that forward signature and
register it in ``ARCHITECTURES``.
"""

from __future__ import annotations

from typing import Any, Callable

import torch
import torch.nn.functional as F
from torch import nn

from rlgames.chess.az_encoding import NUM_ACTIONS, NUM_PLANES

N_FROM_TO = 64 * 64
N_UNDERPROMO = NUM_ACTIONS - N_FROM_TO


class ConvHeads(nn.Module):
    """Heads on top of a (B, C, 8, 8) feature map.

    Policy (convolutional, as in Leela Chess Zero): each square outputs 64 "move to"
    logits plus 9 under-promotion logits, so the head stays small (~C*C*9 weights)
    instead of a dense layer to 4168 outputs that would dwarf the body.
    """

    def __init__(self, channels: int, value_hidden: int = 128):
        super().__init__()
        self.policy = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False), nn.BatchNorm2d(channels), nn.ReLU(),
            nn.Conv2d(channels, 64 + 9, 1),
        )
        self.value_conv = nn.Sequential(nn.Conv2d(channels, 4, 1, bias=False), nn.BatchNorm2d(4), nn.ReLU())
        self.value_fc = nn.Sequential(nn.Linear(4 * 64, value_hidden), nn.ReLU(), nn.Linear(value_hidden, 1), nn.Tanh())

    def forward(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        out = self.policy(h)  # (B, 73, 8, 8): channel = target, spatial position = from-square
        from_to = out[:, :64].flatten(2).transpose(1, 2)  # (B, from = rank*8+file, to)
        promo = out[:, 64:, 6, :].transpose(1, 2)  # 7th-rank squares -> (B, 8 files, 9)
        v = self.value_fc(self.value_conv(h).flatten(1)).squeeze(-1)
        return torch.cat([from_to.flatten(1), promo.flatten(1)], dim=1), v


class MLPNet(nn.Module):
    """Baseline: no spatial structure at all, just the flattened planes."""

    def __init__(self, hidden: list[int] | tuple[int, ...] = (1024, 512)):
        super().__init__()
        layers: list[nn.Module] = [nn.Flatten()]
        d = NUM_PLANES * 64
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU()]
            d = h
        self.body = nn.Sequential(*layers)
        self.policy = nn.Linear(d, NUM_ACTIONS)
        self.value = nn.Sequential(nn.Linear(d, 128), nn.ReLU(), nn.Linear(128, 1), nn.Tanh())

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.body(x)
        return self.policy(h), self.value(h).squeeze(-1)


class CNNNet(nn.Module):
    """Plain convolution stack (no skip connections)."""

    def __init__(self, layers: int = 4, channels: int = 64):
        super().__init__()
        mods: list[nn.Module] = []
        c_in = NUM_PLANES
        for _ in range(layers):
            mods += [nn.Conv2d(c_in, channels, 3, padding=1, bias=False), nn.BatchNorm2d(channels), nn.ReLU()]
            c_in = channels
        self.body = nn.Sequential(*mods)
        self.heads = ConvHeads(channels)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.heads(self.body(x))


class ResidualBlock(nn.Module):
    def __init__(self, channels: int, se: bool = False):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        # optional squeeze-and-excitation (as in Leela Chess Zero)
        self.se = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(channels, channels // 4), nn.ReLU(),
                                nn.Linear(channels // 4, channels), nn.Sigmoid()) if se else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.bn1(self.conv1(x)))
        h = self.bn2(self.conv2(h))
        if self.se is not None:
            h = h * self.se(h)[:, :, None, None]
        return F.relu(x + h)


class ResNet(nn.Module):
    """The AlphaZero architecture (Silver et al. 2018), scaled down: a residual tower of
    3x3 convolutions with batch norm, then policy and value heads."""

    def __init__(self, blocks: int = 6, channels: int = 64, se: bool = False):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(NUM_PLANES, channels, 3, padding=1, bias=False), nn.BatchNorm2d(channels), nn.ReLU())
        self.tower = nn.Sequential(*[ResidualBlock(channels, se) for _ in range(blocks)])
        self.heads = ConvHeads(channels)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.heads(self.tower(self.stem(x)))


class TransformerNet(nn.Module):
    """Each square is a token; self-attention lets any square attend to any other.

    The from->to policy is a bilinear score between the "from" and "to" square
    embeddings (the attention-policy idea used by Leela Chess Zero's transformer nets);
    under-promotions come from the 8 pawn squares on the 7th rank.
    """

    def __init__(self, layers: int = 4, dim: int = 96, heads: int = 4, ff_mult: int = 2):
        super().__init__()
        self.embed = nn.Linear(NUM_PLANES, dim)
        self.pos = nn.Parameter(torch.zeros(1, 64, dim))
        nn.init.normal_(self.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(dim, heads, dim * ff_mult, dropout=0.0, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(dim)
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.promo = nn.Linear(dim, 9)  # 3 directions x 3 pieces, from each 7th-rank square
        self.value = nn.Sequential(nn.Linear(dim, 128), nn.ReLU(), nn.Linear(128, 1), nn.Tanh())
        self.dim = dim

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        tokens = x.flatten(2).transpose(1, 2)  # (B, 64 squares, 20 features); square = rank * 8 + file
        h = self.norm(self.encoder(self.embed(tokens) + self.pos))
        from_to = torch.einsum("bfd,btd->bft", self.q(h), self.k(h)) / self.dim ** 0.5
        promo = self.promo(h[:, 48:56])  # rank 7 squares a7..h7 -> (B, 8 files, 9)
        policy = torch.cat([from_to.flatten(1), promo.flatten(1)], dim=1)
        return policy, self.value(h.mean(dim=1)).squeeze(-1)


ARCHITECTURES: dict[str, Callable[..., nn.Module]] = {
    "mlp": MLPNet,
    "cnn": CNNNet,
    "resnet": ResNet,
    "transformer": TransformerNet,
}


def build_net(model_cfg: dict[str, Any]) -> nn.Module:
    cfg = dict(model_cfg)
    arch = cfg.pop("arch", "resnet")
    if arch not in ARCHITECTURES:
        raise ValueError(f"Unknown chess architecture {arch!r}; choose from {sorted(ARCHITECTURES)}")
    return ARCHITECTURES[arch](**cfg)


def count_params(net: nn.Module) -> int:
    return sum(p.numel() for p in net.parameters())
