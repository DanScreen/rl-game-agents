"""Neural network building blocks shared by the learners."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
from torch import nn


def layer_init(layer: nn.Module, std: float = np.sqrt(2), bias: float = 0.0) -> nn.Module:
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias)
    return layer


class NatureCNN(nn.Module):
    """The convolutional torso from Mnih et al. (2015). Expects uint8 (C, 84, 84) input."""

    def __init__(self, in_channels: int, out_dim: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            layer_init(nn.Conv2d(in_channels, 32, 8, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(64 * 7 * 7, out_dim)),
            nn.ReLU(),
        )
        self.out_dim = out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.float() / 255.0)


class MLPEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden: Sequence[int] = (128, 128), activation: type[nn.Module] = nn.ReLU):
        super().__init__()
        layers: list[nn.Module] = [nn.Flatten()]
        d = in_dim
        for h in hidden:
            layers += [layer_init(nn.Linear(d, h)), activation()]
            d = h
        self.net = nn.Sequential(*layers)
        self.out_dim = d

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.float())


def make_encoder(obs_shape: Sequence[int], hidden: Sequence[int] = (128, 128), activation: str = "relu") -> nn.Module:
    if len(obs_shape) == 3:  # stacked image frames
        return NatureCNN(obs_shape[0])
    act = {"relu": nn.ReLU, "tanh": nn.Tanh}[activation]
    return MLPEncoder(int(np.prod(obs_shape)), hidden, act)


class QNetwork(nn.Module):
    """Q(s, .) head on top of an encoder. ``dueling=True`` uses Wang et al. (2016)."""

    def __init__(self, obs_shape: Sequence[int], n_actions: int, hidden: Sequence[int] = (128, 128), dueling: bool = False):
        super().__init__()
        self.encoder = make_encoder(obs_shape, hidden)
        d = self.encoder.out_dim
        self.dueling = dueling
        if dueling:
            self.value = layer_init(nn.Linear(d, 1), std=1.0)
            self.advantage = layer_init(nn.Linear(d, n_actions), std=1.0)
        else:
            self.head = layer_init(nn.Linear(d, n_actions), std=1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        if self.dueling:
            a = self.advantage(z)
            return self.value(z) + a - a.mean(dim=1, keepdim=True)
        return self.head(z)


class ActorCritic(nn.Module):
    """Policy + value network. Image inputs share a CNN torso; vector inputs use two MLPs."""

    def __init__(self, obs_shape: Sequence[int], n_actions: int, hidden: Sequence[int] = (64, 64)):
        super().__init__()
        self.shared = len(obs_shape) == 3
        if self.shared:
            self.torso = make_encoder(obs_shape)
            d = self.torso.out_dim
            self.actor_head = layer_init(nn.Linear(d, n_actions), std=0.01)
            self.critic_head = layer_init(nn.Linear(d, 1), std=1.0)
        else:
            self.actor_body = make_encoder(obs_shape, hidden, activation="tanh")
            self.critic_body = make_encoder(obs_shape, hidden, activation="tanh")
            self.actor_head = layer_init(nn.Linear(self.actor_body.out_dim, n_actions), std=0.01)
            self.critic_head = layer_init(nn.Linear(self.critic_body.out_dim, 1), std=1.0)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (logits, value)."""
        if self.shared:
            z = self.torso(x)
            return self.actor_head(z), self.critic_head(z).squeeze(-1)
        return self.actor_head(self.actor_body(x)), self.critic_head(self.critic_body(x)).squeeze(-1)
