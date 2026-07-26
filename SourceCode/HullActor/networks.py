from __future__ import annotations

from typing import Iterable, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

LOG_STD_MIN = -20.0
LOG_STD_MAX = 2.0
EPS = 1e-6


def mlp(sizes: Iterable[int], activation=nn.ReLU, output_activation=nn.Identity) -> nn.Sequential:
    sizes = list(sizes)
    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j + 1]), act()]
    return nn.Sequential(*layers)


class DeterministicActor(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_sizes: Tuple[int, int] = (256, 256)):
        super().__init__()
        self.net = mlp([state_dim, *hidden_sizes, action_dim], activation=nn.ReLU, output_activation=nn.Tanh)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)


class QNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_sizes: Tuple[int, int] = (256, 256)):
        super().__init__()
        self.q = mlp([state_dim + action_dim, *hidden_sizes, 1], activation=nn.ReLU, output_activation=nn.Identity)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.q(torch.cat([state, action], dim=-1))


class ValueNetwork(nn.Module):
    def __init__(self, state_dim: int, hidden_sizes: Tuple[int, int] = (256, 256)):
        super().__init__()
        self.v = mlp([state_dim, *hidden_sizes, 1], activation=nn.ReLU, output_activation=nn.Identity)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.v(state).squeeze(-1)


class TanhGaussianActor(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_sizes: Tuple[int, int] = (256, 256)):
        super().__init__()
        self.body = mlp([state_dim, *hidden_sizes], activation=nn.ReLU, output_activation=nn.ReLU)
        last_dim = hidden_sizes[-1] if hidden_sizes else state_dim
        self.mean = nn.Linear(last_dim, action_dim)
        self.log_std = nn.Linear(last_dim, action_dim)

    def mean_logstd(self, state: torch.Tensor):
        h = self.body(state)
        mean = self.mean(h)
        log_std = torch.clamp(self.log_std(h), LOG_STD_MIN, LOG_STD_MAX)
        return mean, log_std

    def sample(self, state: torch.Tensor, deterministic: bool = False, with_logprob: bool = True):
        mean, log_std = self.mean_logstd(state)
        std = log_std.exp()
        dist = torch.distributions.Normal(mean, std)
        if deterministic:
            raw = mean
        else:
            raw = dist.rsample()
        action = torch.tanh(raw)
        logp = None
        if with_logprob:
            logp = dist.log_prob(raw).sum(dim=-1)
            logp -= torch.log(1.0 - action.pow(2) + EPS).sum(dim=-1)
        return action, logp, torch.tanh(mean)

    def log_prob(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        action = torch.clamp(action, -1.0 + EPS, 1.0 - EPS)
        raw = 0.5 * torch.log((1.0 + action) / (1.0 - action))
        mean, log_std = self.mean_logstd(state)
        dist = torch.distributions.Normal(mean, log_std.exp())
        logp = dist.log_prob(raw).sum(dim=-1)
        logp -= torch.log(1.0 - action.pow(2) + EPS).sum(dim=-1)
        return logp

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        _, _, mean_action = self.sample(state, deterministic=True, with_logprob=False)
        return mean_action


def soft_update(target: nn.Module, source: nn.Module, tau: float) -> None:
    with torch.no_grad():
        for tp, sp in zip(target.parameters(), source.parameters()):
            tp.data.mul_(1.0 - tau).add_(sp.data, alpha=tau)


def hard_update(target: nn.Module, source: nn.Module) -> None:
    target.load_state_dict(source.state_dict())


def to_tensor(x, device: torch.device, dtype=torch.float32) -> torch.Tensor:
    return torch.as_tensor(x, dtype=dtype, device=device)
