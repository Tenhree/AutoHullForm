from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import torch

from networks import DeterministicActor, TanhGaussianActor
from utils import resolve_device


class LoadedPolicy:
    def __init__(self, checkpoint_path: str, device: str = "auto"):
        self.path = str(Path(checkpoint_path).resolve())
        self.device = resolve_device(device)
        self.ckpt = torch.load(self.path, map_location=self.device)
        self.algo = str(self.ckpt.get("algo", "UNKNOWN")).upper()
        self.state_dim = int(self.ckpt.get("state_dim", 10))
        self.action_dim = int(self.ckpt.get("action_dim", 3))
        self.hidden_sizes = tuple(int(x) for x in self.ckpt.get("hidden_sizes", [256, 256]))
        self.actor = self._build_actor()
        self.actor.load_state_dict(self.ckpt["actor_state_dict"], strict=True)
        self.actor.to(self.device)
        self.actor.eval()

    def _build_actor(self):
        if self.algo in {"DDPG", "TD3"}:
            return DeterministicActor(self.state_dim, self.action_dim, self.hidden_sizes)
        if self.algo in {"SAC", "PPO"}:
            return TanhGaussianActor(self.state_dim, self.action_dim, self.hidden_sizes)
        raise ValueError(f"Unsupported policy algo in checkpoint: {self.algo}")

    @torch.no_grad()
    def act(self, state: np.ndarray, deterministic: bool = True) -> np.ndarray:
        s = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        if self.algo in {"DDPG", "TD3"}:
            action = self.actor(s)
        else:
            action, _, mean_action = self.actor.sample(s, deterministic=deterministic, with_logprob=False)
            action = mean_action if deterministic else action
        return action.squeeze(0).detach().cpu().numpy().astype(np.float32)
