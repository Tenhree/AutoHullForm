from __future__ import annotations
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F

from networks import (
    DeterministicActor,
    QNetwork,
    TanhGaussianActor,
    hard_update,
    soft_update,
)
from settings import DDPGConfig, SACConfig, TD3Config


class DDPGAgent:
    algo = "DDPG"

    def __init__(self, state_dim: int, action_dim: int, cfg: DDPGConfig, device: torch.device):
        self.cfg = cfg
        self.device = device
        hidden = tuple(cfg.hidden_sizes)
        self.actor = DeterministicActor(state_dim, action_dim, hidden).to(device)
        self.actor_target = DeterministicActor(state_dim, action_dim, hidden).to(device)
        self.critic = QNetwork(state_dim, action_dim, hidden).to(device)
        self.critic_target = QNetwork(state_dim, action_dim, hidden).to(device)
        hard_update(self.actor_target, self.actor)
        hard_update(self.critic_target, self.critic)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=cfg.actor_lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=cfg.critic_lr)
        self.action_dim = action_dim
        self.state_dim = state_dim

    @torch.no_grad()
    def select_action(self, state: np.ndarray, explore: bool = True) -> np.ndarray:
        s = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        action = self.actor(s).squeeze(0).cpu().numpy()
        if explore:
            action += np.random.normal(0.0, self.cfg.exploration_noise, size=self.action_dim)
        return np.clip(action, -1.0, 1.0).astype(np.float32)

    def update(self, replay, batch_size: int, global_step: int) -> Dict[str, float]:
        b = replay.sample(batch_size)
        s, a, r, ns, d = b["state"], b["action"], b["reward"], b["next_state"], b["done"]
        with torch.no_grad():
            next_a = self.actor_target(ns)
            target_q = self.critic_target(ns, next_a)
            y = r + self.cfg.gamma * (1.0 - d) * target_q
        q = self.critic(s, a)
        critic_loss = F.mse_loss(q, y)
        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        actor_loss = -self.critic(s, self.actor(s)).mean()
        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        soft_update(self.actor_target, self.actor, self.cfg.tau)
        soft_update(self.critic_target, self.critic, self.cfg.tau)
        return {"critic_loss": float(critic_loss.item()), "actor_loss": float(actor_loss.item())}

    def save(self, path: str, extra: Dict | None = None) -> None:
        payload = {
            "algo": self.algo,
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "hidden_sizes": list(self.cfg.hidden_sizes),
            "actor_state_dict": self.actor.state_dict(),
            "config": self.cfg.to_dict(),
        }
        if extra:
            payload.update(extra)
        torch.save(payload, path)


class TD3Agent:
    algo = "TD3"

    def __init__(self, state_dim: int, action_dim: int, cfg: TD3Config, device: torch.device):
        self.cfg = cfg
        self.device = device
        hidden = tuple(cfg.hidden_sizes)
        self.actor = DeterministicActor(state_dim, action_dim, hidden).to(device)
        self.actor_target = DeterministicActor(state_dim, action_dim, hidden).to(device)
        self.q1 = QNetwork(state_dim, action_dim, hidden).to(device)
        self.q2 = QNetwork(state_dim, action_dim, hidden).to(device)
        self.q1_target = QNetwork(state_dim, action_dim, hidden).to(device)
        self.q2_target = QNetwork(state_dim, action_dim, hidden).to(device)
        hard_update(self.actor_target, self.actor)
        hard_update(self.q1_target, self.q1)
        hard_update(self.q2_target, self.q2)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=cfg.actor_lr)
        self.q1_opt = torch.optim.Adam(self.q1.parameters(), lr=cfg.critic_lr)
        self.q2_opt = torch.optim.Adam(self.q2.parameters(), lr=cfg.critic_lr)
        self.action_dim = action_dim
        self.state_dim = state_dim

    @torch.no_grad()
    def select_action(self, state: np.ndarray, explore: bool = True) -> np.ndarray:
        s = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        action = self.actor(s).squeeze(0).cpu().numpy()
        if explore:
            action += np.random.normal(0.0, self.cfg.exploration_noise, size=self.action_dim)
        return np.clip(action, -1.0, 1.0).astype(np.float32)

    def update(self, replay, batch_size: int, global_step: int) -> Dict[str, float]:
        b = replay.sample(batch_size)
        s, a, r, ns, d = b["state"], b["action"], b["reward"], b["next_state"], b["done"]
        with torch.no_grad():
            noise = (torch.randn_like(a) * self.cfg.policy_noise).clamp(-self.cfg.noise_clip, self.cfg.noise_clip)
            next_a = (self.actor_target(ns) + noise).clamp(-1.0, 1.0)
            target_q = torch.min(self.q1_target(ns, next_a), self.q2_target(ns, next_a))
            y = r + self.cfg.gamma * (1.0 - d) * target_q
        q1 = self.q1(s, a)
        q2 = self.q2(s, a)
        q1_loss = F.mse_loss(q1, y)
        q2_loss = F.mse_loss(q2, y)
        self.q1_opt.zero_grad()
        q1_loss.backward()
        self.q1_opt.step()
        self.q2_opt.zero_grad()
        q2_loss.backward()
        self.q2_opt.step()

        logs = {"q1_loss": float(q1_loss.item()), "q2_loss": float(q2_loss.item())}
        if global_step % self.cfg.policy_delay == 0:
            actor_loss = -self.q1(s, self.actor(s)).mean()
            self.actor_opt.zero_grad()
            actor_loss.backward()
            self.actor_opt.step()
            soft_update(self.actor_target, self.actor, self.cfg.tau)
            soft_update(self.q1_target, self.q1, self.cfg.tau)
            soft_update(self.q2_target, self.q2, self.cfg.tau)
            logs["actor_loss"] = float(actor_loss.item())
        return logs

    def save(self, path: str, extra: Dict | None = None) -> None:
        payload = {
            "algo": self.algo,
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "hidden_sizes": list(self.cfg.hidden_sizes),
            "actor_state_dict": self.actor.state_dict(),
            "config": self.cfg.to_dict(),
        }
        if extra:
            payload.update(extra)
        torch.save(payload, path)


class SACAgent:
    algo = "SAC"

    def __init__(self, state_dim: int, action_dim: int, cfg: SACConfig, device: torch.device):
        self.cfg = cfg
        self.device = device
        hidden = tuple(cfg.hidden_sizes)
        self.actor = TanhGaussianActor(state_dim, action_dim, hidden).to(device)
        self.q1 = QNetwork(state_dim, action_dim, hidden).to(device)
        self.q2 = QNetwork(state_dim, action_dim, hidden).to(device)
        self.q1_target = QNetwork(state_dim, action_dim, hidden).to(device)
        self.q2_target = QNetwork(state_dim, action_dim, hidden).to(device)
        hard_update(self.q1_target, self.q1)
        hard_update(self.q2_target, self.q2)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=cfg.actor_lr)
        self.q1_opt = torch.optim.Adam(self.q1.parameters(), lr=cfg.critic_lr)
        self.q2_opt = torch.optim.Adam(self.q2.parameters(), lr=cfg.critic_lr)
        self.log_alpha = torch.tensor(np.log(cfg.init_alpha), dtype=torch.float32, device=device, requires_grad=True)
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=cfg.alpha_lr)
        self.action_dim = action_dim
        self.state_dim = state_dim

    @property
    def alpha(self):
        return self.log_alpha.exp()

    @torch.no_grad()
    def select_action(self, state: np.ndarray, explore: bool = True) -> np.ndarray:
        s = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        action, _, mean_action = self.actor.sample(s, deterministic=not explore, with_logprob=False)
        out = action if explore else mean_action
        return out.squeeze(0).cpu().numpy().astype(np.float32)

    def update(self, replay, batch_size: int, global_step: int) -> Dict[str, float]:
        b = replay.sample(batch_size)
        s, a, r, ns, d = b["state"], b["action"], b["reward"], b["next_state"], b["done"]
        with torch.no_grad():
            next_a, next_logp, _ = self.actor.sample(ns, deterministic=False, with_logprob=True)
            target_q = torch.min(self.q1_target(ns, next_a), self.q2_target(ns, next_a)) - self.alpha.detach() * next_logp.unsqueeze(-1)
            y = r + self.cfg.gamma * (1.0 - d) * target_q

        q1_loss = F.mse_loss(self.q1(s, a), y)
        q2_loss = F.mse_loss(self.q2(s, a), y)
        self.q1_opt.zero_grad()
        q1_loss.backward()
        self.q1_opt.step()
        self.q2_opt.zero_grad()
        q2_loss.backward()
        self.q2_opt.step()

        pi, logp, _ = self.actor.sample(s, deterministic=False, with_logprob=True)
        q_pi = torch.min(self.q1(s, pi), self.q2(s, pi))
        actor_loss = (self.alpha.detach() * logp.unsqueeze(-1) - q_pi).mean()
        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        alpha_loss_value = 0.0
        if self.cfg.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (logp + self.cfg.target_entropy).detach()).mean()
            self.alpha_opt.zero_grad()
            alpha_loss.backward()
            self.alpha_opt.step()
            alpha_loss_value = float(alpha_loss.item())

        soft_update(self.q1_target, self.q1, self.cfg.tau)
        soft_update(self.q2_target, self.q2, self.cfg.tau)
        return {
            "q1_loss": float(q1_loss.item()),
            "q2_loss": float(q2_loss.item()),
            "actor_loss": float(actor_loss.item()),
            "alpha": float(self.alpha.detach().item()),
            "alpha_loss": alpha_loss_value,
        }

    def save(self, path: str, extra: Dict | None = None) -> None:
        payload = {
            "algo": self.algo,
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "hidden_sizes": list(self.cfg.hidden_sizes),
            "actor_state_dict": self.actor.state_dict(),
            "config": self.cfg.to_dict(),
        }
        if extra:
            payload.update(extra)
        torch.save(payload, path)
