from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F

from networks import TanhGaussianActor, ValueNetwork
from settings import PPOConfig


@dataclass
class RolloutBatch:
    state: np.ndarray
    action: np.ndarray
    reward: np.ndarray
    done: np.ndarray
    logp: np.ndarray
    value: np.ndarray
    next_value: float


class PPOAgent:
    algo = "PPO"

    def __init__(self, state_dim: int, action_dim: int, cfg: PPOConfig, device: torch.device):
        self.cfg = cfg
        self.device = device
        hidden = tuple(cfg.hidden_sizes)
        self.actor = TanhGaussianActor(state_dim, action_dim, hidden).to(device)
        self.value = ValueNetwork(state_dim, hidden).to(device)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=cfg.actor_lr)
        self.value_opt = torch.optim.Adam(self.value.parameters(), lr=cfg.critic_lr)
        self.state_dim = state_dim
        self.action_dim = action_dim

    @torch.no_grad()
    def select_action(self, state: np.ndarray, explore: bool = True):
        s = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        action, logp, mean_action = self.actor.sample(s, deterministic=not explore, with_logprob=True)
        value = self.value(s)
        a = action if explore else mean_action
        return (
            a.squeeze(0).cpu().numpy().astype(np.float32),
            float(logp.squeeze(0).cpu().item()),
            float(value.squeeze(0).cpu().item()),
        )

    @torch.no_grad()
    def value_of(self, state: np.ndarray) -> float:
        s = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        return float(self.value(s).squeeze(0).cpu().item())

    def compute_gae(self, rewards, dones, values, next_value):
        rewards = np.asarray(rewards, dtype=np.float32)
        dones = np.asarray(dones, dtype=np.float32)
        values = np.asarray(values, dtype=np.float32)
        adv = np.zeros_like(rewards, dtype=np.float32)
        last_gae = 0.0
        for t in reversed(range(len(rewards))):
            next_nonterminal = 1.0 - dones[t]
            next_val = next_value if t == len(rewards) - 1 else values[t + 1]
            delta = rewards[t] + self.cfg.gamma * next_val * next_nonterminal - values[t]
            last_gae = delta + self.cfg.gamma * self.cfg.gae_lambda * next_nonterminal * last_gae
            adv[t] = last_gae
        returns = adv + values
        return adv, returns

    def update(self, states, actions, old_logps, advantages, returns) -> Dict[str, float]:
        states_t = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        actions_t = torch.as_tensor(actions, dtype=torch.float32, device=self.device)
        old_logps_t = torch.as_tensor(old_logps, dtype=torch.float32, device=self.device)
        adv_t = torch.as_tensor(advantages, dtype=torch.float32, device=self.device)
        ret_t = torch.as_tensor(returns, dtype=torch.float32, device=self.device)
        adv_t = (adv_t - adv_t.mean()) / (adv_t.std(unbiased=False) + 1e-8)

        n = states_t.shape[0]
        idx_all = np.arange(n)
        last_logs: Dict[str, float] = {}
        mb = min(self.cfg.minibatch_size, n)
        for _ in range(self.cfg.ppo_epochs):
            np.random.shuffle(idx_all)
            for start in range(0, n, mb):
                idx = torch.as_tensor(idx_all[start : start + mb], dtype=torch.long, device=self.device)
                s = states_t[idx]
                a = actions_t[idx]
                old_logp = old_logps_t[idx]
                adv = adv_t[idx]
                ret = ret_t[idx]

                new_logp = self.actor.log_prob(s, a)
                ratio = torch.exp(new_logp - old_logp)
                clipped = torch.clamp(ratio, 1.0 - self.cfg.clip_ratio, 1.0 + self.cfg.clip_ratio) * adv
                policy_loss = -torch.min(ratio * adv, clipped).mean()

                value_pred = self.value(s)
                value_loss = F.mse_loss(value_pred, ret)
                entropy_proxy = -new_logp.mean()
                actor_loss = policy_loss - self.cfg.entropy_coef * entropy_proxy

                self.actor_opt.zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.cfg.max_grad_norm)
                self.actor_opt.step()

                self.value_opt.zero_grad()
                (self.cfg.value_coef * value_loss).backward()
                torch.nn.utils.clip_grad_norm_(self.value.parameters(), self.cfg.max_grad_norm)
                self.value_opt.step()

                approx_kl = (old_logp - new_logp).mean().detach().abs()
                last_logs = {
                    "policy_loss": float(policy_loss.item()),
                    "value_loss": float(value_loss.item()),
                    "entropy_proxy": float(entropy_proxy.item()),
                    "approx_kl": float(approx_kl.item()),
                }
        return last_logs

    def save(self, path: str, extra: Dict | None = None) -> None:
        payload = {
            "algo": self.algo,
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "hidden_sizes": list(self.cfg.hidden_sizes),
            "actor_state_dict": self.actor.state_dict(),
            "value_state_dict": self.value.state_dict(),
            "config": self.cfg.to_dict(),
        }
        if extra:
            payload.update(extra)
        torch.save(payload, path)
