from __future__ import annotations

import numpy as np
import torch


class ReplayBuffer:
    def __init__(self, state_dim: int, action_dim: int, size: int, device: torch.device):
        self.state = np.zeros((size, state_dim), dtype=np.float32)
        self.action = np.zeros((size, action_dim), dtype=np.float32)
        self.reward = np.zeros((size, 1), dtype=np.float32)
        self.next_state = np.zeros((size, state_dim), dtype=np.float32)
        self.done = np.zeros((size, 1), dtype=np.float32)
        self.max_size = int(size)
        self.ptr = 0
        self.size = 0
        self.device = device

    def add(self, state, action, reward, next_state, done):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.reward[self.ptr] = reward
        self.next_state[self.ptr] = next_state
        self.done[self.ptr] = float(done)
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size: int):
        idx = np.random.randint(0, self.size, size=batch_size)
        return {
            "state": torch.as_tensor(self.state[idx], dtype=torch.float32, device=self.device),
            "action": torch.as_tensor(self.action[idx], dtype=torch.float32, device=self.device),
            "reward": torch.as_tensor(self.reward[idx], dtype=torch.float32, device=self.device),
            "next_state": torch.as_tensor(self.next_state[idx], dtype=torch.float32, device=self.device),
            "done": torch.as_tensor(self.done[idx], dtype=torch.float32, device=self.device),
        }
