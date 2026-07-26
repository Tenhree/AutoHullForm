from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, Tuple


@dataclass
class EnvConfig:
    """Environment hyper-parameters matching the modeling document.

    State: [z1,z2,z3, B/L, D/L, T/D, C_B, V, R/R0, e_volume]
    Action: delta_z in [-1, 1]^3, applied as z <- clip(z + action_scale * action).
    """

    # Latent/action configuration
    z_min: float = -20
    z_max: float = 20
    action_scale: float = 0.10
    episode_len: int = 10

    # Constraint/reward configuration
    eps_volume: float = 0.02
    lambda_volume: float = 50.0
    lambda_geometry: float = 1.0
    lambda_resistance: float = 50.0
    geometry_penalty_threshold: float = 0.10
    invalid_reward: float = -10.0

    # State/action dimensions from the design document
    state_dim: int = 10
    action_dim: int = 3

    # Training case sampler. L is normally fixed to 3 m model scale.
    L_fixed: float = 3.0
    B_range: Tuple[float, float] = (0.3, 2.0)
    D_range: Tuple[float, float] = (0.15, 0.5)
    T_over_D_range: Tuple[float, float] = (0.2, 0.85)
    CB_range: Tuple[float, float] = (0.45, 0.95)
    V_range: Tuple[float, float] = (0.20, 1.80)
    z0_range: Tuple[float, float] = (-10, 10)

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class TrainConfig:
    seed: int = 42
    total_steps: int = 5000000
    start_steps: int = 250000
    update_after: int = 1000
    update_every: int = 1
    batch_size: int = 128
    gamma: float = 0.99
    tau: float = 0.005
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    hidden_sizes: Tuple[int, int] = (256, 256)

    # Logging/evaluation
    log_steps: bool = True
    eval_interval: int = 5_000
    eval_episodes: int = 8
    save_best_pointclouds: bool = False

    def to_dict(self) -> Dict:
        d = asdict(self)
        d["hidden_sizes"] = list(self.hidden_sizes)
        return d


@dataclass
class TD3Config(TrainConfig):
    policy_noise: float = 0.20
    noise_clip: float = 0.50
    policy_delay: int = 2
    exploration_noise: float = 0.10


@dataclass
class DDPGConfig(TrainConfig):
    exploration_noise: float = 0.10


@dataclass
class SACConfig(TrainConfig):
    alpha_lr: float = 3e-4
    init_alpha: float = 0.20
    target_entropy: float = -3.0
    automatic_entropy_tuning: bool = True


@dataclass
class PPOConfig(TrainConfig):
    total_steps: int = 50_000
    rollout_steps: int = 2048
    ppo_epochs: int = 10
    minibatch_size: int = 256
    clip_ratio: float = 0.20
    gae_lambda: float = 0.95
    value_coef: float = 0.50
    entropy_coef: float = 0.00
    max_grad_norm: float = 0.50


DEFAULT_GEOMETRY_WEIGHTS: Dict[str, float] = {
    "x_mono": 5.0,
    "range": 1.0,
    "y0": 10.0,
    "x0": 10.0,
    "line_smooth": 2.0,
    "z_smooth": 2.0,
    "z_osc": 1.0,
}
