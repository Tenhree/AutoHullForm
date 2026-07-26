from dataclasses import dataclass
import torch


@dataclass
class ModelConfig:
    in_channels: int = 2
    height: int = 20
    width: int = 50
    latent_dim: int = 3
    cond_dim: int = 2          # [draft, cb]
    hidden_dim: int = 128
    base_channels: int = 32
    use_structured_output: bool = True


@dataclass
class TrainConfig:
    batch_size: int = 64
    lr_stage1: float = 1e-3
    lr_stage2: float = 5e-4
    epochs_stage1: int = 200
    epochs_stage2: int = 10
    kl_anneal_epochs: int = 50

    lambda_rec: float = 1.0
    lambda_kl: float = 1e-2
    lambda_x_mono: float = 5.0
    lambda_range: float = 1.0
    lambda_x0: float = 10.0
    lambda_y0: float = 10.0
    lambda_line_smooth: float = 2.0
    lambda_z_smooth: float = 2.0
    lambda_z_osc: float = 1.0
    lambda_cb: float = 2.0

    device: str = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class PathConfig:
    geometry_data_path: str = "..//..//Gettraindata//TrainingPC20X50"  # Modify to the corresponding path.
    physical_data_path: str = ".//Model20X50"# Modify to the corresponding path.
    stage1_ckpt_path: str = ".//Model20X50/Dim3/checkpoints/stage1/"# Modify to the corresponding path.
    stage2_ckpt_path: str = ".//Model20X50/Dim3/checkpoints/stage2/"# Modify to the corresponding path.
    inference_path: str = ".//Model20X50/Dim3/inference_path/"# Modify to the corresponding path.