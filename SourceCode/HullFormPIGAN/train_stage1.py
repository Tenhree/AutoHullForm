import os
import torch
from torch.utils.data import DataLoader

from config import ModelConfig, TrainConfig, PathConfig
from datasets import HullGeometryDataset
from models import TwoStageCVAE
from trainers import train_stage1
from utils import ensure_dir


def main():
    model_cfg = ModelConfig()
    train_cfg = TrainConfig()
    path_cfg = PathConfig()

    dataset = HullGeometryDataset(path_cfg.geometry_data_path,20,50)
    loader = DataLoader(
        dataset,
        batch_size=train_cfg.batch_size,
        shuffle=True,
        drop_last=False,
    )

    model = TwoStageCVAE(
        in_channels=model_cfg.in_channels,
        latent_dim=model_cfg.latent_dim,
        cond_dim=model_cfg.cond_dim,
        hidden_dim=model_cfg.hidden_dim,
        base_channels=model_cfg.base_channels,
        use_structured_output=model_cfg.use_structured_output,
    )

    model = train_stage1(model, loader, train_cfg)

    ensure_dir(os.path.dirname(path_cfg.stage1_ckpt_path))
    torch.save({
        "model_state_dict": model.state_dict(),
    },  os.path.join(path_cfg.stage1_ckpt_path, f"stage1_final.pth"))
    print(f"Stage 1 model saved to: {path_cfg.stage1_ckpt_path}")


if __name__ == "__main__":
    main()