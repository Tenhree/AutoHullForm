import torch
import random
import os
from config import ModelConfig, TrainConfig, PathConfig
from models import TwoStageCVAE
from utils import plot_hull_3d,estimate_cb_from_half_hull,hausdorff_distance,build_points,Point2Hull
import csv
import numpy as np

@torch.no_grad()
def sample_geometry_prior(model, num_samples, device):
    model.eval()
    cond = torch.zeros(num_samples, 2, device=device)
    return model.sample(num_samples, cond, device)


@torch.no_grad()
def sample_conditioned_geometry(model, draft_value, cb_value, num_samples, device):
    model.eval()
    cond = torch.tensor([[draft_value, cb_value]], device=device).repeat(num_samples, 1)
    return model.sample(num_samples, cond, device)


def main():
    model_cfg = ModelConfig()
    train_cfg = TrainConfig()
    path_cfg = PathConfig()

    model = TwoStageCVAE(
        in_channels=model_cfg.in_channels,
        latent_dim=model_cfg.latent_dim,
        cond_dim=model_cfg.cond_dim,
        hidden_dim=model_cfg.hidden_dim,
        base_channels=model_cfg.base_channels,
        use_structured_output=model_cfg.use_structured_output,
    )
    checkpoint = torch.load(
        os.path.join(path_cfg.stage2_ckpt_path, "stage2_final.pth"),# Modify to the corresponding path.
        map_location=train_cfg.device
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(train_cfg.device)
    num_samples = 30
    # 示例1：按条件生成
    x_gen = sample_conditioned_geometry(
        model=model,
        draft_value=0.5,
        cb_value=0.6,
        num_samples=num_samples,
        device=train_cfg.device,
    )
    print("Generated conditioned samples shape:", x_gen.shape)
    with open(os.path.join(path_cfg.inference_path, "violation_result.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["cb_pred", "cb_true","violation_result"])

        cb_pred = estimate_cb_from_half_hull(
            x_gen[:, 0, :, :],
            x_gen[:, 1, :, :],
            torch.full((num_samples,), 0.5)
        )

        for i in range(num_samples):
            cb_true = 0.6
            cb_pred_value = cb_pred[i, 0].item()
            writer.writerow([cb_pred_value, cb_true,np.abs(cb_pred_value-cb_true)/cb_true*100])

    D = []
    L=90.00
    Bd=15.6/2
    Dd=7.15
    h, w = x_gen.shape[2], x_gen.shape[3]
    Z = np.linspace(0.0, 1.0, h, dtype=np.float32).reshape(h, 1).repeat(w, axis=1).ravel()

    for i in range(num_samples - 1):
        S1 = x_gen[i].detach().cpu().numpy()
        points1 = build_points(S1, Z,L,Bd,Dd)

        for j in range(i + 1, num_samples):
            S2 = x_gen[j].detach().cpu().numpy()
            points2 = build_points(S2, Z,L,Bd,Dd)

            D.append(hausdorff_distance(points1, points2))

    csv_path = os.path.join(path_cfg.inference_path, "hausdorff_distance.csv")

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["distance"])
        writer.writerows([[d] for d in D])

    for i in range(30):
        suc=Point2Hull(x_gen[i],L,Dd,Bd,20,50,path_cfg.inference_path,"test_"+str(i)+".stl","test_deck_"+str(i)+".stl","test_PC_"+str(i)+".csv")




if __name__ == "__main__":

    seed = 970709

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    main()