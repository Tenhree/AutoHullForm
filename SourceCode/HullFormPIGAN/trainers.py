import os
import re
import torch
from losses import total_stage1_loss, total_stage2_loss
from utils import move_batch_to_device, average_log_dict,plot_hull_3d

def get_latest_checkpoint(ckpt_dir):

    if not os.path.exists(ckpt_dir):
        return None

    ckpt_files = []
    pattern = re.compile(r"stage1_epoch_(\d+)\.pth")# Modify to the corresponding path.

    for fname in os.listdir(ckpt_dir):
        match = pattern.match(fname)
        if match:
            epoch_num = int(match.group(1))
            ckpt_files.append((epoch_num, os.path.join(ckpt_dir, fname)))

    if not ckpt_files:
        return None

    ckpt_files.sort(key=lambda x: x[0])
    return ckpt_files[-1][1]

def train_stage1(model, train_loader, cfg):
    device = cfg.device
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr_stage1)

    ckpt_dir = getattr(cfg, "stage1_ckpt_path", ".//Model20X50/Dim3/checkpoints/stage1/")# Modify to the corresponding path.
    os.makedirs(ckpt_dir, exist_ok=True)

    start_epoch = 1

    # =========================
    # 自动恢复训练
    # =========================
    latest_ckpt = get_latest_checkpoint(ckpt_dir)
    if latest_ckpt is not None:
        print(f"find checkpoint，recover: {latest_ckpt}")
        checkpoint = torch.load(latest_ckpt, map_location=device)

        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1

        print(f"recover，from epoch {start_epoch} retraining")
    else:
        print("No checkpoint!")

    for epoch in range(start_epoch, cfg.epochs_stage1 + 1):
        model.train()
        logs = []
        step=0
        for batch in train_loader:
            batch = move_batch_to_device(batch, device)
            x = batch["x"]


            cond = torch.zeros(x.shape[0], 2, device=device)

            x_hat, mu, logvar = model(x, cond)

            loss, log_dict = total_stage1_loss(x_hat, x, mu, logvar, cfg, epoch)
            printable_log = {k: float(v.item()) for k, v in log_dict.items()}
            print(f"[Stage1][Epoch {epoch:03d}][Batch {step:04d}] {printable_log}")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            step += 1
            logs.append(log_dict)

        avg_logs = average_log_dict(logs)
        print(f"[Stage1][Epoch {epoch:03d}] {avg_logs}")

        if epoch % 100 == 0:
            ckpt_path = os.path.join(ckpt_dir, f"stage1_epoch_{epoch:03d}.pth")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "avg_logs": avg_logs,
            }, ckpt_path)
            print(f"checkpoint save: {ckpt_path}")
    return model


def train_stage2(model, train_loader, cfg):
    device = cfg.device
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr_stage2)
    ckpt_dir = getattr(cfg, "stage2_ckpt_path", ".//Model20X50/Dim3/checkpoints/stage2/")
    os.makedirs(ckpt_dir, exist_ok=True)

    start_epoch = 1


    latest_ckpt = get_latest_checkpoint(ckpt_dir)
    if latest_ckpt is not None:
        print(f"find checkpoint，recover: {latest_ckpt}")
        checkpoint = torch.load(latest_ckpt, map_location=device)

        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1

        print(f"recover，from epoch {start_epoch} retraining")
    else:
        print("No checkpoint!")


    for epoch in range(start_epoch, cfg.epochs_stage2 + 1):
        model.train()
        logs = []
        step=0
        for batch in train_loader:
            batch = move_batch_to_device(batch, device)
            x = batch["x"].to(device=device, dtype=torch.float32)
            draft = batch["draft"].to(device=device, dtype=torch.float32)
            cb = batch["cb"].to(device=device, dtype=torch.float32)

            cond = torch.stack([draft, cb], dim=1)

            x_hat, mu, logvar = model(x, cond)

            loss, log_dict = total_stage2_loss(
                x_hat=x_hat,
                x=x,
                mu=mu,
                logvar=logvar,
                draft=draft,
                cb_target=cb,
                cfg=cfg,
                epoch=epoch,
            )
            printable_log = {k: float(v.item()) for k, v in log_dict.items()}
            print(f"[Stage2][Epoch {epoch:03d}][Batch {step:04d}] {printable_log}")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            logs.append(log_dict)
            step += 1

        avg_logs = average_log_dict(logs)
        print(f"[Stage2][Epoch {epoch:03d}] {avg_logs}")

        if epoch % 1 == 0:
            ckpt_path = os.path.join(ckpt_dir, f"stage2_epoch_{epoch:03d}.pth")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "avg_logs": avg_logs,
            }, ckpt_path)
            print(f"checkpoint save: {ckpt_path}")

    return model