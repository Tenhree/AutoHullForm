import torch
import torch.nn.functional as F
from utils import estimate_cb_from_half_hull

def reconstruction_loss(x_hat: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    loss_x = F.l1_loss(x_hat[:, 0:1], x[:, 0:1])
    loss_y = F.l1_loss(x_hat[:, 1:2], x[:, 1:2])
    return loss_x + loss_y


def kl_loss(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())


def x_monotonicity_loss(x_hat: torch.Tensor) -> torch.Tensor:
    x_ch = x_hat[:, 0, :, :]                 # [B,10,10]
    diff = x_ch[:, :, :-1] - x_ch[:, :, 1:]
    return F.relu(diff).mean()


def range_loss(x_hat: torch.Tensor) -> torch.Tensor:
    lower = F.relu(-x_hat)
    upper = F.relu(x_hat - 1.0)
    return (lower + upper).mean()


def y_first_row_zero_loss(x_hat: torch.Tensor) -> torch.Tensor:
    y_ch = x_hat[:, 1, :, :]
    return torch.abs(y_ch[:, 0, :]).mean()


def line_smoothness_loss(x_hat: torch.Tensor) -> torch.Tensor:

    y_ch = x_hat[:, 1, :, :]
    d2 = y_ch[:, :, 2:] - 2.0 * y_ch[:, :, 1:-1] + y_ch[:, :, :-2]
    return torch.abs(d2).mean()


def z_smoothness_loss(x_hat: torch.Tensor) -> torch.Tensor:

    y_ch = x_hat[:, 1, :, :]
    d2 = y_ch[:, 2:, :] - 2.0 * y_ch[:, 1:-1, :] + y_ch[:, :-2, :]
    return torch.abs(d2).mean()


def z_oscillation_loss(x_hat: torch.Tensor) -> torch.Tensor:

    y_ch = x_hat[:, 1, :, :]
    d = y_ch[:, 1:, :] - y_ch[:, :-1, :]
    prod = d[:, :-1, :] * d[:, 1:, :]
    return F.relu(-prod).mean()

def first_point_no_oscillation_loss(x):


    y = x[:, 0, :, 0]          # [B, 10]

    d = y[:, 1:] - y[:, :-1]   # [B, 9]

    prod = d[:, 1:] * d[:, :-1]  # [B, 8]

    loss = F.relu(-prod)

    return loss.mean()

def cb_condition_loss(
    cb_pred: torch.Tensor,
    cb_target: torch.Tensor,
) -> torch.Tensor:
    return F.l1_loss(cb_pred, cb_target)


def geometry_loss_bundle(x_hat: torch.Tensor):
    return {
        "x_mono": x_monotonicity_loss(x_hat),
        "range": range_loss(x_hat),
        "y0": y_first_row_zero_loss(x_hat),
        "x0": first_point_no_oscillation_loss(x_hat),
        "line_smooth": line_smoothness_loss(x_hat),
        "z_smooth": z_smoothness_loss(x_hat),
        "z_osc": z_oscillation_loss(x_hat),
    }


def total_stage1_loss(x_hat, x, mu, logvar, cfg, epoch):
    rec = reconstruction_loss(x_hat, x)
    kl = kl_loss(mu, logvar)
    geo = geometry_loss_bundle(x_hat)

    kl_weight = cfg.lambda_kl * min(1.0, epoch / max(1, cfg.kl_anneal_epochs))

    total = (
        cfg.lambda_rec * rec
        + kl_weight * kl
        + cfg.lambda_x_mono * geo["x_mono"]
        + cfg.lambda_range * geo["range"]
        + cfg.lambda_x0 * geo["x0"]
        + cfg.lambda_y0 * geo["y0"]
        + cfg.lambda_line_smooth * geo["line_smooth"]
        + cfg.lambda_z_smooth * geo["z_smooth"]
        + cfg.lambda_z_osc * geo["z_osc"]
    )

    log_dict = {
        "loss_total": total.detach(),
        "loss_rec": rec.detach(),
        "loss_kl": kl.detach(),
        "loss_x_mono": geo["x_mono"].detach(),
        "loss_range": geo["range"].detach(),
        "loss_x0": geo["x0"].detach(),
        "loss_y0": geo["y0"].detach(),
        "loss_line_smooth": geo["line_smooth"].detach(),
        "loss_z_smooth": geo["z_smooth"].detach(),
        "loss_z_osc": geo["z_osc"].detach(),
    }
    return total, log_dict


def total_stage2_loss(x_hat, x, mu, logvar, draft, cb_target, cfg, epoch):
    rec = reconstruction_loss(x_hat, x)
    kl = kl_loss(mu, logvar)
    geo = geometry_loss_bundle(x_hat)

    cb_pred=estimate_cb_from_half_hull(x_hat[:,0,:,:],x_hat[:,1,:,:],draft)
    cb_loss = cb_condition_loss(cb_pred.squeeze(1), cb_target)

    kl_weight = cfg.lambda_kl * min(1.0, epoch / max(1, cfg.kl_anneal_epochs))

    total = (
        cfg.lambda_rec * rec
        + kl_weight * kl
        + cfg.lambda_x_mono * geo["x_mono"]
        + cfg.lambda_range * geo["range"]
        + cfg.lambda_x0 * geo["x0"]
        + cfg.lambda_y0 * geo["y0"]
        + cfg.lambda_line_smooth * geo["line_smooth"]
        + cfg.lambda_z_smooth * geo["z_smooth"]
        + cfg.lambda_z_osc * geo["z_osc"]
        + cfg.lambda_cb * cb_loss
    )

    log_dict = {
        "loss_total": total.detach(),
        "loss_rec": rec.detach(),
        "loss_kl": kl.detach(),
        "loss_x_mono": geo["x_mono"].detach(),
        "loss_range": geo["range"].detach(),
        "loss_x0": geo["x0"].detach(),
        "loss_y0": geo["y0"].detach(),
        "loss_line_smooth": geo["line_smooth"].detach(),
        "loss_z_smooth": geo["z_smooth"].detach(),
        "loss_z_osc": geo["z_osc"].detach(),
        "loss_cb": cb_loss.detach(),
    }
    return total, log_dict