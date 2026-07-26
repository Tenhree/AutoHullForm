from __future__ import annotations
from typing import Callable, Dict, Tuple
import numpy as np
import torch
from settings import DEFAULT_GEOMETRY_WEIGHTS

CBEstimator = Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]


def _safe_divide_by_max(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    arr = np.nan_to_num(np.asarray(x, dtype=np.float32), nan=0.0, posinf=1.0, neginf=0.0)
    max_value = float(np.max(arr)) if arr.size else 0.0
    if abs(max_value) < eps:
        return np.zeros_like(arr, dtype=np.float32)
    return (arr / max_value).astype(np.float32)


def build_half_hull_points_from_surface(surface: np.ndarray, L: float, B: float, D: float) -> np.ndarray:
    surface = np.asarray(surface, dtype=np.float32)
    if surface.ndim != 3 or surface.shape[0] != 2:
        raise ValueError(f"surface is [2,H,W]，current {surface.shape}")
    _, h, w = surface.shape
    x = _safe_divide_by_max(surface[0]) * float(L)
    y = _safe_divide_by_max(surface[1]) * (float(B) / 2.0)
    z = np.linspace(0.0, float(D), h, dtype=np.float32).reshape(h, 1).repeat(w, axis=1)
    return np.stack([x.ravel(), y.ravel(), z.ravel()], axis=1).astype(np.float32)


def build_points_from_surface(surface: np.ndarray, L: float, B: float, D: float) -> np.ndarray:
    return build_half_hull_points_from_surface(surface=surface, L=L, B=B, D=D)


def _waterline_area_torch_fallback(x_layer: torch.Tensor, y_layer: torch.Tensor) -> torch.Tensor:
    if x_layer.numel() < 2:
        return torch.zeros((), device=x_layer.device, dtype=x_layer.dtype)
    x_sorted, idx = torch.sort(x_layer, dim=-1)
    y_sorted = torch.gather(y_layer, dim=-1, index=idx)
    x_sorted = torch.nan_to_num(x_sorted, nan=0.0, posinf=1.0, neginf=0.0)
    y_sorted = torch.nan_to_num(y_sorted, nan=0.0, posinf=1.0, neginf=0.0)
    area = 2.0 * torch.trapz(y_sorted, x_sorted, dim=-1)
    return torch.clamp(area, min=0.0)


def estimate_cb_from_half_hull_fallback(x_channel: torch.Tensor, y_channel: torch.Tensor, draft: torch.Tensor) -> torch.Tensor:
    bsz, h, _ = y_channel.shape
    device = y_channel.device
    out = []
    for b in range(bsz):
        X = x_channel[b, :, :]
        Y = y_channel[b, :, :]
        hv = X.shape[0]
        z_values = torch.linspace(0.0, 1.0, hv, device=device)
        Tb = torch.clamp(draft[b], 0.0, 1.0)
        if hv == 1:
            out.append(torch.zeros((1,), device=device))
            continue
        pos = Tb * (hv - 1)
        i0 = torch.floor(pos).long().clamp(0, hv - 1)
        i1 = (i0 + 1).clamp(0, hv - 1)
        z0 = z_values[i0]
        z1 = z_values[i1]
        denom_z = (z1 - z0).clamp_min(1e-8)
        alpha = ((Tb - z0) / denom_z).clamp(0.0, 1.0)
        X0 = X[i0]
        X1 = X[i1]
        Y0 = Y[i0]
        Y1 = Y[i1]
        X_new = X0 + alpha * (X1 - X0)
        Y_new = Y0 + alpha * (Y1 - Y0)
        row_areas = []
        row_beams = []
        for k in range(hv):
            xk = X[k]
            yk = Y[k]
            if xk.numel() < 2:
                Ak = torch.zeros((), device=device)
                Bk = torch.zeros((), device=device)
            else:
                Ak = _waterline_area_torch_fallback(xk, yk)
                Bk = torch.clamp(torch.max(yk), min=0.0)
            row_areas.append(Ak)
            row_beams.append(Bk)
        row_areas = torch.stack(row_areas, dim=0)
        row_beams = torch.stack(row_beams, dim=0)
        if X_new.numel() < 2:
            A_new = torch.zeros((), device=device)
            B_new = torch.zeros((), device=device)
        else:
            A_new = _waterline_area_torch_fallback(X_new, Y_new)
            B_new = torch.clamp(torch.max(Y_new), min=1e-8)
        if i0 <= 0:
            vol_below = torch.zeros((), device=device)
        else:
            z_seg = z_values[: i0 + 1]
            A_seg = row_areas[: i0 + 1]
            vol_below = torch.trapz(A_seg, z_seg)
        dz_last = (Tb - z0).clamp_min(0.0)
        A_i0 = row_areas[i0]
        vol_last = 0.5 * (A_i0 + A_new) * dz_last
        volume = torch.clamp(vol_below + vol_last, min=0.0)
        if X_new.numel() >= 2:
            Lwl = (torch.max(X_new) - torch.min(X_new)).clamp_min(1e-8)
        else:
            Lwl = torch.tensor(1e-8, device=device)
        if i0 <= 0:
            B_under = row_beams[0]
        else:
            B_under = torch.max(row_beams[: i0 + 1])
        B_T = torch.maximum(B_under, B_new).clamp_min(1e-8)
        T_safe = Tb.clamp_min(1e-8)
        Cb = volume / (Lwl * B_T * T_safe * 2.0 + 1e-8)
        Cb = torch.clamp(Cb, min=0.0, max=5.0)
        out.append(Cb.view(1))
    return torch.stack(out, dim=0)


def estimate_cb_with_code1(surface: np.ndarray, draft_ratio: float, estimator: CBEstimator | None, device: torch.device | str = "cpu") -> float:

    surface = np.asarray(surface, dtype=np.float32)
    if surface.ndim != 3 or surface.shape[0] != 2:
        raise ValueError(f"surface is [2,H,W]，current {surface.shape}")
    device = torch.device(device)
    estimator = estimator or estimate_cb_from_half_hull_fallback
    x_tensor = torch.from_numpy(surface[0:1]).to(device=device, dtype=torch.float32)
    y_tensor = torch.from_numpy(surface[1:2]).to(device=device, dtype=torch.float32)
    draft_tensor = torch.tensor([float(draft_ratio)], device=device, dtype=torch.float32)
    with torch.no_grad():
        cb_tensor = estimator(x_tensor, y_tensor, draft_tensor)
    cb_value = float(cb_tensor.detach().float().cpu().reshape(-1)[0].item())
    return cb_value


def displacement_from_cb(L: float, B: float, T: float, cb_estimated: float) -> float:
    return float(L) * float(B) * float(T) * float(cb_estimated)


def volume_error_from_cb(L: float, B: float, T: float, CB: float, cb_estimated: float) -> Tuple[float, float, float]:
    target = float(L) * float(B) * float(T) * float(CB)
    actual = displacement_from_cb(L=L, B=B, T=T, cb_estimated=cb_estimated)
    if abs(target) < 1e-12:
        return actual, target, float("inf")
    return actual, target, float((actual - target) / target)


def geometry_loss_bundle_np(surface: np.ndarray) -> Dict[str, float]:
    s = np.nan_to_num(np.asarray(surface, dtype=np.float32), nan=0.0, posinf=1.0, neginf=0.0)
    if s.ndim != 3 or s.shape[0] != 2:
        raise ValueError(f"surface 必须是 [2,H,W]，当前得到 {s.shape}")
    x = s[0]
    y = s[1]

    def relu(a: np.ndarray) -> np.ndarray:
        return np.maximum(a, 0.0)

    x_mono = float(np.mean(relu(x[:, :-1] - x[:, 1:]))) if x.shape[1] > 1 else 0.0
    range_loss = float(np.mean(relu(-s) + relu(s - 1.0)))
    y0 = float(np.mean(np.abs(y[0, :]))) if y.shape[0] > 0 else 0.0
    if x.shape[0] >= 3:
        d = x[1:, 0] - x[:-1, 0]
        x0 = float(np.mean(relu(-(d[1:] * d[:-1])))) if d.size >= 2 else 0.0
    else:
        x0 = 0.0
    line_smooth = float(np.mean(np.abs(y[:, 2:] - 2.0 * y[:, 1:-1] + y[:, :-2]))) if y.shape[1] >= 3 else 0.0
    z_smooth = float(np.mean(np.abs(y[2:, :] - 2.0 * y[1:-1, :] + y[:-2, :]))) if y.shape[0] >= 3 else 0.0
    if y.shape[0] >= 3:
        dz = y[1:, :] - y[:-1, :]
        z_osc = float(np.mean(relu(-(dz[:-1, :] * dz[1:, :]))))
    else:
        z_osc = 0.0
    return {
        "x_mono": x_mono,
        "range": range_loss,
        "y0": y0,
        "x0": x0,
        "line_smooth": line_smooth,
        "z_smooth": z_smooth,
        "z_osc": z_osc,
    }


def weighted_geometry_penalty(surface: np.ndarray, weights: Dict[str, float] | None = None) -> Tuple[float, Dict[str, float]]:
    weights = weights or DEFAULT_GEOMETRY_WEIGHTS
    losses = geometry_loss_bundle_np(surface)
    total = 0.0
    for k, v in losses.items():
        total += float(weights.get(k, 1.0)) * float(v)
    return float(total), losses

def ensure_num_points(points: np.ndarray, num_points: int) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"points need[N,3]，This is {pts.shape}")
    n = pts.shape[0]
    if n == num_points:
        return pts
    if n <= 0:
        raise ValueError("None")
    idx = np.linspace(0, n - 1, num_points).round().astype(np.int64)
    return pts[idx].astype(np.float32)
