import os
import torch
import numpy as np
import pandas as pd
import scipy.interpolate as interp
from stl import mesh
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def move_batch_to_device(batch, device: str):
    return {k: v.to(device) for k, v in batch.items()}


def average_log_dict(log_list):
    if len(log_list) == 0:
        return {}

    keys = log_list[0].keys()
    avg = {}
    for k in keys:
        avg[k] = torch.stack([d[k] for d in log_list]).mean().item()
    return avg

def set_axes_equal(ax):

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = x_limits[1] - x_limits[0]
    y_range = y_limits[1] - y_limits[0]
    z_range = z_limits[1] - z_limits[0]

    max_range = max(x_range, y_range, z_range)

    x_middle = np.mean(x_limits)
    y_middle = np.mean(y_limits)
    z_middle = np.mean(z_limits)

    ax.set_xlim3d([x_middle - max_range/2, x_middle + max_range/2])
    ax.set_ylim3d([y_middle - max_range/2, y_middle + max_range/2])
    ax.set_zlim3d([z_middle - max_range/2, z_middle + max_range/2])

def plot_hull_3d(S_hat, step, epoch, sample_idx=0):

    os.makedirs("./trainpic", exist_ok=True)

    S = S_hat[sample_idx].detach().cpu().numpy()
    h=S.shape[1]
    w=S.shape[2]

    X = S[0, :, :]
    Y = S[1, :, :]
    z = np.linspace(0.0, 1.0, h, dtype=np.float32).reshape(h, 1).repeat(w, axis=1)

    fig = plt.figure(figsize=(18, 5), dpi=300)

    # 1) Scatter
    ax1 = fig.add_subplot(131, projection='3d')
    for j in range(h):
        ax1.scatter(X[j, :], Y[j, :], z[j, :], s=5, alpha=0.6)
    ax1.view_init(elev=10, azim=40)
    ax1.set_title('Scatter Plot')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    set_axes_equal(ax1)

    # 2) Curve
    ax2 = fig.add_subplot(132, projection='3d')
    for j in range(h):
        ax2.plot(X[j, :], Y[j, :], z[j, :], linewidth=1.0, alpha=0.7)
    ax2.view_init(elev=10, azim=40)
    ax2.set_title('Curve Plot')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    set_axes_equal(ax2)

    # 3) Surface
    ax3 = fig.add_subplot(133, projection='3d')
    ax3.plot_surface(X, Y, z, alpha=0.7, edgecolor='k')
    ax3.view_init(elev=10, azim=40)
    ax3.set_title('Surface Plot')
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.set_zlabel('Z')
    set_axes_equal(ax3)

    plt.tight_layout()
    path = f"./trainpic/my_plotepoch{epoch}step{step}.png"
    if os.path.exists(path):
        os.remove(path)
    plt.savefig(path)
    plt.close(fig)


def _waterline_area_torch(x_layer: torch.Tensor, y_layer: torch.Tensor) -> torch.Tensor:

    if x_layer.numel() < 2:
        return torch.zeros((), device=x_layer.device, dtype=x_layer.dtype)

    x_sorted, idx = torch.sort(x_layer, dim=-1)
    y_sorted = torch.gather(y_layer, dim=-1, index=idx)

    # 防止出现非有限值
    x_sorted = torch.nan_to_num(x_sorted, nan=0.0, posinf=1.0, neginf=0.0)
    y_sorted = torch.nan_to_num(y_sorted, nan=0.0, posinf=1.0, neginf=0.0)

    area = 2.0 * torch.trapz(y_sorted, x_sorted, dim=-1)
    return torch.clamp(area, min=0.0)

def estimate_cb_from_half_hull(
    x_channel: torch.Tensor,
    y_channel: torch.Tensor,
    draft: torch.Tensor,
) -> torch.Tensor:

    bsz, h, w = y_channel.shape
    device = y_channel.device
    out = []

    for b in range(bsz):
        X = x_channel[b, :, :]          # (h,w)
        Y = y_channel[b, :, :]           # (h,w)
        hv = X.shape[0]

        # z坐标按“有效层数”均匀分布到 [0,1]
        z_values = torch.linspace(0.0, 1.0, hv, device=device)

        Tb = torch.clamp(draft[b,], 0.0, 1.0)

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

        X0 = X[i0]   # (w,)
        X1 = X[i1]
        Y0 = Y[i0]
        Y1 = Y[i1]
        # 仅保留插值层中双侧都有效的位置
        X_new = (X0 + alpha * (X1 - X0))
        Y_new = (Y0 + alpha * (Y1 - Y0))

        # 每层面积
        row_areas = []
        row_beams = []
        for k in range(hv):
            xk = X[k]
            yk = Y[k]

            if xk.numel() < 2:
                Ak = torch.zeros((), device=device)
                Bk = torch.zeros((), device=device)
            else:
                Ak = _waterline_area_torch(xk, yk)
                Bk = torch.clamp(torch.max(yk), min=0.0)
            row_areas.append(Ak)
            row_beams.append(Bk)

        row_areas = torch.stack(row_areas, dim=0)   # (hv,)
        row_beams = torch.stack(row_beams, dim=0)   # (hv,)


        x_new_valid = X_new
        y_new_valid = Y_new
        if x_new_valid.numel() < 2:
            A_new = torch.zeros((), device=device)
            B_new = torch.zeros((), device=device)
        else:
            A_new = _waterline_area_torch(x_new_valid, y_new_valid)
            B_new = torch.clamp(torch.max(y_new_valid), min=1e-8)


        if i0 <= 0:
            vol_below = torch.zeros((), device=device)
        else:
            z_seg = z_values[:i0 + 1]
            A_seg = row_areas[:i0 + 1]
            vol_below = torch.trapz(A_seg, z_seg)

        dz_last = (Tb - z0).clamp_min(0.0)
        A_i0 = row_areas[i0]
        vol_last = 0.5 * (A_i0 + A_new) * dz_last

        volume = torch.clamp(vol_below + vol_last, min=0.0)


        if x_new_valid.numel() >= 2:
            Lwl = (torch.max(x_new_valid) - torch.min(x_new_valid)).clamp_min(1e-8)
        else:
            Lwl = torch.tensor(1e-8, device=device)


        if i0 <= 0:
            B_under = row_beams[0]
        else:
            B_under = torch.max(row_beams[:i0 + 1])

        B_T = torch.maximum(B_under, B_new).clamp_min(1e-8)
        T_safe = Tb.clamp_min(1e-8)

        Cb = volume / (Lwl * B_T * T_safe * 2.0 + 1e-8)
        Cb = torch.clamp(Cb, min=0.0, max=5.0)

        out.append(Cb.view(1))

    return torch.stack(out, dim=0)   # (B,1)

def hausdorff_distance(A, B):


    A = np.asarray(A)
    B = np.asarray(B)

    dists_A_to_B = np.sqrt(((A[:, None, :] - B[None, :, :]) ** 2).sum(axis=2))
    min_A_to_B = dists_A_to_B.min(axis=1)


    dists_B_to_A = np.sqrt(((B[:, None, :] - A[None, :, :]) ** 2).sum(axis=2))
    min_B_to_A = dists_B_to_A.min(axis=1)

    return max(min_A_to_B.max(), min_B_to_A.max())

def build_points(S, Z,L,B,D):
    X = S[0].ravel()
    Y = S[1].ravel()
    X = X / max(X)
    Y = Y / max(Y)
    Z = Z/ max(Z)
    X=X*L
    Y=Y*B
    Z=Z*D
    return np.stack((X, Y, Z), axis=1)


def Point2Hull(PC, LOA, Dd, Bd, num_wl, num_x, Outfilepath, Outfilename, Outfilename1, OutPCfilename):
    Z = np.linspace(0, 1, num_wl).round(3)
    X = PC[0, :, :].detach().cpu().numpy()
    Y = PC[1, :, :].detach().cpu().numpy()
    mypts = np.empty((0, 3))
    wfb = []
    for j in range(num_wl):
        x_line = X[j, :]
        y_line = Y[j, :]
        if y_line[-1] >= 1e-2:
            wfb.append(j)
        z_line = np.full_like(x_line, Z[j])
        new_pts = np.stack([x_line, y_line, z_line], axis=1)
        if mypts is None or mypts.size == 0:
            mypts = new_pts
        else:
            mypts = np.vstack((mypts, new_pts))
    wfb.append(num_wl)
    wl_above = num_wl - wfb[0]

    PC = PC.detach().cpu().numpy()
    PC = PC.transpose(1, 2, 0).reshape(-1, 2)
    Z = np.linspace(0, 1, num_wl).reshape(-1, 1).repeat(num_x, axis=1)

    Z_flat = Z.flatten().reshape(-1, 1)  # shape: (num_wl * num_x, 1)

    PC = np.hstack([PC, Z_flat])  # shape: (N, 4) → [x, y, z, wavelength_norm]

    Min_Pos = min(PC[:, 0])
    PC[:, 0] = PC[:, 0] - (Min_Pos)
    PC[:, 0] = PC[:, 0] / max(PC[:, 0])
    PC[:, 1] = PC[:, 1] / max(PC[:, 1])  # 半宽
    PC[:, 2] = PC[:, 2] / max(PC[:, 2])

    PC[:, 0] = PC[:, 0] * LOA
    PC[:, 1] = PC[:, 1] * Bd
    PC[:, 2] = PC[:, 2] * Dd

    csv_path = os.path.join(Outfilepath, OutPCfilename)

    df = pd.DataFrame(PC, columns=["x", "y", "z"])


    df.to_csv(csv_path, index=False)



    pts = [PC[i * num_x:(i + 1) * num_x] for i in range(num_wl)]
    x_ship_pos = np.linspace(0, max(PC[:, 0]), num_x)

    NUM_WL = num_wl

    for i in range(0, NUM_WL):
        indices = np.where((x_ship_pos > pts[i][0, 0]) & (x_ship_pos < pts[i][-1, 0]))[0]
        _, idx = np.unique(pts[i][:, 0], return_index=True)
        WL_curve = interp.interp1d(pts[i][idx, 0], pts[i][idx, 1], kind='linear')
        ydata = WL_curve(x_ship_pos[indices])
        pts[i] = np.vstack(
            [pts[i][0, :], np.stack([x_ship_pos[indices], ydata, pts[i][0, 2] * np.ones(len(indices))], axis=1),
             pts[i][-1, :]])
        # start to assemble the triangles into vectors of indices from pts
        TriVec = []

    for i in range(0, NUM_WL - 1):

        # Find idx where the mesh grids begin to align between two rows returns a zero or 1:

        bow = np.argmax([pts[i][0, 0], pts[i + 1][0, 0]])

        stern = np.argmin([pts[i][-1, 0], pts[i + 1][-1, 0]])

        # Find index where mesh grid lines up and ends between each WL

        if bow:
            idx_WLB1 = 1
            idx_WLB0 = np.where(pts[i][:, 0] == pts[i + 1][idx_WLB1, 0])[0][0]
        else:
            idx_WLB0 = 1
            aaa = pts[i + 1][:, 0]
            bbb = pts[i][idx_WLB0, 0]
            idx_WLB1 = np.where(pts[i + 1][:, 0] == pts[i][idx_WLB0, 0])[0][0]

        if stern:
            idx_WLS1 = len(pts[i + 1]) - 2
            idx_WLS0 = np.where(pts[i][:, 0] == pts[i + 1][idx_WLS1, 0])[0][0]
        else:
            idx_WLS0 = len(pts[i]) - 2
            idx_WLS1 = np.where(pts[i + 1][:, 0] == pts[i][idx_WLS0, 0])[0][0]

            # check that these two are the same size:

            # Build the bow triangles Includes Port assignments

        if bow:
            TriVec.append([pts[i + 1][idx_WLB1], pts[i][0], pts[i + 1][0]])

            for j in range(0, idx_WLB0):
                TriVec.append([pts[i + 1][idx_WLB1], pts[i][j + 1], pts[i][j]])



        else:

            for j in range(0, idx_WLB1):
                TriVec.append([pts[i][0], pts[i + 1][j], pts[i + 1][j + 1]])

            TriVec.append([pts[i][0], pts[i + 1][idx_WLB1], pts[i][idx_WLB0]])

            # Build main part of hull triangles. Port Assignments
        for j in range(0, idx_WLS1 - idx_WLB1):
            TriVec.append([pts[i][idx_WLB0 + j], pts[i + 1][idx_WLB1 + j], pts[i + 1][idx_WLB1 + j + 1]])
            TriVec.append([pts[i][idx_WLB0 + j], pts[i + 1][idx_WLB1 + j + 1], pts[i][idx_WLB0 + j + 1]])

            # Build the stern:
        if stern:

            for j in range(idx_WLS0, len(pts[i]) - 1):
                TriVec.append([pts[i + 1][idx_WLS1], pts[i][j + 1], pts[i][j]])

            TriVec.append([pts[i + 1][idx_WLS1], pts[i + 1][-1], pts[i][-1]])

        else:

            TriVec.append([pts[i][idx_WLS0], pts[i + 1][idx_WLS1], pts[i][-1]])

            for j in range(idx_WLS1, len(pts[i + 1]) - 1):
                TriVec.append([pts[i][-1], pts[i + 1][j], pts[i + 1][j + 1]])

    TriVec = np.array(TriVec)

    hullTriangles = 2 * len(TriVec)
    numTriangles = hullTriangles

    z_idx = NUM_WL - wl_above - 1

    transomTriangles = 2 * wl_above - 1

    numTriangles += transomTriangles

    numTriangles_fordeck = 2 * len(pts[-1]) - 3

    HULL = mesh.Mesh(np.zeros(numTriangles, dtype=mesh.Mesh.dtype))

    HULL.vectors[0:len(TriVec)] = np.copy(TriVec)

    TriVec_stbd = np.copy(TriVec[:, ::-1])
    TriVec_stbd[:, :, 1] *= -1
    HULL.vectors[len(TriVec):hullTriangles] = np.copy(TriVec_stbd)

    # NowBuild the transom:
    pts_trans = np.zeros((wl_above + 1, 3))

    for i in range(0, len(pts_trans)):
        pts_trans[i] = pts[z_idx + i][-1, :]

    pts_tranp = np.array(pts_trans)

    pts_tranp[:, 1] *= -1.0

    HULL.vectors[hullTriangles] = np.array([pts_trans[0], pts_trans[1], pts_tranp[1]])
    for i in range(1, wl_above):
        HULL.vectors[hullTriangles + 2 * i - 1] = np.array([pts_trans[i], pts_trans[i + 1], pts_tranp[i]])
        HULL.vectors[hullTriangles + 2 * i] = np.array([pts_tranp[i], pts_trans[i + 1], pts_tranp[i + 1]])
    os.makedirs(Outfilepath, exist_ok=True)
    HULL.save(Outfilepath + Outfilename)
    # Add the deck lid
    Deck = mesh.Mesh(np.zeros(numTriangles_fordeck, dtype=mesh.Mesh.dtype))
    pts_Lids = pts[NUM_WL - 1]

    pts_Lidp = np.array(pts_Lids)
    pts_Lidp[:, 1] *= -1.0

    # startTriangles = hullTriangles + transomTriangles

    # Points are orered so the right hand rule points the lid in positive z
    Deck.vectors[0] = np.array([pts_Lids[0], pts_Lidp[1], pts_Lids[1]])

    for i in range(1, len(pts_Lids) - 1):
        Deck.vectors[0 + 2 * i - 1] = np.array([pts_Lids[i], pts_Lidp[i], pts_Lids[i + 1]])
        Deck.vectors[0 + 2 * i] = np.array([pts_Lids[i + 1], pts_Lidp[i], pts_Lidp[i + 1]])

    # utfilepath+Outfilename"
    os.makedirs(Outfilepath, exist_ok=True)
    Deck.save(Outfilepath + Outfilename1)
    return 1
