from __future__ import annotations
import importlib
import importlib.util
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import pandas as pd
import numpy as np
import os
import torch
from stl import mesh
import scipy.interpolate as interp
from geometry import build_points_from_surface, ensure_num_points, volume_error_from_cb, estimate_cb_from_half_hull_fallback, estimate_cb_with_code1
from utils import resolve_device


@dataclass
class GenerationResult:
    points: np.ndarray
    surface: np.ndarray
    cb_estimated: float
    volume_actual: float
    volume_target: float
    volume_error: float
    is_half_hull: bool = True


class HullGenerator:
    def __init__(self, code1_dir: str, checkpoint_path: str, device: str = "auto"):
        self.code1_dir = str(Path(code1_dir).resolve())
        self.checkpoint_path = str(Path(checkpoint_path).resolve())
        self.device = resolve_device(device)
        self.config_mod, self.models_mod, self.code1_utils = self._import_code1_modules()

        self.model = self._load_model()

    def _module_belongs_to_code1(self, module_name: str) -> bool:
        module = sys.modules.get(module_name)
        if module is None:
            return True
        file_name = getattr(module, "__file__", "") or ""
        if not file_name:
            return False
        try:
            Path(file_name).resolve().relative_to(Path(self.code1_dir).resolve())
            return True
        except Exception:
            return False

    def _prepare_code1_imports(self) -> None:
        if self.code1_dir in sys.path:
            sys.path.remove(self.code1_dir)
        sys.path.insert(0, self.code1_dir)
        for name in ("config", "models", "utils"):
            if not self._module_belongs_to_code1(name):
                sys.modules.pop(name, None)
        if "stl" not in sys.modules:
            stl_stub = types.ModuleType("stl")
            stl_stub.mesh = types.SimpleNamespace()
            sys.modules["stl"] = stl_stub
        importlib.invalidate_caches()

    def _import_code1_modules(self):
        self._prepare_code1_imports()
        config_mod = importlib.import_module("config")
        models_mod = importlib.import_module("models")
        utils_mod = importlib.import_module("utils")
        if not hasattr(utils_mod, "estimate_cb_from_half_hull"):
            raise AttributeError("no estimate_cb_from_half_hull")
        return config_mod, models_mod, utils_mod

    def _load_model(self):
        model_cfg = self.config_mod.ModelConfig()
        model = self.models_mod.TwoStageCVAE(
            in_channels=model_cfg.in_channels,
            latent_dim=model_cfg.latent_dim,
            cond_dim=model_cfg.cond_dim,
            hidden_dim=model_cfg.hidden_dim,
            base_channels=model_cfg.base_channels,
            use_structured_output=model_cfg.use_structured_output,
        )
        ckpt = torch.load(self.checkpoint_path, map_location=self.device)
        state = ckpt.get("model_state_dict", ckpt)
        model.load_state_dict(state, strict=True)
        model.to(self.device)
        model.eval()
        if int(getattr(model, "latent_dim", 3)) != 3:
            raise ValueError(f"need latent_dim=3，but latent_dim={model.latent_dim}")
        return model

    def _build_half_points_with_code1_logic(self, surface: np.ndarray, L: float, B: float, D: float) -> np.ndarray:

        _, h, w = surface.shape
        Z = np.linspace(0.0, 1.0, h, dtype=np.float32).reshape(h, 1).repeat(w, axis=1).ravel()
        if hasattr(self.code1_utils, "build_points"):
            points = self.code1_utils.build_points(surface, Z, float(L), float(B) / 2.0, float(D))
            points = np.asarray(points, dtype=np.float32)
            if points.ndim == 2 and points.shape[1] == 3 and np.isfinite(points).all():
                return points
        return build_points_from_surface(surface, L=L, B=B, D=D)

    @torch.no_grad()
    def _estimate_cb_with_code1(self, surface_tensor: torch.Tensor, draft_ratio: float) -> float:
        x_channel = surface_tensor[:, 0, :, :]
        y_channel = surface_tensor[:, 1, :, :]
        draft = torch.full((surface_tensor.shape[0],), float(draft_ratio), dtype=torch.float32, device=self.device)
        cb_tensor = self.code1_utils.estimate_cb_from_half_hull(x_channel, y_channel, draft)
        return float(cb_tensor.detach().float().cpu().reshape(-1)[0].item())

    @torch.no_grad()
    def estimate_cb(self, surface: np.ndarray, draft_ratio: float) -> float:
        surface_tensor = torch.from_numpy(np.asarray(surface, dtype=np.float32)).unsqueeze(0).to(self.device)
        return self._estimate_cb_with_code1(surface_tensor, draft_ratio=float(draft_ratio))

    @torch.no_grad()
    def generate(self, L: float, B: float, D: float, T: float, CB: float, z: np.ndarray) -> GenerationResult:
        z_arr = np.asarray(z, dtype=np.float32).reshape(1, 3)
        draft_ratio = float(T) / max(float(D), 1e-8)
        cond = torch.tensor([[draft_ratio, float(CB)]], dtype=torch.float32, device=self.device)
        z_tensor = torch.from_numpy(z_arr).to(self.device)
        surface_tensor = self.model.decode(z_tensor, cond)
        cb_estimated = self._estimate_cb_with_code1(surface_tensor, draft_ratio=draft_ratio)
        surface = surface_tensor.squeeze(0).detach().float().cpu().numpy().astype(np.float32)
        points = self._build_half_points_with_code1_logic(surface, L=L, B=B, D=D)
        volume_actual, volume_target, volume_error = volume_error_from_cb(L=L, B=B, T=T, CB=CB, cb_estimated=cb_estimated)
        return GenerationResult(
            points=points,
            surface=surface,
            cb_estimated=cb_estimated,
            volume_actual=volume_actual,
            volume_target=volume_target,
            volume_error=volume_error,
            is_half_hull=True,
        )
    def Point2Hull(self, LOA, Dd,T, Bd,CB,z0, num_wl, num_x, Outfilepath, Outfilename, Outfilename1, OutPCfilename):
        z_arr = np.asarray(z0, dtype=np.float32).reshape(1, 3)
        draft_ratio = float(T) / max(float(Dd), 1e-8)
        cond = torch.tensor([[draft_ratio, float(CB)]], dtype=torch.float32, device=self.device)
        z_tensor = torch.from_numpy(z_arr).to(self.device)
        PC1 = self.model.decode(z_tensor, cond)
        PC=PC1[0]

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

        Z_flat = Z.flatten().reshape(-1, 1)

        PC = np.hstack([PC, Z_flat])

        Min_Pos = min(PC[:, 0])
        PC[:, 0] = PC[:, 0] - (Min_Pos)
        PC[:, 0] = PC[:, 0] / max(PC[:, 0])
        PC[:, 1] = PC[:, 1] / max(PC[:, 1])
        PC[:, 2] = PC[:, 2] / max(PC[:, 2])

        PC[:, 0] = PC[:, 0] * LOA
        PC[:, 1] = PC[:, 1] * Bd
        PC[:, 2] = PC[:, 2] * Dd

        csv_path = Outfilepath+OutPCfilename

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

        os.makedirs(Outfilepath, exist_ok=True)
        Deck.save(Outfilepath + Outfilename1)
        return 1


class MockHullGenerator:
    def __init__(self, *args, **kwargs):
        # mock 模式只使用 CPU。 
        self.device = torch.device("cpu")

    def estimate_cb(self, surface: np.ndarray, draft_ratio: float) -> float:
        return estimate_cb_with_code1(surface, draft_ratio=float(draft_ratio), estimator=estimate_cb_from_half_hull_fallback, device=self.device)

    def generate(self, L: float, B: float, D: float, T: float, CB: float, z: np.ndarray) -> GenerationResult:
        z = np.asarray(z, dtype=np.float32).reshape(3)
        h, w = 20, 50
        x = np.linspace(0.0, 1.0, w, dtype=np.float32)
        zz = np.linspace(0.0, 1.0, h, dtype=np.float32)

        X = np.tile(x.reshape(1, w), (h, 1))
        Z = np.tile(zz.reshape(h, 1), (1, w))
        fullness = np.clip(float(CB) + 0.04 * np.tanh(float(z[0])), 0.35, 0.85)
        bow_exp = np.clip(0.75 + 0.08 * float(z[1]), 0.45, 1.20)
        vertical_exp = np.clip(0.60 + 0.08 * float(z[2]), 0.35, 1.30)
        length_base = np.clip(np.sin(np.pi * np.clip(X, 0.0, 1.0)), 0.0, 1.0)
        vertical_base = np.clip(np.sin(0.5 * np.pi * np.clip(Z, 0.0, 1.0)), 0.0, 1.0)
        length_shape = length_base ** bow_exp
        vertical_shape = vertical_base ** vertical_exp
        Y = np.clip(fullness * length_shape * vertical_shape, 0.0, 1.0).astype(np.float32)
        Y[0, :] = 0.0
        Y[:, 0] = 0.0
        surface = np.stack([X.astype(np.float32), Y], axis=0)
        points = build_points_from_surface(surface, L=L, B=B, D=D)
        draft_ratio = float(T) / max(float(D), 1e-8)
        cb_estimated = self.estimate_cb(surface, draft_ratio=draft_ratio)
        volume_actual, volume_target, volume_error = volume_error_from_cb(L=L, B=B, T=T, CB=CB, cb_estimated=cb_estimated)
        return GenerationResult(
            points=points,
            surface=surface,
            cb_estimated=cb_estimated,
            volume_actual=volume_actual,
            volume_target=volume_target,
            volume_error=volume_error,
            is_half_hull=True,
        )


class ResistancePredictor:
    TARGET_NAME = {
        0: "friction_resistance",
        1: "total_resistance",
        2: "wave_resistance",
    }

    def __init__(self, code2_dir: str, checkpoint_path: str, device: str = "auto", train_file: Optional[str] = None, amp: bool = False):
        self.code2_dir = str(Path(code2_dir).resolve())
        self.checkpoint_path = str(Path(checkpoint_path).resolve())
        self.device = resolve_device(device)
        self.amp = bool(amp)
        self.train_file = train_file or self._default_train_file()
        self.train_mod = self._load_train_module()
        self.checkpoint = self._torch_load(self.checkpoint_path)
        self.model = self._build_model_from_checkpoint()
        self.condition_normalizer, self.target_normalizer = self._load_normalizers()
        self.saved_args = self.checkpoint.get("args", {})
        self.num_points = int(self.saved_args.get("num_points", 1000))
        self.target_column = int(self.saved_args.get("target_column", 1))

    def _default_train_file(self) -> str:
        candidates = [
            Path(self.code2_dir) / "Hull2Hydro.py",
        ]
        for p in candidates:
            if p.exists():
                return str(p)
        raise FileNotFoundError(f"In {self.code2_dir} no Hull2Hydro.py")

    def _load_train_module(self):

        path = str(Path(self.train_file).resolve())
        spec = importlib.util.spec_from_file_location("train_prediction", path)
        if spec is None or spec.loader is None:
            raise ImportError(f"cant load：{path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules["train_prediction"] = module
        spec.loader.exec_module(module)
        return module

    def _torch_load(self, path: str) -> Dict[str, Any]:
        try:
            return torch.load(path, map_location=self.device, weights_only=False)
        except TypeError:
            return torch.load(path, map_location=self.device)

    def _get_arg(self, name: str, default):
        return self.checkpoint.get("args", {}).get(name, default)

    def _build_model_from_checkpoint(self):
        m = self.train_mod
        if "serialization_orders" in self.checkpoint:
            serialization_orders = tuple(self.checkpoint["serialization_orders"])
        else:
            order_str = self._get_arg("serialization_orders", ",".join(m.DEFAULT_HULL_PRIOR_SERIALIZATION_ORDERS))
            serialization_orders = m.parse_serialization_orders(order_str)
        model = m.HullResistancePredictor(
            num_points=int(self._get_arg("num_points", 1000)),
            patch_size=int(self._get_arg("patch_size", 20)),
            embed_dim=int(self._get_arg("embed_dim", 256)),
            encoder_depth=int(self._get_arg("encoder_depth", 6)),
            num_heads=int(self._get_arg("num_heads", 8)),
            window_size=int(self._get_arg("window_size", 16)),
            dropout=float(self._get_arg("dropout", 0.0)),
            serialization_orders=serialization_orders,
            shuffle_orders=False,
            serialization_bits=int(self._get_arg("serialization_bits", 10)),
            xcpe_kernel_size=int(self._get_arg("xcpe_kernel_size", 3)),
            xcpe_grid_size=self._get_arg("xcpe_grid_size", None),
            xcpe_grid_bits=int(self._get_arg("xcpe_grid_bits", 10)),
            xcpe_sparse_padding=int(self._get_arg("xcpe_sparse_padding", 96)),
            condition_dim=2,
            condition_embed_dim=int(self._get_arg("condition_embed_dim", 64)),
        ).to(self.device)
        if "model" not in self.checkpoint:
            raise KeyError("Code 2 checkpoint no mode, need prediction_best.pth 或 prediction_last.pth")
        model.load_state_dict(self.checkpoint["model"], strict=True)
        model.eval()
        return model

    def _load_normalizers(self) -> Tuple[Any, Any]:
        if self.checkpoint.get("target_transform", None) != "log1p":
            raise ValueError(" Code 2 checkpoint'")
        if "condition_normalizer" not in self.checkpoint:
            raise KeyError("Code 2 checkpoint need condition_normalizer")
        if "target_normalizer" not in self.checkpoint:
            raise KeyError("Code 2 checkpoint need target_normalizer")
        return (
            self.train_mod.ZScoreNormalizer.from_state_dict(self.checkpoint["condition_normalizer"]),
            self.train_mod.ZScoreNormalizer.from_state_dict(self.checkpoint["target_normalizer"]),
        )

    @torch.no_grad()
    def predict(self, points: np.ndarray, draft_ratio: float, speed: float) -> float:
        pts = ensure_num_points(points, self.num_points)
        condition_raw = np.asarray([float(draft_ratio), float(speed)], dtype=np.float32)
        condition_norm = self.condition_normalizer.transform(condition_raw)
        points_t = torch.from_numpy(pts).float().unsqueeze(0).to(self.device)
        cond_t = torch.from_numpy(condition_norm).float().unsqueeze(0).to(self.device)
        amp_enabled = bool(self.amp and self.device.type == "cuda")
        with self.train_mod.autocast_context(self.device, amp_enabled):
            pred_norm = self.model(points_t, cond_t)
        pred_norm_np = pred_norm.detach().float().cpu().numpy()
        pred_log_np = self.target_normalizer.inverse_transform(pred_norm_np)
        pred_raw_np = self.train_mod.log1p_target_to_raw_target(pred_log_np)
        return float(pred_raw_np.reshape(-1)[0])


class MockResistancePredictor:

    def __init__(self, *args, **kwargs):

        self.num_points = 1000

        self.target_column = 1

    def predict(self, points: np.ndarray, draft_ratio: float, speed: float) -> float:

        pts = np.asarray(points, dtype=np.float32)

        x_span = float(np.ptp(pts[:, 0]) + 1e-6)

        y_span = float(np.ptp(pts[:, 1]) + 1e-6)

        z_span = float(np.ptp(pts[:, 2]) + 1e-6)

        slenderness = x_span / max(2.0 * y_span, 1e-6)

        mean_half_breadth = float(np.mean(np.abs(pts[:, 1])))

        fullness = mean_half_breadth / max(y_span, 1e-6)

        friction = 0.04 * float(speed) ** 2 * (x_span * z_span + x_span * y_span)

        wave = 0.03 * float(speed) ** 4 * (1.0 + 1.5 * fullness) / max(slenderness, 0.5)

        draft_term = 1.0 + 0.25 * abs(float(draft_ratio) - 0.65)

        return float(max((friction + wave) * draft_term, 1e-8))


def make_generator(code1_dir: str, generator_ckpt: str, device: str = "auto", mock: bool = False):

    if mock:
        return MockHullGenerator()
    return HullGenerator(code1_dir=code1_dir, checkpoint_path=generator_ckpt, device=device)


def make_resistance_predictor(code2_dir: str, resistance_ckpt: str, device: str = "auto", mock: bool = False, code2_train_file: Optional[str] = None, amp: bool = False):

    if mock:
        return MockResistancePredictor()
    return ResistancePredictor(code2_dir=code2_dir, checkpoint_path=resistance_ckpt, device=device, train_file=code2_train_file, amp=amp)
