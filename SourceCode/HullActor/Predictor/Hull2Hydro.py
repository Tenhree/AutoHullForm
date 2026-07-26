import argparse
import csv
import glob
import os
import random
import re
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

try:
    import spconv.pytorch as spconv
except ImportError:
    spconv = None



def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def autocast_context(device: torch.device, enabled: bool):
    if enabled and device.type == "cuda":
        return torch.amp.autocast(device_type="cuda", enabled=True)
    return nullcontext()


def sample_id_from_path(path: str, prefix: str) -> int:
    name = os.path.basename(path)
    match = re.search(rf"{re.escape(prefix)}(\d+)\.csv$", name)
    if match is None:
        raise ValueError(f"Cannot parse sample id from file name: {name}")
    return int(match.group(1))


def sample_file_sort_key(path: str, prefix: str) -> Tuple[int, str]:
    try:
        return (sample_id_from_path(path, prefix), os.path.basename(path))
    except ValueError:
        return (10**18, os.path.basename(path))


def hull_id_from_sample_id(sample_id: int, samples_per_hull: int) -> int:
    return (sample_id - 1) // samples_per_hull


def read_point_cloud_csv(file_path: Path, num_points: int) -> np.ndarray:
    points = np.loadtxt(file_path, delimiter=",", dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"{file_path} must have shape [N, 3], but got {points.shape}")
    if points.shape[0] != num_points:
        raise ValueError(f"{file_path} must have {num_points} points, but got {points.shape[0]}")
    return points


def read_condition_csv(file_path: Path) -> np.ndarray:
    condition = np.loadtxt(file_path, delimiter=",", dtype=np.float32)
    condition = np.asarray(condition, dtype=np.float32).reshape(-1)
    if condition.shape[0] != 2:
        raise ValueError(f"{file_path} must contain 2 values: draft_ratio, speed. Got shape {condition.shape}")
    return condition


def read_hydro_csv(file_path: Path) -> np.ndarray:
    hydro = np.loadtxt(file_path, delimiter=",", dtype=np.float32)
    hydro = np.asarray(hydro, dtype=np.float32).reshape(-1)
    if hydro.shape[0] != 3:
        raise ValueError(f"{file_path} must contain 3 values: friction_resistance, total_resistance, wave_resistance. Got shape {hydro.shape}")
    return hydro


class ZScoreNormalizer:
    def __init__(self, mean: np.ndarray, std: np.ndarray, eps: float = 1e-8):
        self.mean = np.asarray(mean, dtype=np.float32)
        self.std = np.asarray(std, dtype=np.float32)
        self.std = np.where(self.std < eps, 1.0, self.std).astype(np.float32)

    @classmethod
    def fit(cls, values: np.ndarray) -> "ZScoreNormalizer":
        values = np.asarray(values, dtype=np.float32)
        return cls(mean=values.mean(axis=0), std=values.std(axis=0))

    def transform(self, values: np.ndarray) -> np.ndarray:
        return (np.asarray(values, dtype=np.float32) - self.mean) / self.std

    def inverse_transform(self, values: np.ndarray) -> np.ndarray:
        return np.asarray(values, dtype=np.float32) * self.std + self.mean

    def state_dict(self) -> Dict[str, np.ndarray]:
        return {"mean": self.mean, "std": self.std}

    @classmethod
    def from_state_dict(cls, state: Dict[str, np.ndarray]) -> "ZScoreNormalizer":
        return cls(mean=np.asarray(state["mean"], dtype=np.float32), std=np.asarray(state["std"], dtype=np.float32))



def raw_target_to_log1p_target(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if np.any(values < 0.0):
        raise ValueError(
            "Resistance target contains negative values, but log1p target transform "
            "requires non-negative physical targets."
        )
    return np.log1p(values).astype(np.float32)


def log1p_target_to_raw_target(values: np.ndarray) -> np.ndarray:
    raw = np.expm1(np.asarray(values, dtype=np.float32))
    return np.maximum(raw, 0.0).astype(np.float32)


class PredictionDataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        sample_ids: Sequence[int],
        condition_normalizer: ZScoreNormalizer,
        target_normalizer: ZScoreNormalizer,
        num_points: int = 1000,
        samples_per_hull: int = 21,
        target_column: int = 1,
        preload_shapes: bool = True,
    ):
        self.root_dir = Path(root_dir)
        self.condition_dir = self.root_dir / "Conditiondata"
        self.hydro_dir = self.root_dir / "Hydrodata"
        self.shape_dir = self.root_dir / "Shapedata"
        self.sample_ids = list(sample_ids)
        self.condition_normalizer = condition_normalizer
        self.target_normalizer = target_normalizer
        self.num_points = num_points
        self.samples_per_hull = samples_per_hull
        self.target_column = target_column
        self.preload_shapes = preload_shapes
        self.shape_cache: Dict[int, torch.Tensor] = {}

        if not self.condition_dir.exists():
            raise FileNotFoundError(f"Conditiondata directory not found: {self.condition_dir}")
        if not self.hydro_dir.exists():
            raise FileNotFoundError(f"Hydrodata directory not found: {self.hydro_dir}")
        if not self.shape_dir.exists():
            raise FileNotFoundError(f"Shapedata directory not found: {self.shape_dir}")

        if self.preload_shapes:
            self._preload_unique_shapes()

    def __len__(self) -> int:
        return len(self.sample_ids)

    def _condition_path(self, sample_id: int) -> Path:
        return self.condition_dir / f"CDSample{sample_id}.csv"

    def _hydro_path(self, sample_id: int) -> Path:
        return self.hydro_dir / f"HydroSample{sample_id}.csv"

    def _shape_path(self, sample_id: int) -> Path:
        return self.shape_dir / f"PCSample{sample_id}.csv"

    def _load_shape_for_sample(self, sample_id: int) -> torch.Tensor:
        points = read_point_cloud_csv(self._shape_path(sample_id), num_points=self.num_points)
        return torch.from_numpy(points)

    def _preload_unique_shapes(self) -> None:
        hull_to_first_sample: Dict[int, int] = {}
        for sample_id in self.sample_ids:
            hull_id = hull_id_from_sample_id(sample_id, self.samples_per_hull)
            hull_to_first_sample.setdefault(hull_id, sample_id)

        for hull_id, sample_id in hull_to_first_sample.items():
            self.shape_cache[hull_id] = self._load_shape_for_sample(sample_id)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        sample_id = self.sample_ids[index]
        hull_id = hull_id_from_sample_id(sample_id, self.samples_per_hull)

        if hull_id in self.shape_cache:
            points = self.shape_cache[hull_id]
        else:
            points = self._load_shape_for_sample(sample_id)
            if self.preload_shapes:
                self.shape_cache[hull_id] = points

        condition_raw = read_condition_csv(self._condition_path(sample_id))
        hydro = read_hydro_csv(self._hydro_path(sample_id))
        target_raw = np.asarray([hydro[self.target_column]], dtype=np.float32)
        target_log = raw_target_to_log1p_target(target_raw)

        condition_norm = self.condition_normalizer.transform(condition_raw)
        target_norm = self.target_normalizer.transform(target_log)

        return {
            "points": points.float(),
            "condition": torch.from_numpy(condition_norm).float(),
            "target": torch.from_numpy(target_norm).float(),
            "target_raw": torch.from_numpy(target_raw).float(),
            "condition_raw": torch.from_numpy(condition_raw).float(),
            "sample_id": torch.tensor(sample_id, dtype=torch.long),
            "hull_id": torch.tensor(hull_id, dtype=torch.long),
        }


def find_available_sample_ids(root_dir: str) -> List[int]:
    root = Path(root_dir)
    condition_dir = root / "Conditiondata"
    hydro_dir = root / "Hydrodata"
    shape_dir = root / "Shapedata"

    condition_files = glob.glob(str(condition_dir / "CDSample*.csv"))
    hydro_files = glob.glob(str(hydro_dir / "HydroSample*.csv"))
    shape_files = glob.glob(str(shape_dir / "PCSample*.csv"))

    condition_ids = {sample_id_from_path(path, "CDSample") for path in condition_files}
    hydro_ids = {sample_id_from_path(path, "HydroSample") for path in hydro_files}
    shape_ids = {sample_id_from_path(path, "PCSample") for path in shape_files}

    common_ids = sorted(condition_ids & hydro_ids & shape_ids)
    missing_from_hydro = sorted((condition_ids | shape_ids) - hydro_ids)
    missing_from_condition = sorted((hydro_ids | shape_ids) - condition_ids)
    missing_from_shape = sorted((condition_ids | hydro_ids) - shape_ids)

    if len(missing_from_hydro) > 0 or len(missing_from_condition) > 0 or len(missing_from_shape) > 0:
        print("warning: sample id mismatch across folders")
        print(f"missing_from_hydro_count={len(missing_from_hydro)}")
        print(f"missing_from_condition_count={len(missing_from_condition)}")
        print(f"missing_from_shape_count={len(missing_from_shape)}")

    if len(common_ids) == 0:
        raise FileNotFoundError(f"No matched samples found in {root_dir}")
    return common_ids


def check_hull_groups(sample_ids: Sequence[int], samples_per_hull: int, strict: bool) -> Dict[int, List[int]]:
    hull_to_samples: Dict[int, List[int]] = {}
    for sample_id in sample_ids:
        hull_id = hull_id_from_sample_id(sample_id, samples_per_hull)
        hull_to_samples.setdefault(hull_id, []).append(sample_id)

    for hull_id in hull_to_samples:
        hull_to_samples[hull_id] = sorted(hull_to_samples[hull_id])

    bad_groups = {hull_id: ids for hull_id, ids in hull_to_samples.items() if len(ids) != samples_per_hull}
    if strict and len(bad_groups) > 0:
        preview = list(bad_groups.items())[:5]
        raise ValueError(
            f"Found {len(bad_groups)} incomplete hull groups. Each hull must have {samples_per_hull} samples. "
            f"Preview: {preview}"
        )
    if len(bad_groups) > 0:
        print(f"warning: found {len(bad_groups)} incomplete hull groups; strict_group_check=False, continuing")

    return hull_to_samples


def split_train_test_by_hull(
    sample_ids: Sequence[int],
    samples_per_hull: int,
    train_ratio: float,
    seed: int,
    strict_group_check: bool,
) -> Tuple[List[int], List[int], List[int], List[int]]:
    if train_ratio <= 0 or train_ratio >= 1.0:
        raise ValueError("Need train_ratio in (0, 1). The remaining hulls are used as the test set.")

    hull_to_samples = check_hull_groups(sample_ids, samples_per_hull=samples_per_hull, strict=strict_group_check)
    hull_ids = sorted(hull_to_samples.keys())

    rng = random.Random(seed)
    rng.shuffle(hull_ids)

    num_hulls = len(hull_ids)
    if num_hulls < 2:
        raise ValueError("Need at least 2 hulls to split into train and test sets")

    num_train = max(1, int(num_hulls * train_ratio))
    if num_train >= num_hulls:
        num_train = num_hulls - 1

    train_hulls = sorted(hull_ids[:num_train])
    test_hulls = sorted(hull_ids[num_train:])
    train_ids = sorted([sid for hid in train_hulls for sid in hull_to_samples[hid]])
    test_ids = sorted([sid for hid in test_hulls for sid in hull_to_samples[hid]])
    return train_ids, test_ids, train_hulls, test_hulls


def sample_ids_from_hulls(sample_ids: Sequence[int], samples_per_hull: int, hulls: Sequence[int]) -> List[int]:
    hull_set = {int(hull_id) for hull_id in hulls}
    selected_ids = [
        int(sample_id)
        for sample_id in sample_ids
        if hull_id_from_sample_id(int(sample_id), samples_per_hull) in hull_set
    ]
    return sorted(selected_ids)


def fit_normalizers(root_dir: str, sample_ids: Sequence[int], target_column: int) -> Tuple[ZScoreNormalizer, ZScoreNormalizer]:
    root = Path(root_dir)
    condition_values = []
    target_values = []
    for sample_id in sample_ids:
        condition = read_condition_csv(root / "Conditiondata" / f"CDSample{sample_id}.csv")
        hydro = read_hydro_csv(root / "Hydrodata" / f"HydroSample{sample_id}.csv")
        target_raw = np.asarray([hydro[target_column]], dtype=np.float32)
        target_log = raw_target_to_log1p_target(target_raw)

        condition_values.append(condition)
        target_values.append(target_log)

    condition_values = np.asarray(condition_values, dtype=np.float32)
    target_values = np.asarray(target_values, dtype=np.float32)
    return ZScoreNormalizer.fit(condition_values), ZScoreNormalizer.fit(target_values)



SPACE_FILLING_SERIALIZATION_ORDERS = ("z", "z-trans", "hilbert", "hilbert-trans")

HULL_PRIOR_SERIALIZATION_ORDER_SPECS: Dict[str, Tuple[Tuple[str, str], ...]] = {

    "deck2bottom_bow2stern": (("z", "desc"), ("x", "asc"), ("y", "asc")),

    "deck2bottom_stern2bow": (("z", "desc"), ("x", "desc"), ("y", "asc")),

    "bottom2deck_bow2stern": (("z", "asc"), ("x", "asc"), ("y", "asc")),

    "bottom2deck_stern2bow": (("z", "asc"), ("x", "desc"), ("y", "asc")),

    "bow2stern_deck2bottom": (("x", "asc"), ("z", "desc"), ("y", "asc")),

    "stern2bow_deck2bottom": (("x", "desc"), ("z", "desc"), ("y", "asc")),

    "bow2stern_bottom2deck": (("x", "asc"), ("z", "asc"), ("y", "asc")),

    "stern2bow_bottom2deck": (("x", "desc"), ("z", "asc"), ("y", "asc")),
}

DEFAULT_HULL_PRIOR_SERIALIZATION_ORDERS = tuple(HULL_PRIOR_SERIALIZATION_ORDER_SPECS.keys())

HULL_PRIOR_SERIALIZATION_ALIASES: Dict[str, str] = {
    "hull_order1": "deck2bottom_bow2stern",
    "hull_order2": "deck2bottom_stern2bow",
    "hull_order3": "bottom2deck_bow2stern",
    "hull_order4": "bottom2deck_stern2bow",
    "hull_order5": "bow2stern_deck2bottom",
    "hull_order6": "stern2bow_deck2bottom",
    "hull_order7": "bow2stern_bottom2deck",
    "hull_order8": "stern2bow_bottom2deck",
}

VALID_SERIALIZATION_ORDERS = SPACE_FILLING_SERIALIZATION_ORDERS + DEFAULT_HULL_PRIOR_SERIALIZATION_ORDERS
SUPPORTED_SERIALIZATION_ORDERS = VALID_SERIALIZATION_ORDERS + tuple(HULL_PRIOR_SERIALIZATION_ALIASES.keys())


def parse_serialization_orders(order_string: str) -> Tuple[str, ...]:
    raw_orders = tuple(item.strip() for item in order_string.split(",") if item.strip())
    if len(raw_orders) == 0:
        raise ValueError("serialization_orders cannot be empty")

    orders: List[str] = []
    for raw_order in raw_orders:
        order = HULL_PRIOR_SERIALIZATION_ALIASES.get(raw_order, raw_order)
        if order not in VALID_SERIALIZATION_ORDERS:
            raise ValueError(f"Unsupported serialization order: {raw_order}. Supported: {SUPPORTED_SERIALIZATION_ORDERS}")
        orders.append(order)
    return tuple(orders)


def maybe_shuffle_orders(order_names: Sequence[str], shuffle_orders: bool, training: bool) -> Tuple[str, ...]:
    order_names = list(order_names)
    if shuffle_orders and training and len(order_names) > 1:
        random.shuffle(order_names)
    return tuple(order_names)


def quantize_coordinates(coords: torch.Tensor, bits: int = 10, eps: float = 1e-6) -> torch.Tensor:
    if coords.ndim != 3 or coords.shape[-1] != 3:
        raise ValueError(f"Expected coords with shape [B, N, 3], but got {coords.shape}")
    if bits <= 0 or bits > 16:
        raise ValueError("serialization bits must be in [1, 16]")
    mins = coords.amin(dim=1, keepdim=True)
    maxs = coords.amax(dim=1, keepdim=True)
    normalized = (coords - mins) / (maxs - mins).clamp_min(eps)
    grid_max = 2**bits - 1
    grid_coord = (normalized * grid_max).clamp(0, grid_max).long()
    return grid_coord


def z_order_encode_grid(grid_coord: torch.Tensor, bits: int = 10, trans: bool = False) -> torch.Tensor:
    if trans:
        grid_coord = grid_coord[..., [1, 0, 2]]
    x = grid_coord[..., 0].long()
    y = grid_coord[..., 1].long()
    z = grid_coord[..., 2].long()
    code = torch.zeros_like(x, dtype=torch.long)
    for bit in range(bits):
        mask = 1 << bit
        code = code | ((x & mask) << (2 * bit + 2))
        code = code | ((y & mask) << (2 * bit + 1))
        code = code | ((z & mask) << (2 * bit + 0))
    return code


def right_shift(binary: torch.Tensor, k: int = 1) -> torch.Tensor:
    if binary.shape[-1] <= k:
        return torch.zeros_like(binary)
    shifted = F.pad(binary[..., :-k], (k, 0), mode="constant", value=0)
    return shifted


def gray2binary(gray: torch.Tensor) -> torch.Tensor:
    shift = 1 << (gray.shape[-1].bit_length() - 1)
    while shift > 0:
        gray = torch.logical_xor(gray, right_shift(gray, shift))
        shift = shift // 2
    return gray


def hilbert_encode_grid(grid_coord: torch.Tensor, bits: int = 10, trans: bool = False) -> torch.Tensor:
    if trans:
        grid_coord = grid_coord[..., [1, 0, 2]]
    if bits * 3 > 63:
        raise ValueError("Hilbert code needs bits * 3 <= 63")

    original_shape = grid_coord.shape[:-1]
    locs = grid_coord.reshape(-1, 3).long()
    bitpack_mask = 1 << torch.arange(0, 8, device=locs.device, dtype=torch.int64)
    bitpack_mask_rev = bitpack_mask.flip(-1)

    locs_uint8 = locs.view(torch.uint8).reshape((-1, 3, 8)).flip(-1)
    gray = (
        locs_uint8.unsqueeze(-1)
        .bitwise_and(bitpack_mask_rev)
        .ne(0)
        .byte()
        .flatten(-2, -1)[..., -bits:]
    )

    for bit in range(0, bits):
        for dim in range(0, 3):
            mask = gray[:, dim, bit]
            gray[:, 0, bit + 1 :] = torch.logical_xor(gray[:, 0, bit + 1 :], mask[:, None])
            to_flip = torch.logical_and(
                torch.logical_not(mask[:, None]).repeat(1, gray.shape[2] - bit - 1),
                torch.logical_xor(gray[:, 0, bit + 1 :], gray[:, dim, bit + 1 :]),
            )
            gray[:, dim, bit + 1 :] = torch.logical_xor(gray[:, dim, bit + 1 :], to_flip)
            gray[:, 0, bit + 1 :] = torch.logical_xor(gray[:, 0, bit + 1 :], to_flip)

    gray = gray.swapaxes(1, 2).reshape((-1, bits * 3))
    hh_bin = gray2binary(gray)
    padded = F.pad(hh_bin, (64 - bits * 3, 0), "constant", 0)
    hh_uint8 = (padded.flip(-1).reshape((-1, 8, 8)) * bitpack_mask).sum(2).squeeze().type(torch.uint8)
    hh_code = hh_uint8.view(torch.int64).reshape(-1)
    return hh_code.reshape(original_shape)


def hull_prior_encode_grid(grid_coord: torch.Tensor, order: str, bits: int = 10) -> torch.Tensor:

    order = HULL_PRIOR_SERIALIZATION_ALIASES.get(order, order)
    if order not in HULL_PRIOR_SERIALIZATION_ORDER_SPECS:
        raise ValueError(f"Unsupported hull-prior serialization order: {order}")

    axis_to_index = {"x": 0, "y": 1, "z": 2}
    base = 2**bits
    grid_max = base - 1
    code = torch.zeros(grid_coord.shape[:-1], dtype=torch.long, device=grid_coord.device)

    for axis_name, direction in HULL_PRIOR_SERIALIZATION_ORDER_SPECS[order]:
        value = grid_coord[..., axis_to_index[axis_name]].long()
        if direction == "desc":
            value = grid_max - value
        elif direction != "asc":
            raise ValueError(f"Unsupported direction {direction} in order {order}")
        code = code * base + value
    return code


def serialization_codes(coords: torch.Tensor, order: str, bits: int = 10) -> torch.Tensor:
    order = HULL_PRIOR_SERIALIZATION_ALIASES.get(order, order)
    grid_coord = quantize_coordinates(coords, bits=bits)
    if order == "z":
        return z_order_encode_grid(grid_coord, bits=bits, trans=False)
    if order == "z-trans":
        return z_order_encode_grid(grid_coord, bits=bits, trans=True)
    if order == "hilbert":
        return hilbert_encode_grid(grid_coord, bits=bits, trans=False)
    if order == "hilbert-trans":
        return hilbert_encode_grid(grid_coord, bits=bits, trans=True)
    if order in HULL_PRIOR_SERIALIZATION_ORDER_SPECS:
        return hull_prior_encode_grid(grid_coord, order=order, bits=bits)
    raise ValueError(f"Unsupported serialization order: {order}")


def compute_serialized_orders(coords: torch.Tensor, order_names: Sequence[str], bits: int = 10) -> Dict[str, torch.Tensor]:
    code_list: List[torch.Tensor] = []
    for order in order_names:
        code_list.append(serialization_codes(coords, order=order, bits=bits))
    codes = torch.stack(code_list, dim=0)
    orders = torch.argsort(codes, dim=2)
    base = torch.arange(coords.shape[1], device=coords.device).view(1, 1, -1).expand_as(orders)
    inverses = torch.empty_like(orders)
    inverses.scatter_(dim=2, index=orders, src=base)
    return {"codes": codes, "orders": orders, "inverses": inverses}


def batched_gather(x: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    expand_shape = list(index.shape) + [x.shape[-1]]
    index_expand = index.unsqueeze(-1).expand(*expand_shape)
    return torch.gather(x, dim=1, index=index_expand)


def make_serialized_patches(points: torch.Tensor, patch_size: int, order_name: str, bits: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch_size, num_points, coord_dim = points.shape
    if coord_dim != 3:
        raise ValueError(f"Expected points with shape [B, N, 3], but got {points.shape}")
    if num_points % patch_size != 0:
        raise ValueError(f"num_points={num_points} must be divisible by patch_size={patch_size}")

    order_pack = compute_serialized_orders(points, order_names=(order_name,), bits=bits)
    sort_index = order_pack["orders"][0]
    sorted_points = batched_gather(points, sort_index)

    num_patches = num_points // patch_size
    patches = sorted_points.view(batch_size, num_patches, patch_size, 3)
    centers = patches.mean(dim=2)
    relative_patches = patches - centers.unsqueeze(2)
    return relative_patches, centers, sort_index


class MLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class OfficialSparseXCPE(nn.Module):

    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        grid_size: Optional[float] = None,
        grid_bits: int = 10,
        sparse_padding: int = 96,
        indice_key: str = "xcpe",
    ):
        super().__init__()
        if spconv is None:
            raise ImportError(
                "OfficialSparseXCPE requires spconv. Install a CUDA-matched package, "
                "for example: pip install spconv-cu120, spconv-cu118, or spconv-cu117."
            )
        if kernel_size % 2 == 0:
            raise ValueError("xcpe_kernel_size must be odd")
        if grid_bits <= 0 or grid_bits > 16:
            raise ValueError("xcpe_grid_bits must be in [1, 16]")

        self.channels = channels
        self.grid_size = grid_size
        self.grid_bits = grid_bits
        self.sparse_padding = sparse_padding

        self.subm_conv = spconv.SubMConv3d(
            channels,
            channels,
            kernel_size=kernel_size,
            bias=True,
            indice_key=indice_key,
        )
        self.linear = nn.Linear(channels, channels)
        self.norm = nn.LayerNorm(channels)

    def _make_grid_coord(self, centers: torch.Tensor) -> torch.Tensor:
        if self.grid_size is None:
            return quantize_coordinates(centers, bits=self.grid_bits).int()

        mins = centers.amin(dim=1, keepdim=True)
        shifted = centers - mins
        grid_coord = torch.div(shifted, self.grid_size, rounding_mode="trunc")
        grid_coord = grid_coord.clamp_min(0).int()
        return grid_coord

    def forward(self, tokens: torch.Tensor, centers: torch.Tensor) -> torch.Tensor:
        if tokens.ndim != 3 or centers.ndim != 3:
            raise ValueError(f"Expected tokens [B, P, C] and centers [B, P, 3], got {tokens.shape} and {centers.shape}")
        if tokens.shape[:2] != centers.shape[:2] or centers.shape[-1] != 3:
            raise ValueError(f"Token and center shapes are incompatible: {tokens.shape}, {centers.shape}")

        batch_size, num_tokens, channels = tokens.shape
        if channels != self.channels:
            raise ValueError(f"Expected token channels {self.channels}, but got {channels}")

        shortcut = tokens
        grid_coord = self._make_grid_coord(centers)

        feat = tokens.reshape(batch_size * num_tokens, channels).contiguous()
        batch = torch.arange(batch_size, device=tokens.device, dtype=torch.int32)
        batch = batch.view(batch_size, 1).expand(batch_size, num_tokens).reshape(-1)

        indices = torch.cat(
            [
                batch.unsqueeze(1),
                grid_coord.reshape(batch_size * num_tokens, 3).int(),
            ],
            dim=1,
        ).contiguous()

        max_coord = grid_coord.reshape(-1, 3).amax(dim=0)
        spatial_shape = (max_coord + 1 + self.sparse_padding).tolist()

        sparse_tensor = spconv.SparseConvTensor(
            features=feat,
            indices=indices,
            spatial_shape=spatial_shape,
            batch_size=batch_size,
        )
        sparse_tensor = self.subm_conv(sparse_tensor)

        out = sparse_tensor.features
        out = self.linear(out)
        out = self.norm(out)
        out = out.view(batch_size, num_tokens, channels)

        return shortcut + out


class SerializedAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, window_size: int, dropout: float):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}")
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, order: torch.Tensor, inverse: torch.Tensor) -> torch.Tensor:
        batch_size, length, dim = x.shape
        x_ordered = batched_gather(x, order)

        pad_len = (self.window_size - length % self.window_size) % self.window_size
        if pad_len > 0:
            pad = torch.zeros(batch_size, pad_len, dim, device=x.device, dtype=x.dtype)
            x_ordered = torch.cat([x_ordered, pad], dim=1)

        padded_length = x_ordered.shape[1]
        num_windows = padded_length // self.window_size

        padding_mask = torch.zeros(batch_size, padded_length, dtype=torch.bool, device=x.device)
        if pad_len > 0:
            padding_mask[:, -pad_len:] = True

        qkv = self.qkv(x_ordered)
        qkv = qkv.view(batch_size, padded_length, 3, self.num_heads, self.head_dim)
        qkv = qkv.view(batch_size, num_windows, self.window_size, 3, self.num_heads, self.head_dim)
        qkv = qkv.reshape(batch_size * num_windows, self.window_size, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)

        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        attn = (q * self.scale) @ k.transpose(-2, -1)
        if pad_len > 0:
            mask_windows = padding_mask.view(batch_size, num_windows, self.window_size)
            mask_windows = mask_windows.reshape(batch_size * num_windows, self.window_size)
            attn = attn.masked_fill(mask_windows[:, None, None, :], torch.finfo(attn.dtype).min)

        attn = torch.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)
        out = (attn @ v).transpose(1, 2).reshape(batch_size * num_windows, self.window_size, dim)
        out = out.view(batch_size, padded_length, dim)
        out = out[:, :length, :]

        out = self.proj(out)
        out = self.proj_drop(out)
        out = batched_gather(out, inverse)
        return out


class SerializedAttentionBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        order_index: int = 0,
        xcpe_kernel_size: int = 3,
        xcpe_grid_size: Optional[float] = None,
        xcpe_grid_bits: int = 10,
        xcpe_sparse_padding: int = 96,
        cpe_indice_key: Optional[str] = None,
    ):
        super().__init__()
        self.order_index = order_index
        self.xcpe = OfficialSparseXCPE(
            channels=dim,
            kernel_size=xcpe_kernel_size,
            grid_size=xcpe_grid_size,
            grid_bits=xcpe_grid_bits,
            sparse_padding=xcpe_sparse_padding,
            indice_key=cpe_indice_key or f"xcpe_{order_index}",
        )
        self.norm1 = nn.LayerNorm(dim)
        self.attn = SerializedAttention(dim=dim, num_heads=num_heads, window_size=window_size, dropout=dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim=dim, hidden_dim=int(dim * mlp_ratio), dropout=dropout)

    def forward(self, x: torch.Tensor, centers: torch.Tensor, order_pack: Dict[str, torch.Tensor]) -> torch.Tensor:
        order_count = order_pack["orders"].shape[0]
        selected_order_index = self.order_index % order_count
        order = order_pack["orders"][selected_order_index]
        inverse = order_pack["inverses"][selected_order_index]

        x = self.xcpe(x, centers)
        x = x + self.attn(self.norm1(x), order=order, inverse=inverse)
        x = x + self.mlp(self.norm2(x))
        return x

class PatchEmbed(nn.Module):
    def __init__(self, patch_size: int, embed_dim: int):
        super().__init__()
        self.patch_size = patch_size
        self.point_mlp = nn.Sequential(
            nn.Linear(3, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, embed_dim),
        )
        self.center_mlp = nn.Sequential(
            nn.Linear(3, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, relative_patches: torch.Tensor, centers: torch.Tensor) -> torch.Tensor:
        point_features = self.point_mlp(relative_patches)
        patch_features = point_features.max(dim=2).values
        center_features = self.center_mlp(centers)
        return patch_features + center_features


class HullPTv3Backbone(nn.Module):
    def __init__(
        self,
        num_points: int,
        patch_size: int,
        embed_dim: int,
        depth: int,
        num_heads: int,
        window_size: int,
        dropout: float,
        serialization_orders: Sequence[str],
        shuffle_orders: bool,
        serialization_bits: int,
        xcpe_kernel_size: int,
        xcpe_grid_size: Optional[float],
        xcpe_grid_bits: int,
        xcpe_sparse_padding: int,
    ):
        super().__init__()
        if num_points % patch_size != 0:
            raise ValueError(f"num_points={num_points} must be divisible by patch_size={patch_size}")
        self.num_points = num_points
        self.patch_size = patch_size
        self.num_patches = num_points // patch_size
        self.serialization_orders = tuple(serialization_orders)
        self.shuffle_orders = shuffle_orders
        self.serialization_bits = serialization_bits

        self.patch_embed = PatchEmbed(patch_size=patch_size, embed_dim=embed_dim)
        self.blocks = nn.ModuleList(
            [
                SerializedAttentionBlock(
                    dim=embed_dim,
                    num_heads=num_heads,
                    window_size=window_size,
                    dropout=dropout,
                    order_index=i,
                    xcpe_kernel_size=xcpe_kernel_size,
                    xcpe_grid_size=xcpe_grid_size,
                    xcpe_grid_bits=xcpe_grid_bits,
                    xcpe_sparse_padding=xcpe_sparse_padding,
                    cpe_indice_key=f"encoder_xcpe_{i}",
                )
                for i in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        active_order_names = maybe_shuffle_orders(
            self.serialization_orders,
            shuffle_orders=self.shuffle_orders,
            training=self.training,
        )
        patching_order_name = active_order_names[0]
        relative_patches, centers, _ = make_serialized_patches(
            points,
            patch_size=self.patch_size,
            order_name=patching_order_name,
            bits=self.serialization_bits,
        )
        tokens = self.patch_embed(relative_patches, centers)
        order_pack = compute_serialized_orders(centers, order_names=active_order_names, bits=self.serialization_bits)

        for block in self.blocks:
            tokens = block(tokens, centers, order_pack)
        return self.norm(tokens)


class HullResistancePredictor(nn.Module):
    def __init__(
        self,
        num_points: int = 1000,
        patch_size: int = 20,
        embed_dim: int = 256,
        encoder_depth: int = 6,
        num_heads: int = 8,
        window_size: int = 16,
        dropout: float = 0.0,
        serialization_orders: Sequence[str] = DEFAULT_HULL_PRIOR_SERIALIZATION_ORDERS,
        shuffle_orders: bool = True,
        serialization_bits: int = 10,
        xcpe_kernel_size: int = 3,
        xcpe_grid_size: Optional[float] = None,
        xcpe_grid_bits: int = 10,
        xcpe_sparse_padding: int = 96,
        condition_dim: int = 2,
        condition_embed_dim: int = 64,
    ):
        super().__init__()
        self.encoder = HullPTv3Backbone(
            num_points=num_points,
            patch_size=patch_size,
            embed_dim=embed_dim,
            depth=encoder_depth,
            num_heads=num_heads,
            window_size=window_size,
            dropout=dropout,
            serialization_orders=serialization_orders,
            shuffle_orders=shuffle_orders,
            serialization_bits=serialization_bits,
            xcpe_kernel_size=xcpe_kernel_size,
            xcpe_grid_size=xcpe_grid_size,
            xcpe_grid_bits=xcpe_grid_bits,
            xcpe_sparse_padding=xcpe_sparse_padding,
        )

        self.condition_mlp = nn.Sequential(
            nn.Linear(condition_dim, condition_embed_dim),
            nn.GELU(),
            nn.Linear(condition_embed_dim, condition_embed_dim),
            nn.GELU(),
        )

        self.reg_head = nn.Sequential(
            nn.Linear(embed_dim + condition_embed_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, 1),
        )

    def forward(self, points: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        patch_tokens = self.encoder(points)
        geometry_feature = patch_tokens.mean(dim=1)
        condition_feature = self.condition_mlp(condition)
        fused_feature = torch.cat([geometry_feature, condition_feature], dim=-1)
        pred = self.reg_head(fused_feature)
        return pred


def load_pretrained_encoder(model: HullResistancePredictor, checkpoint_path: str, device: torch.device) -> None:
    if checkpoint_path is None or checkpoint_path == "":
        print("pretrained_ckpt is empty; training predictor from scratch")
        return

    checkpoint = torch.load(checkpoint_path, map_location=device)
    source_state = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    target_state = model.state_dict()

    loaded_state = {}
    skipped = []
    for key, value in source_state.items():
        if not key.startswith("encoder."):
            continue
        if key in target_state and target_state[key].shape == value.shape:
            loaded_state[key] = value
        else:
            skipped.append(key)

    target_state.update(loaded_state)
    model.load_state_dict(target_state, strict=True)
    print(f"loaded_encoder_keys={len(loaded_state)} from {checkpoint_path}")
    if len(skipped) > 0:
        print(f"skipped_encoder_keys={len(skipped)}")


def set_encoder_trainable(model: HullResistancePredictor, trainable: bool) -> None:
    for param in model.encoder.parameters():
        param.requires_grad = trainable


def build_optimizer(model: HullResistancePredictor, args) -> torch.optim.Optimizer:
    encoder_params = [p for p in model.encoder.parameters() if p.requires_grad]
    head_params = [p for name, p in model.named_parameters() if not name.startswith("encoder.") and p.requires_grad]
    param_groups = []
    if len(encoder_params) > 0:
        param_groups.append({"params": encoder_params, "lr": args.encoder_lr})
    if len(head_params) > 0:
        param_groups.append({"params": head_params, "lr": args.head_lr})
    return torch.optim.AdamW(param_groups, weight_decay=args.weight_decay)


def compute_regression_metrics(pred: np.ndarray, target: np.ndarray) -> Dict[str, float]:
    pred = np.asarray(pred, dtype=np.float64).reshape(-1)
    target = np.asarray(target, dtype=np.float64).reshape(-1)
    if pred.shape[0] == 0 or target.shape[0] == 0:
        raise ValueError("Cannot compute metrics on empty predictions or targets")

    error = pred - target
    abs_error = np.abs(error)
    mae = np.mean(abs_error)
    mse = np.mean(error**2)
    rmse = np.sqrt(mse)

    target_abs = np.abs(target)
    safe_target_abs = np.maximum(target_abs, 1e-8)
    mape_percent = np.mean(abs_error / safe_target_abs) * 100.0

    target_mean = np.mean(target)
    ss_res = np.sum(error**2)
    ss_tot = np.sum((target - target_mean) ** 2)
    r2 = 1.0 - ss_res / max(ss_tot, 1e-12)

    target_mean_abs = max(abs(float(target_mean)), 1e-8)
    rrmse_percent = rmse / target_mean_abs * 100.0

    return {
        "mae": float(mae),
        "mse": float(mse),
        "rmse": float(rmse),
        "mape_percent": float(mape_percent),
        "r2": float(r2),
        "rrmse_percent": float(rrmse_percent),
    }


def train_one_epoch(model, loader, optimizer, scaler, device, args, epoch: int) -> float:
    model.train()
    running_loss = 0.0
    seen = 0
    amp_enabled = args.amp and device.type == "cuda"

    for step, batch in enumerate(loader, start=1):
        points = batch["points"].to(device, non_blocking=True)
        condition = batch["condition"].to(device, non_blocking=True)
        target = batch["target"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with autocast_context(device, amp_enabled):
            pred = model(points, condition)
            if args.loss == "mse":
                loss = F.mse_loss(pred, target)
            elif args.loss == "huber":
                loss = F.smooth_l1_loss(pred, target, beta=args.huber_beta)
            else:
                raise ValueError(f"Unsupported loss: {args.loss}")

        scaler.scale(loss).backward()
        if args.grad_clip > 0:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(optimizer)
        scaler.update()

        batch_size = points.shape[0]
        running_loss += loss.item() * batch_size
        seen += batch_size

        if step % args.log_interval == 0:
            avg_loss = running_loss / max(seen, 1)
            print(f"epoch={epoch:04d} step={step:05d} train_loss={avg_loss:.6f}")

    return running_loss / max(seen, 1)


@torch.no_grad()
def evaluate(model, loader, device, args, target_normalizer: ZScoreNormalizer) -> Dict[str, float]:
    model.eval()
    running_loss = 0.0
    seen = 0
    amp_enabled = args.amp and device.type == "cuda"
    pred_raw_list = []
    target_raw_list = []

    for batch in loader:
        points = batch["points"].to(device, non_blocking=True)
        condition = batch["condition"].to(device, non_blocking=True)
        target = batch["target"].to(device, non_blocking=True)

        with autocast_context(device, amp_enabled):
            pred = model(points, condition)
            if args.loss == "mse":
                loss = F.mse_loss(pred, target)
            elif args.loss == "huber":
                loss = F.smooth_l1_loss(pred, target, beta=args.huber_beta)
            else:
                raise ValueError(f"Unsupported loss: {args.loss}")

        batch_size = points.shape[0]
        running_loss += loss.item() * batch_size
        seen += batch_size

        pred_norm_np = pred.detach().float().cpu().numpy()
        target_raw_np = batch["target_raw"].detach().float().cpu().numpy()

        pred_log_np = target_normalizer.inverse_transform(pred_norm_np)
        pred_raw_np = log1p_target_to_raw_target(pred_log_np)

        pred_raw_list.append(pred_raw_np)
        target_raw_list.append(target_raw_np)

    pred_raw = np.concatenate(pred_raw_list, axis=0)
    target_raw = np.concatenate(target_raw_list, axis=0)
    metrics = compute_regression_metrics(pred_raw, target_raw)
    metrics["loss"] = running_loss / max(seen, 1)
    return metrics


@torch.no_grad()
def save_predictions_csv(model, loader, device, args, target_normalizer: ZScoreNormalizer, output_path: Path) -> None:
    model.eval()
    amp_enabled = args.amp and device.type == "cuda"
    rows = []

    for batch in loader:
        points = batch["points"].to(device, non_blocking=True)
        condition = batch["condition"].to(device, non_blocking=True)
        with autocast_context(device, amp_enabled):
            pred = model(points, condition)

        pred_log = target_normalizer.inverse_transform(pred.detach().float().cpu().numpy())
        pred_raw = log1p_target_to_raw_target(pred_log).reshape(-1)
        target_raw = batch["target_raw"].detach().float().cpu().numpy().reshape(-1)
        condition_raw = batch["condition_raw"].detach().float().cpu().numpy()
        sample_ids = batch["sample_id"].detach().cpu().numpy().reshape(-1)
        hull_ids = batch["hull_id"].detach().cpu().numpy().reshape(-1)

        for i in range(len(sample_ids)):
            abs_error = float(abs(pred_raw[i] - target_raw[i]))
            rel_error = float(abs_error / max(abs(float(target_raw[i])), 1e-8))
            rows.append(
                {
                    "sample_id": int(sample_ids[i]),
                    "hull_id": int(hull_ids[i]),
                    "draft_depth_ratio": float(condition_raw[i, 0]),
                    "speed": float(condition_raw[i, 1]),
                    "target_total_resistance": float(target_raw[i]),
                    "pred_total_resistance": float(pred_raw[i]),
                    "abs_error": abs_error,
                    "rel_error": rel_error,
                }
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "sample_id",
                "hull_id",
                "draft_depth_ratio",
                "speed",
                "target_total_resistance",
                "pred_total_resistance",
                "abs_error",
                "rel_error",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def write_metrics_csv(output_path: Path, rows: List[Dict[str, float]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "stage",
        "epoch",
        "loss",
        "mae",
        "mse",
        "rmse",
        "mape_percent",
        "r2",
        "rrmse_percent",
    ]
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            clean_row = {name: row.get(name, "") for name in fieldnames}
            writer.writerow(clean_row)


def append_metrics_csv(output_path: Path, row: Dict[str, float]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "stage",
        "epoch",
        "loss",
        "mae",
        "mse",
        "rmse",
        "mape_percent",
        "r2",
        "rrmse_percent",
    ]
    file_exists = output_path.exists()
    with open(output_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        clean_row = {name: row.get(name, "") for name in fieldnames}
        writer.writerow(clean_row)


def save_checkpoint(
    path: Path,
    model,
    optimizer,
    epoch: int,
    best_test_mae: float,
    best_test_r2: float,
    args,
    serialization_orders: Sequence[str],
    condition_normalizer: ZScoreNormalizer,
    target_normalizer: ZScoreNormalizer,
    train_hulls: Sequence[int],
    test_hulls: Sequence[int],
    scaler=None,
    best_epoch: int = 0,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict() if scaler is not None else None,
            "best_test_mae": best_test_mae,
            "best_test_r2": best_test_r2,
            "best_epoch": best_epoch,
            "args": vars(args),
            "serialization_orders": tuple(serialization_orders),
            "target_transform": "log1p",
            "condition_normalizer": condition_normalizer.state_dict(),
            "target_normalizer": target_normalizer.state_dict(),
            "train_hulls": list(train_hulls),
            "test_hulls": list(test_hulls),
        },
        path,
    )


def load_checkpoint_metadata(checkpoint_path: str):
    if checkpoint_path is None or checkpoint_path == "":
        return None
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"resume checkpoint not found: {path}")
    return torch.load(path, map_location="cpu")


def move_optimizer_state_to_device(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)


def load_training_checkpoint(
    checkpoint_path: str,
    model,
    optimizer,
    scaler,
    device: torch.device,
    resume_model_only: bool = False,
    strict: bool = True,
) -> Tuple[int, float, float, int]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    source_state = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint

    load_result = model.load_state_dict(source_state, strict=strict)
    if not strict:
        print(f"resume_missing_keys={len(load_result.missing_keys)} resume_unexpected_keys={len(load_result.unexpected_keys)}")

    if resume_model_only:
        print(f"loaded model weights from {checkpoint_path}; optimizer/epoch/best metric were not restored")
        return 1, float("inf"), -float("inf"), 0

    if not isinstance(checkpoint, dict):
        raise ValueError("Full resume requires a checkpoint dict with optimizer and epoch states")

    if "optimizer" in checkpoint and checkpoint["optimizer"] is not None:
        try:
            optimizer.load_state_dict(checkpoint["optimizer"])
            move_optimizer_state_to_device(optimizer, device)
        except ValueError as exc:
            print(f"warning: could not restore optimizer state: {exc}")
            print("continuing with a freshly initialized optimizer")
    else:
        print("warning: optimizer state is missing in checkpoint; continuing with a freshly initialized optimizer")

    if scaler is not None and checkpoint.get("scaler") is not None:
        try:
            scaler.load_state_dict(checkpoint["scaler"])
        except Exception as exc:
            print(f"warning: could not restore AMP scaler state: {exc}")

    last_epoch = int(checkpoint.get("epoch", 0))
    start_epoch = last_epoch + 1
    best_test_mae = float(checkpoint.get("best_test_mae", float("inf")))
    best_test_r2 = float(checkpoint.get("best_test_r2", -float("inf")))
    best_epoch = int(checkpoint.get("best_epoch", 0))

    print(
        f"resumed training from {checkpoint_path}: "
        f"last_epoch={last_epoch} start_epoch={start_epoch} "
        f"best_test_mae={best_test_mae:.6f} best_test_r2={best_test_r2:.6f} best_epoch={best_epoch}"
    )
    return start_epoch, best_test_mae, best_test_r2, best_epoch


def build_loaders(args, condition_normalizer, target_normalizer, train_ids, test_ids):
    pin_memory = torch.cuda.is_available()

    train_dataset = PredictionDataset(
        root_dir=args.data_dir,
        sample_ids=train_ids,
        condition_normalizer=condition_normalizer,
        target_normalizer=target_normalizer,
        num_points=args.num_points,
        samples_per_hull=args.samples_per_hull,
        target_column=args.target_column,
        preload_shapes=args.preload_shapes,
    )
    test_dataset = PredictionDataset(
        root_dir=args.data_dir,
        sample_ids=test_ids,
        condition_normalizer=condition_normalizer,
        target_normalizer=target_normalizer,
        num_points=args.num_points,
        samples_per_hull=args.samples_per_hull,
        target_column=args.target_column,
        preload_shapes=args.preload_shapes,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        drop_last=True if len(train_dataset) >= args.batch_size else False,
    )
    train_eval_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    return train_loader, train_eval_loader, test_loader



def parse_args():
    parser = argparse.ArgumentParser(description="Supervised hull total resistance prediction with pretrained PTV3-style encoder and official xCPE")

    parser.add_argument("--data_dir", type=str, default="../PredictionTraindata")#Modify to the corresponding path.
    parser.add_argument("--output_dir", type=str, default="./prediction_outputs")#Modify to the corresponding path.
    parser.add_argument("--pretrained_ckpt", type=str, default="./pretrain_outputs/pretrain_best.pth")#Modify to the corresponding path.
    parser.add_argument("--resume", type=str, default="", help="Path to prediction checkpoint for resuming training, e.g. prediction_last.pth")
    parser.add_argument("--resume_model_only", action="store_true", help="Only load model weights from --resume; do not restore optimizer, scaler, epoch, or best metric")
    parser.add_argument("--resume_non_strict", action="store_true", help="Load --resume with strict=False, useful after small architecture/key changes")
    parser.add_argument("--reset_metrics_history", action="store_true", help="Overwrite metrics_history.csv/final_metrics.csv instead of appending when resuming")

    parser.add_argument("--num_points", type=int, default=1000)
    parser.add_argument("--samples_per_hull", type=int, default=21)
    parser.add_argument("--target_column", type=int, default=1, help="0=friction resistance, 1=total resistance, 2=wave resistance")
    parser.add_argument("--max_hulls", type=int, default=None, help="Debug only. Use first N hulls after parsing sample ids.")

    parser.add_argument("--patch_size", type=int, default=20)
    parser.add_argument("--embed_dim", type=int, default=256)
    parser.add_argument("--encoder_depth", type=int, default=10)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--window_size", type=int, default=16)
    parser.add_argument("--dropout", type=float, default=0.2)

    parser.add_argument("--serialization_orders", type=str, default=",".join(DEFAULT_HULL_PRIOR_SERIALIZATION_ORDERS))
    parser.add_argument("--serialization_bits", type=int, default=10)
    parser.add_argument("--shuffle_orders", dest="shuffle_orders", action="store_true")
    parser.add_argument("--no_shuffle_orders", dest="shuffle_orders", action="store_false")
    parser.set_defaults(shuffle_orders=True)

    parser.add_argument("--xcpe_kernel_size", type=int, default=3)
    parser.add_argument("--xcpe_grid_size", type=float, default=None)
    parser.add_argument("--xcpe_grid_bits", type=int, default=10)
    parser.add_argument("--xcpe_sparse_padding", type=int, default=96)

    parser.add_argument("--condition_embed_dim", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--encoder_lr", type=float, default=1e-4)
    parser.add_argument("--head_lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--loss", type=str, default="huber", choices=["huber", "mse"])
    parser.add_argument("--huber_beta", type=float, default=1.0)

    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=970709)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--log_interval", type=int, default=50)
    parser.add_argument("--save_interval", type=int, default=20)
    parser.add_argument("--amp", action="store_true")

    parser.add_argument("--preload_shapes", dest="preload_shapes", action="store_true")
    parser.add_argument("--no_preload_shapes", dest="preload_shapes", action="store_false")
    parser.set_defaults(preload_shapes=True)

    parser.add_argument("--strict_group_check", dest="strict_group_check", action="store_true")
    parser.add_argument("--no_strict_group_check", dest="strict_group_check", action="store_false")
    parser.set_defaults(strict_group_check=True)

    parser.add_argument("--freeze_encoder_epochs", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    serialization_orders = parse_serialization_orders(args.serialization_orders)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_sample_ids = find_available_sample_ids(args.data_dir)
    if args.max_hulls is not None:
        max_sample_id = args.max_hulls * args.samples_per_hull
        all_sample_ids = [sid for sid in all_sample_ids if sid <= max_sample_id]

    resume_checkpoint = load_checkpoint_metadata(args.resume) if args.resume != "" else None

    train_ids, test_ids, train_hulls, test_hulls = split_train_test_by_hull(
        sample_ids=all_sample_ids,
        samples_per_hull=args.samples_per_hull,
        train_ratio=args.train_ratio,
        seed=args.seed,
        strict_group_check=args.strict_group_check,
    )

    condition_normalizer, target_normalizer = fit_normalizers(
        root_dir=args.data_dir,
        sample_ids=train_ids,
        target_column=args.target_column,
    )

    if resume_checkpoint is not None and not args.resume_model_only:
        if "train_hulls" in resume_checkpoint and "test_hulls" in resume_checkpoint:
            train_hulls = sorted(int(hull_id) for hull_id in resume_checkpoint["train_hulls"])
            test_hulls = sorted(int(hull_id) for hull_id in resume_checkpoint["test_hulls"])
            train_ids = sample_ids_from_hulls(all_sample_ids, args.samples_per_hull, train_hulls)
            test_ids = sample_ids_from_hulls(all_sample_ids, args.samples_per_hull, test_hulls)
            print("restored train/test hull split from resume checkpoint")
        else:
            print("warning: resume checkpoint has no train_hulls/test_hulls; using split from current args")

        checkpoint_target_transform = resume_checkpoint.get("target_transform", "raw_zscore")
        if checkpoint_target_transform != "log1p":
            raise ValueError(
                "The resume checkpoint was not trained with target_transform='log1p'. "
                "Please retrain from the pretrained encoder, or use a checkpoint produced by this log1p version. "
                "For old checkpoints, do not use full --resume with this script."
            )

        if "condition_normalizer" in resume_checkpoint and "target_normalizer" in resume_checkpoint:
            condition_normalizer = ZScoreNormalizer.from_state_dict(resume_checkpoint["condition_normalizer"])
            target_normalizer = ZScoreNormalizer.from_state_dict(resume_checkpoint["target_normalizer"])
            print("restored normalizers from resume checkpoint")
        else:
            print("warning: resume checkpoint has no normalizers; using normalizers fitted from current train split")

    train_loader, train_eval_loader, test_loader = build_loaders(
        args=args,
        condition_normalizer=condition_normalizer,
        target_normalizer=target_normalizer,
        train_ids=train_ids,
        test_ids=test_ids,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")
    print(f"sample_count={len(all_sample_ids)} hull_count={len(set(hull_id_from_sample_id(sid, args.samples_per_hull) for sid in all_sample_ids))}")
    print(f"train_hulls={len(train_hulls)} test_hulls={len(test_hulls)}")
    print(f"train_samples={len(train_ids)} test_samples={len(test_ids)}")
    print(f"condition_mean={condition_normalizer.mean.tolist()} condition_std={condition_normalizer.std.tolist()}")
    print(f"target_transform=log1p")
    print(f"target_log1p_resistance_mean={target_normalizer.mean.tolist()} target_log1p_resistance_std={target_normalizer.std.tolist()}")
    print(f"serialization_orders={serialization_orders} shuffle_orders={args.shuffle_orders}")
    print("xCPE=official-style spconv.SubMConv3d + Linear + LayerNorm + residual")
    print("split_mode=train_test_only; validation set is used as test set")

    model = HullResistancePredictor(
        num_points=args.num_points,
        patch_size=args.patch_size,
        embed_dim=args.embed_dim,
        encoder_depth=args.encoder_depth,
        num_heads=args.num_heads,
        window_size=args.window_size,
        dropout=args.dropout,
        serialization_orders=serialization_orders,
        shuffle_orders=args.shuffle_orders,
        serialization_bits=args.serialization_bits,
        xcpe_kernel_size=args.xcpe_kernel_size,
        xcpe_grid_size=args.xcpe_grid_size,
        xcpe_grid_bits=args.xcpe_grid_bits,
        xcpe_sparse_padding=args.xcpe_sparse_padding,
        condition_dim=2,
        condition_embed_dim=args.condition_embed_dim,
    ).to(device)

    if args.resume == "":
        load_pretrained_encoder(model, args.pretrained_ckpt, device=device)
    else:
        print("resume is set; skipping pretrained encoder loading")

    start_epoch = 1
    if resume_checkpoint is not None and not args.resume_model_only:
        start_epoch = int(resume_checkpoint.get("epoch", 0)) + 1

    if args.freeze_encoder_epochs > 0 and start_epoch <= args.freeze_encoder_epochs + 1:
        set_encoder_trainable(model, False)
        print(f"encoder frozen until epoch {args.freeze_encoder_epochs}")
    else:
        set_encoder_trainable(model, True)

    optimizer = build_optimizer(model, args)
    amp_enabled = args.amp and device.type == "cuda"
    try:
        scaler = torch.cuda.amp.GradScaler("cuda", enabled=amp_enabled)
    except TypeError:
        scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    best_test_mae = float("inf")
    best_test_r2 = -float("inf")
    best_epoch = 0
    if args.resume != "":
        start_epoch, best_test_mae, best_test_r2, best_epoch = load_training_checkpoint(
            checkpoint_path=args.resume,
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            resume_model_only=args.resume_model_only,
            strict=not args.resume_non_strict,
        )

    metrics_history_path = output_dir / "metrics_history.csv"
    final_metrics_path = output_dir / "final_metrics.csv"
    reset_metrics = args.reset_metrics_history or args.resume == "" or args.resume_model_only
    if reset_metrics and metrics_history_path.exists():
        metrics_history_path.unlink()
    if reset_metrics and final_metrics_path.exists():
        final_metrics_path.unlink()

    if start_epoch > args.epochs:
        print(f"start_epoch={start_epoch} is greater than epochs={args.epochs}; skip training loop and run final evaluation")

    for epoch in range(start_epoch, args.epochs + 1):
        if args.freeze_encoder_epochs > 0 and epoch == args.freeze_encoder_epochs + 1:
            set_encoder_trainable(model, True)
            optimizer = build_optimizer(model, args)
            print("encoder unfrozen")

        train_loss = train_one_epoch(model, train_loader, optimizer, scaler, device, args, epoch)
        test_metrics = evaluate(model, test_loader, device, args, target_normalizer)
        print(
            f"epoch={epoch:04d} train_loss={train_loss:.6f} "
            f"test_loss={test_metrics['loss']:.6f} test_mae={test_metrics['mae']:.6f} "
            f"test_rmse={test_metrics['rmse']:.6f} test_mape={test_metrics['mape_percent']:.4f}% "
            f"test_r2={test_metrics['r2']:.6f} test_rrmse={test_metrics['rrmse_percent']:.4f}%"
        )

        append_metrics_csv(
            metrics_history_path,
            {
                "stage": "train_epoch_loss",
                "epoch": epoch,
                "loss": train_loss,
            },
        )
        append_metrics_csv(
            metrics_history_path,
            {
                "stage": "test_epoch",
                "epoch": epoch,
                **test_metrics,
            },
        )

        # Use larger R2 as the best-checkpoint criterion.
        if test_metrics["r2"] > best_test_r2:
            best_test_mae = test_metrics["mae"]
            best_test_r2 = test_metrics["r2"]
            best_epoch = epoch
            save_checkpoint(
                output_dir / "prediction_best.pth",
                model,
                optimizer,
                epoch,
                best_test_mae,
                best_test_r2,
                args,
                serialization_orders,
                condition_normalizer,
                target_normalizer,
                train_hulls,
                test_hulls,
                scaler=scaler,
                best_epoch=best_epoch,
            )
            print(f"saved best checkpoint with test_mae={best_test_mae:.6f} test_r2={best_test_r2:.6f}")

        save_checkpoint(
            output_dir / "prediction_last.pth",
            model,
            optimizer,
            epoch,
            best_test_mae,
            best_test_r2,
            args,
            serialization_orders,
            condition_normalizer,
            target_normalizer,
            train_hulls,
            test_hulls,
            scaler=scaler,
            best_epoch=best_epoch,
        )

        if epoch % args.save_interval == 0:
            save_checkpoint(
                output_dir / f"prediction_epoch_{epoch:04d}.pth",
                model,
                optimizer,
                epoch,
                best_test_mae,
                best_test_r2,
                args,
                serialization_orders,
                condition_normalizer,
                target_normalizer,
                train_hulls,
                test_hulls,
                scaler=scaler,
                best_epoch=best_epoch,
            )

    best_checkpoint_path = output_dir / "prediction_best.pth"
    if not best_checkpoint_path.exists():
        best_checkpoint_path = output_dir / "prediction_last.pth"
    print(f"loading best checkpoint for final evaluation: {best_checkpoint_path}")
    best_checkpoint = torch.load(best_checkpoint_path, map_location=device)
    model.load_state_dict(best_checkpoint["model"], strict=True)

    best_epoch = int(best_checkpoint.get("epoch", best_epoch))
    train_metrics = evaluate(model, train_eval_loader, device, args, target_normalizer)
    test_metrics = evaluate(model, test_loader, device, args, target_normalizer)

    print(
        f"final_train_loss={train_metrics['loss']:.6f} final_train_mae={train_metrics['mae']:.6f} "
        f"final_train_rmse={train_metrics['rmse']:.6f} final_train_mape={train_metrics['mape_percent']:.4f}% "
        f"final_train_r2={train_metrics['r2']:.6f} final_train_rrmse={train_metrics['rrmse_percent']:.4f}%"
    )
    print(
        f"final_test_loss={test_metrics['loss']:.6f} final_test_mae={test_metrics['mae']:.6f} "
        f"final_test_rmse={test_metrics['rmse']:.6f} final_test_mape={test_metrics['mape_percent']:.4f}% "
        f"final_test_r2={test_metrics['r2']:.6f} final_test_rrmse={test_metrics['rrmse_percent']:.4f}%"
    )

    write_metrics_csv(
        final_metrics_path,
        [
            {"stage": "final_train_best_checkpoint", "epoch": best_epoch, **train_metrics},
            {"stage": "final_test_best_checkpoint", "epoch": best_epoch, **test_metrics},
        ],
    )

    save_predictions_csv(
        model=model,
        loader=train_eval_loader,
        device=device,
        args=args,
        target_normalizer=target_normalizer,
        output_path=output_dir / "train_predictions.csv",
    )
    save_predictions_csv(
        model=model,
        loader=test_loader,
        device=device,
        args=args,
        target_normalizer=target_normalizer,
        output_path=output_dir / "test_predictions.csv",
    )

    print(f"best_checkpoint={output_dir / 'prediction_best.pth'}")
    print(f"metrics_history={metrics_history_path}")
    print(f"final_metrics={final_metrics_path}")
    print(f"train_predictions={output_dir / 'train_predictions.csv'}")
    print(f"test_predictions={output_dir / 'test_predictions.csv'}")


if __name__ == "__main__":
    main()
