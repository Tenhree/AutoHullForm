import argparse
import glob
import math
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
from torch.utils.data import DataLoader, Dataset, random_split

try:
    import spconv.pytorch as spconv
except ImportError:
    spconv = None



def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def hull_file_sort_key(path: str) -> Tuple[int, str]:
    name = os.path.basename(path)
    match = re.search(r"Hull_PC_(\d+)\.csv$", name)
    if match is None:
        return (10**18, name)
    return (int(match.group(1)), name)


def autocast_context(device: torch.device, enabled: bool):
    if enabled and device.type == "cuda":
        return torch.amp.autocast(device_type="cuda", enabled=True)
    return nullcontext()



class HullPointCloudDataset(Dataset):
    def __init__(self, data_dir: str, pattern: str = "Hull_PC_*.csv", num_points: int = 1000, max_files: int = None):
        self.data_dir = Path(data_dir)
        self.pattern = pattern
        self.num_points = num_points
        self.files = sorted(glob.glob(str(self.data_dir / self.pattern)), key=hull_file_sort_key)
        if max_files is not None:
            self.files = self.files[:max_files]
        if len(self.files) == 0:
            raise FileNotFoundError(f"No files found in {self.data_dir} with pattern {self.pattern}")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> torch.Tensor:
        file_path = self.files[index]
        points = np.loadtxt(file_path, delimiter=",", dtype=np.float32)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError(f"{file_path} must have shape [N, 3], but got {points.shape}")
        if points.shape[0] != self.num_points:
            raise ValueError(f"{file_path} must have {self.num_points} points, but got {points.shape[0]}")
        return torch.from_numpy(points)


SPACE_FILLING_SERIALIZATION_ORDERS = ("z", "z-trans", "hilbert", "hilbert-trans")
HULL_PRIOR_SERIALIZATION_ORDER_SPECS: Dict[str, Tuple[Tuple[str, str], ...]] = {
    # 1) deck -> bottom, then bow -> stern
    "deck2bottom_bow2stern": (("z", "desc"), ("x", "asc"), ("y", "asc")),
    # 2) deck -> bottom, then stern -> bow
    "deck2bottom_stern2bow": (("z", "desc"), ("x", "desc"), ("y", "asc")),
    # 3) bottom -> deck, then bow -> stern
    "bottom2deck_bow2stern": (("z", "asc"), ("x", "asc"), ("y", "asc")),
    # 4) bottom -> deck, then stern -> bow
    "bottom2deck_stern2bow": (("z", "asc"), ("x", "desc"), ("y", "asc")),
    # 5) bow -> stern, then deck -> bottom
    "bow2stern_deck2bottom": (("x", "asc"), ("z", "desc"), ("y", "asc")),
    # 6) stern -> bow, then deck -> bottom
    "stern2bow_deck2bottom": (("x", "desc"), ("z", "desc"), ("y", "asc")),
    # 7) bow -> stern, then bottom -> deck
    "bow2stern_bottom2deck": (("x", "asc"), ("z", "asc"), ("y", "asc")),
    # 8) stern -> bow, then bottom -> deck
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


def random_patch_mask(batch_size: int, num_patches: int, mask_ratio: float, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    num_keep = max(1, int(num_patches * (1.0 - mask_ratio)))
    noise = torch.rand(batch_size, num_patches, device=device)
    ids_shuffle = torch.argsort(noise, dim=1)
    ids_keep = torch.sort(ids_shuffle[:, :num_keep], dim=1).values
    mask = torch.ones(batch_size, num_patches, dtype=torch.bool, device=device)
    mask.scatter_(dim=1, index=ids_keep, value=False)
    return ids_keep, mask

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


class HullPTv3Encoder(nn.Module):
    def __init__(
        self,
        patch_size: int,
        embed_dim: int,
        depth: int,
        num_heads: int,
        window_size: int,
        dropout: float,
        xcpe_kernel_size: int,
        xcpe_grid_size: Optional[float],
        xcpe_grid_bits: int,
        xcpe_sparse_padding: int,
    ):
        super().__init__()
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

    def forward(
        self,
        relative_patches: torch.Tensor,
        centers: torch.Tensor,
        ids_keep: torch.Tensor,
        order_names: Sequence[str],
        serialization_bits: int,
    ) -> torch.Tensor:
        tokens = self.patch_embed(relative_patches, centers)
        visible_tokens = batched_gather(tokens, ids_keep)
        visible_centers = batched_gather(centers, ids_keep)
        visible_order_pack = compute_serialized_orders(visible_centers, order_names=order_names, bits=serialization_bits)

        for block in self.blocks:
            visible_tokens = block(visible_tokens, visible_centers, visible_order_pack)
        return self.norm(visible_tokens)


class HullPTv3MAE(nn.Module):
    def __init__(
        self,
        num_points: int = 1000,
        patch_size: int = 20,
        embed_dim: int = 256,
        encoder_depth: int = 6,
        decoder_dim: int = 256,
        decoder_depth: int = 4,
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

        self.encoder = HullPTv3Encoder(
            patch_size=patch_size,
            embed_dim=embed_dim,
            depth=encoder_depth,
            num_heads=num_heads,
            window_size=window_size,
            dropout=dropout,
            xcpe_kernel_size=xcpe_kernel_size,
            xcpe_grid_size=xcpe_grid_size,
            xcpe_grid_bits=xcpe_grid_bits,
            xcpe_sparse_padding=xcpe_sparse_padding,
        )

        self.decoder_embed = nn.Linear(embed_dim, decoder_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_dim))
        self.decoder_pos = nn.Sequential(
            nn.Linear(3, decoder_dim),
            nn.GELU(),
            nn.Linear(decoder_dim, decoder_dim),
        )
        self.decoder_blocks = nn.ModuleList(
            [
                SerializedAttentionBlock(
                    dim=decoder_dim,
                    num_heads=num_heads,
                    window_size=window_size,
                    dropout=dropout,
                    order_index=i,
                    xcpe_kernel_size=xcpe_kernel_size,
                    xcpe_grid_size=xcpe_grid_size,
                    xcpe_grid_bits=xcpe_grid_bits,
                    xcpe_sparse_padding=xcpe_sparse_padding,
                    cpe_indice_key=f"decoder_xcpe_{i}",
                )
                for i in range(decoder_depth)
            ]
        )
        self.decoder_norm = nn.LayerNorm(decoder_dim)
        self.pred_head = nn.Linear(decoder_dim, patch_size * 3)
        nn.init.trunc_normal_(self.mask_token, std=0.02)

    def forward(self, points: torch.Tensor, mask_ratio: float) -> Dict[str, torch.Tensor]:
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

        batch_size = points.shape[0]
        ids_keep, mask = random_patch_mask(
            batch_size=batch_size,
            num_patches=self.num_patches,
            mask_ratio=mask_ratio,
            device=points.device,
        )

        encoded_visible = self.encoder(
            relative_patches=relative_patches,
            centers=centers,
            ids_keep=ids_keep,
            order_names=active_order_names,
            serialization_bits=self.serialization_bits,
        )
        decoded_visible = self.decoder_embed(encoded_visible)

        full_tokens = self.mask_token.expand(batch_size, self.num_patches, -1).clone()
        index_expand = ids_keep.unsqueeze(-1).expand(-1, -1, decoded_visible.shape[-1])
        full_tokens.scatter_(dim=1, index=index_expand, src=decoded_visible)
        full_tokens = full_tokens + self.decoder_pos(centers)

        decoder_order_pack = compute_serialized_orders(centers, order_names=active_order_names, bits=self.serialization_bits)
        for block in self.decoder_blocks:
            full_tokens = block(full_tokens, centers, decoder_order_pack)

        full_tokens = self.decoder_norm(full_tokens)
        pred_relative = self.pred_head(full_tokens).view(batch_size, self.num_patches, self.patch_size, 3)
        loss = chamfer_l2_loss(pred_relative[mask], relative_patches[mask])

        return {
            "loss": loss,
            "pred_relative": pred_relative,
            "target_relative": relative_patches,
            "mask": mask,
        }

def chamfer_l2_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if pred.numel() == 0:
        return pred.sum() * 0.0
    pred = pred.float()
    target = target.float()
    dist = torch.cdist(pred, target, p=2).pow(2)
    loss_pred_to_target = dist.min(dim=2).values.mean()
    loss_target_to_pred = dist.min(dim=1).values.mean()
    return loss_pred_to_target + loss_target_to_pred


def train_one_epoch(model, loader, optimizer, scaler, device, args, epoch: int) -> float:
    model.train()
    running_loss = 0.0
    seen = 0
    amp_enabled = args.amp and device.type == "cuda"

    for step, points in enumerate(loader, start=1):
        points = points.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        with autocast_context(device, amp_enabled):
            output = model(points, mask_ratio=args.mask_ratio)
            loss = output["loss"]

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
def validate(model, loader, device, args) -> float:
    if loader is None:
        return math.nan
    model.eval()
    running_loss = 0.0
    seen = 0
    amp_enabled = args.amp and device.type == "cuda"

    for points in loader:
        points = points.to(device, non_blocking=True)
        with autocast_context(device, amp_enabled):
            output = model(points, mask_ratio=args.mask_ratio)
            loss = output["loss"]
        batch_size = points.shape[0]
        running_loss += loss.item() * batch_size
        seen += batch_size

    return running_loss / max(seen, 1)


def save_checkpoint(
    path: Path,
    model,
    optimizer,
    scaler,
    epoch: int,
    best_val_loss: float,
    args,
    serialization_orders: Sequence[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict() if scaler is not None else None,
            "best_val_loss": best_val_loss,
            "args": vars(args),
            "serialization_orders": tuple(serialization_orders),
        },
        path,
    )


def load_checkpoint(
    path: Path,
    model,
    optimizer=None,
    scaler=None,
    device: torch.device = torch.device("cpu"),
    strict: bool = True,
    model_only: bool = False,
):
    if not path.is_file():
        raise FileNotFoundError(f"Resume checkpoint not found: {path}")

    checkpoint = torch.load(path, map_location=device)
    if "model" not in checkpoint:
        raise KeyError(f"Checkpoint {path} does not contain a 'model' state_dict")

    missing_keys, unexpected_keys = model.load_state_dict(checkpoint["model"], strict=strict)
    if not strict:
        if missing_keys:
            print(f"[resume] missing model keys: {missing_keys}")
        if unexpected_keys:
            print(f"[resume] unexpected model keys: {unexpected_keys}")

    if not model_only and optimizer is not None and checkpoint.get("optimizer") is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
        # Move optimizer states to the current device after loading from CPU/GPU checkpoints.
        for state in optimizer.state.values():
            for key, value in state.items():
                if torch.is_tensor(value):
                    state[key] = value.to(device)
    elif not model_only:
        print("[resume] optimizer state not found; optimizer will start fresh")

    if not model_only and scaler is not None and checkpoint.get("scaler") is not None:
        scaler.load_state_dict(checkpoint["scaler"])
    elif not model_only and scaler is not None:
        print("[resume] AMP scaler state not found; scaler will start fresh")

    loaded_epoch = int(checkpoint.get("epoch", 0))
    best_val_loss = float(checkpoint.get("best_val_loss", float("inf")))
    return loaded_epoch, best_val_loss, checkpoint

def build_loaders(args):
    dataset = HullPointCloudDataset(
        data_dir=args.data_dir,
        pattern=args.pattern,
        num_points=args.num_points,
        max_files=args.max_files,
    )
    dataset_size = len(dataset)
    val_size = int(dataset_size * args.val_ratio)
    if dataset_size >= 10 and args.val_ratio > 0:
        val_size = max(1, val_size)
    else:
        val_size = 0
    train_size = dataset_size - val_size

    generator = torch.Generator().manual_seed(args.seed)
    if val_size > 0:
        train_set, val_set = random_split(dataset, [train_size, val_size], generator=generator)
    else:
        train_set, val_set = dataset, None

    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        drop_last=True if train_size >= args.batch_size else False,
    )

    val_loader = None
    if val_set is not None:
        val_loader = DataLoader(
            val_set,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=pin_memory,
            drop_last=False,
        )

    return train_loader, val_loader, dataset_size, train_size, val_size


def parse_args():
    parser = argparse.ArgumentParser(description="Self-supervised hull point cloud pretraining with PTV3-style MAE and official xCPE")
    parser.add_argument("--data_dir", type=str, default="../PreTrainingdata/TrainingPC20X50")#Modify to the corresponding path.
    parser.add_argument("--pattern", type=str, default="Hull_PC_*.csv")#Modify to the corresponding path.
    parser.add_argument("--output_dir", type=str, default="./pretrain_outputs")#Modify to the corresponding path.
    parser.add_argument("--num_points", type=int, default=1000)
    parser.add_argument("--patch_size", type=int, default=20)
    parser.add_argument("--mask_ratio", type=float, default=0.6)

    parser.add_argument("--embed_dim", type=int, default=256)
    parser.add_argument("--encoder_depth", type=int, default=6)
    parser.add_argument("--decoder_dim", type=int, default=256)
    parser.add_argument("--decoder_depth", type=int, default=10)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--window_size", type=int, default=16)
    parser.add_argument("--dropout", type=float, default=0.0)

    parser.add_argument("--serialization_orders", type=str, default=",".join(VALID_SERIALIZATION_ORDERS))
    parser.add_argument("--serialization_bits", type=int, default=10)
    parser.add_argument("--shuffle_orders", dest="shuffle_orders", action="store_true")
    parser.add_argument("--no_shuffle_orders", dest="shuffle_orders", action="store_false")
    parser.set_defaults(shuffle_orders=True)

    parser.add_argument("--xcpe_kernel_size", type=int, default=3)
    parser.add_argument("--xcpe_grid_size", type=float, default=None)
    parser.add_argument("--xcpe_grid_bits", type=int, default=10)
    parser.add_argument("--xcpe_sparse_padding", type=int, default=96)

    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--val_ratio", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=970709)
    parser.add_argument("--log_interval", type=int, default=50)
    parser.add_argument("--save_interval", type=int, default=10)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--max_files", type=int, default=None)

    # Resume training options.
    parser.add_argument(
        "--resume",
        type=str,
        default="",
        help="Path to a checkpoint to resume from, for example ./pretrain_outputs/pretrain_last.pth",
    )
    parser.add_argument(
        "--resume_model_only",
        action="store_true",
        help="Only load model weights from --resume and start optimizer/scaler/epoch from scratch",
    )
    parser.add_argument(
        "--resume_non_strict",
        action="store_true",
        help="Load model weights with strict=False; useful only after small architecture changes",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    serialization_orders = parse_serialization_orders(args.serialization_orders)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, dataset_size, train_size, val_size = build_loaders(args)

    print(f"device={device}")
    print(f"dataset_size={dataset_size} train_size={train_size} val_size={val_size}")
    print(f"serialization_orders={serialization_orders} shuffle_orders={args.shuffle_orders}")
    print("xCPE=official-style spconv.SubMConv3d + Linear + LayerNorm + residual")

    model = HullPTv3MAE(
        num_points=args.num_points,
        patch_size=args.patch_size,
        embed_dim=args.embed_dim,
        encoder_depth=args.encoder_depth,
        decoder_dim=args.decoder_dim,
        decoder_depth=args.decoder_depth,
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
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    amp_enabled = args.amp and device.type == "cuda"
    try:
        scaler = torch.cuda.amp.GradScaler("cuda", enabled=amp_enabled)
    except TypeError:
        scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    best_val_loss = float("inf")
    start_epoch = 1
    if args.resume:
        loaded_epoch, best_val_loss, checkpoint = load_checkpoint(
            path=Path(args.resume),
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            strict=not args.resume_non_strict,
            model_only=args.resume_model_only,
        )

        if args.resume_model_only:
            start_epoch = 1
            best_val_loss = float("inf")
            print(f"[resume] loaded model weights from {args.resume}; training will start from epoch 1")
        else:
            start_epoch = loaded_epoch + 1
            ckpt_orders = checkpoint.get("serialization_orders")
            if ckpt_orders is not None and tuple(ckpt_orders) != tuple(serialization_orders):
                print(
                    f"[resume] warning: checkpoint serialization_orders={tuple(ckpt_orders)} "
                    f"but current serialization_orders={tuple(serialization_orders)}"
                )
            print(
                f"[resume] loaded checkpoint={args.resume} "
                f"epoch={loaded_epoch} next_epoch={start_epoch} best_val_loss={best_val_loss:.6f}"
            )

    if start_epoch > args.epochs:
        print(
            f"[resume] checkpoint epoch is {start_epoch - 1}, but --epochs={args.epochs}. "
            "Increase --epochs if you want to continue training more epochs."
        )
        return

    for epoch in range(start_epoch, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, scaler, device, args, epoch)
        val_loss = validate(model, val_loader, device, args)
        print(f"epoch={epoch:04d} train_loss={train_loss:.6f} val_loss={val_loss:.6f}")

        score = val_loss if not math.isnan(val_loss) else train_loss
        if score < best_val_loss:
            best_val_loss = score
            save_checkpoint(output_dir / "pretrain_best.pth", model, optimizer, scaler, epoch, best_val_loss, args, serialization_orders)

        save_checkpoint(output_dir / "pretrain_last.pth", model, optimizer, scaler, epoch, best_val_loss, args, serialization_orders)

        if epoch % args.save_interval == 0:
            save_checkpoint(output_dir / f"pretrain_epoch_{epoch:04d}.pth", model, optimizer, scaler, epoch, best_val_loss, args, serialization_orders)

    print(f"finished best_loss={best_val_loss:.6f}")
    print(f"best_checkpoint={output_dir / 'pretrain_best.pth'}")


if __name__ == "__main__":
    main()
