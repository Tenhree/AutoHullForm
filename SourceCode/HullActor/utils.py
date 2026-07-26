from __future__ import annotations

import csv
import json
import os
import random
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(device: str = "auto") -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(path: str | Path, data: Dict) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


class CSVLogger:
    """Append dictionaries to a CSV file while keeping a stable header."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        ensure_dir(self.path.parent)
        self.fieldnames: Optional[List[str]] = None
        self._created = False

    def write(self, row: Dict) -> None:
        clean = {}
        for k, v in row.items():
            if isinstance(v, np.generic):
                v = v.item()
            if isinstance(v, (np.ndarray, list, tuple)):
                v = json.dumps(np.asarray(v).tolist(), ensure_ascii=False)
            clean[k] = v

        if self.fieldnames is None:
            self.fieldnames = list(clean.keys())
            exists = self.path.exists() and self.path.stat().st_size > 0
            if exists:
                # Existing logs are allowed, but this simple logger assumes same header.
                with self.path.open("r", newline="", encoding="utf-8") as f:
                    reader = csv.reader(f)
                    first = next(reader, None)
                if first:
                    self.fieldnames = list(first)
            else:
                with self.path.open("w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=self.fieldnames)
                    writer.writeheader()
                self._created = True

        for k in clean.keys():
            if k not in self.fieldnames:
                raise KeyError(
                    f"CSVLogger {self.path} got a new field {k!r}. "
                    f"Initial fields: {self.fieldnames}. Keep log rows schema-stable."
                )

        with self.path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writerow({k: clean.get(k, "") for k in self.fieldnames})


def now_string() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def moving_average(values: Iterable[float], window: int) -> np.ndarray:
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size == 0:
        return arr
    window = max(1, int(window))
    out = np.empty_like(arr, dtype=np.float64)
    for i in range(arr.size):
        j0 = max(0, i - window + 1)
        out[i] = np.nanmean(arr[j0 : i + 1])
    return out


def safe_float(x, default: float = float("nan")) -> float:
    try:
        y = float(x)
        if np.isfinite(y):
            return y
    except Exception:
        pass
    return default
