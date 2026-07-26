from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np

from settings import EnvConfig


@dataclass
class HullCase:
    L: float
    B: float
    D: float
    T: float
    CB: float
    V: float
    z0: np.ndarray

    def to_dict(self):
        d = asdict(self)
        z = np.asarray(self.z0, dtype=np.float32).reshape(3)
        d.pop("z0", None)
        d.update({"z0_1": float(z[0]), "z0_2": float(z[1]), "z0_3": float(z[2])})
        return d


def sample_case(rng: np.random.Generator, cfg: EnvConfig) -> HullCase:
    L = float(cfg.L_fixed)
    B = float(rng.uniform(*cfg.B_range))
    D = float(rng.uniform(*cfg.D_range))
    T_over_D = float(rng.uniform(*cfg.T_over_D_range))
    T = float(T_over_D * D)
    CB = float(rng.uniform(*cfg.CB_range))
    V = float(rng.uniform(*cfg.V_range))
    z0 = rng.uniform(cfg.z0_range[0], cfg.z0_range[1], size=3).astype(np.float32)
    return HullCase(L=L, B=B, D=D, T=T, CB=CB, V=V, z0=z0)


def _get(row: dict, names: Iterable[str], default=None):
    for name in names:
        if name in row and row[name] not in ("", None):
            return row[name]
    return default


def case_from_row(row: dict) -> HullCase:
    L = float(_get(row, ["L", "length"], 3.0))
    B = float(_get(row, ["B", "beam"], None))
    D = float(_get(row, ["D", "depth"], None))
    T = float(_get(row, ["T", "draft"], None))
    CB = float(_get(row, ["CB", "C_B", "cb"], None))
    V = float(_get(row, ["V", "speed"], None))

    z_values = []
    for candidates in (["z0_1", "z1", "z_1"], ["z0_2", "z2", "z_2"], ["z0_3", "z3", "z_3"]):
        val = _get(row, candidates, None)
        if val is None:
            raise ValueError(f"case csv row lacks latent component from candidates {candidates}: {row}")
        z_values.append(float(val))

    return HullCase(L=L, B=B, D=D, T=T, CB=CB, V=V, z0=np.asarray(z_values, dtype=np.float32))


def load_cases_csv(path: str | Path) -> List[HullCase]:
    path = Path(path)
    cases: List[HullCase] = []
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cases.append(case_from_row(row))
    if not cases:
        raise ValueError(f"No cases found in {path}")
    return cases


def write_example_cases(path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        HullCase(3.0, 0.50, 0.36, 0.24, 0.60, 0.90, np.array([0.0, 0.0, 0.0], dtype=np.float32)),
        HullCase(3.0, 0.58, 0.40, 0.28, 0.66, 1.20, np.array([0.5, -0.2, 0.1], dtype=np.float32)),
        HullCase(3.0, 0.46, 0.32, 0.21, 0.55, 1.50, np.array([-0.3, 0.4, -0.1], dtype=np.float32)),
    ]
    fieldnames = list(rows[0].to_dict().keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row.to_dict())
