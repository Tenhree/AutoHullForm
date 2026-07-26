import os
import hashlib
import glob
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from multiprocessing import Pool, cpu_count

class HullGeometryDataset(Dataset):

    def __init__(self, root_dir, num_wl, num_x, transform=None, dtype=torch.float32, return_empty_cond=True):
        self.root_dir = root_dir
        self.transform = transform
        self.num_wl = num_wl
        self.num_x = num_x
        self.dtype = dtype
        self.return_empty_cond = return_empty_cond

        self.file_paths = glob.glob(
            os.path.join(root_dir, "**", "*.csv"),
            recursive=True
        )

        if len(self.file_paths) == 0:
            raise RuntimeError(f"No CSV files found in {root_dir}")

        self.file_paths = sorted(self.file_paths)

    def __len__(self):
        return len(self.file_paths)

    def _normalize_columnwise(self, data: np.ndarray) -> np.ndarray:

        data = data.copy()

        # x column
        data[:, 0] = data[:, 0] - np.min(data[:, 0])
        x_max = np.max(data[:, 0])
        if x_max > 0:
            data[:, 0] = data[:, 0] / x_max
        else:
            data[:, 0] = 0.0

        # y column
        data[:, 1] = data[:, 1] - np.min(data[:, 1])
        y_max = np.max(data[:, 1])
        if y_max > 0:
            data[:, 1] = data[:, 1] / y_max
        else:
            data[:, 1] = 0.0

        return data

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]

        try:

            data = pd.read_csv(file_path, header=None).values.astype(np.float32)


            if data.shape[1] < 2:
                raise ValueError(f"CSV must contain at least 2 columns, got shape {data.shape}")


            data = data[:, :2]


            expected_rows = self.num_wl * self.num_x
            if data.shape[0] != expected_rows:
                raise ValueError(
                    f"CSV row count mismatch in {file_path}. "
                    f"Expected {expected_rows} rows (= num_wl*num_x), got {data.shape[0]}"
                )


            data = np.round(data, 5)
            data = self._normalize_columnwise(data)

            # reshape: [num_wl*num_x, 2] -> [num_wl, num_x, 2]
            data = data.reshape(self.num_wl, self.num_x, 2)

            # transpose -> [2, num_wl, num_x]
            data = data.transpose(2, 0, 1)

            # transform
            if self.transform is not None:
                data = self.transform(data)

            # 转 tensor
            if not isinstance(data, torch.Tensor):
                data = torch.tensor(data, dtype=self.dtype)
            else:
                data = data.to(self.dtype)


            data = torch.clamp(data, 0.0, 1.0)


            if data.ndim != 3 or data.shape[0] != 2:
                raise ValueError(f"Expected sample shape [2, H, W], got {tuple(data.shape)}")

        except Exception as e:
            raise RuntimeError(f"Error reading {file_path}: {e}")

        if self.return_empty_cond:
            cond = torch.empty(0, dtype=self.dtype)
        else:
            cond = None

        return {
            "x": data,
        }


def _hash_file(path: str) -> str:

    st = os.stat(path)
    s = f"{path}|{st.st_size}|{st.st_mtime}"
    return hashlib.md5(s.encode("utf-8")).hexdigest()


def _normalize_xy_columnwise(data: np.ndarray) -> np.ndarray:

    data = data.copy()

    # x
    data[:, 0] = data[:, 0] - np.min(data[:, 0])
    x_max = np.max(data[:, 0])
    if x_max > 0:
        data[:, 0] = data[:, 0] / x_max
    else:
        data[:, 0] = 0.0

    # y
    data[:, 1] = data[:, 1] - np.min(data[:, 1])
    y_max = np.max(data[:, 1])
    if y_max > 0:
        data[:, 1] = data[:, 1] / y_max
    else:
        data[:, 1] = 0.0

    return data


def _read_geometry_csv_as_numpy(file_path: str, num_wl: int, num_x: int) -> np.ndarray:

    data = pd.read_csv(file_path, header=None).values.astype(np.float32)

    if data.shape[1] < 2:
        raise ValueError(f"CSV must contain at least 2 columns, got shape {data.shape}")

    data = data[:, :2]

    expected_rows = num_wl * num_x
    if data.shape[0] != expected_rows:
        raise ValueError(
            f"CSV row count mismatch in {file_path}. "
            f"Expected {expected_rows} rows (= num_wl*num_x), got {data.shape[0]}"
        )

    data = np.round(data, 5)
    data = _normalize_xy_columnwise(data)

    data = data.reshape(num_wl, num_x, 2)
    data = data.transpose(2, 0, 1)   # -> [2, num_wl, num_x]
    data = np.clip(data, 0.0, 1.0)

    return data.astype(np.float32)


def calc_displacement(S_xy: np.ndarray, T: float, num_wl: int, num_x: int) -> float:

    X = S_xy[0]
    Y = S_xy[1]

    Z = np.linspace(0, 1, num_wl, dtype=np.float32).reshape(-1, 1).repeat(num_x, axis=1)
    z_values = np.unique(Z)

    T = float(np.clip(T, float(z_values.min()), float(z_values.max())))

    z_below = z_values[z_values <= T].max()
    z_above = z_values[z_values >= T].min()

    X_np = X
    Y_np = Y
    Z_np = Z

    def waterline_area(X_layer, Y_layer):
        idx = np.argsort(X_layer)
        X_layer = X_layer[idx]
        Y_layer = Y_layer[idx]
        return 2.0 * np.trapz(Y_layer, X_layer)

    if np.isclose(z_above, z_below):
        mask = np.abs(Z_np - z_below) < 1e-6
        X_new = X_np[mask]
        Y_new = Y_np[mask]
    else:
        mask_below = (Z_np == z_below)
        mask_above = (Z_np == z_above)

        X_below = X_np[mask_below]
        X_above = X_np[mask_above]
        Y_below = Y_np[mask_below]
        Y_above = Y_np[mask_above]

        alpha = (T - z_below) / (z_above - z_below)
        X_new = X_below + alpha * (X_above - X_below)
        Y_new = Y_below + alpha * (Y_above - Y_below)

    z_all = np.sort(np.append(z_values[z_values < T], T))

    areas = []
    Btemp = []

    for z in z_all:
        if np.isclose(z, T):
            A = waterline_area(X_new, Y_new)
            Btemptemp = np.max(Y_new) if len(Y_new) > 0 else 0.0
        else:
            mask = np.abs(Z_np - z) < 1e-6
            X_layer = X_np[mask]
            Y_layer = Y_np[mask]
            A = waterline_area(X_layer, Y_layer)
            Btemptemp = np.max(Y_layer) if len(Y_layer) > 0 else 0.0

        areas.append(A)
        Btemp.append(Btemptemp)

    areas = np.asarray(areas, dtype=np.float32)
    Btemp = np.asarray(Btemp, dtype=np.float32)

    volume = np.trapz(areas, z_all)

    Lwl = float(np.max(X_new) - np.min(X_new)) if len(X_new) > 0 else 0.0
    B = float(np.max(Btemp)) if len(Btemp) > 0 else 0.0

    eps = 1e-8
    Cb = volume / max(Lwl * B * T * 2.0, eps)
    return float(Cb)


def _compute_one_file_all_drafts(args):

    file_path, file_hash, num_wl, num_x, draft_list = args

    S = _read_geometry_csv_as_numpy(file_path, num_wl, num_x)

    rows = []
    for T in draft_list:
        cb = calc_displacement(S, float(T), num_wl, num_x)
        rows.append({
            "file_hash": file_hash,
            "T": float(T),
            "cb": float(cb),
        })

    return {
        "file_path": file_path,
        "file_hash": file_hash,
        "rows": rows,
    }


class HullPhysicalConditionDataset(Dataset):


    def __init__(
        self,
        root_dir,
        num_wl,
        num_x,
        draft_list=(0.3, 0.4, 0.5, 0.6, 0.7, 0.8),
        transform=None,
        dtype=torch.float32,
        cache_dir=None,
        cache_name="cb_cache.csv",
        force_recompute=False,
        num_cb_workers=None,
    ):
        self.root_dir = root_dir
        self.transform = transform
        self.num_wl = int(num_wl)
        self.num_x = int(num_x)
        self.draft_list = [float(t) for t in draft_list]
        self.dtype = dtype

        self.file_paths = sorted(
            glob.glob(os.path.join(root_dir, "**", "*.csv"), recursive=True)
        )
        if len(self.file_paths) == 0:
            raise RuntimeError(f"No CSV files found in {root_dir}")

        if cache_dir is None:
            cache_dir = root_dir
        os.makedirs(cache_dir, exist_ok=True)
        self.cache_path = os.path.join(cache_dir, cache_name)


        self.file_hashes = [_hash_file(fp) for fp in self.file_paths]


        self.geometry_cache = []
        for fp in self.file_paths:
            arr = _read_geometry_csv_as_numpy(fp, self.num_wl, self.num_x)
            self.geometry_cache.append(arr)


        self.cb_map = {}   # key: (file_hash, T) -> cb

        if (not force_recompute) and os.path.exists(self.cache_path):
            try:
                df_cache = pd.read_csv(self.cache_path)
                required_cols = {"file_hash", "T", "cb"}
                if not required_cols.issubset(df_cache.columns):
                    raise RuntimeError(
                        f"Cache file missing columns. Need {required_cols}, got {set(df_cache.columns)}"
                    )

                for _, row in df_cache.iterrows():
                    self.cb_map[(str(row["file_hash"]), float(row["T"]))] = float(row["cb"])
            except Exception as e:
                print(f"[Warn] Failed to read cache {self.cache_path}: {e}. Will recompute.")
                self.cb_map = {}


        missing_files = []
        for fp, fh in zip(self.file_paths, self.file_hashes):
            need_this_file = False
            for T in self.draft_list:
                if (fh, T) not in self.cb_map:
                    need_this_file = True
                    break
            if need_this_file:
                missing_files.append((fp, fh, self.num_wl, self.num_x, self.draft_list))

        if len(missing_files) > 0:
            print(f"[Info] Need to compute cb for {len(missing_files)} files...")

            if num_cb_workers is None:
                n_workers = min(cpu_count(), 8)
            else:
                n_workers = max(1, int(num_cb_workers))

            new_rows = []

            if n_workers == 1:
                results = map(_compute_one_file_all_drafts, missing_files)
            else:
                pool = Pool(processes=n_workers)
                results = pool.imap_unordered(_compute_one_file_all_drafts, missing_files, chunksize=8)

            try:
                for item in results:
                    file_hash = item["file_hash"]
                    for row in item["rows"]:
                        self.cb_map[(file_hash, float(row["T"]))] = float(row["cb"])
                        new_rows.append({
                            "file_path": item["file_path"],
                            "file_hash": file_hash,
                            "T": float(row["T"]),
                            "cb": float(row["cb"]),
                        })
            finally:
                if n_workers != 1:
                    pool.close()
                    pool.join()

            df_new = pd.DataFrame(new_rows)

            if os.path.exists(self.cache_path) and (not force_recompute):
                try:
                    df_old = pd.read_csv(self.cache_path)
                    df_all = pd.concat([df_old, df_new], ignore_index=True)
                except Exception:
                    df_all = df_new
            else:
                df_all = df_new

            df_all = df_all.drop_duplicates(subset=["file_hash", "T"], keep="last")
            df_all.to_csv(self.cache_path, index=False)
            print(f"[Info] Saved cb cache to: {self.cache_path}")


        self.cb_by_index = {}
        for i, fh in enumerate(self.file_hashes):
            for T in self.draft_list:
                key = (fh, T)
                if key not in self.cb_map:
                    raise RuntimeError(f"Missing cb for file={self.file_paths[i]}, T={T}")
                self.cb_by_index[(i, T)] = float(self.cb_map[key])

        self.index = []
        for file_idx in range(len(self.file_paths)):
            for t_idx in range(len(self.draft_list)):
                self.index.append((file_idx, t_idx))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        file_idx, t_idx = self.index[idx]


        data = self.geometry_cache[file_idx]

        # transform 放这里，保留随机增强能力
        if self.transform is not None:
            data = self.transform(data)

        if not isinstance(data, torch.Tensor):
            data = torch.tensor(data, dtype=self.dtype)
        else:
            data = data.to(self.dtype)

        data = torch.clamp(data, 0.0, 1.0)

        if data.ndim != 3 or data.shape[0] != 2:
            raise ValueError(f"Expected geometry shape [2, H, W], got {tuple(data.shape)}")

        T = self.draft_list[t_idx]
        cb = self.cb_by_index[(file_idx, T)]

        return {
            "x": data,
            "draft": torch.tensor(T, dtype=self.dtype),
            "cb": torch.tensor(cb, dtype=self.dtype),
        }


