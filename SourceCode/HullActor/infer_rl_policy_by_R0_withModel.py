from __future__ import annotations
import argparse
import csv
import json
import os

import sys

from pathlib import Path

from typing import Iterable, List, Optional, Sequence, Tuple


import numpy as np

import torch
import time


if __package__ is None or __package__ == "":

    sys.path.append(os.path.dirname(os.path.dirname(__file__)))


from cases import HullCase

from env import HullOptimizationEnv

from hull_interfaces import make_generator, make_resistance_predictor

from policy_loader import LoadedPolicy

from training import add_common_args, env_config_from_args

from utils import ensure_dir



EPS = 1.0e-8


def parse_args() -> argparse.Namespace:


    parser = argparse.ArgumentParser(    )

    add_common_args(parser)


    parser.add_argument(
        "--policy_ckpt",
        type=str,
        default="./outputs/ppo_old/model_final.pt", #Modify to the corresponding path.
    )

    parser.add_argument("--L", type=float, default=3.0)

    parser.add_argument("--B", type=float, default=0.45)

    parser.add_argument("--D", type=float, default=0.289090909)

    parser.add_argument("--T", type=float, default=0.143181818)

    parser.add_argument("--CB", type=float, default=0.57)

    parser.add_argument("--V", type=float, default=1.8)

    parser.add_argument(
        "--R0",
        type=float,
        default=9.434036255,
    )


    parser.add_argument("--K", type=int, default=50)

    parser.add_argument(
        "--num_restarts",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--z_init_low",
        type=float,
        default=-1.5,
    )

    parser.add_argument(
        "--z_init_high",
        type=float,
        default=1.5,
    )

    parser.add_argument(
        "--init_sampling",
        type=str,
        choices=["lhs", "uniform"],
        default="lhs",
    )

    parser.add_argument(
        "--no_zero_start",
        action="store_true",
    )

    parser.add_argument(
        "--min_improvement_ratio",
        type=float,
        default=0.1,
    )

    parser.add_argument(
        "--stochastic_policy",
        action="store_true",
        default=True,
    )

    parser.add_argument(
        "--save_all_valid_pointclouds",
        action="store_true",
    )

    parser.add_argument(
        "--save_all_better_pointclouds",
        action="store_true",
    )


    parser.set_defaults(output_dir="outputs/inference_by_R0_Ship1V3", log_steps=True, total_steps=10)

    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:


    if not np.isfinite(args.R0) or float(args.R0) <= 0.0:
        raise ValueError("--R0 ")

    if int(args.K) <= 0:

        raise ValueError("--K > 0")

    if int(args.num_restarts) <= 0:

        raise ValueError("--num_restarts> 0")

    if float(args.z_init_high) <= float(args.z_init_low):

        raise ValueError("--z_init_high > --z_init_low")

    if not 0.0 <= float(args.min_improvement_ratio) < 1.0:

        raise ValueError("--min_improvement_ratio in [0,1)")

    for name in ("L", "B", "D", "T", "CB", "V"):

        value = float(getattr(args, name))

        if not np.isfinite(value) or value <= 0.0:

            raise ValueError(f"--{name}  {value}")


def latin_hypercube_samples(
    rng: np.random.Generator,
    sample_count: int,
    dimension: int,
    low: float,
    high: float,
) -> np.ndarray:

    if sample_count <= 0:

        return np.empty((0, dimension), dtype=np.float32)

    unit_samples = np.empty((sample_count, dimension), dtype=np.float64)

    for dim in range(dimension):

        permutation = rng.permutation(sample_count)

        offsets = rng.random(sample_count)

        unit_samples[:, dim] = (permutation + offsets) / float(sample_count)

    samples = float(low) + unit_samples * (float(high) - float(low))

    return samples.astype(np.float32)


def build_initial_latents(args: argparse.Namespace, env) -> np.ndarray:

    rng = np.random.default_rng(int(args.seed))

    include_zero = not bool(args.no_zero_start)

    include_zero = include_zero and float(args.z_init_low) <= 0.0 <= float(args.z_init_high)

    random_count = int(args.num_restarts) - (1 if include_zero else 0)


    if args.init_sampling == "lhs":

        random_latents = latin_hypercube_samples(
            rng=rng,
            sample_count=random_count,
            dimension=3,
            low=float(args.z_init_low),
            high=float(args.z_init_high),
        )
    else:

        random_latents = rng.uniform(
            low=float(args.z_init_low),
            high=float(args.z_init_high),
            size=(random_count, 3),
        ).astype(np.float32)


    starts: List[np.ndarray] = []

    if include_zero:

        starts.append(np.zeros(3, dtype=np.float32))

    for row in random_latents:

        starts.append(np.asarray(row, dtype=np.float32).copy())


    start_array = np.stack(starts, axis=0).astype(np.float32)

    start_array = np.clip(start_array, float(env.cfg.z_min), float(env.cfg.z_max)).astype(np.float32)

    return start_array


def is_target_candidate(
    rec,
    target_resistance: float,
    min_improvement_ratio: float,
) -> bool:


    required_resistance = float(target_resistance) * (1.0 - float(min_improvement_ratio))

    return bool(rec.constraint_ok and np.isfinite(rec.resistance) and float(rec.resistance) < required_resistance)


def candidate_row(
    restart_id: int,
    step: int,
    rec,
    case: HullCase,
    search_start_z: np.ndarray,
    reference_resistance: float,
    restart_initial_resistance: float,
    min_improvement_ratio: float,
    reward: Optional[float] = None,
    valid_rank: Optional[int] = None,
    target_rank: Optional[int] = None,
    point_cloud_path: str = "",
) -> dict:


    current_z = np.asarray(rec.z, dtype=np.float32).reshape(3)

    start_z = np.asarray(search_start_z, dtype=np.float32).reshape(3)

    required_resistance = float(reference_resistance) * (1.0 - float(min_improvement_ratio))

    better_than_reference = bool(float(rec.resistance) < float(reference_resistance))

    meets_target = is_target_candidate(
        rec=rec,
        target_resistance=float(reference_resistance),
        min_improvement_ratio=float(min_improvement_ratio),
    )

    reduction_vs_reference = (
        float(reference_resistance) - float(rec.resistance)
    ) / max(float(reference_resistance), EPS)

    reduction_vs_restart_initial = (
        float(restart_initial_resistance) - float(rec.resistance)
    ) / max(float(restart_initial_resistance), EPS)


    return {

        "restart_id": int(restart_id),

        "step": int(step),

        "valid_rank": "" if valid_rank is None else int(valid_rank),

        "target_rank": "" if target_rank is None else int(target_rank),

        "L": float(case.L),
        "B": float(case.B),
        "D": float(case.D),
        "T": float(case.T),
        "CB": float(case.CB),
        "V": float(case.V),

        "R0_reference": float(reference_resistance),

        "required_resistance_threshold": float(required_resistance),

        "search_start_z": start_z.tolist(),
        "search_start_z1": float(start_z[0]),
        "search_start_z2": float(start_z[1]),
        "search_start_z3": float(start_z[2]),

        "z": current_z.tolist(),
        "z1": float(current_z[0]),
        "z2": float(current_z[1]),
        "z3": float(current_z[2]),

        "R": float(rec.resistance),
        "resistance": float(rec.resistance),

        "restart_initial_predicted_resistance": float(restart_initial_resistance),

        "resistance_ratio_to_reference_R0": float(rec.resistance) / max(float(reference_resistance), EPS),

        "resistance_ratio_to_restart_initial": float(rec.resistance) / max(float(restart_initial_resistance), EPS),

        "resistance_reduction_vs_reference_R0": float(reduction_vs_reference),

        "resistance_reduction_vs_restart_initial": float(reduction_vs_restart_initial),

        "better_than_reference_R0": bool(better_than_reference),

        "meets_target": bool(meets_target),

        "cb_estimated": float(rec.cb_estimated),

        "volume_actual": float(rec.volume_actual),

        "volume_target": float(rec.volume_target),

        "volume_error": float(rec.volume_error),

        "abs_volume_error": float(abs(rec.volume_error)),

        "geometry_penalty": float(rec.geometry_penalty),

        "constraint_ok": bool(rec.constraint_ok),

        "is_half_hull": bool(getattr(rec, "is_half_hull", True)),

        "reward": "" if reward is None else float(reward),

        "point_cloud_path": str(point_cloud_path),
    }


def write_rows(path: Path, rows: Sequence[dict], fieldnames: Optional[Sequence[str]] = None) -> None:


    if fieldnames is None:

        keys: List[str] = []

        for row in rows:

            for key in row.keys():

                if key not in keys:

                    keys.append(key)

        fieldnames = keys

    if not fieldnames:

        raise ValueError(f"cant  CSV：{path}")

    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="", encoding="utf-8") as file_obj:

        writer = csv.DictWriter(file_obj, fieldnames=list(fieldnames))

        writer.writeheader()

        for row in rows:

            clean: dict = {}

            for key, value in row.items():

                if isinstance(value, (list, tuple, np.ndarray)):

                    clean[key] = json.dumps(np.asarray(value).tolist(), ensure_ascii=False)
                else:

                    clean[key] = value

            writer.writerow({key: clean.get(key, "") for key in fieldnames})


def save_candidate_pointcloud(
    out_dir: Path,
    rec,
    restart_id: int,
    step: int,
    category: str,
) -> str:
    """保存指定候选的半船点云并返回文件路径。"""


    path = out_dir / (
        f"{category}_restart_{int(restart_id):03d}_step_{int(step):03d}_half_hull_pointcloud.csv"
    )

    np.savetxt(path, np.asarray(rec.points), delimiter=",", fmt="%.8f")

    return str(path)


def record_candidate(
    *,
    out_dir: Path,
    all_rows: List[dict],
    valid_rows: List[dict],
    target_rows: List[dict],
    restart_id: int,
    step: int,
    rec,
    case: HullCase,
    search_start_z: np.ndarray,
    reference_resistance: float,
    restart_initial_resistance: float,
    min_improvement_ratio: float,
    reward: Optional[float],
    save_all_valid_pointclouds: bool,
    save_all_better_pointclouds: bool,
) -> dict:


    is_valid = bool(rec.constraint_ok)

    meets_target = is_target_candidate(
        rec=rec,
        target_resistance=float(reference_resistance),
        min_improvement_ratio=float(min_improvement_ratio),
    )

    valid_rank = len(valid_rows) + 1 if is_valid else None

    target_rank = len(target_rows) + 1 if meets_target else None

    point_cloud_path = ""


    if meets_target and bool(save_all_better_pointclouds):

        point_cloud_path = save_candidate_pointcloud(
            out_dir=out_dir,
            rec=rec,
            restart_id=restart_id,
            step=step,
            category="better_than_R0",
        )

    elif is_valid and bool(save_all_valid_pointclouds):

        point_cloud_path = save_candidate_pointcloud(
            out_dir=out_dir,
            rec=rec,
            restart_id=restart_id,
            step=step,
            category="valid",
        )


    row = candidate_row(
        restart_id=restart_id,
        step=step,
        rec=rec,
        case=case,
        search_start_z=search_start_z,
        reference_resistance=reference_resistance,
        restart_initial_resistance=restart_initial_resistance,
        min_improvement_ratio=min_improvement_ratio,
        reward=reward,
        valid_rank=valid_rank,
        target_rank=target_rank,
        point_cloud_path=point_cloud_path,
    )

    all_rows.append(row)

    if is_valid:

        valid_rows.append(dict(row))

    if meets_target:

        target_rows.append(dict(row))

    return row


def candidate_summary(rec, reference_resistance: float, point_cloud_path: str) -> dict:


    z_value = np.asarray(rec.z, dtype=np.float32).reshape(3)

    return {

        "optimal_z": [float(z_value[0]), float(z_value[1]), float(z_value[2])],

        "best_half_hull_pointcloud_csv": str(point_cloud_path),

        "cb_estimated_by_code1_function": float(rec.cb_estimated),

        "volume_actual": float(rec.volume_actual),

        "volume_target": float(rec.volume_target),

        "volume_error": float(rec.volume_error),

        "abs_volume_error": float(abs(rec.volume_error)),

        "resistance": float(rec.resistance),

        "relative_reference_resistance_reduction": float(
            (float(reference_resistance) - float(rec.resistance))
            / max(float(reference_resistance), EPS)
        ),

        "geometry_penalty": float(rec.geometry_penalty),

        "constraint_ok": bool(rec.constraint_ok),

        "is_half_hull": bool(getattr(rec, "is_half_hull", True)),
    }


def main() -> None:


    args = parse_args()

    validate_args(args)


    if getattr(args, "torch_num_threads", 0) and int(args.torch_num_threads) > 0:

        torch.set_num_threads(int(args.torch_num_threads))


    out_dir = ensure_dir(args.output_dir)

    env_cfg = env_config_from_args(args)

    env_cfg.episode_len = int(args.K)


    if not args.mock and (not args.generator_ckpt or not args.resistance_ckpt):

        raise ValueError("非 mock 模式必须提供 --generator_ckpt 和 --resistance_ckpt")


    generator = make_generator(
        args.code1_dir,
        args.generator_ckpt,
        device=args.device,
        mock=args.mock,
    )

    predictor = make_resistance_predictor(
        args.code2_dir,
        args.resistance_ckpt,
        device=args.device,
        mock=args.mock,
        code2_train_file=args.code2_train_file or None,
        amp=args.amp,
    )

    env = HullOptimizationEnv(
        generator,
        predictor,
        cfg=env_cfg,
        seed=args.seed,
    )

    policy = LoadedPolicy(args.policy_ckpt, device=args.device)


    if int(policy.state_dim) != int(env_cfg.state_dim):

        raise ValueError(
            f" error"
        )

    if int(policy.action_dim) != int(env_cfg.action_dim):

        raise ValueError(
            f"error"
        )


    initial_latents = build_initial_latents(args=args, env=env)

    np.savetxt(
        out_dir / "sampled_initial_latents.csv",
        initial_latents,
        delimiter=",",
        header="z1,z2,z3",
        comments="",
        fmt="%.8f",
    )


    required_resistance = float(args.R0) * (1.0 - float(args.min_improvement_ratio))

    all_rows: List[dict] = []

    valid_rows: List[dict] = []

    target_rows: List[dict] = []

    restart_rows: List[dict] = []


    best_target_rec = None

    best_target_meta: Optional[Tuple[int, int]] = None

    best_valid_rec = None

    best_valid_meta: Optional[Tuple[int, int]] = None

    best_any_rec = None

    best_any_meta: Optional[Tuple[int, int]] = None

    start_time = time.time()

    for restart_index, start_z in enumerate(initial_latents, start=1):

        case = HullCase(
            L=float(args.L),
            B=float(args.B),
            D=float(args.D),
            T=float(args.T),
            CB=float(args.CB),
            V=float(args.V),
            z0=np.asarray(start_z, dtype=np.float32),
        )


        state = env.reset(case=case)

        initial_rec = env.initial

        if initial_rec is None:

            raise RuntimeError(f"{restart_index}")

        restart_initial_resistance = float(env.R0)


        record_candidate(
            out_dir=out_dir,
            all_rows=all_rows,
            valid_rows=valid_rows,
            target_rows=target_rows,
            restart_id=restart_index,
            step=0,
            rec=initial_rec,
            case=case,
            search_start_z=start_z,
            reference_resistance=float(args.R0),
            restart_initial_resistance=restart_initial_resistance,
            min_improvement_ratio=float(args.min_improvement_ratio),
            reward=None,
            save_all_valid_pointclouds=bool(args.save_all_valid_pointclouds),
            save_all_better_pointclouds=bool(args.save_all_better_pointclouds),
        )


        if best_any_rec is None or float(initial_rec.resistance) < float(best_any_rec.resistance):

            best_any_rec = initial_rec

            best_any_meta = (restart_index, 0)

        if initial_rec.constraint_ok and (
            best_valid_rec is None or float(initial_rec.resistance) < float(best_valid_rec.resistance)
        ):

            best_valid_rec = initial_rec

            best_valid_meta = (restart_index, 0)

        if is_target_candidate(
            rec=initial_rec,
            target_resistance=float(args.R0),
            min_improvement_ratio=float(args.min_improvement_ratio),
        ) and (
            best_target_rec is None or float(initial_rec.resistance) < float(best_target_rec.resistance)
        ):

            best_target_rec = initial_rec

            best_target_meta = (restart_index, 0)


        done = False

        step = 0

        restart_best_valid_resistance = (
            float(initial_rec.resistance) if initial_rec.constraint_ok else float("inf")
        )

        restart_target_count_before = len(target_rows)


        while not done and step < int(args.K):

            action = policy.act(
                state,
                deterministic=not bool(args.stochastic_policy),
            )

            state, reward, done, info = env.step(action)

            step += 1

            rec = env.current

            if rec is None:

                raise RuntimeError(f"error")


            record_candidate(
                out_dir=out_dir,
                all_rows=all_rows,
                valid_rows=valid_rows,
                target_rows=target_rows,
                restart_id=restart_index,
                step=step,
                rec=rec,
                case=case,
                search_start_z=start_z,
                reference_resistance=float(args.R0),
                restart_initial_resistance=restart_initial_resistance,
                min_improvement_ratio=float(args.min_improvement_ratio),
                reward=float(reward),
                save_all_valid_pointclouds=bool(args.save_all_valid_pointclouds),
                save_all_better_pointclouds=bool(args.save_all_better_pointclouds),
            )


            if best_any_rec is None or float(rec.resistance) < float(best_any_rec.resistance):

                best_any_rec = rec

                best_any_meta = (restart_index, step)

            if rec.constraint_ok:

                restart_best_valid_resistance = min(
                    restart_best_valid_resistance,
                    float(rec.resistance),
                )

                if best_valid_rec is None or float(rec.resistance) < float(best_valid_rec.resistance):

                    best_valid_rec = rec

                    best_valid_meta = (restart_index, step)

            if is_target_candidate(
                rec=rec,
                target_resistance=float(args.R0),
                min_improvement_ratio=float(args.min_improvement_ratio),
            ) and (
                best_target_rec is None or float(rec.resistance) < float(best_target_rec.resistance)
            ):

                best_target_rec = rec

                best_target_meta = (restart_index, step)


        restart_target_count = len(target_rows) - restart_target_count_before

        restart_best_valid_output = (
            None if not np.isfinite(restart_best_valid_resistance) else float(restart_best_valid_resistance)
        )

        restart_rows.append(
            {
                "restart_id": int(restart_index),
                "start_z": np.asarray(start_z, dtype=np.float32).tolist(),
                "start_z1": float(start_z[0]),
                "start_z2": float(start_z[1]),
                "start_z3": float(start_z[2]),
                "restart_initial_predicted_resistance": float(restart_initial_resistance),
                "steps_executed": int(step),
                "num_all_candidates": int(len(env.all_candidates)),
                "num_valid_candidates": int(len(env.valid_candidates)),
                "num_target_candidates": int(restart_target_count),
                "best_valid_resistance": restart_best_valid_output,
                "target_achieved": bool(restart_target_count > 0),
            }
        )


    fieldnames = list(all_rows[0].keys())

    write_rows(out_dir / "all_candidates.csv", all_rows, fieldnames=fieldnames)

    write_rows(out_dir / "valid_candidates.csv", valid_rows, fieldnames=fieldnames)

    write_rows(out_dir / "better_than_R0_candidates.csv", target_rows, fieldnames=fieldnames)

    write_rows(out_dir / "restart_summary.csv", restart_rows)


    target_achieved = best_target_rec is not None

    best_target_output = None

    best_available_output = None


    if best_target_rec is not None:
        suc =generator.Point2Hull(float(args.L),
                                  float(args.D),
                                  float(args.T),
                                  float(args.B)/2,
                                  float(args.CB),
                                  best_target_rec.z,
                                  20,
                                  50,
                                  args.output_dir,
                                  "/Shipresult.stl",
                                  "/Shipresult_deck.stl",
                                  "/Shipresult_PC.csv")
        end_time = time.time()
        duration = end_time - start_time
        print('time of code：', duration, 's')
        print('Cb='+str(best_target_rec.cb_estimated))
        with open(args.output_dir+"/execution_time.csv", 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['duration_seconds','CB'])
            writer.writerow([duration,best_target_rec.cb_estimated])

        best_target_pc_path = out_dir / "best_better_than_R0_half_hull_pointcloud.csv"

        np.savetxt(
            best_target_pc_path,
            np.asarray(best_target_rec.points),
            delimiter=",",
            fmt="%.8f",
        )

        best_target_output = candidate_summary(
            rec=best_target_rec,
            reference_resistance=float(args.R0),
            point_cloud_path=str(best_target_pc_path),
        )

        best_target_output["restart_id"] = int(best_target_meta[0]) if best_target_meta else None
        best_target_output["step"] = int(best_target_meta[1]) if best_target_meta else None

        best_target_output["meets_target"] = True





    diagnostic_rec = best_valid_rec if best_valid_rec is not None else best_any_rec

    diagnostic_meta = best_valid_meta if best_valid_rec is not None else best_any_meta

    if diagnostic_rec is not None:

        best_available_pc_path = out_dir / "best_available_diagnostic_half_hull_pointcloud.csv"

        np.savetxt(
            best_available_pc_path,
            np.asarray(diagnostic_rec.points),
            delimiter=",",
            fmt="%.8f",
        )

        best_available_output = candidate_summary(
            rec=diagnostic_rec,
            reference_resistance=float(args.R0),
            point_cloud_path=str(best_available_pc_path),
        )

        best_available_output["restart_id"] = int(diagnostic_meta[0]) if diagnostic_meta else None
        best_available_output["step"] = int(diagnostic_meta[1]) if diagnostic_meta else None

        best_available_output["meets_target"] = bool(
            diagnostic_rec.constraint_ok
            and float(diagnostic_rec.resistance) < float(required_resistance)
        )

        best_available_output["selected_from_valid_candidates"] = bool(best_valid_rec is not None)


    summary = {

        "input": {
            "L": float(args.L),
            "B": float(args.B),
            "D": float(args.D),
            "T": float(args.T),
            "C_B": float(args.CB),
            "V": float(args.V),
            "R0_reference": float(args.R0),
        },

        "algorithm": policy.algo,

        "search": {
            "K_per_restart": int(args.K),
            "num_restarts": int(args.num_restarts),
            "init_sampling": str(args.init_sampling),
            "z_init_low": float(args.z_init_low),
            "z_init_high": float(args.z_init_high),
            "zero_start_included": bool(not args.no_zero_start and args.z_init_low <= 0.0 <= args.z_init_high),
            "deterministic_policy": bool(not args.stochastic_policy),
            "min_improvement_ratio": float(args.min_improvement_ratio),
            "required_resistance_threshold": float(required_resistance),
        },

        "policy_state_normalization": (  ),

        "point_cloud_logic": (        ),

        "target_achieved": bool(target_achieved),

        "counts": {
            "all_candidates": int(len(all_rows)),
            "valid_candidates": int(len(valid_rows)),
            "better_than_R0_candidates": int(len(target_rows)),
        },

        "best_target_output": best_target_output,

        "best_available_diagnostic": best_available_output,

        "files": {
            "sampled_initial_latents_csv": str(out_dir / "sampled_initial_latents.csv"),
            "all_candidates_csv": str(out_dir / "all_candidates.csv"),
            "valid_candidates_csv": str(out_dir / "valid_candidates.csv"),
            "better_than_R0_candidates_csv": str(out_dir / "better_than_R0_candidates.csv"),
            "restart_summary_csv": str(out_dir / "restart_summary.csv"),
            "best_target_pointcloud_csv": (
                str(out_dir / "best_better_than_R0_half_hull_pointcloud.csv")
                if target_achieved
                else None
            ),
            "best_available_diagnostic_pointcloud_csv": (
                str(out_dir / "best_available_diagnostic_half_hull_pointcloud.csv")
                if best_available_output is not None
                else None
            ),
        },
    }


    with (out_dir / "inference_summary.json").open("w", encoding="utf-8") as file_obj:

        json.dump(summary, file_obj, ensure_ascii=False, indent=2)


    if target_achieved:

        best_resistance = float(best_target_rec.resistance)

        reduction = (float(args.R0) - best_resistance) / max(float(args.R0), EPS)


    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
