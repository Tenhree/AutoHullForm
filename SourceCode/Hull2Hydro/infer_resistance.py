# infer_resistance.py
import argparse
import csv
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import torch

from Hull2Hydro import (
    HullResistancePredictor,
    ZScoreNormalizer,
    read_point_cloud_csv,
    log1p_target_to_raw_target,
    autocast_context,
    parse_serialization_orders,
    DEFAULT_HULL_PRIOR_SERIALIZATION_ORDERS,
)


TARGET_NAME = {
    0: "friction_resistance",
    1: "total_resistance",
    2: "wave_resistance",
}


def torch_load_checkpoint(checkpoint_path: str, device: torch.device) -> Dict[str, Any]:
    try:
        return torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(checkpoint_path, map_location=device)


def get_arg(saved_args: Dict[str, Any], name: str, default):
    return saved_args.get(name, default)


def build_model_from_checkpoint(checkpoint: Dict[str, Any], device: torch.device) -> HullResistancePredictor:
    saved_args = checkpoint.get("args", {})

    if "serialization_orders" in checkpoint:
        serialization_orders = tuple(checkpoint["serialization_orders"])
    else:
        order_str = get_arg(
            saved_args,
            "serialization_orders",
            ",".join(DEFAULT_HULL_PRIOR_SERIALIZATION_ORDERS),
        )
        serialization_orders = parse_serialization_orders(order_str)

    model = HullResistancePredictor(
        num_points=int(get_arg(saved_args, "num_points", 1000)),
        patch_size=int(get_arg(saved_args, "patch_size", 20)),
        embed_dim=int(get_arg(saved_args, "embed_dim", 256)),
        encoder_depth=int(get_arg(saved_args, "encoder_depth", 6)),
        num_heads=int(get_arg(saved_args, "num_heads", 8)),
        window_size=int(get_arg(saved_args, "window_size", 16)),
        dropout=float(get_arg(saved_args, "dropout", 0.0)),
        serialization_orders=serialization_orders,
        # 推理阶段建议关闭随机顺序；model.eval() 下本来也不会 shuffle，这里只是双保险。
        shuffle_orders=False,
        serialization_bits=int(get_arg(saved_args, "serialization_bits", 10)),
        xcpe_kernel_size=int(get_arg(saved_args, "xcpe_kernel_size", 3)),
        xcpe_grid_size=get_arg(saved_args, "xcpe_grid_size", None),
        xcpe_grid_bits=int(get_arg(saved_args, "xcpe_grid_bits", 10)),
        xcpe_sparse_padding=int(get_arg(saved_args, "xcpe_sparse_padding", 96)),
        condition_dim=2,
        condition_embed_dim=int(get_arg(saved_args, "condition_embed_dim", 64)),
    ).to(device)


    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    return model


def load_normalizers_from_checkpoint(checkpoint: Dict[str, Any]):
    target_transform = checkpoint.get("target_transform", None)
    condition_normalizer = ZScoreNormalizer.from_state_dict(checkpoint["condition_normalizer"])
    target_normalizer = ZScoreNormalizer.from_state_dict(checkpoint["target_normalizer"])
    return condition_normalizer, target_normalizer


@torch.no_grad()
def predict_one_case(
    model: HullResistancePredictor,
    checkpoint: Dict[str, Any],
    condition_normalizer: ZScoreNormalizer,
    target_normalizer: ZScoreNormalizer,
    points_csv: str,
    draft_ratio: float,
    speed: float,
    device: torch.device,
    amp: bool = False,
) -> Dict[str, float]:
    saved_args = checkpoint.get("args", {})
    num_points = int(get_arg(saved_args, "num_points", 1000))
    target_column = int(get_arg(saved_args, "target_column", 1))

    # 你的训练代码要求点云 CSV 形状为 [N, 3]，且 N 等于 num_points。
    points_np = read_point_cloud_csv(Path(points_csv), num_points=num_points)

    # 你的训练代码里工况顺序是 [draft_ratio, speed]。
    condition_raw = np.asarray([draft_ratio, speed], dtype=np.float32)
    condition_norm = condition_normalizer.transform(condition_raw)

    points = torch.from_numpy(points_np).float().unsqueeze(0).to(device)
    condition = torch.from_numpy(condition_norm).float().unsqueeze(0).to(device)

    amp_enabled = bool(amp and device.type == "cuda")
    with autocast_context(device, amp_enabled):
        pred_norm = model(points, condition)

    # 模型输出是归一化后的 log1p 阻力，需要先 inverse z-score，再 expm1 回到真实阻力。
    pred_norm_np = pred_norm.detach().float().cpu().numpy()
    pred_log_np = target_normalizer.inverse_transform(pred_norm_np)
    pred_raw_np = log1p_target_to_raw_target(pred_log_np)

    pred_value = float(pred_raw_np.reshape(-1)[0])
    target_name = TARGET_NAME.get(target_column, f"target_column_{target_column}")

    return {
        "draft_ratio": float(draft_ratio),
        "speed": float(speed),
        f"pred_{target_name}": pred_value,
    }


def predict_batch_from_cases_csv(
    model: HullResistancePredictor,
    checkpoint: Dict[str, Any],
    condition_normalizer: ZScoreNormalizer,
    target_normalizer: ZScoreNormalizer,
    cases_csv: str,
    output_csv: str,
    device: torch.device,
    amp: bool = False,
) -> None:
    rows_out: List[Dict[str, float]] = []

    with open(cases_csv, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        required_cols = {"points_csv", "draft_ratio", "speed"}
        missing = required_cols - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{cases_csv} missing: {sorted(missing)}. need: points_csv,draft_ratio,speed")

        for row in reader:
            result = predict_one_case(
                model=model,
                checkpoint=checkpoint,
                condition_normalizer=condition_normalizer,
                target_normalizer=target_normalizer,
                points_csv=row["points_csv"],
                draft_ratio=float(row["draft_ratio"]),
                speed=float(row["speed"]),
                device=device,
                amp=amp,
            )
            result["points_csv"] = row["points_csv"]
            rows_out.append(result)

    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = list(rows_out[0].keys()) if rows_out else ["points_csv", "draft_ratio", "speed"]
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_out)


def parse_args():
    parser = argparse.ArgumentParser(description="Inference for hull resistance predictor")

    parser.add_argument(
        "--checkpoint",
        type=str,

        required=True,
        help="checkpoint, like prediction_outputs/prediction_best.pth",
    )

    # 单个样本推理
    parser.add_argument("--points_csv", type=str, default="")
    parser.add_argument("--draft_ratio", type=float, default=None)
    parser.add_argument("--speed", type=float, default=None)

    # 批量推理
    parser.add_argument(
        "--cases_csv",
        type=str,
        default="",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="inference_predictions.csv",#Modify to the corresponding path.
    )

    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--amp", action="store_true")

    return parser.parse_args()


def main():
    args = parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    checkpoint = torch_load_checkpoint(args.checkpoint, device=device)
    model = build_model_from_checkpoint(checkpoint, device=device)
    condition_normalizer, target_normalizer = load_normalizers_from_checkpoint(checkpoint)

    if args.cases_csv:
        predict_batch_from_cases_csv(
            model=model,
            checkpoint=checkpoint,
            condition_normalizer=condition_normalizer,
            target_normalizer=target_normalizer,
            cases_csv=args.cases_csv,
            output_csv=args.output_csv,
            device=device,
            amp=args.amp,
        )
        print(f"result save to: {args.output_csv}")
        return

    if not args.points_csv:
        raise ValueError("need --points_csv，or --cases_csv。")
    if args.draft_ratio is None:
        raise ValueError("need --draft_ratio。")
    if args.speed is None:
        raise ValueError("need--speed。")

    result = predict_one_case(
        model=model,
        checkpoint=checkpoint,
        condition_normalizer=condition_normalizer,
        target_normalizer=target_normalizer,
        points_csv=args.points_csv,
        draft_ratio=args.draft_ratio,
        speed=args.speed,
        device=device,
        amp=args.amp,
    )

    print("result:")
    for k, v in result.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()