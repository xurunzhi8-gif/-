from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
import yaml

from src.dataset import build_dataloader
from src.metrics import evaluate_predictions, save_predictions
from src.model import ModelConfig, NitrogenNet
from src.scaler import TargetScaler


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Cross-device testing.")
    parser.add_argument("--config", required=True, help="Path to config.yaml")
    parser.add_argument("--source", required=True, choices=["iphone", "nova"])
    parser.add_argument("--target", required=True, choices=["iphone", "nova"])
    parser.add_argument("--run-tag", default="base")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    out_root = _project_root() / cfg["output"]["root"]
    out_models = out_root / "models"
    out_preds = out_root / "preds"
    _ensure_dir(out_preds)

    model_path = out_models / f"best_{args.source}_{args.run_tag}.pt"
    scaler_path = out_models / f"scaler_{args.source}_{args.run_tag}.json"
    if not model_path.exists():
        raise FileNotFoundError(f"Missing model: {model_path}")

    scaler = TargetScaler(cfg["targets"])
    scaler.load(scaler_path)

    model_cfg = ModelConfig(
        backbone=cfg["model"]["backbone"],
        use_fpn=cfg["model"]["use_fpn"],
        use_film=cfg["model"]["use_film"],
        use_task_decouple=cfg["model"]["use_task_decouple"],
        use_spatial_head=cfg["model"]["use_spatial_head"],
    )
    model = NitrogenNet(model_cfg, num_targets=len(cfg["targets"]))
    bundle = torch.load(model_path, map_location="cpu")
    model.load_state_dict(bundle["model_state"])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    data_root = out_root / "data"
    test_csv = data_root / f"test_target_{args.target}.csv"
    if not test_csv.exists():
        raise FileNotFoundError(f"Missing test CSV: {test_csv}")

    test_df = pd.read_csv(test_csv)
    test_loader = build_dataloader(test_df, cfg, scaler, training=False)

    preds, trues, masks, meta = evaluate_predictions(model, test_loader, device, scaler)
    out_file = out_preds / f"preds_{args.source}_to_{args.target}_{args.run_tag}.csv"
    save_predictions(out_file, preds, trues, masks, meta, cfg["targets"])


if __name__ == "__main__":
    main()
