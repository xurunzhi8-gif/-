from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from src.dataset import build_dataloader
from src.model import ModelConfig, NitrogenNet
from src.scaler import TargetScaler


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _corr_table(df: pd.DataFrame, label: str) -> pd.DataFrame:
    corr = df.corr()
    rows = []
    for i in corr.columns:
        for j in corr.columns:
            rows.append({"label": label, "var1": i, "var2": j, "corr": corr.loc[i, j]})
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze prediction correlations.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-tag", default="base")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    out_root = _project_root() / cfg["output"]["root"]
    preds_dir = out_root / "preds"
    tables_dir = out_root / "tables"
    _ensure_dir(tables_dir)

    within_iphone = pd.read_csv(preds_dir / f"preds_within_iphone_{args.run_tag}.csv")
    within_nova = pd.read_csv(preds_dir / f"preds_within_nova_{args.run_tag}.csv")
    combined = pd.concat([within_iphone, within_nova], ignore_index=True)

    pred_cols = [f"{t}_pred" for t in cfg["targets"]]
    true_cols = [f"{t}_true" for t in cfg["targets"]]

    pred_corr = _corr_table(combined[pred_cols], "pred")
    err_df = combined[pred_cols].to_numpy() - combined[true_cols].to_numpy()
    err_df = pd.DataFrame(err_df, columns=pred_cols)
    err_corr = _corr_table(err_df, "error")

    pd.concat([pred_corr, err_corr], ignore_index=True).to_csv(
        tables_dir / "Table_prediction_correlation.csv", index=False
    )

    # Feature similarity
    data_root = out_root / "data"
    val_df = pd.read_csv(data_root / "val_full_iphone.csv")
    scaler = TargetScaler(cfg["targets"])
    scaler.fit(val_df)
    loader = build_dataloader(val_df, cfg, scaler, training=False)

    model_cfg = ModelConfig(
        backbone=cfg["model"]["backbone"],
        use_fpn=cfg["model"]["use_fpn"],
        use_film=cfg["model"]["use_film"],
        use_task_decouple=cfg["model"]["use_task_decouple"],
        use_spatial_head=cfg["model"]["use_spatial_head"],
    )
    model = NitrogenNet(model_cfg, num_targets=len(cfg["targets"]))
    model_path = out_root / "models" / f"best_iphone_{args.run_tag}.pt"
    bundle = torch.load(model_path, map_location="cpu")
    model.load_state_dict(bundle["model_state"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    sims = []
    for images, angle_id, _, _, _ in loader:
        images = images.to(device)
        angle_id = angle_id.to(device)
        with torch.no_grad():
            _, features = model(images, angle_id, return_features=True)
        if features is None:
            continue
        feats = [f.detach().cpu().numpy() for f in features]
        if len(feats) < 3:
            continue
        f1, f2, f3 = feats
        def cos(a, b):
            a_n = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-8)
            b_n = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-8)
            return np.mean(np.sum(a_n * b_n, axis=1))
        sims.append({
            "pair": "SNC_shootN",
            "cosine": cos(f1, f2),
        })
        sims.append({
            "pair": "SNC_leafN",
            "cosine": cos(f1, f3),
        })
        sims.append({
            "pair": "shootN_leafN",
            "cosine": cos(f2, f3),
        })
        break

    pd.DataFrame(sims).to_csv(tables_dir / "Table_feature_similarity.csv", index=False)


if __name__ == "__main__":
    main()
