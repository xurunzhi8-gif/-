from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.metrics import compute_metrics


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _metrics_from_df(df: pd.DataFrame, targets: list[str]) -> list[dict]:
    y_true = df[[f"{t}_true" for t in targets]].to_numpy()
    y_pred = df[[f"{t}_pred" for t in targets]].to_numpy()
    mask = df[[f"{t}_mask" for t in targets]].to_numpy()
    return compute_metrics(y_true, y_pred, mask)


def _avg_metrics(metrics_list: list[list[dict]]) -> list[dict]:
    metrics_avg = []
    for idx in range(len(metrics_list[0])):
        r2 = np.nanmean([m[idx]["r2"] for m in metrics_list])
        rmse = np.nanmean([m[idx]["rmse"] for m in metrics_list])
        mae = np.nanmean([m[idx]["mae"] for m in metrics_list])
        metrics_avg.append({"r2": float(r2), "rmse": float(rmse), "mae": float(mae)})
    return metrics_avg


def _rows_for_variant(variant: str, preds_dir: Path, targets: list[str]) -> dict:
    wi = pd.read_csv(preds_dir / f"preds_within_iphone_{variant}.csv")
    wn = pd.read_csv(preds_dir / f"preds_within_nova_{variant}.csv")
    ci = pd.read_csv(preds_dir / f"preds_iphone_to_nova_{variant}.csv")
    cn = pd.read_csv(preds_dir / f"preds_nova_to_iphone_{variant}.csv")

    within_metrics = _avg_metrics([_metrics_from_df(wi, targets), _metrics_from_df(wn, targets)])
    cross_metrics = _avg_metrics([_metrics_from_df(ci, targets), _metrics_from_df(cn, targets)])

    row = {"variant": variant}
    for t, m in zip(targets, within_metrics):
        row[f"within_{t}_r2"] = m["r2"]
        row[f"within_{t}_rmse"] = m["rmse"]
        row[f"within_{t}_mae"] = m["mae"]
    for t, m in zip(targets, cross_metrics):
        row[f"cross_{t}_r2"] = m["r2"]
        row[f"cross_{t}_rmse"] = m["rmse"]
        row[f"cross_{t}_mae"] = m["mae"]
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description="Make paper tables.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    out_root = _project_root() / cfg["output"]["root"]
    preds_dir = out_root / "preds"
    tables_dir = out_root / "tables"
    _ensure_dir(tables_dir)

    # Overall within-device
    wi = pd.read_csv(preds_dir / "preds_within_iphone_base.csv")
    wn = pd.read_csv(preds_dir / "preds_within_nova_base.csv")
    rows = []
    for name, df in [("iphone", wi), ("nova", wn)]:
        metrics = _metrics_from_df(df, cfg["targets"])
        row = {"device": name}
        for t, m in zip(cfg["targets"], metrics):
            row[f"{t}_r2"] = m["r2"]
            row[f"{t}_rmse"] = m["rmse"]
            row[f"{t}_mae"] = m["mae"]
        rows.append(row)
    pd.DataFrame(rows).to_csv(tables_dir / "Table_overall_within_device.csv", index=False)

    # Cross-device
    ci = pd.read_csv(preds_dir / "preds_iphone_to_nova_base.csv")
    cn = pd.read_csv(preds_dir / "preds_nova_to_iphone_base.csv")
    rows = []
    for name, df in [("iphone_to_nova", ci), ("nova_to_iphone", cn)]:
        metrics = _metrics_from_df(df, cfg["targets"])
        row = {"transfer": name}
        for t, m in zip(cfg["targets"], metrics):
            row[f"{t}_r2"] = m["r2"]
            row[f"{t}_rmse"] = m["rmse"]
            row[f"{t}_mae"] = m["mae"]
        rows.append(row)
    pd.DataFrame(rows).to_csv(tables_dir / "Table_cross_device.csv", index=False)

    # Ablation
    ablation_rows = []
    for variant in cfg["experiments"]["ablations"]:
        ablation_rows.append(_rows_for_variant(variant, preds_dir, cfg["targets"]))
    pd.DataFrame(ablation_rows).to_csv(tables_dir / "Table_ablation.csv", index=False)

    # Backbone comparison
    backbone_rows = []
    for backbone in cfg["experiments"]["backbones"]:
        variant = f"backbone_{backbone}"
        backbone_rows.append(_rows_for_variant(variant, preds_dir, cfg["targets"]))
    pd.DataFrame(backbone_rows).to_csv(tables_dir / "Table_backbone_comparison.csv", index=False)


if __name__ == "__main__":
    main()
