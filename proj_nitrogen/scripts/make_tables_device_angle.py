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


def _group_metrics(df: pd.DataFrame, group_cols: list[str], targets: list[str]) -> pd.DataFrame:
    rows = []
    for keys, group in df.groupby(group_cols):
        y_true = group[[f"{t}_true" for t in targets]].to_numpy()
        y_pred = group[[f"{t}_pred" for t in targets]].to_numpy()
        mask = group[[f"{t}_mask" for t in targets]].to_numpy()
        metrics = compute_metrics(y_true, y_pred, mask)
        row = {}
        if isinstance(keys, tuple):
            for key, value in zip(group_cols, keys):
                row[key] = value
        else:
            row[group_cols[0]] = keys
        for t, m in zip(targets, metrics):
            row[f"{t}_r2"] = m["r2"]
            row[f"{t}_rmse"] = m["rmse"]
            row[f"{t}_mae"] = m["mae"]
        row["n"] = len(group)
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Make device/angle/stage/variety tables.")
    parser.add_argument("--config", required=True, help="Path to config.yaml")
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
    cross_i2n = pd.read_csv(preds_dir / f"preds_iphone_to_nova_{args.run_tag}.csv")
    cross_n2i = pd.read_csv(preds_dir / f"preds_nova_to_iphone_{args.run_tag}.csv")

    within_iphone["split"] = "within_iphone"
    within_nova["split"] = "within_nova"
    cross_i2n["split"] = "iphone_to_nova"
    cross_n2i["split"] = "nova_to_iphone"

    within = pd.concat([within_iphone, within_nova], ignore_index=True)
    cross = pd.concat([cross_i2n, cross_n2i], ignore_index=True)

    table_device_angle = _group_metrics(within, ["device", "angle"], cfg["targets"])
    table_device_angle.to_csv(tables_dir / "Table_device_angle.csv", index=False)

    table_transfer_angle = _group_metrics(cross, ["split", "angle"], cfg["targets"])
    table_transfer_angle.to_csv(tables_dir / "Table_transfer_angle.csv", index=False)

    table_transfer_stage = _group_metrics(cross, ["split", "stage_en"], cfg["targets"])
    table_transfer_stage.to_csv(tables_dir / "Table_transfer_stage.csv", index=False)

    table_transfer_variety = _group_metrics(cross, ["split", "variety"], cfg["targets"])
    table_transfer_variety.to_csv(tables_dir / "Table_transfer_variety.csv", index=False)


if __name__ == "__main__":
    main()
