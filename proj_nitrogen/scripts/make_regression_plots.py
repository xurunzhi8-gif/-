from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
import matplotlib.pyplot as plt


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _regression_stats(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    slope, intercept = np.polyfit(y_true, y_pred, 1)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    mae = np.mean(np.abs(y_true - y_pred))
    return {
        "slope": float(slope),
        "intercept": float(intercept),
        "r2": float(r2),
        "rmse": float(rmse),
        "mae": float(mae),
        "n": int(len(y_true)),
    }


def _plot_regression(y_true: np.ndarray, y_pred: np.ndarray, title: str, out_png: Path, out_pdf: Path) -> dict:
    stats = _regression_stats(y_true, y_pred)
    plt.figure(figsize=(5, 5))
    plt.scatter(y_true, y_pred, s=8, alpha=0.6)
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], "k--", linewidth=1)
    x = np.array([min_val, max_val])
    y = stats["slope"] * x + stats["intercept"]
    plt.plot(x, y, "r-", linewidth=1)
    plt.xlabel("True")
    plt.ylabel("Pred")
    plt.title(title)
    text = (
        f"slope={stats['slope']:.3f}\n"
        f"intercept={stats['intercept']:.3f}\n"
        f"R2={stats['r2']:.3f}\n"
        f"RMSE={stats['rmse']:.3f}\n"
        f"MAE={stats['mae']:.3f}\n"
        f"N={stats['n']}"
    )
    plt.text(0.05, 0.95, text, transform=plt.gca().transAxes, va="top")
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.savefig(out_pdf)
    plt.close()
    return stats


def _collect_and_plot(df: pd.DataFrame, target: str, label: str, out_dir: Path) -> dict:
    y_true = df[f"{target}_true"].to_numpy()
    y_pred = df[f"{target}_pred"].to_numpy()
    mask = df[f"{target}_mask"].to_numpy().astype(bool)
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    if len(y_true) == 0:
        return {}
    out_png = out_dir / f"{label}_{target}.png"
    out_pdf = out_dir / f"{label}_{target}.pdf"
    stats = _plot_regression(y_true, y_pred, f"{label} - {target}", out_png, out_pdf)
    stats["label"] = label
    stats["target"] = target
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Make regression plots.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-tag", default="base")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    out_root = _project_root() / cfg["output"]["root"]
    preds_dir = out_root / "preds"
    figs_dir = out_root / "figs" / "regression"
    _ensure_dir(figs_dir)

    within_iphone = pd.read_csv(preds_dir / f"preds_within_iphone_{args.run_tag}.csv")
    within_nova = pd.read_csv(preds_dir / f"preds_within_nova_{args.run_tag}.csv")
    cross_i2n = pd.read_csv(preds_dir / f"preds_iphone_to_nova_{args.run_tag}.csv")
    cross_n2i = pd.read_csv(preds_dir / f"preds_nova_to_iphone_{args.run_tag}.csv")

    stats_rows = []
    for target in cfg["targets"]:
        for device, df in [("iphone", within_iphone), ("nova", within_nova)]:
            for angle in ["a", "b"]:
                subset = df[df["angle"].astype(str).str.lower() == angle]
                label = f"within_{device}_{angle}"
                stats = _collect_and_plot(subset, target, label, figs_dir)
                if stats:
                    stats_rows.append(stats)

        for label, df in [("iphone_to_nova", cross_i2n), ("nova_to_iphone", cross_n2i)]:
            for angle in ["a", "b"]:
                subset = df[df["angle"].astype(str).str.lower() == angle]
                stats = _collect_and_plot(subset, target, f"{label}_{angle}", figs_dir)
                if stats:
                    stats_rows.append(stats)

    stats_df = pd.DataFrame(stats_rows)
    stats_df.to_csv(out_root / "tables" / "Table_regression_stats.csv", index=False)


if __name__ == "__main__":
    main()
