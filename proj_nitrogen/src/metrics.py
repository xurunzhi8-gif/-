from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, mask: np.ndarray) -> list[dict]:
    results = []
    for i in range(y_true.shape[1]):
        m = mask[:, i].astype(bool)
        if m.sum() == 0:
            results.append({"r2": np.nan, "rmse": np.nan, "mae": np.nan})
            continue
        t = y_true[m, i]
        p = y_pred[m, i]
        ss_res = np.sum((t - p) ** 2)
        ss_tot = np.sum((t - np.mean(t)) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
        rmse = np.sqrt(np.mean((t - p) ** 2))
        mae = np.mean(np.abs(t - p))
        results.append({"r2": float(r2), "rmse": float(rmse), "mae": float(mae)})
    return results


def evaluate_predictions(
    model,
    loader,
    device: str,
    scaler,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict]]:
    preds_all = []
    trues_all = []
    masks_all = []
    meta_all = []
    model.eval()
    for images, angle_id, targets, mask, meta in loader:
        images = images.to(device)
        angle_id = angle_id.to(device)
        with torch.no_grad():
            pred, _ = model(images, angle_id)
        pred = scaler.inverse(pred.cpu().numpy())
        true = scaler.inverse(targets.numpy())
        preds_all.append(pred)
        trues_all.append(true)
        masks_all.append(mask.numpy())
        meta_all.extend(meta)
    return (
        np.concatenate(preds_all, axis=0),
        np.concatenate(trues_all, axis=0),
        np.concatenate(masks_all, axis=0),
        meta_all,
    )


def save_predictions(
    path: Path,
    preds: np.ndarray,
    trues: np.ndarray,
    masks: np.ndarray,
    meta: list[dict],
    targets: list[str],
) -> None:
    rows = []
    for i in range(len(preds)):
        row = dict(meta[i])
        for j, t in enumerate(targets):
            row[f"{t}_true"] = trues[i, j]
            row[f"{t}_pred"] = preds[i, j]
            row[f"{t}_mask"] = masks[i, j]
        rows.append(row)
    pd.DataFrame(rows).to_csv(path, index=False)


def summarize_metrics(csv_path: Path, targets: list[str]) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    y_true = df[[f"{t}_true" for t in targets]].to_numpy()
    y_pred = df[[f"{t}_pred" for t in targets]].to_numpy()
    mask = df[[f"{t}_mask" for t in targets]].to_numpy()
    metrics = compute_metrics(y_true, y_pred, mask)
    row = {}
    for t, m in zip(targets, metrics):
        row[f"{t}_r2"] = m["r2"]
        row[f"{t}_rmse"] = m["rmse"]
        row[f"{t}_mae"] = m["mae"]
    return pd.DataFrame([row])
