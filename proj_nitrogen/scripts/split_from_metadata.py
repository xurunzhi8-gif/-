from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _stratified_split_by_image_id(
    df: pd.DataFrame,
    strata_cols: list[str],
    val_ratio: float,
    test_ratio: float,
    seed: int,
):
    rng = np.random.default_rng(seed)
    strata = df[strata_cols[0]].astype(str)
    for col in strata_cols[1:]:
        strata = strata + "_" + df[col].astype(str)

    unique = df[["image_id"] + strata_cols].drop_duplicates("image_id").copy()
    unique["strata"] = unique[strata_cols[0]].astype(str)
    for col in strata_cols[1:]:
        unique["strata"] = unique["strata"] + "_" + unique[col].astype(str)

    train_ids = []
    val_ids = []
    test_ids = []
    for s in unique["strata"].unique():
        ids = unique.loc[unique["strata"] == s, "image_id"].to_numpy()
        rng.shuffle(ids)
        n_val = max(1, int(len(ids) * val_ratio)) if len(ids) > 1 else 0
        remaining = ids[n_val:]
        n_test = max(1, int(len(remaining) * test_ratio)) if len(remaining) > 1 else 0
        val_ids.extend(ids[:n_val])
        test_ids.extend(remaining[:n_test])
        train_ids.extend(remaining[n_test:])

    train_df = df[df["image_id"].isin(train_ids)].reset_index(drop=True)
    val_df = df[df["image_id"].isin(val_ids)].reset_index(drop=True)
    test_df = df[df["image_id"].isin(test_ids)].reset_index(drop=True)
    return train_df, val_df, test_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Split metadata by device with stratified sampling.")
    parser.add_argument("--config", required=True, help="Path to config.yaml")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    metadata_path = Path(cfg["data"]["metadata_full"])
    if not metadata_path.exists():
        raise FileNotFoundError(f"metadata_full.csv not found: {metadata_path}")

    out_root = _project_root() / cfg["output"]["root"] / "data"
    _ensure_dir(out_root)

    data = pd.read_csv(metadata_path)
    required = {"image_id", "image_path", "device", "angle", "stage_en", "variety"}
    if not required.issubset(data.columns):
        missing = ", ".join(sorted(required - set(data.columns)))
        raise ValueError(f"metadata_full.csv missing columns: {missing}")

    for device in ["iphone", "nova"]:
        subset = data[data["device"].astype(str).str.lower() == device].copy()
        if subset.empty:
            raise ValueError(f"No data found for device: {device}")

        train_df, val_df, test_df = _stratified_split_by_image_id(
            subset,
            strata_cols=["variety", "angle"],
            val_ratio=0.2,
            test_ratio=cfg["train"]["test_ratio"],
            seed=cfg["train"]["seed"],
        )

        (out_root / f"train_full_{device}.csv").write_text(train_df.to_csv(index=False), encoding="utf-8")
        (out_root / f"val_full_{device}.csv").write_text(val_df.to_csv(index=False), encoding="utf-8")
        (out_root / f"test_full_{device}.csv").write_text(test_df.to_csv(index=False), encoding="utf-8")
        (out_root / f"all_full_{device}.csv").write_text(subset.to_csv(index=False), encoding="utf-8")

    train_iphone = pd.read_csv(out_root / "train_full_iphone.csv")
    train_nova = pd.read_csv(out_root / "train_full_nova.csv")
    all_iphone = pd.read_csv(out_root / "all_full_iphone.csv")
    all_nova = pd.read_csv(out_root / "all_full_nova.csv")

    (out_root / "train_source_iphone.csv").write_text(train_iphone.to_csv(index=False), encoding="utf-8")
    (out_root / "test_target_nova.csv").write_text(all_nova.to_csv(index=False), encoding="utf-8")

    (out_root / "train_source_nova.csv").write_text(train_nova.to_csv(index=False), encoding="utf-8")
    (out_root / "test_target_iphone.csv").write_text(all_iphone.to_csv(index=False), encoding="utf-8")


if __name__ == "__main__":
    main()
