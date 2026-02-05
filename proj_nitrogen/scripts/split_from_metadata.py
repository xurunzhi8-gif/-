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


def _build_strata(df: pd.DataFrame) -> np.ndarray:
    return (
        df["variety"].astype(str)
        + "_"
        + df["stage_en"].astype(str)
        + "_"
        + df["angle"].astype(str)
    ).to_numpy()


def _stratified_split_by_image_id(
    df: pd.DataFrame,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
):
    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        raise ValueError("train/val/test ratios must sum to 1.0")

    rng = np.random.default_rng(seed)
    unique = df[["image_id", "variety", "stage_en", "angle"]].drop_duplicates("image_id").copy()
    unique["strata"] = (
        unique["variety"].astype(str)
        + "_"
        + unique["stage_en"].astype(str)
        + "_"
        + unique["angle"].astype(str)
    )

    train_ids: list[str] = []
    val_ids: list[str] = []
    test_ids: list[str] = []

    for strata in unique["strata"].unique():
        ids = unique.loc[unique["strata"] == strata, "image_id"].to_numpy()
        rng.shuffle(ids)
        n_train = max(1, int(len(ids) * train_ratio)) if len(ids) > 1 else len(ids)
        remaining = ids[n_train:]
        n_val = max(1, int(len(ids) * val_ratio)) if len(remaining) > 1 else 0
        val_ids.extend(remaining[:n_val])
        test_ids.extend(remaining[n_val:])
        train_ids.extend(ids[:n_train])

    train_df = df[df["image_id"].isin(train_ids)].reset_index(drop=True)
    val_df = df[df["image_id"].isin(val_ids)].reset_index(drop=True)
    test_df = df[df["image_id"].isin(test_ids)].reset_index(drop=True)
    return train_df, val_df, test_df


def _print_split_stats(name: str, df: pd.DataFrame) -> None:
    strata = _build_strata(df)
    print(f"{name}: samples={len(df)} strata_unique={len(np.unique(strata))}")


def _check_no_overlap(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    train_ids = set(train_df["image_id"].astype(str))
    val_ids = set(val_df["image_id"].astype(str))
    test_ids = set(test_df["image_id"].astype(str))
    overlap_tv = train_ids.intersection(val_ids)
    overlap_tt = train_ids.intersection(test_ids)
    overlap_vt = val_ids.intersection(test_ids)
    print(f"overlap train/val: {len(overlap_tv)}")
    print(f"overlap train/test: {len(overlap_tt)}")
    print(f"overlap val/test: {len(overlap_vt)}")
    if overlap_tv or overlap_tt or overlap_vt:
        raise ValueError("image_id overlap detected across splits.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Split metadata by device with stratified sampling.")
    parser.add_argument("--config", required=True, help="Path to config.yaml")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--ratios",
        type=str,
        default="0.7,0.1,0.2",
        help="Train/val/test ratios, e.g. 0.7,0.1,0.2",
    )
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    metadata_path = Path(cfg["data"]["metadata_full"])
    if not metadata_path.exists():
        raise FileNotFoundError(f"metadata_full.csv not found: {metadata_path}")

    out_root = _project_root() / cfg["output"]["root"] / "data"
    _ensure_dir(out_root)

    data = pd.read_csv(metadata_path)
    required = {
        "image_id",
        "image_path",
        "device",
        "angle",
        "stage_en",
        "variety",
        "SNC_pct",
        "shootN_gm2",
        "leafN_gm2",
    }
    if not required.issubset(data.columns):
        missing = ", ".join(sorted(required - set(data.columns)))
        raise ValueError(f"metadata_full.csv missing columns: {missing}")

    ratios = [float(x) for x in args.ratios.split(",")]
    if len(ratios) != 3:
        raise ValueError("--ratios must provide three comma-separated values")
    train_ratio, val_ratio, test_ratio = ratios

    for device in ["iphone", "nova"]:
        subset = data[data["device"].astype(str).str.lower() == device].copy()
        if subset.empty:
            raise ValueError(f"No data found for device: {device}")

        train_df, val_df, test_df = _stratified_split_by_image_id(
            subset,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            seed=args.seed,
        )

        (out_root / f"train_full_{device}.csv").write_text(train_df.to_csv(index=False), encoding="utf-8")
        (out_root / f"val_full_{device}.csv").write_text(val_df.to_csv(index=False), encoding="utf-8")
        (out_root / f"test_full_{device}.csv").write_text(test_df.to_csv(index=False), encoding="utf-8")

        print(f"Device: {device}")
        _print_split_stats("train", train_df)
        _print_split_stats("val", val_df)
        _print_split_stats("test", test_df)
        _check_no_overlap(train_df, val_df, test_df)

    train_iphone = pd.read_csv(out_root / "train_full_iphone.csv")
    train_nova = pd.read_csv(out_root / "train_full_nova.csv")
    test_iphone = pd.read_csv(out_root / "test_full_iphone.csv")
    test_nova = pd.read_csv(out_root / "test_full_nova.csv")

    (out_root / "train_source_iphone.csv").write_text(train_iphone.to_csv(index=False), encoding="utf-8")
    (out_root / "test_target_nova.csv").write_text(test_nova.to_csv(index=False), encoding="utf-8")

    (out_root / "train_source_nova.csv").write_text(train_nova.to_csv(index=False), encoding="utf-8")
    (out_root / "test_target_iphone.csv").write_text(test_iphone.to_csv(index=False), encoding="utf-8")


if __name__ == "__main__":
    main()
