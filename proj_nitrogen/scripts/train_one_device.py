from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from src.dataset import build_dataloader
from src.losses import MaskedSmoothL1, UncertaintyWeighting, consistency_loss
from src.metrics import compute_metrics, evaluate_predictions, save_predictions
from src.model import ModelConfig, NitrogenNet
from src.scaler import TargetScaler


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _set_seed(seed: int) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _run_tag(ablation: str | None, backbone: str | None) -> str:
    if ablation:
        return ablation
    if backbone:
        return f"backbone_{backbone}"
    return "base"


def _apply_ablation(cfg: dict, ablation: str | None) -> dict:
    cfg = json.loads(json.dumps(cfg))
    if not ablation:
        return cfg
    if ablation == "no_fpn":
        cfg["model"]["use_fpn"] = False
    elif ablation == "no_film":
        cfg["model"]["use_film"] = False
    elif ablation == "no_task_decouple":
        cfg["model"]["use_task_decouple"] = False
    elif ablation == "no_uncertainty":
        cfg["loss"]["use_uncertainty_weighting"] = False
    elif ablation == "no_consistency":
        cfg["loss"]["use_consistency_loss"] = False
    elif ablation == "baseline":
        pass
    else:
        raise ValueError(f"Unknown ablation: {ablation}")
    return cfg


def _estimate_flops(model: torch.nn.Module, image_size: int) -> float | None:
    try:
        from thop import profile

        dummy = torch.randn(1, 3, image_size, image_size)
        angle = torch.zeros(1, dtype=torch.long)
        flops, _ = profile(model, inputs=(dummy, angle), verbose=False)
        return float(flops)
    except Exception:
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Train AMDR-Net on one device.")
    parser.add_argument("--config", required=True, help="Path to config.yaml")
    parser.add_argument("--device", required=True, choices=["iphone", "nova"])
    parser.add_argument("--ablation", default=None)
    parser.add_argument("--backbone", default=None)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    cfg = _apply_ablation(cfg, args.ablation)
    if args.backbone:
        cfg["model"]["backbone"] = args.backbone

    run_tag = _run_tag(args.ablation, args.backbone)

    out_root = _project_root() / cfg["output"]["root"]
    out_models = out_root / "models"
    out_logs = out_root / "logs"
    out_preds = out_root / "preds"
    for path in [out_models, out_logs, out_preds]:
        _ensure_dir(path)

    data_root = out_root / "data"
    train_csv = data_root / f"train_full_{args.device}.csv"
    val_csv = data_root / f"val_full_{args.device}.csv"
    if not train_csv.exists() or not val_csv.exists():
        raise FileNotFoundError("Please run split_from_metadata.py first.")

    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)

    scaler = TargetScaler(cfg["targets"])
    scaler.fit(train_df)

    train_loader = build_dataloader(train_df, cfg, scaler, training=True)
    val_loader = build_dataloader(val_df, cfg, scaler, training=False)

    model_cfg = ModelConfig(
        backbone=cfg["model"]["backbone"],
        use_fpn=cfg["model"]["use_fpn"],
        use_film=cfg["model"]["use_film"],
        use_task_decouple=cfg["model"]["use_task_decouple"],
        use_spatial_head=cfg["model"]["use_spatial_head"],
    )
    model = NitrogenNet(model_cfg, num_targets=len(cfg["targets"]))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    params = model.count_params()
    flops = _estimate_flops(model, cfg["data"]["image_size"])
    print(f"Model params: {params}")
    if flops is not None:
        print(f"Approx FLOPs: {flops}")

    _set_seed(cfg["train"]["seed"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["train"]["epochs"]) if cfg["train"]["scheduler"] == "cosine" else None
    loss_fn = MaskedSmoothL1()
    uq = UncertaintyWeighting(len(cfg["targets"])) if cfg["loss"]["use_uncertainty_weighting"] else None
    scaler_amp = torch.cuda.amp.GradScaler(enabled=cfg["train"]["use_amp"])

    best_rmse = float("inf")
    patience = 0

    log_path = out_logs / f"log_{args.device}_{run_tag}.csv"
    best_path = out_models / f"best_{args.device}_{run_tag}.pt"
    scaler_path = out_models / f"scaler_{args.device}_{run_tag}.json"

    with open(log_path, "w", encoding="utf-8") as f:
        f.write("epoch,train_loss,val_loss,val_rmse_mean\n")

    for epoch in range(1, cfg["train"]["epochs"] + 1):
        model.train()
        total_loss = 0.0
        for images, angle_id, targets, mask, meta in train_loader:
            images = images.to(device)
            angle_id = angle_id.to(device)
            targets = targets.to(device)
            mask = mask.to(device)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=cfg["train"]["use_amp"]):
                preds, _ = model(images, angle_id)
                if uq is not None:
                    task_losses = [loss_fn(preds[:, i:i+1], targets[:, i:i+1], mask[:, i:i+1]) for i in range(preds.shape[1])]
                    loss = uq(task_losses)
                else:
                    loss = loss_fn(preds, targets, mask)
                if cfg["loss"]["use_consistency_loss"]:
                    loss = loss + consistency_loss(preds, meta, cfg["loss"]["lambda_consistency"])

            scaler_amp.scale(loss).backward()
            scaler_amp.step(optimizer)
            scaler_amp.update()
            total_loss += loss.item() * images.size(0)

        train_loss = total_loss / len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        preds_all, trues_all, masks_all, _ = evaluate_predictions(model, val_loader, device, scaler)
        metrics = compute_metrics(trues_all, preds_all, masks_all)
        val_rmse_mean = float(np.nanmean([m["rmse"] for m in metrics]))

        for images, angle_id, targets, mask, meta in val_loader:
            images = images.to(device)
            angle_id = angle_id.to(device)
            targets = targets.to(device)
            mask = mask.to(device)
            with torch.no_grad():
                preds, _ = model(images, angle_id)
                if uq is not None:
                    task_losses = [loss_fn(preds[:, i:i+1], targets[:, i:i+1], mask[:, i:i+1]) for i in range(preds.shape[1])]
                    loss = uq(task_losses)
                else:
                    loss = loss_fn(preds, targets, mask)
            val_loss += loss.item() * images.size(0)
        val_loss = val_loss / len(val_loader.dataset)

        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"{epoch},{train_loss:.6f},{val_loss:.6f},{val_rmse_mean:.6f}\n")

        print(f"Epoch {epoch}/{cfg['train']['epochs']} - train_loss: {train_loss:.4f} - val_rmse_mean: {val_rmse_mean:.4f}")

        if val_rmse_mean < best_rmse:
            best_rmse = val_rmse_mean
            patience = 0
            torch.save({"model_state": model.state_dict(), "config": cfg}, best_path)
            scaler.save(scaler_path)
        else:
            patience += 1
            if patience >= cfg["train"]["early_stop_patience"]:
                print("Early stopping triggered.")
                break

        if scheduler is not None:
            scheduler.step()

    bundle = torch.load(best_path, map_location=device)
    model.load_state_dict(bundle["model_state"])
    preds, trues, masks, meta = evaluate_predictions(model, val_loader, device, scaler)
    out_file = out_preds / f"preds_within_{args.device}_{run_tag}.csv"
    save_predictions(out_file, preds, trues, masks, meta, cfg["targets"])


if __name__ == "__main__":
    main()
