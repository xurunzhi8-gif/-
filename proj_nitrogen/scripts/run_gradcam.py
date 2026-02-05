from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from PIL import Image

from src.dataset import build_dataloader
from src.gradcam import compute_gradcam
from src.model import ModelConfig, NitrogenNet
from src.scaler import TargetScaler


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _overlay_cam(image: Image.Image, cam: np.ndarray) -> Image.Image:
    cam_resized = Image.fromarray((cam * 255).astype(np.uint8)).resize(image.size, resample=Image.BILINEAR)
    cam_rgb = Image.merge("RGB", (cam_resized, cam_resized, cam_resized))
    overlay = Image.blend(image.convert("RGB"), cam_rgb, alpha=0.4)
    return overlay


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Grad-CAM for AMDR-Net.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", required=True, choices=["iphone", "nova"])
    parser.add_argument("--split", default="val", choices=["train", "val"])
    parser.add_argument("--task", required=True, choices=["SNC_pct", "shootN_gm2", "leafN_gm2"])
    parser.add_argument("--run-tag", default="base")
    parser.add_argument("--max-samples", type=int, default=8)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    out_root = _project_root() / cfg["output"]["root"]
    figs_dir = out_root / "figs" / "gradcam"
    _ensure_dir(figs_dir)

    data_root = out_root / "data"
    split_name = f"{args.split}_full_{args.device}.csv"
    df = pd.read_csv(data_root / split_name)

    scaler = TargetScaler(cfg["targets"])
    scaler.fit(df)

    loader = build_dataloader(df.head(args.max_samples), cfg, scaler, training=False)

    model_cfg = ModelConfig(
        backbone=cfg["model"]["backbone"],
        use_fpn=cfg["model"]["use_fpn"],
        use_film=cfg["model"]["use_film"],
        use_task_decouple=cfg["model"]["use_task_decouple"],
        use_spatial_head=cfg["model"]["use_spatial_head"],
    )
    model = NitrogenNet(model_cfg, num_targets=len(cfg["targets"]))
    model_path = out_root / "models" / f"best_{args.device}_{args.run_tag}.pt"
    bundle = torch.load(model_path, map_location="cpu")
    model.load_state_dict(bundle["model_state"])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    task_idx = cfg["targets"].index(args.task)

    for batch_idx, (images, angle_id, _, _, meta) in enumerate(loader):
        image = images[0]
        angle = angle_id[0]
        result = compute_gradcam(model, image, angle, task_idx, target_layer=cfg["gradcam"]["target_layers"][0])
        image_path = meta[0]["image_path"]
        raw_image = Image.open(image_path).convert("RGB")
        overlay = _overlay_cam(raw_image, result.cam)
        out_path = figs_dir / f"gradcam_{args.device}_{args.task}_{batch_idx}.png"
        overlay.save(out_path)


if __name__ == "__main__":
    main()
