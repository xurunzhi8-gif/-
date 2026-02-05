from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets.folder import default_loader

from src.scaler import TargetScaler


class WheatDataset(Dataset):
    def __init__(self, df: pd.DataFrame, cfg: dict, scaler: TargetScaler, training: bool) -> None:
        self.df = df.reset_index(drop=True)
        self.targets = cfg["targets"]
        self.scaler = scaler

        if training:
            self.transform = transforms.Compose(
                [
                    transforms.Resize((cfg["data"]["image_size"], cfg["data"]["image_size"])),
                    transforms.ColorJitter(0.2, 0.2, 0.2, 0.05),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ]
            )
        else:
            self.transform = transforms.Compose(
                [
                    transforms.Resize((cfg["data"]["image_size"], cfg["data"]["image_size"])),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ]
            )

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        image_path = Path(row["image_path"])
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        image = default_loader(image_path)
        image = self.transform(image)

        angle = str(row["angle"]).strip().lower()
        angle_id = 0 if angle == "a" else 1

        targets = row[self.targets].to_numpy(dtype=np.float32)
        targets, mask = self.scaler.transform_with_mask(targets)

        meta = {
            "image_id": row.get("image_id", ""),
            "image_path": str(image_path),
            "device": row.get("device", ""),
            "angle": row.get("angle", ""),
            "stage_en": row.get("stage_en", ""),
            "variety": row.get("variety", ""),
        }

        return image, torch.tensor(angle_id, dtype=torch.long), torch.tensor(targets), torch.tensor(mask), meta


def _collate(batch):
    images, angle_ids, targets, masks, meta = zip(*batch)
    return (
        torch.stack(images),
        torch.stack(angle_ids),
        torch.stack(targets),
        torch.stack(masks),
        list(meta),
    )


def build_dataloader(df: pd.DataFrame, cfg: dict, scaler: TargetScaler, training: bool) -> DataLoader:
    dataset = WheatDataset(df, cfg, scaler, training=training)
    return DataLoader(
        dataset,
        batch_size=cfg["train"]["batch_size"],
        shuffle=training,
        num_workers=cfg["data"]["num_workers"],
        collate_fn=_collate,
    )
