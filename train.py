from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import timm
from torchvision.datasets.folder import default_loader


@dataclass
class TrainConfig:
    image_size: int = 224
    batch_size: int = 16
    epochs: int = 10
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    device: str = "cpu"


class WheatNitrogenDataset(Dataset):
    def __init__(self, csv_path: Path, transform: transforms.Compose) -> None:
        data = pd.read_csv(csv_path)
        required = {"image_path", "target", "angle", "variety"}
        if not required.issubset(data.columns):
            missing = ", ".join(sorted(required - set(data.columns)))
            raise ValueError(f"CSV must contain columns: {missing}")
        self.csv_dir = csv_path.resolve().parent
        self.image_paths = [self.csv_dir / path for path in data["image_path"].astype(str)]
        self.targets = data["target"].astype(float).to_numpy()
        self.angles = data["angle"].astype(str).to_numpy()
        self.varieties = data["variety"].astype(str).to_numpy()
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int):
        image = default_loader(self.image_paths[index])
        image = self.transform(image)
        target = torch.tensor(self.targets[index], dtype=torch.float32)
        return image, target, self.angles[index], self.varieties[index]


def build_model(model_name: str, pretrained: bool) -> nn.Module:
    try:
        model = timm.create_model(model_name, pretrained=pretrained, num_classes=1)
    except RuntimeError as exc:
        available = ", ".join(timm.list_models("mobilenetv4*"))
        raise ValueError(
            f"Unknown model: {model_name}. Available MobileNetV4 variants: {available}"
        ) from exc
    return model


def train_epoch(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer, device: str) -> float:
    model.train()
    total_loss = 0.0
    loss_fn = nn.MSELoss()
    for images, targets, _, _ in loader:
        images = images.to(device)
        targets = targets.to(device)
        optimizer.zero_grad()
        outputs = model(images).squeeze(1)
        loss = loss_fn(outputs, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
    return total_loss / len(loader.dataset)


def evaluate(model: nn.Module, loader: DataLoader, device: str) -> tuple[float, float, dict[str, dict[str, float]]]:
    model.eval()
    preds = []
    targets_all = []
    angles = []
    varieties = []
    with torch.no_grad():
        for images, targets, angle, variety in loader:
            images = images.to(device)
            outputs = model(images).squeeze(1).cpu().numpy()
            preds.append(outputs)
            targets_all.append(targets.numpy())
            angles.extend(angle)
            varieties.extend(variety)
    preds = np.concatenate(preds)
    targets_all = np.concatenate(targets_all)
    mae = np.mean(np.abs(preds - targets_all))
    ss_res = np.sum((targets_all - preds) ** 2)
    ss_tot = np.sum((targets_all - np.mean(targets_all)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    group_metrics: dict[str, dict[str, float]] = {}
    for key, labels in {"angle": angles, "variety": varieties}.items():
        group_metrics[key] = {}
        for group in sorted(set(labels)):
            mask = np.array([label == group for label in labels])
            if mask.sum() == 0:
                continue
            group_mae = np.mean(np.abs(preds[mask] - targets_all[mask]))
            group_ss_res = np.sum((targets_all[mask] - preds[mask]) ** 2)
            group_ss_tot = np.sum((targets_all[mask] - np.mean(targets_all[mask])) ** 2)
            group_r2 = 1 - group_ss_res / group_ss_tot if group_ss_tot > 0 else 0.0
            group_metrics[key][group] = {"mae": float(group_mae), "r2": float(group_r2)}
    return mae, r2, group_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train deep learning model for nitrogen diagnosis")
    parser.add_argument("--csv", required=True, help="CSV file with image_path and target")
    parser.add_argument("--output", required=True, help="Output model path")
    parser.add_argument("--model", default="mobilenetv4_conv_small")
    parser.add_argument("--pretrained", action="store_true", help="Use ImageNet pretrained weights")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    config = TrainConfig(
        image_size=args.image_size,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        device=args.device,
    )

    train_transform = transforms.Compose(
        [
            transforms.Resize((config.image_size, config.image_size)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    val_transform = transforms.Compose(
        [
            transforms.Resize((config.image_size, config.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    dataset = WheatNitrogenDataset(Path(args.csv), train_transform)
    val_dataset = WheatNitrogenDataset(Path(args.csv), val_transform)
    indices = np.random.permutation(len(dataset))
    split = int(len(dataset) * 0.8)
    train_indices, val_indices = indices[:split], indices[split:]
    train_subset = torch.utils.data.Subset(dataset, train_indices)
    val_subset = torch.utils.data.Subset(val_dataset, val_indices)

    train_loader = DataLoader(train_subset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=config.batch_size, shuffle=False)

    model = build_model(args.model, args.pretrained)
    model.to(config.device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    for epoch in range(config.epochs):
        train_loss = train_epoch(model, train_loader, optimizer, config.device)
        mae, r2, group_metrics = evaluate(model, val_loader, config.device)
        print(f"Epoch {epoch + 1}/{config.epochs} - loss: {train_loss:.4f} - MAE: {mae:.4f} - R2: {r2:.4f}")
        for key, metrics in group_metrics.items():
            for group, values in metrics.items():
                print(
                    f"  [{key}={group}] MAE: {values['mae']:.4f} - R2: {values['r2']:.4f}"
                )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "model_name": args.model,
            "config": config,
        },
        output_path,
    )
    print(f"Model saved to {output_path}")


if __name__ == "__main__":
    main()
