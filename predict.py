from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn
from torchvision import transforms
import timm
from torchvision.datasets.folder import default_loader


def build_model(model_name: str) -> nn.Module:
    try:
        return timm.create_model(model_name, pretrained=False, num_classes=1)
    except RuntimeError as exc:
        available = ", ".join(timm.list_models("mobilenetv4*"))
        raise ValueError(
            f"Unknown model: {model_name}. Available MobileNetV4 variants: {available}"
        ) from exc


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict nitrogen level from RGB images")
    parser.add_argument("--model", required=True, help="Model path")
    parser.add_argument("--images", nargs="+", required=True, help="Image paths")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    bundle = torch.load(args.model, map_location=args.device)
    model = build_model(bundle["model_name"])
    model.load_state_dict(bundle["model_state"])
    model.to(args.device)
    model.eval()

    config = bundle["config"]
    transform = transforms.Compose(
        [
            transforms.Resize((config.image_size, config.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    with torch.no_grad():
        for image_path in args.images:
            image = default_loader(Path(image_path))
            image = transform(image).unsqueeze(0).to(args.device)
            pred = model(image).squeeze().item()
            print(f"{image_path}: {pred:.4f}")


if __name__ == "__main__":
    main()
