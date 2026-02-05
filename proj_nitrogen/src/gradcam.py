from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
from torch import nn


@dataclass
class GradCamResult:
    cam: np.ndarray
    pred: float


def _normalize_cam(cam: np.ndarray) -> np.ndarray:
    cam = np.maximum(cam, 0)
    if cam.max() > 0:
        cam = cam / cam.max()
    return cam


def compute_gradcam(
    model: nn.Module,
    image: torch.Tensor,
    angle_id: torch.Tensor,
    task_idx: int,
    target_layer: str = "C5",
) -> GradCamResult:
    model.eval()
    image = image.unsqueeze(0)
    angle_id = angle_id.unsqueeze(0)

    image = image.to(next(model.parameters()).device)
    angle_id = angle_id.to(next(model.parameters()).device)

    feats = model.backbone(image)
    if model.cfg.use_film:
        feats = [film(f, angle_id) for film, f in zip(model.film_blocks, feats)]

    target_feat = feats[-1] if target_layer == "C5" else feats[-1]
    target_feat.retain_grad()

    if model.cfg.use_fpn:
        fused = model.fpn(feats)
    else:
        fused = feats[-1]

    outputs = []
    for idx in range(model.num_targets):
        task_feat = fused
        if model.task_convs is not None:
            task_feat = model.task_convs[idx](task_feat)
        pred, _ = model.heads[idx](task_feat)
        outputs.append(pred)
    preds = torch.cat(outputs, dim=1)

    score = preds[:, task_idx].sum()
    score.backward()

    gradients = target_feat.grad
    weights = gradients.mean(dim=(2, 3), keepdim=True)
    cam = (weights * target_feat).sum(dim=1, keepdim=True)
    cam = cam.detach().cpu().numpy()[0, 0]
    cam = _normalize_cam(cam)

    pred_value = preds.detach().cpu().numpy()[0, task_idx]
    return GradCamResult(cam=cam, pred=float(pred_value))
