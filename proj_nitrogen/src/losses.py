from __future__ import annotations

import torch
from torch import nn


class MaskedSmoothL1(nn.Module):
    def __init__(self, beta: float = 1.0) -> None:
        super().__init__()
        self.loss = nn.SmoothL1Loss(reduction="none", beta=beta)

    def forward(self, preds: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        raw = self.loss(preds, targets)
        masked = raw * mask
        denom = mask.sum().clamp(min=1.0)
        return masked.sum() / denom


class UncertaintyWeighting(nn.Module):
    def __init__(self, num_tasks: int) -> None:
        super().__init__()
        self.log_sigma = nn.Parameter(torch.zeros(num_tasks))

    def forward(self, task_losses: list[torch.Tensor]) -> torch.Tensor:
        total = 0.0
        for i, loss in enumerate(task_losses):
            total = total + torch.exp(-self.log_sigma[i]) * loss + self.log_sigma[i]
        return total


def consistency_loss(
    preds: torch.Tensor,
    meta: list[dict],
    lambda_consistency: float,
) -> torch.Tensor:
    if lambda_consistency <= 0 or preds.numel() == 0:
        return torch.tensor(0.0, device=preds.device)

    loss = 0.0
    count = 0
    for i, row in enumerate(meta):
        target_angle = "b" if str(row.get("angle", "")).lower() == "a" else "a"
        candidates = [
            j
            for j, r in enumerate(meta)
            if str(r.get("angle", "")).lower() == target_angle
            and str(r.get("stage_en", "")) == str(row.get("stage_en", ""))
            and str(r.get("variety", "")) == str(row.get("variety", ""))
        ]
        if not candidates:
            continue
        j = candidates[0]
        loss = loss + torch.mean(torch.abs(preds[i] - preds[j]))
        count += 1

    if count == 0:
        return torch.tensor(0.0, device=preds.device)
    return lambda_consistency * (loss / count)
