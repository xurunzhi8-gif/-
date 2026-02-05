from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import timm
import torch
from torch import nn


@dataclass
class ModelConfig:
    backbone: str
    use_fpn: bool
    use_film: bool
    use_task_decouple: bool
    use_spatial_head: bool


class AngleFiLM(nn.Module):
    def __init__(self, channels: int, num_angles: int = 2, hidden: int = 64) -> None:
        super().__init__()
        self.embedding = nn.Embedding(num_angles, hidden)
        self.mlp = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, channels * 2),
        )
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, x: torch.Tensor, angle_id: torch.Tensor) -> torch.Tensor:
        emb = self.embedding(angle_id)
        params = self.mlp(emb)
        gamma, beta = params.chunk(2, dim=-1)
        gamma = gamma.view(-1, x.shape[1], 1, 1)
        beta = beta.view(-1, x.shape[1], 1, 1)
        return (1.0 + gamma) * x + beta


class LiteFPN(nn.Module):
    def __init__(self, in_channels: List[int], out_ch: int = 96) -> None:
        super().__init__()
        self.lateral = nn.ModuleList([nn.Conv2d(c, out_ch, 1) for c in in_channels])
        self.smooth = nn.ModuleList([nn.Conv2d(out_ch, out_ch, 3, padding=1) for _ in in_channels])

    def forward(self, feats: List[torch.Tensor]) -> torch.Tensor:
        c3, c4, c5 = feats
        p5 = self.lateral[2](c5)
        p4 = self.lateral[1](c4) + nn.functional.interpolate(p5, size=c4.shape[-2:], mode="nearest")
        p3 = self.lateral[0](c3) + nn.functional.interpolate(p4, size=c3.shape[-2:], mode="nearest")
        return self.smooth[0](p3)


class SpatialRegressionHead(nn.Module):
    def __init__(self, in_ch: int, hidden: int = 128) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_ch, hidden, 1)
        self.attn = nn.Conv2d(hidden, 1, 1)
        self.fc = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        feat = torch.relu(self.conv(x))
        attn_map = self.attn(feat).flatten(2)
        attn = torch.softmax(attn_map, dim=-1).unsqueeze(1)
        feat_flat = feat.flatten(2)
        pooled = torch.sum(feat_flat * attn, dim=-1)
        out = self.fc(pooled)
        return out, pooled


class MLPHead(nn.Module):
    def __init__(self, in_ch: int) -> None:
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_ch, in_ch),
            nn.ReLU(),
            nn.Linear(in_ch, 1),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        pooled = torch.mean(x.flatten(2), dim=-1)
        return self.fc(pooled), pooled


class NitrogenNet(nn.Module):
    def __init__(self, cfg: ModelConfig, num_targets: int = 3, out_ch: int = 96) -> None:
        super().__init__()
        self.cfg = cfg
        self.num_targets = num_targets
        self.backbone, channels = self._build_backbone(cfg.backbone)
        self.fpn = LiteFPN(channels, out_ch=out_ch)
        self.film_blocks = nn.ModuleList([AngleFiLM(c) for c in channels])

        if cfg.use_task_decouple:
            self.task_convs = nn.ModuleList([nn.Conv2d(out_ch, out_ch, 1) for _ in range(num_targets)])
        else:
            self.task_convs = None

        if cfg.use_spatial_head:
            self.heads = nn.ModuleList([SpatialRegressionHead(out_ch) for _ in range(num_targets)])
        else:
            self.heads = nn.ModuleList([MLPHead(out_ch) for _ in range(num_targets)])

    def _build_backbone(self, backbone_name: str):
        name_map = {
            "mobilenetv3_small": "mobilenetv3_small_100",
            "mobilenetv4_small": "mobilenetv4_small",
            "efficientnet_b0": "efficientnet_b0",
            "mobilevit_xs": "mobilevit_xs",
        }
        model_name = name_map.get(backbone_name, backbone_name)
        try:
            backbone = timm.create_model(model_name, pretrained=True, features_only=True, out_indices=(2, 3, 4))
        except Exception:
            backbone = timm.create_model("mobilenetv3_small_100", pretrained=True, features_only=True, out_indices=(2, 3, 4))
        channels = backbone.feature_info.channels()
        return backbone, channels

    def forward(
        self,
        x: torch.Tensor,
        angle_id: torch.Tensor,
        return_features: bool = False,
    ) -> Tuple[torch.Tensor, List[torch.Tensor] | None]:
        feats = self.backbone(x)
        if self.cfg.use_film:
            feats = [film(f, angle_id) for film, f in zip(self.film_blocks, feats)]
        if self.cfg.use_fpn:
            fused = self.fpn(feats)
        else:
            fused = feats[-1]

        outputs = []
        task_features: List[torch.Tensor] = []
        for idx in range(self.num_targets):
            task_feat = fused
            if self.task_convs is not None:
                task_feat = self.task_convs[idx](task_feat)
            pred, pooled = self.heads[idx](task_feat)
            outputs.append(pred)
            task_features.append(pooled)
        preds = torch.cat(outputs, dim=1)
        return (preds, task_features) if return_features else (preds, None)

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
