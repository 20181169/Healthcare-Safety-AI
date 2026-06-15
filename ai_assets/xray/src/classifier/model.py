"""
Multi-Label 질병 진단 모델.

- 백본: DenseNet-121 (의료 영상에서 CheXNet 등으로 검증된 구조)
- 입력 1채널 (X-ray) → 3채널로 확장하여 ImageNet pretrained 사용
- 출력: 각 질병 클래스별 sigmoid 확률
"""

from typing import List

import torch
import torch.nn as nn
from torchvision.models import DenseNet121_Weights, densenet121

from src.data import DISEASE_LABELS


class MultiLabelXrayClassifier(nn.Module):
    def __init__(self, num_labels: int = len(DISEASE_LABELS), pretrained: bool = True):
        super().__init__()
        weights = DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = densenet121(weights=weights)
        in_features = backbone.classifier.in_features

        # 1채널 입력을 위한 conv0 가중치 평균
        old_conv = backbone.features.conv0
        new_conv = nn.Conv2d(
            in_channels=1,
            out_channels=old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=False,
        )
        with torch.no_grad():
            new_conv.weight[:] = old_conv.weight.mean(dim=1, keepdim=True)
        backbone.features.conv0 = new_conv

        backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features, num_labels),
        )
        self.backbone = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.forward(x))


def compute_pos_weight(label_counts: List[int], total: int) -> torch.Tensor:
    """클래스 불균형 보정을 위한 pos_weight (BCEWithLogitsLoss 용)."""
    weights = []
    for c in label_counts:
        neg = max(1, total - c)
        weights.append(neg / max(1, c))
    return torch.tensor(weights, dtype=torch.float32)
