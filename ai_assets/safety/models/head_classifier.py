"""Head ROI 이중분류 모델 - (1) Helmet 유/무  (2) 올바른 착용 여부"""
import torch
import torch.nn as nn
import torchvision.models as tvm


class HeadDualClassifier(nn.Module):
    """
    Backbone: MobileNetV3-Small (경량, 실시간)
    Head 1: helmet_presence (0=no_helmet, 1=helmet)
    Head 2: helmet_correctness (0=incorrect, 1=correct)  - 턱끈 미체결 등
    """

    def __init__(self, pretrained: bool = True):
        super().__init__()
        backbone = tvm.mobilenet_v3_small(weights="DEFAULT" if pretrained else None)
        feat_dim = backbone.classifier[0].in_features
        self.features = backbone.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.shared = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feat_dim, 256),
            nn.Hardswish(inplace=True),
            nn.Dropout(0.2),
        )
        self.head_presence = nn.Linear(256, 2)
        self.head_correctness = nn.Linear(256, 2)

    def forward(self, x: torch.Tensor):
        f = self.features(x)
        f = self.pool(f)
        f = self.shared(f)
        return {
            "presence": self.head_presence(f),
            "correctness": self.head_correctness(f),
        }
