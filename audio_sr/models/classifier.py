"""log-mel + SE-CNN 기반 심폐음 4-class 분류기."""
import torch
import torch.nn as nn

from ..utils.features import LogMelSpec
from ..utils.audio import HR_SR


class _SEBlock(nn.Module):
    def __init__(self, channels: int, r: int = 8):
        super().__init__()
        self.fc1 = nn.Conv2d(channels, channels // r, 1)
        self.fc2 = nn.Conv2d(channels // r, channels, 1)

    def forward(self, x):
        s = x.mean(dim=(2, 3), keepdim=True)
        s = torch.relu(self.fc1(s))
        s = torch.sigmoid(self.fc2(s))
        return x * s


class _ConvBNAct(nn.Sequential):
    def __init__(self, in_c: int, out_c: int, k: int = 3, s: int = 1, p: int = 1):
        super().__init__(
            nn.Conv2d(in_c, out_c, k, s, p, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )


class HeartLungCNN(nn.Module):
    def __init__(self, n_classes: int = 4, sr: int = HR_SR,
                 n_mels: int = 128, n_fft: int = 1024, hop: int = 256,
                 fmin: float = 20.0, fmax: float = 2000.0):
        super().__init__()
        self.spec = LogMelSpec(sr=sr, n_fft=n_fft, hop=hop,
                               n_mels=n_mels, fmin=fmin, fmax=fmax)

        self.stem = _ConvBNAct(1, 32)
        self.block1 = nn.Sequential(_ConvBNAct(32, 64), _SEBlock(64), nn.MaxPool2d(2))
        self.block2 = nn.Sequential(_ConvBNAct(64, 128), _SEBlock(128), nn.MaxPool2d(2))
        self.block3 = nn.Sequential(_ConvBNAct(128, 256), _SEBlock(256), nn.MaxPool2d(2))
        self.block4 = nn.Sequential(_ConvBNAct(256, 384), _SEBlock(384), nn.AdaptiveAvgPool2d(1))

        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(384, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, n_classes),
        )

    def forward(self, wave: torch.Tensor) -> torch.Tensor:
        spec = self.spec(wave)
        x = self.stem(spec)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        return self.head(x)
