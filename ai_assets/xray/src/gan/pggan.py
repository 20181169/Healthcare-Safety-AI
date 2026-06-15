"""
PGGAN (Progressive Growing of GANs) - 소아 X-ray 데이터 증강 모델.

Karras et al., 2017 "Progressive Growing of GANs for Improved Quality,
Stability, and Variation" 논문 구조를 1채널 X-ray 용으로 단순화하여 구현.

핵심 구성:
- Equalized Learning Rate (EqualizedConv2d, EqualizedLinear)
- Pixelwise Feature Vector Normalization (PixelNorm)
- Minibatch Standard Deviation Layer (Discriminator 마지막 블록)
- 해상도를 4 → 8 → 16 → ... → 256 으로 점진적으로 키우며 fade-in 보간
"""

import math
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------- 기본 빌딩 블록 ----------------------------- #

class PixelNorm(nn.Module):
    """채널 방향 픽셀 정규화."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x / torch.sqrt(x.pow(2).mean(dim=1, keepdim=True) + 1e-8)


class EqualizedConv2d(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, k: int, stride: int = 1, padding: int = 0):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_ch, in_ch, k, k))
        self.bias = nn.Parameter(torch.zeros(out_ch))
        self.scale = math.sqrt(2.0 / (in_ch * k * k))
        self.stride = stride
        self.padding = padding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.conv2d(x, self.weight * self.scale, self.bias, self.stride, self.padding)


class EqualizedLinear(nn.Module):
    def __init__(self, in_f: int, out_f: int):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_f, in_f))
        self.bias = nn.Parameter(torch.zeros(out_f))
        self.scale = math.sqrt(2.0 / in_f)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight * self.scale, self.bias)


class MinibatchStdDev(nn.Module):
    """Discriminator 마지막에 미니배치 통계량을 채널로 추가."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _, h, w = x.shape
        std = x.std(dim=0, unbiased=False).mean().expand(b, 1, h, w)
        return torch.cat([x, std], dim=1)


# ----------------------------- Generator ----------------------------- #

class GenInitialBlock(nn.Module):
    """4x4 출력을 만드는 초기 블록 (latent z → 4x4 feature map)."""

    def __init__(self, latent_dim: int, channels: int):
        super().__init__()
        self.pixel_norm = PixelNorm()
        self.dense = EqualizedLinear(latent_dim, channels * 4 * 4)
        self.conv = EqualizedConv2d(channels, channels, k=3, padding=1)
        self.channels = channels

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.pixel_norm(z)
        x = self.dense(x).view(-1, self.channels, 4, 4)
        x = F.leaky_relu(x, 0.2)
        x = self.pixel_norm(x)
        x = F.leaky_relu(self.conv(x), 0.2)
        x = self.pixel_norm(x)
        return x


class GenBlock(nn.Module):
    """업샘플링 후 conv 2회 적용."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv1 = EqualizedConv2d(in_ch, out_ch, k=3, padding=1)
        self.conv2 = EqualizedConv2d(out_ch, out_ch, k=3, padding=1)
        self.pixel_norm = PixelNorm()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        x = self.pixel_norm(F.leaky_relu(self.conv1(x), 0.2))
        x = self.pixel_norm(F.leaky_relu(self.conv2(x), 0.2))
        return x


class Generator(nn.Module):
    """
    해상도별로 toRGB(=toGray) 헤드를 따로 가지며, 단계가 올라갈 때
    이전 해상도와 새 해상도를 alpha 로 보간하여 fade-in.
    """

    def __init__(self, latent_dim: int = 256, channels: List[int] = None, img_channels: int = 1):
        super().__init__()
        self.latent_dim = latent_dim
        self.channels = channels or [256, 256, 256, 128, 64, 32, 16]  # 4 → 256

        self.initial = GenInitialBlock(latent_dim, self.channels[0])
        self.blocks = nn.ModuleList([
            GenBlock(self.channels[i], self.channels[i + 1])
            for i in range(len(self.channels) - 1)
        ])
        self.to_gray = nn.ModuleList([
            EqualizedConv2d(c, img_channels, k=1) for c in self.channels
        ])

    def forward(self, z: torch.Tensor, step: int, alpha: float) -> torch.Tensor:
        x = self.initial(z)
        if step == 0:
            return torch.tanh(self.to_gray[0](x))

        for i in range(step - 1):
            x = self.blocks[i](x)

        prev = F.interpolate(self.to_gray[step - 1](x), scale_factor=2, mode="nearest")
        x = self.blocks[step - 1](x)
        new = self.to_gray[step](x)
        return torch.tanh(alpha * new + (1 - alpha) * prev)


# ----------------------------- Discriminator ----------------------------- #

class DiscBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv1 = EqualizedConv2d(in_ch, in_ch, k=3, padding=1)
        self.conv2 = EqualizedConv2d(in_ch, out_ch, k=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.leaky_relu(self.conv1(x), 0.2)
        x = F.leaky_relu(self.conv2(x), 0.2)
        return F.avg_pool2d(x, 2)


class DiscFinalBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.mbstd = MinibatchStdDev()
        self.conv1 = EqualizedConv2d(channels + 1, channels, k=3, padding=1)
        self.conv2 = EqualizedConv2d(channels, channels, k=4)
        self.dense = EqualizedLinear(channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mbstd(x)
        x = F.leaky_relu(self.conv1(x), 0.2)
        x = F.leaky_relu(self.conv2(x), 0.2)
        return self.dense(x.view(x.size(0), -1))


class Discriminator(nn.Module):
    def __init__(self, channels: List[int] = None, img_channels: int = 1):
        super().__init__()
        self.channels = channels or [256, 256, 256, 128, 64, 32, 16]

        self.from_gray = nn.ModuleList([
            EqualizedConv2d(img_channels, c, k=1) for c in self.channels
        ])
        self.blocks = nn.ModuleList([
            DiscBlock(self.channels[i + 1], self.channels[i])
            for i in range(len(self.channels) - 1)
        ])
        self.final = DiscFinalBlock(self.channels[0])

    def forward(self, img: torch.Tensor, step: int, alpha: float) -> torch.Tensor:
        if step == 0:
            x = F.leaky_relu(self.from_gray[0](img), 0.2)
            return self.final(x)

        x_new = F.leaky_relu(self.from_gray[step](img), 0.2)
        x_new = self.blocks[step - 1](x_new)

        x_prev = F.avg_pool2d(img, 2)
        x_prev = F.leaky_relu(self.from_gray[step - 1](x_prev), 0.2)

        x = alpha * x_new + (1 - alpha) * x_prev
        for i in range(step - 2, -1, -1):
            x = self.blocks[i](x)
        return self.final(x)


# ------------------------ WGAN-GP 손실 보조 함수 ------------------------ #

def gradient_penalty(
    disc: Discriminator,
    real: torch.Tensor,
    fake: torch.Tensor,
    step: int,
    alpha: float,
) -> torch.Tensor:
    b = real.size(0)
    eps = torch.rand(b, 1, 1, 1, device=real.device)
    interp = (eps * real + (1 - eps) * fake).requires_grad_(True)
    d_interp = disc(interp, step, alpha)
    grads = torch.autograd.grad(
        outputs=d_interp,
        inputs=interp,
        grad_outputs=torch.ones_like(d_interp),
        create_graph=True,
        retain_graph=True,
    )[0]
    grads = grads.view(b, -1)
    return ((grads.norm(2, dim=1) - 1) ** 2).mean()
