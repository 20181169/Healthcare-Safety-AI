"""1D Wave-U-Net 형태의 Audio Super-Resolution 모델."""
from pathlib import Path
from typing import Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F


def detect_sr_channels(state_dict: Mapping[str, torch.Tensor]) -> tuple:
    """state_dict로부터 AudioSRUNet 의 channels 튜플을 자동 감지.

    저장된 체크포인트가 어떤 channels 설정으로 학습됐는지 알 수 없을 때
    in_conv / downs.{i}.block.0 의 weight shape을 보고 복원한다.
    """
    chs = [state_dict["in_conv.0.weight"].shape[0]]
    i = 0
    while f"downs.{i}.block.0.weight" in state_dict:
        chs.append(state_dict[f"downs.{i}.block.0.weight"].shape[0])
        i += 1
    return tuple(chs)


def load_sr_model(ckpt_path: str | Path, device: str = "cpu") -> "tuple[AudioSRUNet, dict]":
    """체크포인트에서 SR 모델을 채널 자동 감지로 로드.

    반환: (model, raw_state_dict_or_meta)
    """
    state = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    sd = state["model"] if isinstance(state, dict) and "model" in state else state
    channels = detect_sr_channels(sd)
    model = AudioSRUNet(channels=channels).to(device)
    model.load_state_dict(sd)
    return model, state


def _conv_block(in_c: int, out_c: int, k: int = 15) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv1d(in_c, out_c, kernel_size=k, padding=k // 2),
        nn.GroupNorm(8, out_c),
        nn.PReLU(),
    )


class _Down(nn.Module):
    def __init__(self, in_c: int, out_c: int, k: int = 15):
        super().__init__()
        self.block = _conv_block(in_c, out_c, k)
        self.pool = nn.Conv1d(out_c, out_c, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        skip = self.block(x)
        return self.pool(skip), skip


class _Up(nn.Module):
    def __init__(self, in_c: int, out_c: int, k: int = 15):
        super().__init__()
        self.up = nn.ConvTranspose1d(in_c, out_c, kernel_size=4, stride=2, padding=1)
        # 인코더 skip은 in_c 채널(= 2*out_c), ConvTranspose 결과는 out_c 채널
        # concat 후 입력 채널 = in_c + out_c
        self.block = _conv_block(in_c + out_c, out_c, k)

    def forward(self, x, skip):
        x = self.up(x)
        if x.size(-1) != skip.size(-1):
            x = F.pad(x, (0, skip.size(-1) - x.size(-1)))
        return self.block(torch.cat([x, skip], dim=1))


class AudioSRUNet(nn.Module):
    """LR(zero-stuff 업샘플) → HR 파형 복원. residual 학습."""

    def __init__(self, channels=(32, 64, 128, 256, 512), kernel: int = 15):
        super().__init__()
        chs = tuple(channels)
        self.in_conv = _conv_block(1, chs[0], k=kernel)
        self.downs = nn.ModuleList([
            _Down(chs[i], chs[i + 1], k=kernel) for i in range(len(chs) - 1)
        ])
        self.bottleneck = _conv_block(chs[-1], chs[-1], k=kernel)
        rev = chs[::-1]
        self.ups = nn.ModuleList([
            _Up(rev[i], rev[i + 1], k=kernel) for i in range(len(rev) - 1)
        ])
        self.out_conv = nn.Conv1d(chs[0], 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        h = self.in_conv(x)

        skips = []
        for d in self.downs:
            h, skip = d(h)
            skips.append(skip)

        h = self.bottleneck(h)

        for u, skip in zip(self.ups, reversed(skips)):
            h = u(h, skip)

        delta = self.out_conv(h)
        return torch.tanh(delta + residual)
