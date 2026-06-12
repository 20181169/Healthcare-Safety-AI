"""심폐음 분류용 log-mel 스펙트로그램."""
import torch
import torch.nn as nn
import torchaudio

from .audio import HR_SR


class LogMelSpec(nn.Module):
    """저주파 대역(20–2000 Hz)을 강조한 mel-spectrogram."""
    def __init__(self, sr: int = HR_SR, n_fft: int = 1024, hop: int = 256,
                 n_mels: int = 128, fmin: float = 20.0, fmax: float = 2000.0):
        super().__init__()
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sr, n_fft=n_fft, hop_length=hop,
            n_mels=n_mels, f_min=fmin, f_max=fmax, power=2.0,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.log1p(self.mel(x))
