"""심폐음 SR을 위한 다해상도 STFT 손실 (spectral convergence + log-magnitude L1)."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiResolutionSTFTLoss(nn.Module):
    def __init__(self, n_ffts=(512, 1024, 2048), hops=(128, 256, 512)):
        super().__init__()
        assert len(n_ffts) == len(hops)
        self.n_ffts = tuple(n_ffts)
        self.hops = tuple(hops)

    def _stft_mag(self, x: torch.Tensor, n_fft: int, hop: int) -> torch.Tensor:
        window = torch.hann_window(n_fft, device=x.device)
        S = torch.stft(x.squeeze(1), n_fft=n_fft, hop_length=hop,
                       window=window, return_complex=True)
        return S.abs().clamp_min(1e-7)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = 0.0
        for n_fft, hop in zip(self.n_ffts, self.hops):
            P = self._stft_mag(pred, n_fft, hop)
            T = self._stft_mag(target, n_fft, hop)
            sc = torch.norm(T - P, p="fro") / torch.norm(T, p="fro")
            mag = F.l1_loss(torch.log(P), torch.log(T))
            loss = loss + sc + mag
        return loss / len(self.n_ffts)
