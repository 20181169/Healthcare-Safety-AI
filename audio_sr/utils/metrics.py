"""평가지표 — SR(SNR/LSD)과 분류(ICBHI)."""
import numpy as np
import torch


def snr_db(reference: torch.Tensor, estimate: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    noise = reference - estimate
    return 10.0 * torch.log10(
        (reference.pow(2).sum(-1) + eps) / (noise.pow(2).sum(-1) + eps)
    )


def lsd_db(reference: torch.Tensor, estimate: torch.Tensor,
           n_fft: int = 1024, hop: int = 256, eps: float = 1e-8) -> torch.Tensor:
    """Log-Spectral Distance (오디오 SR 표준 평가지표)."""
    window = torch.hann_window(n_fft, device=reference.device)
    R = torch.stft(reference, n_fft=n_fft, hop_length=hop,
                   window=window, return_complex=True).abs()
    E = torch.stft(estimate, n_fft=n_fft, hop_length=hop,
                   window=window, return_complex=True).abs()
    diff = torch.log10(R.pow(2) + eps) - torch.log10(E.pow(2) + eps)
    return diff.pow(2).mean(dim=(-1, -2)).sqrt().mean()


def icbhi_score(cm: np.ndarray) -> dict:
    """ICBHI 2017 표준 스코어 — Specificity + Sensitivity 의 평균."""
    spec = cm[0, 0] / max(cm[0].sum(), 1)
    sens = (cm[1, 1] + cm[2, 2] + cm[3, 3]) / max(cm[1:].sum(), 1)
    return {"specificity": float(spec), "sensitivity": float(sens),
            "icbhi": float((spec + sens) / 2)}
