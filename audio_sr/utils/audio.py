"""심폐음 파형 I/O 및 LR 시뮬레이션."""
import numpy as np
import librosa
from scipy.signal import butter, sosfiltfilt


HR_SR = 16000   # 고해상도 샘플링 레이트
LR_SR = 4000    # 가정용 디지털 청진기 측정 가정 (모사)


def butter_lowpass_sos(cutoff: float, sr: int, order: int = 8):
    nyq = sr / 2
    return butter(order, cutoff / nyq, btype="low", output="sos")


def simulate_lr_from_hr(hr_wave: np.ndarray, hr_sr: int = HR_SR,
                       lr_sr: int = LR_SR, add_noise: bool = True) -> np.ndarray:
    """
    HR → LR 시뮬레이션:
      1) anti-aliasing 저역통과
      2) 정수배 다운샘플
      3) (옵션) 신체/측정 잡음 가산
      4) HR 길이 정합용 zero-stuff 업샘플
    """
    sos = butter_lowpass_sos(cutoff=lr_sr / 2 * 0.9, sr=hr_sr)
    filt = sosfiltfilt(sos, hr_wave).astype(np.float32)
    decim = filt[:: hr_sr // lr_sr].copy()

    if add_noise:
        t = np.arange(len(decim)) / lr_sr
        body_noise = 0.01 * np.sin(2 * np.pi * np.random.uniform(2, 8) * t)
        white = np.random.randn(len(decim)).astype(np.float32) * 0.005
        decim = decim + body_noise.astype(np.float32) + white

    up = np.zeros(len(decim) * (hr_sr // lr_sr), dtype=np.float32)
    up[:: hr_sr // lr_sr] = decim
    return up


def peak_normalize(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    m = np.max(np.abs(x))
    return x / (m + eps)


def load_wave(path: str, target_sr: int = HR_SR) -> np.ndarray:
    wav, _ = librosa.load(path, sr=target_sr, mono=True)
    return peak_normalize(wav.astype(np.float32))


def random_crop(wave: np.ndarray, length: int) -> np.ndarray:
    if len(wave) <= length:
        out = np.zeros(length, dtype=np.float32)
        out[: len(wave)] = wave
        return out
    s = np.random.randint(0, len(wave) - length)
    return wave[s : s + length]
