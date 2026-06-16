"""LR/SR 비교용 시각화 — 결과 페이지에 띄울 PNG 3종 생성.

  1) 파형(waveform): 시간 도메인. 실제 4 kHz LR vs 16 kHz SR 복원본.
  2) 스펙트로그램(STFT log-magnitude): 시간×주파수. LR/SR을 각자의 실제 샘플링레이트로 표시.
  3) 평균 magnitude spectrum (dB): 어느 주파수까지 살아났는지 보조 비교.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from scipy.signal import stft

# 한글 라벨이 깨지지 않도록 시스템에 있는 폰트를 자동 선택
_KOR_FONT_CANDIDATES = ["Malgun Gothic", "NanumGothic", "AppleGothic", "Pretendard"]
_available = {f.name for f in fm.fontManager.ttflist}
for _f in _KOR_FONT_CANDIDATES:
    if _f in _available:
        plt.rcParams["font.family"] = _f
        break
plt.rcParams["axes.unicode_minus"] = False


# UI 다크 테마와 어울리는 컬러
BG = "#1e293b"
PANEL = "#273449"
FG = "#f1f5f9"
MUTE = "#94a3b8"
LR_C = "#94a3b8"
SR_C = "#38bdf8"
GRID = "#334155"


def _style(ax):
    ax.set_facecolor(PANEL)
    ax.tick_params(colors=MUTE, labelsize=8)
    for spine in ax.spines.values():
        spine.set_color(GRID)
    ax.grid(True, color=GRID, alpha=0.4, linewidth=0.5)


def _new_fig(rows: int, cols: int = 1, height: float = 4.0, width: float = 9.0):
    fig, axes = plt.subplots(rows, cols, figsize=(width, height), dpi=110)
    fig.patch.set_facecolor(BG)
    return fig, axes


def _save(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(str(path), facecolor=BG, edgecolor="none")
    plt.close(fig)


def plot_waveform(lr: np.ndarray, lr_sr: int, sr_wave: np.ndarray, sr_sr: int,
                  path: Path) -> None:
    """LR vs SR 파형 (위/아래 2단)."""
    fig, axes = _new_fig(rows=2, height=4.0)
    t = np.arange(len(lr)) / lr_sr

    axes[0].plot(t, lr, color=LR_C, linewidth=0.6)
    axes[0].set_title("① 원본 LR 측정본 (4 kHz)", color=FG, fontsize=10, loc="left")
    axes[0].set_ylabel("진폭", color=MUTE, fontsize=9)
    _style(axes[0])

    t2 = np.arange(len(sr_wave)) / sr_sr
    axes[1].plot(t2, sr_wave, color=SR_C, linewidth=0.6)
    axes[1].set_title("② SR 복원본 (16 kHz)", color=FG, fontsize=10, loc="left")
    axes[1].set_xlabel("시간 (s)", color=MUTE, fontsize=9)
    axes[1].set_ylabel("진폭", color=MUTE, fontsize=9)
    _style(axes[1])
    _save(fig, path)


def _stft_log_mag(wave: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_fft = min(1024, max(128, 2 ** int(np.floor(np.log2(max(len(wave), 128))))))
    hop = max(n_fft // 4, 1)
    f, t, Z = stft(wave, fs=sr, nperseg=n_fft, noverlap=n_fft - hop,
                   boundary=None, padded=False)
    mag = np.abs(Z)
    log_mag = 20.0 * np.log10(mag + 1e-6)
    return f, t, log_mag


def plot_spectrogram(lr: np.ndarray, lr_sr: int, sr_wave: np.ndarray, sr_sr: int,
                     path: Path) -> None:
    """LR vs SR 스펙트로그램 (위/아래 2단). dB 스케일."""
    fig, axes = _new_fig(rows=2, height=5.0)

    for ax, wave, sample_rate, title in [
        (axes[0], lr, lr_sr, "① LR 스펙트로그램 (4 kHz 원 신호)"),
        (axes[1], sr_wave, sr_sr, "② SR 복원 스펙트로그램 (16 kHz 출력)"),
    ]:
        f, t, log_mag = _stft_log_mag(wave, sample_rate)
        vmax = float(np.max(log_mag))
        vmin = vmax - 80.0
        im = ax.pcolormesh(t, f, log_mag, shading="auto",
                           cmap="magma", vmin=vmin, vmax=vmax)
        ax.set_title(title, color=FG, fontsize=10, loc="left")
        ax.set_ylabel("주파수 (Hz)", color=MUTE, fontsize=9)
        ax.set_ylim(0, sample_rate / 2)
        _style(ax)
        cb = fig.colorbar(im, ax=ax, pad=0.01, fraction=0.04)
        cb.ax.tick_params(colors=MUTE, labelsize=7)
        cb.outline.set_edgecolor(GRID)
        cb.set_label("dB", color=MUTE, fontsize=8)

    axes[1].set_xlabel("시간 (s)", color=MUTE, fontsize=9)
    _save(fig, path)


def plot_avg_spectrum(lr: np.ndarray, lr_sr: int, sr_wave: np.ndarray, sr_sr: int,
                      path: Path) -> None:
    """평균 magnitude 스펙트럼 — 어느 주파수까지 살아났는지 한눈에 비교."""
    def _spectrum(x, sample_rate):
        n = 2 ** int(np.ceil(np.log2(max(len(x), 1024))))
        X = np.fft.rfft(x * np.hanning(len(x)), n=n)
        freqs = np.fft.rfftfreq(n, d=1.0 / sample_rate)
        db = 20 * np.log10(np.abs(X) + 1e-6)
        return freqs, db

    lf, Ldb = _spectrum(lr, lr_sr)
    sf, Sdb = _spectrum(sr_wave, sr_sr)

    fig, ax = _new_fig(rows=1, height=3.5)
    ax.semilogx(lf[1:], Ldb[1:], color=LR_C, linewidth=1.0, label="원본 LR (4 kHz)")
    ax.semilogx(sf[1:], Sdb[1:], color=SR_C, linewidth=1.0, label="SR 복원 (16 kHz)")
    ax.axvline(lr_sr / 2, color=MUTE, linestyle="--", linewidth=0.8, alpha=0.6)
    ax.text(lr_sr / 2, ax.get_ylim()[0] + 6, "LR Nyquist", color=MUTE, fontsize=7,
            rotation=90, va="bottom", ha="right")
    ax.set_xlim(20, sr_sr / 2)
    ax.set_xlabel("주파수 (Hz, log)", color=MUTE, fontsize=9)
    ax.set_ylabel("magnitude (dB)", color=MUTE, fontsize=9)
    ax.set_title("③ 평균 스펙트럼 비교 — 2 kHz 위 대역 복원 여부 확인", color=FG,
                 fontsize=10, loc="left")
    leg = ax.legend(facecolor=PANEL, edgecolor=GRID, labelcolor=FG, fontsize=9)
    for txt in leg.get_texts():
        txt.set_color(FG)
    _style(ax)
    _save(fig, path)


def render_all(lr: np.ndarray, lr_sr: int, sr_wave: np.ndarray, sr_sr: int,
               out_dir: Path, stem: str) -> dict[str, Path]:
    """3개 PNG를 한 번에 생성하고 경로 dict 반환."""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "waveform": out_dir / f"{stem}_waveform.png",
        "spectrogram": out_dir / f"{stem}_spectrogram.png",
        "spectrum": out_dir / f"{stem}_spectrum.png",
    }
    plot_waveform(lr, lr_sr, sr_wave, sr_sr, paths["waveform"])
    plot_spectrogram(lr, lr_sr, sr_wave, sr_sr, paths["spectrogram"])
    plot_avg_spectrum(lr, lr_sr, sr_wave, sr_sr, paths["spectrum"])
    return paths
