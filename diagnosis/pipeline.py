"""웹 요청에서 호출하는 SR + 분류 추론 파이프라인 (싱글톤).

- 첫 요청 시 모델을 메모리에 로드, 이후 재사용
- 체크포인트가 없으면 무작위 가중치로 데모 모드 동작 (UI 흐름 확인용)
"""
from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import librosa
import soundfile as sf

from django.conf import settings

from audio_sr.models import AudioSRUNet, HeartLungCNN, detect_sr_channels
from audio_sr.data import LABELS
from audio_sr.utils.audio import HR_SR, LR_SR, peak_normalize, simulate_lr_from_hr

from .visualize import render_all


@dataclass
class InferenceResult:
    restored_path: str               # MEDIA_ROOT 기준 상대경로
    predicted_label: str             # normal/crackle/wheeze/both
    confidence: float                # Top-1 확률 (0..1)
    probabilities: dict              # {라벨: 확률}
    snr_in_db: float                 # 입력 SNR (HR 시뮬레이션 가능 시)
    snr_out_db: float                # 복원 SNR
    demo_mode: bool                  # 체크포인트 없이 동작했는지 여부
    waveform_png: str = ""           # 시각화 PNG (MEDIA 상대경로)
    spectrogram_png: str = ""
    spectrum_png: str = ""


def _resolve_device() -> str:
    pref = settings.INFERENCE_DEVICE
    if pref == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return pref


def _safe_load_state(model: torch.nn.Module, ckpt: str, device: str) -> bool:
    if not ckpt or not Path(ckpt).is_file():
        return False
    state = torch.load(ckpt, map_location=device, weights_only=False)
    model.load_state_dict(state["model"] if "model" in state else state)
    return True


def _build_sr_model(ckpt: str, device: str) -> "tuple[AudioSRUNet, bool]":
    """체크포인트의 채널 설정을 자동 감지해 SR 모델 구성.
    체크포인트 없으면 기본 채널로 무작위 가중치 모델 반환."""
    if ckpt and Path(ckpt).is_file():
        state = torch.load(ckpt, map_location=device, weights_only=False)
        sd = state["model"] if "model" in state else state
        channels = detect_sr_channels(sd)
        model = AudioSRUNet(channels=channels).to(device).eval()
        model.load_state_dict(sd)
        return model, True
    return AudioSRUNet().to(device).eval(), False


class DiagnosisPipeline:
    """프로세스당 1회 로드되는 추론 파이프라인."""

    _instance: "DiagnosisPipeline | None" = None
    _lock = threading.Lock()

    @classmethod
    def get(cls) -> "DiagnosisPipeline":
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    def __init__(self):
        self.device = _resolve_device()
        self.sr_model, sr_loaded = _build_sr_model(settings.SR_CKPT, self.device)
        self.cls_model = HeartLungCNN(n_classes=len(LABELS)).to(self.device).eval()
        cls_loaded = _safe_load_state(self.cls_model, settings.CLS_CKPT, self.device)
        self.demo_mode = not (sr_loaded and cls_loaded)

        for p in self.sr_model.parameters():
            p.requires_grad_(False)
        for p in self.cls_model.parameters():
            p.requires_grad_(False)

    # ------------------- 내부 헬퍼 -------------------

    @staticmethod
    def _load_as_lr(path: str) -> tuple[np.ndarray, np.ndarray | None]:
        """업로드 파형을 HR 길이 zero-stuff LR로 정합. 16k 입력이면 HR도 보존."""
        wav, sr = librosa.load(path, sr=None, mono=True)
        wav = peak_normalize(wav.astype(np.float32))

        if sr >= HR_SR:
            hr = librosa.resample(wav, orig_sr=sr, target_sr=HR_SR) if sr != HR_SR else wav
            lr = simulate_lr_from_hr(hr, add_noise=False)
            return lr.astype(np.float32), hr.astype(np.float32)

        lr_native = librosa.resample(wav, orig_sr=sr, target_sr=LR_SR) if sr != LR_SR else wav
        up = np.zeros(len(lr_native) * (HR_SR // LR_SR), dtype=np.float32)
        up[:: HR_SR // LR_SR] = lr_native
        return up, None

    def _overlap_add_sr(self, lr_wave: np.ndarray, seg: int,
                        hop_ratio: float = 0.5) -> np.ndarray:
        hop = max(int(seg * hop_ratio), 1)
        out = np.zeros(len(lr_wave), dtype=np.float32)
        norm = np.zeros(len(lr_wave), dtype=np.float32)
        window = np.hanning(seg).astype(np.float32)

        with torch.no_grad():
            for s in range(0, max(len(lr_wave) - seg, 0) + 1, hop):
                chunk = lr_wave[s : s + seg]
                x = torch.from_numpy(chunk).float().to(self.device).view(1, 1, -1)
                y = self.sr_model(x).cpu().numpy().reshape(-1)
                out[s : s + seg] += y * window
                norm[s : s + seg] += window

        norm[norm < 1e-6] = 1.0
        return out / norm

    @staticmethod
    def _snr_db(reference: np.ndarray, estimate: np.ndarray, eps: float = 1e-8) -> float:
        n = min(len(reference), len(estimate))
        ref, est = reference[:n], estimate[:n]
        noise = ref - est
        return float(10.0 * np.log10(
            (np.sum(ref ** 2) + eps) / (np.sum(noise ** 2) + eps)
        ))

    # ------------------- 외부 API -------------------

    def run(self, upload_path: str, restored_rel_path: str,
            segment_sec: float | None = None) -> InferenceResult:
        """
        upload_path        : MEDIA_ROOT 기준 절대경로 (원본 wav)
        restored_rel_path  : MEDIA_ROOT 하위 저장 상대경로 (예: 'restored/abc.wav')
        """
        seg = int((segment_sec or settings.INFERENCE_SEGMENT_SEC) * HR_SR)
        lr_wave, hr_ref = self._load_as_lr(upload_path)

        if len(lr_wave) < seg:
            pad = np.zeros(seg, dtype=np.float32)
            pad[: len(lr_wave)] = lr_wave
            lr_wave = pad

        restored = self._overlap_add_sr(lr_wave, seg=seg)
        restored = peak_normalize(restored).astype(np.float32)

        out_abs = Path(settings.MEDIA_ROOT) / restored_rel_path
        out_abs.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(out_abs), restored, HR_SR)

        snr_in = snr_out = float("nan")
        if hr_ref is not None:
            snr_in = self._snr_db(hr_ref, lr_wave[: len(hr_ref)])
            snr_out = self._snr_db(hr_ref, restored[: len(hr_ref)])

        x = torch.from_numpy(restored).float().to(self.device).view(1, 1, -1)
        with torch.no_grad():
            prob = F.softmax(self.cls_model(x), dim=-1).cpu().numpy()[0]

        # 시각화 PNG 3종 생성
        visuals_root = Path(settings.MEDIA_ROOT) / "visuals"
        stem = Path(restored_rel_path).stem
        paths = render_all(lr_wave, restored, HR_SR, visuals_root, stem)
        rel = lambda p: str(p.relative_to(settings.MEDIA_ROOT)).replace("\\", "/")

        idx = int(prob.argmax())
        return InferenceResult(
            restored_path=restored_rel_path,
            predicted_label=LABELS[idx],
            confidence=float(prob[idx]),
            probabilities={lab: float(p) for lab, p in zip(LABELS, prob)},
            snr_in_db=snr_in,
            snr_out_db=snr_out,
            demo_mode=self.demo_mode,
            waveform_png=rel(paths["waveform"]),
            spectrogram_png=rel(paths["spectrogram"]),
            spectrum_png=rel(paths["spectrum"]),
        )
