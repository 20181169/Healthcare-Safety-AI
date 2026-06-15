"""
심폐음 분류 모델 실측 평가 — ICBHI validation split 에 대한 진짜 metric 산출.

비교 대상:
  • Baseline   : LR 신호를 그대로 baseline.pt 에 통과 (SR 없음)
  • SR-boost  : LR → AudioSR U-Net 으로 복원 → best_sr.pt 분류기에 통과

산출 metric:
  • Macro F1, accuracy
  • Per-class precision / recall
  • Confusion matrix (4×4)
  • ICBHI score (Specificity + Sensitivity / 2)

실행:
  python scripts/eval_diagnosis.py
  python scripts/eval_diagnosis.py --device cpu --max 200   # 빠른 sanity check
"""
import argparse
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


# win/ 자체를 sys.path 에 — audio_sr 모듈 import
THIS = Path(__file__).resolve()
WIN = THIS.parent.parent
sys.path.insert(0, str(WIN))

HEART_LUNG_ROOT = Path(os.environ.get("HEART_LUNG_ROOT", str(WIN.parent / "수상관리_심폐음")))
VAL_CSV = HEART_LUNG_ROOT / "data" / "cls_val.csv"
SR_CKPT = HEART_LUNG_ROOT / "checkpoints" / "sr" / "best.pt"
CLS_BASELINE = HEART_LUNG_ROOT / "checkpoints" / "cls" / "best_baseline.pt"
CLS_SR = HEART_LUNG_ROOT / "checkpoints" / "cls" / "best_sr.pt"

from audio_sr.data import LABELS  # noqa: E402
from audio_sr.models import HeartLungCNN, load_sr_model  # noqa: E402
from audio_sr.utils.audio import HR_SR, peak_normalize, simulate_lr_from_hr, load_wave  # noqa: E402
from audio_sr.utils.metrics import icbhi_score  # noqa: E402

import pandas as pd  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    confusion_matrix, f1_score, precision_recall_fscore_support, accuracy_score,
)


def center_crop_or_pad(wav: np.ndarray, length: int) -> np.ndarray:
    """랜덤 crop 대신 deterministic center crop (평가 reproducibility)."""
    if len(wav) <= length:
        out = np.zeros(length, dtype=np.float32)
        out[: len(wav)] = wav
        return out
    s = (len(wav) - length) // 2
    return wav[s : s + length]


def _resolve_csv_path(rel: str) -> str:
    """CSV 의 filepath 가 'data/icbhi_cycles/xxx.wav' 형식 — HEART_LUNG_ROOT 와 결합."""
    p = Path(rel)
    if p.is_absolute() and p.is_file():
        return str(p)
    cand = HEART_LUNG_ROOT / p
    if cand.is_file():
        return str(cand)
    # 마지막 시도 — CSV 가 'icbhi_cycles/...' 일 수도
    cand2 = HEART_LUNG_ROOT / "data" / p
    return str(cand2)


def load_classifier(ckpt_path: Path, device: str) -> HeartLungCNN:
    model = HeartLungCNN(n_classes=len(LABELS)).to(device).eval()
    state = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    state = state["model"] if isinstance(state, dict) and "model" in state else state
    model.load_state_dict(state)
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def evaluate(mode: str, df: pd.DataFrame, sr_model, cls_model, device: str,
             seg: int, log_every: int = 100):
    """mode: 'baseline' (no SR) or 'sr-boost' (SR 후 분류)."""
    y_true, y_pred = [], []
    label2idx = {l: i for i, l in enumerate(LABELS)}
    t0 = time.time()
    total = len(df)

    for i, row in enumerate(df.itertuples(index=False)):
        path = _resolve_csv_path(row.filepath)
        try:
            wav = load_wave(path, target_sr=HR_SR)
        except Exception as e:
            print(f"  [skip] {path} ({e})", flush=True)
            continue

        # LR 시뮬레이션 (deterministic — no noise)
        np.random.seed(0)
        lr = simulate_lr_from_hr(wav, add_noise=False)
        lr = peak_normalize(lr)
        lr = center_crop_or_pad(lr, seg)

        x = torch.from_numpy(lr).float().to(device).view(1, 1, -1)
        if mode == "sr-boost":
            x = sr_model(x)

        with torch.no_grad():
            logits = cls_model(x)
            pred = int(logits.argmax(dim=-1).item())
        y_true.append(label2idx[row.label])
        y_pred.append(pred)

        if (i + 1) % log_every == 0:
            dt = time.time() - t0
            eta = dt / (i + 1) * (total - i - 1)
            print(f"  [{mode}] {i+1}/{total} · elapsed {dt:.0f}s · ETA {eta:.0f}s", flush=True)

    y_true = np.array(y_true); y_pred = np.array(y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(LABELS))))
    pre, rec, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(len(LABELS))), zero_division=0,
    )
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    acc = float(accuracy_score(y_true, y_pred))
    icbhi = icbhi_score(cm)

    return {
        "n": int(len(y_true)),
        "accuracy": acc,
        "macro_f1": macro_f1,
        "confusion_matrix": cm.tolist(),
        "icbhi": icbhi,
        "per_class": [
            {"label": LABELS[i], "precision": float(pre[i]),
             "recall": float(rec[i]), "f1": float(f1[i]),
             "support": int(support[i])}
            for i in range(len(LABELS))
        ],
        "elapsed_sec": time.time() - t0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--segment_sec", type=float, default=6.0)
    ap.add_argument("--max", type=int, default=0, help="0 이면 전체 val set")
    ap.add_argument("--out", default=str(WIN / "scripts" / "eval_diagnosis_result.json"))
    args = ap.parse_args()

    device = "cuda" if (args.device == "auto" and torch.cuda.is_available()) else \
             ("cuda" if args.device == "cuda" else "cpu")
    print(f"device       : {device}")
    print(f"VAL_CSV      : {VAL_CSV}")
    print(f"SR_CKPT      : {SR_CKPT}")
    print(f"CLS_BASELINE : {CLS_BASELINE}")
    print(f"CLS_SR       : {CLS_SR}")

    df = pd.read_csv(VAL_CSV)
    if args.max:
        df = df.head(args.max)
    print(f"val rows     : {len(df)}")

    # class distribution
    counts = Counter(df["label"])
    print(f"class dist   : {dict(counts)}")

    seg = int(args.segment_sec * HR_SR)
    print(f"\n[1/2] loading SR model ...")
    sr_model, _ = load_sr_model(str(SR_CKPT), device=device)
    sr_model.eval()
    for p in sr_model.parameters():
        p.requires_grad_(False)

    print("[2/2] loading classifiers ...")
    cls_baseline = load_classifier(CLS_BASELINE, device)
    cls_sr       = load_classifier(CLS_SR,       device)

    print(f"\n>>> evaluating Baseline (LR → baseline.pt)")
    res_baseline = evaluate("baseline", df, sr_model, cls_baseline, device, seg)

    print(f"\n>>> evaluating SR-boost (LR → SR → best_sr.pt)")
    res_srboost  = evaluate("sr-boost", df, sr_model, cls_sr,       device, seg)

    result = {
        "val_csv": str(VAL_CSV),
        "n_total": len(df),
        "class_distribution": dict(counts),
        "device": device,
        "segment_sec": args.segment_sec,
        "baseline":  res_baseline,
        "sr_boost":  res_srboost,
        "improvement": {
            "macro_f1_abs": res_srboost["macro_f1"] - res_baseline["macro_f1"],
            "macro_f1_rel": (res_srboost["macro_f1"] - res_baseline["macro_f1"]) / max(res_baseline["macro_f1"], 1e-9),
            "accuracy_abs": res_srboost["accuracy"] - res_baseline["accuracy"],
            "icbhi_abs":    res_srboost["icbhi"]["icbhi"] - res_baseline["icbhi"]["icbhi"],
        },
    }

    Path(args.out).write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n========= 결과 =========")
    print(f"  Baseline  : macro F1 = {res_baseline['macro_f1']:.4f} · acc = {res_baseline['accuracy']:.4f} · ICBHI = {res_baseline['icbhi']['icbhi']:.4f}")
    print(f"  SR-boost  : macro F1 = {res_srboost['macro_f1']:.4f} · acc = {res_srboost['accuracy']:.4f} · ICBHI = {res_srboost['icbhi']['icbhi']:.4f}")
    print(f"  Δ macro F1 = {result['improvement']['macro_f1_abs']:+.4f}  ({result['improvement']['macro_f1_rel']*100:+.2f}% relative)")
    print(f"\n  per-class (SR-boost):")
    for row in res_srboost["per_class"]:
        print(f"    {row['label']:10s}  P={row['precision']:.3f}  R={row['recall']:.3f}  F1={row['f1']:.3f}  n={row['support']}")
    print(f"\n  saved → {args.out}")


if __name__ == "__main__":
    main()
