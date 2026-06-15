"""
소아 X-ray 분류 모델 실측 평가 — Kermany test split 625장에 대한 진짜 metric 산출.

모델은 14-label multi-label classifier 이지만, 본 평가는 Pneumonia channel(index 6) 만 사용.
(Kermany 가 NORMAL / PNEUMONIA binary 라벨만 제공)

산출 metric:
  • AUROC (ROC curve area)
  • F1, precision, recall (threshold 0.5)
  • Sensitivity (recall on PNEUMONIA)
  • Specificity (true negative rate on NORMAL)
  • Confusion matrix
  • Inference latency 통계

실행:
  python scripts/eval_studies.py
  python scripts/eval_studies.py --kermany "C:/Users/user/Downloads/archive (1)/chest_xray"
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image


THIS = Path(__file__).resolve()
WIN = THIS.parent.parent
sys.path.insert(0, str(WIN))

# 소아 X-ray src 패키지 위치 — settings 와 같은 방식으로
XRAY_ROOT = Path(os.environ.get("XRAY_ROOT", str(WIN.parent / "수상관리_소아엑스레이")))
if str(XRAY_ROOT) not in sys.path:
    sys.path.append(str(XRAY_ROOT))

CKPT_DEFAULT = XRAY_ROOT / "outputs" / "classifier" / "best.pt"
KERMANY_DEFAULT = Path("C:/Users/user/Downloads/archive (1)/chest_xray")

from src.classifier import XrayDiagnosticEngine  # noqa: E402
from src.data import DISEASE_LABELS, build_transform  # noqa: E402
from src.classifier.model import MultiLabelXrayClassifier  # noqa: E402

from sklearn.metrics import (  # noqa: E402
    roc_auc_score, f1_score, precision_score, recall_score,
    confusion_matrix, average_precision_score,
)


PNEUMONIA_IDX = DISEASE_LABELS.index("Pneumonia")  # 6
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def gather_images(root: Path):
    """test/{NORMAL,PNEUMONIA} 폴더에서 (path, label) 리스트 수집. label: 0=Normal, 1=Pneumonia."""
    items = []
    for cls, lab in (("NORMAL", 0), ("PNEUMONIA", 1)):
        d = root / cls
        if not d.is_dir():
            raise FileNotFoundError(f"{d} 없음")
        for p in sorted(d.iterdir()):
            if p.suffix.lower() in IMAGE_EXTS:
                items.append((p, lab))
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kermany", default=str(KERMANY_DEFAULT),
                    help="Kermany chest_xray 폴더 (test/ 가 직계 자식이어야 함)")
    ap.add_argument("--split", default="test", choices=["test", "val", "train"])
    ap.add_argument("--ckpt", default=str(CKPT_DEFAULT))
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--max", type=int, default=0)
    ap.add_argument("--out", default=str(WIN / "scripts" / "eval_studies_result.json"))
    args = ap.parse_args()

    device = "cuda" if (args.device == "auto" and torch.cuda.is_available()) else \
             ("cuda" if args.device == "cuda" else "cpu")
    print(f"device  : {device}")
    print(f"ckpt    : {args.ckpt}")
    print(f"kermany : {args.kermany} (split={args.split})")
    print(f"thr     : {args.threshold}")

    split_dir = Path(args.kermany) / args.split
    items = gather_images(split_dir)
    if args.max:
        items = items[:args.max]
    print(f"images  : {len(items)} (NORMAL={sum(1 for _,l in items if l==0)} · "
          f"PNEUMONIA={sum(1 for _,l in items if l==1)})")

    # 모델 로드 — XrayDiagnosticEngine 은 Grad-CAM 까지 등록하므로 그냥 model 만 직접 로드
    print("\nloading classifier...")
    model = MultiLabelXrayClassifier(pretrained=False).to(device).eval()
    state = torch.load(args.ckpt, map_location=device, weights_only=False)
    state = state["model"] if isinstance(state, dict) and "model" in state else state
    model.load_state_dict(state)
    transform = build_transform(image_size=224, train=False)

    # 추론 루프
    y_true, y_score = [], []
    latencies = []
    t0 = time.time()
    log_every = max(1, len(items) // 20)
    with torch.no_grad():
        for i, (path, lab) in enumerate(items):
            try:
                img = Image.open(path).convert("L")
            except Exception as e:
                print(f"  [skip] {path.name} ({e})")
                continue
            x = transform(img).unsqueeze(0).to(device)
            t1 = time.perf_counter()
            logits = model(x)
            probs = torch.sigmoid(logits).cpu().numpy()[0]
            latencies.append((time.perf_counter() - t1) * 1000.0)
            y_true.append(lab)
            y_score.append(float(probs[PNEUMONIA_IDX]))
            if (i + 1) % log_every == 0:
                dt = time.time() - t0
                eta = dt / (i + 1) * (len(items) - i - 1)
                print(f"  {i+1}/{len(items)} · {dt:.0f}s · ETA {eta:.0f}s", flush=True)

    y_true = np.array(y_true); y_score = np.array(y_score)
    y_pred = (y_score >= args.threshold).astype(np.int32)

    auroc = float(roc_auc_score(y_true, y_score))
    ap_   = float(average_precision_score(y_true, y_score))
    f1    = float(f1_score(y_true, y_pred, zero_division=0))
    prec  = float(precision_score(y_true, y_pred, zero_division=0))
    rec   = float(recall_score(y_true, y_pred, zero_division=0))
    cm    = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel().tolist()
    spec  = tn / max(tn + fp, 1)
    sens  = tp / max(tp + fn, 1)

    lat = np.array(latencies)
    result = {
        "kermany_root": args.kermany,
        "split": args.split,
        "threshold": args.threshold,
        "device": device,
        "n_total": int(len(y_true)),
        "n_normal": int((y_true == 0).sum()),
        "n_pneumonia": int((y_true == 1).sum()),
        "auroc": auroc,
        "average_precision": ap_,
        "f1": f1,
        "precision": prec,
        "recall_sensitivity": rec,
        "specificity": spec,
        "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
        "latency_ms": {
            "mean": float(lat.mean()),
            "median": float(np.median(lat)),
            "p95": float(np.percentile(lat, 95)),
            "min": float(lat.min()),
            "max": float(lat.max()),
        },
        "elapsed_sec": time.time() - t0,
    }
    Path(args.out).write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n========= 결과 =========")
    print(f"  N           : {result['n_total']} (NORMAL {result['n_normal']} · PNEUMONIA {result['n_pneumonia']})")
    print(f"  AUROC       : {auroc:.4f}")
    print(f"  AP          : {ap_:.4f}")
    print(f"  F1 @ {args.threshold} : {f1:.4f}")
    print(f"  Sensitivity : {sens:.4f}   (Pneumonia recall)")
    print(f"  Specificity : {spec:.4f}   (Normal TNR)")
    print(f"  Precision   : {prec:.4f}")
    print(f"  Confusion   : TN={tn} FP={fp} FN={fn} TP={tp}")
    print(f"  Latency     : mean={lat.mean():.1f}ms · p95={np.percentile(lat,95):.1f}ms")
    print(f"\n  saved → {args.out}")


if __name__ == "__main__":
    main()
