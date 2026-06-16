"""
공사현장 안전 YOLO 모델 helmet AP 실측 — Roboflow Hard Hat Workers v6 test 88장.

3 조건 비교:
  ① Baseline       — image → YOLO
  ② 조건부 (운영)  — should_enhance(image) True → Zero-DCE → YOLO
                     False → image → YOLO
  ③ 무조건 (ablation) — image → Zero-DCE 강제 → YOLO

산출 metric (helmet 클래스 한정):
  • AP @ IoU 0.5 (PR curve 면적)
  • Precision / Recall / F1 (conf threshold 0.20)
  • TP / FP / FN
  • Latency per image (mean / median / p95)
  • 조건부 트리거 통계

dataset 한계:
  • Roboflow Hard Hat Workers v6 = single class (helmet) 라벨만 보유
  • 우리 YOLOv5s 모델은 실제 2 class (Hardhat / NO-Hardhat) 출력
  → 모델 예측 중 Hardhat 클래스만 필터하여 GT helmet 과 매칭
  → NO-Hardhat 탐지 정확도는 본 평가로 측정 불가 (데이터셋에 라벨 없음)
"""
import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import cv2

THIS = Path(__file__).resolve()
WIN = THIS.parent.parent
sys.path.insert(0, str(WIN))

SAFETY_ROOT = Path(os.environ.get("SAFETY_ROOT", str(WIN.parent / "수상관리")))
if str(SAFETY_ROOT) not in sys.path:
    sys.path.insert(0, str(SAFETY_ROOT))

DATASET_DEFAULT = Path("C:/Users/user/Downloads/Hard hat workers.v6i.yolov5pytorch")


# ─── 평가 보조 함수 ───────────────────────────────────────────────────

def yolo_to_xyxy(cx, cy, w, h, img_w, img_h):
    x1 = (cx - w / 2) * img_w
    y1 = (cy - h / 2) * img_h
    x2 = (cx + w / 2) * img_w
    y2 = (cy + h / 2) * img_h
    return [x1, y1, x2, y2]


def iou(b1, b2):
    x1 = max(b1[0], b2[0]); y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2]); y2 = min(b1[3], b2[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    a1 = max(0, b1[2] - b1[0]) * max(0, b1[3] - b1[1])
    a2 = max(0, b2[2] - b2[0]) * max(0, b2[3] - b2[1])
    return inter / (a1 + a2 - inter + 1e-9)


def compute_ap(preds, gts_by_img, iou_thr=0.5):
    """
    preds: list of (img_id, score, [x1,y1,x2,y2])
    gts_by_img: dict img_id → list of [x1,y1,x2,y2]
    반환: AP, precision_curve, recall_curve, TP/FP 누적, n_gt
    """
    preds_sorted = sorted(preds, key=lambda x: -x[1])
    n_gt = sum(len(b) for b in gts_by_img.values())
    if n_gt == 0:
        return {"ap": 0.0, "n_gt": 0, "n_pred": len(preds), "tp": 0, "fp": len(preds)}

    matched = {img_id: [False] * len(b) for img_id, b in gts_by_img.items()}
    tp_arr = np.zeros(len(preds_sorted))
    fp_arr = np.zeros(len(preds_sorted))

    for i, (img_id, _, box) in enumerate(preds_sorted):
        gts = gts_by_img.get(img_id, [])
        if not gts:
            fp_arr[i] = 1
            continue
        best, bi = 0.0, -1
        for j, gt in enumerate(gts):
            if matched[img_id][j]:
                continue
            v = iou(box, gt)
            if v > best:
                best, bi = v, j
        if best >= iou_thr:
            tp_arr[i] = 1
            matched[img_id][bi] = True
        else:
            fp_arr[i] = 1

    tp_cum = np.cumsum(tp_arr)
    fp_cum = np.cumsum(fp_arr)
    recall = tp_cum / n_gt
    precision = tp_cum / np.maximum(tp_cum + fp_cum, 1)

    # COCO-style exact area under PR curve (monotone-decreasing precision envelope)
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    ap = float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))

    return {
        "ap": ap,
        "n_gt": n_gt,
        "n_pred": len(preds),
        "tp": int(tp_cum[-1]) if len(tp_cum) else 0,
        "fp": int(fp_cum[-1]) if len(fp_cum) else 0,
        "fn": int(n_gt - (tp_cum[-1] if len(tp_cum) else 0)),
    }


def threshold_metrics(preds, gts_by_img, conf_thr=0.20, iou_thr=0.5):
    """주어진 confidence threshold 에서의 precision / recall / F1."""
    filtered = [(i, s, b) for i, s, b in preds if s >= conf_thr]
    stats = compute_ap(filtered, gts_by_img, iou_thr)
    p = stats["tp"] / max(stats["tp"] + stats["fp"], 1)
    r = stats["tp"] / max(stats["n_gt"], 1)
    f1 = 2 * p * r / max(p + r, 1e-9)
    return {**stats, "precision": p, "recall": r, "f1": f1, "conf_thr": conf_thr}


# ─── 메인 ───────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=str(DATASET_DEFAULT))
    ap.add_argument("--split", default="test", choices=["test", "valid"])
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--conf", type=float, default=0.20)
    ap.add_argument("--max", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    # 게이트 임계값 (CONFIG override) — 기본은 config.py 의 값 그대로
    ap.add_argument("--brightness", type=float, default=None,
                    help="밝기 트리거 임계값 (기본 70). 이 값 미만이면 enhance 트리거")
    ap.add_argument("--contrast", type=float, default=None,
                    help="대비 트리거 임계값 (기본 35)")
    ap.add_argument("--noise", type=float, default=None,
                    help="노이즈 트리거 임계값 (기본 0.02). 이 값 초과면 trigger")
    ap.add_argument("--out", default=str(WIN / "scripts" / "eval_safety_result.json"))
    args = ap.parse_args()

    # CPU 강제 + 게이트 임계값 override (CONFIG 의 기본값을 import 시점에 override)
    import config as safety_config
    safety_config.CONFIG.detection.device = args.device
    if args.brightness is not None:
        safety_config.CONFIG.enhancer.brightness_trigger = args.brightness
    if args.contrast is not None:
        safety_config.CONFIG.enhancer.contrast_trigger = args.contrast
    if args.noise is not None:
        safety_config.CONFIG.enhancer.noise_trigger = args.noise

    from services.detection_service import DetectionService
    from services.enhancer_service import LowLightEnhancerService
    from utils.image_quality import quality_report

    img_dir = Path(args.dataset) / args.split / "images"
    lbl_dir = Path(args.dataset) / args.split / "labels"
    images = sorted(list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png")))
    if args.max:
        images = images[: args.max]
    print(f"dataset    : {args.dataset}")
    print(f"split      : {args.split}")
    print(f"images     : {len(images)}")
    print(f"device     : {args.device}")
    print(f"IoU thr    : {args.iou} · conf thr (for P/R/F1): {args.conf}")
    print(f"게이트 임계값: brightness<{safety_config.CONFIG.enhancer.brightness_trigger} "
          f"OR contrast<{safety_config.CONFIG.enhancer.contrast_trigger} "
          f"OR noise>{safety_config.CONFIG.enhancer.noise_trigger}")

    print("\nloading YOLO + Zero-DCE ...")
    detector = DetectionService()
    enhancer = LowLightEnhancerService(device=args.device)

    # GT 로드
    gts_by_img = {}
    img_meta = {}
    for path in images:
        img_id = path.stem
        # 이미지 크기 (라벨 변환용)
        img = cv2.imread(str(path))
        H, W = img.shape[:2]
        img_meta[img_id] = (W, H, path)

        lbl = lbl_dir / (path.stem + ".txt")
        boxes = []
        if lbl.exists():
            for line in lbl.read_text().strip().splitlines():
                if not line:
                    continue
                parts = line.split()
                # 데이터셋이 single-class (helmet) 이므로 모든 box 사용
                cx, cy, bw, bh = map(float, parts[1:5])
                boxes.append(yolo_to_xyxy(cx, cy, bw, bh, W, H))
        gts_by_img[img_id] = boxes

    total_gt = sum(len(b) for b in gts_by_img.values())
    print(f"total GT helmet boxes: {total_gt}")

    # ── 3 조건 추론 ──
    preds_baseline = []
    preds_gated    = []
    preds_forced   = []
    lat_baseline = []
    lat_gated_total = []   # quality check + (optional enhance) + YOLO 전체
    lat_forced_total = []
    lat_enhance_only = []
    triggered_count = 0
    triggered_ids = []
    quality_records = []   # 각 이미지의 brightness/contrast/noise

    print(f"\n>>> running 3 conditions on {len(images)} images ...")
    log_every = max(1, len(images) // 10)
    t_total = time.time()

    for i, path in enumerate(images):
        img_id = path.stem
        img = cv2.imread(str(path))

        # 품질 측정 (게이트 결정용 + 분석)
        q = quality_report(img)
        will_trigger = enhancer.should_enhance(img)
        quality_records.append({
            "img_id": img_id,
            "brightness": float(q["brightness"]),
            "contrast": float(q["contrast"]),
            "noise": float(q["noise"]),
            "triggered": bool(will_trigger),
        })
        if will_trigger:
            triggered_count += 1
            triggered_ids.append(img_id)

        # ── ① Baseline ──
        t0 = time.perf_counter()
        dets_bl = detector.infer(img)
        lat_baseline.append((time.perf_counter() - t0) * 1000)
        for d in dets_bl:
            if d.class_name in ("helmet", "Hardhat"):
                preds_baseline.append((img_id, d.score, list(d.bbox)))

        # ── ③ 무조건 적용 (먼저 enhance 캐시) ──
        t0 = time.perf_counter()
        img_enh = enhancer.enhance(img)
        lat_enh = (time.perf_counter() - t0) * 1000
        lat_enhance_only.append(lat_enh)

        t0 = time.perf_counter()
        dets_force = detector.infer(img_enh)
        lat_force_yolo = (time.perf_counter() - t0) * 1000
        lat_forced_total.append(lat_enh + lat_force_yolo)
        for d in dets_force:
            if d.class_name in ("helmet", "Hardhat"):
                preds_forced.append((img_id, d.score, list(d.bbox)))

        # ── ② 조건부 (운영) — 트리거 여부에 따라 분기 ──
        if will_trigger:
            # 위에서 이미 계산한 enhance + YOLO 결과 재사용
            lat_gated_total.append(lat_enh + lat_force_yolo)
            for d in dets_force:
                if d.class_name in ("helmet", "Hardhat"):
                    preds_gated.append((img_id, d.score, list(d.bbox)))
        else:
            # Baseline 과 동일 — 위에서 계산한 dets_bl 재사용
            lat_gated_total.append(lat_baseline[-1])
            for d in dets_bl:
                if d.class_name in ("helmet", "Hardhat"):
                    preds_gated.append((img_id, d.score, list(d.bbox)))

        if (i + 1) % log_every == 0:
            dt = time.time() - t_total
            eta = dt / (i + 1) * (len(images) - i - 1)
            print(f"  {i+1}/{len(images)} · elapsed {dt:.0f}s · ETA {eta:.0f}s", flush=True)

    # ── 평가 ──
    print("\ncomputing metrics ...")
    ap_base   = compute_ap(preds_baseline, gts_by_img, args.iou)
    ap_gated  = compute_ap(preds_gated,    gts_by_img, args.iou)
    ap_forced = compute_ap(preds_forced,   gts_by_img, args.iou)

    pr_base   = threshold_metrics(preds_baseline, gts_by_img, args.conf, args.iou)
    pr_gated  = threshold_metrics(preds_gated,    gts_by_img, args.conf, args.iou)
    pr_forced = threshold_metrics(preds_forced,   gts_by_img, args.conf, args.iou)

    def lat_stats(arr):
        a = np.array(arr) if len(arr) else np.zeros(1)
        return {
            "mean": float(a.mean()),
            "median": float(np.median(a)),
            "p95": float(np.percentile(a, 95)),
            "min": float(a.min()),
            "max": float(a.max()),
        }

    bright_arr = np.array([r["brightness"] for r in quality_records])
    contr_arr  = np.array([r["contrast"]   for r in quality_records])
    noise_arr  = np.array([r["noise"]      for r in quality_records])

    result = {
        "dataset": str(args.dataset),
        "split": args.split,
        "n_images": len(images),
        "n_gt_helmet": total_gt,
        "iou_threshold": args.iou,
        "conf_threshold": args.conf,
        "device": args.device,
        "gate_thresholds": {
            "brightness": safety_config.CONFIG.enhancer.brightness_trigger,
            "contrast":   safety_config.CONFIG.enhancer.contrast_trigger,
            "noise":      safety_config.CONFIG.enhancer.noise_trigger,
        },
        "gate_stats": {
            "triggered": triggered_count,
            "triggered_ratio": triggered_count / max(len(images), 1),
            "triggered_ids_sample": triggered_ids[:10],
            "brightness": {"mean": float(bright_arr.mean()), "min": float(bright_arr.min()), "max": float(bright_arr.max()), "threshold": safety_config.CONFIG.enhancer.brightness_trigger},
            "contrast":   {"mean": float(contr_arr.mean()),  "min": float(contr_arr.min()),  "max": float(contr_arr.max()),  "threshold": safety_config.CONFIG.enhancer.contrast_trigger},
            "noise":      {"mean": float(noise_arr.mean()),  "min": float(noise_arr.min()),  "max": float(noise_arr.max()),  "threshold": safety_config.CONFIG.enhancer.noise_trigger},
        },
        "baseline": {
            **ap_base, **{k: pr_base[k] for k in ("precision", "recall", "f1", "conf_thr")},
            "latency_ms": lat_stats(lat_baseline),
        },
        "gated": {
            **ap_gated, **{k: pr_gated[k] for k in ("precision", "recall", "f1", "conf_thr")},
            "latency_ms": lat_stats(lat_gated_total),
        },
        "forced": {
            **ap_forced, **{k: pr_forced[k] for k in ("precision", "recall", "f1", "conf_thr")},
            "latency_ms": lat_stats(lat_forced_total),
            "enhance_only_ms": lat_stats(lat_enhance_only),
        },
        "delta": {
            "gated_vs_baseline_ap":   ap_gated["ap"]  - ap_base["ap"],
            "forced_vs_baseline_ap":  ap_forced["ap"] - ap_base["ap"],
            "forced_vs_gated_ap":     ap_forced["ap"] - ap_gated["ap"],
        },
    }
    Path(args.out).write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n=========================== 결과 ===========================")
    print(f"  images: {len(images)} · GT helmet boxes: {total_gt}")
    print(f"  Zero-DCE 게이트 트리거: {triggered_count}/{len(images)} "
          f"({triggered_count/len(images)*100:.1f}%)")
    print(f"  품질 평균: brightness={bright_arr.mean():.1f}  contrast={contr_arr.mean():.1f}  noise={noise_arr.mean():.4f}")
    print()
    print(f"  {'조건':<14}  {'AP@0.5':>8}  {'P':>6}  {'R':>6}  {'F1':>6}  {'TP':>4} {'FP':>4} {'FN':>4}  {'lat(ms)':>9}")
    print("  " + "-" * 80)
    for tag, stats, lat in (
        ("① Baseline",   {**ap_base,   **pr_base},   lat_baseline),
        ("② 조건부",     {**ap_gated,  **pr_gated},  lat_gated_total),
        ("③ 무조건",     {**ap_forced, **pr_forced}, lat_forced_total),
    ):
        mlat = np.mean(lat) if len(lat) else 0.0
        print(f"  {tag:<14}  {stats['ap']:>8.4f}  "
              f"{stats['precision']:>6.3f}  {stats['recall']:>6.3f}  {stats['f1']:>6.3f}  "
              f"{stats['tp']:>4} {stats['fp']:>4} {stats['fn']:>4}  {mlat:>9.1f}")
    print()
    print(f"  Δ AP (② − ①) = {result['delta']['gated_vs_baseline_ap']:+.4f}")
    print(f"  Δ AP (③ − ①) = {result['delta']['forced_vs_baseline_ap']:+.4f}")
    print(f"\n  saved → {args.out}")


if __name__ == "__main__":
    main()
