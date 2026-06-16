"""
합성 저조도 darkening 적용 후 helmet AP 측정 — Zero-DCE 의 진짜 저조도 효과 검증.

원본 88장에 4가지 감마 값 적용:
  • gamma=1.0 — 원본 (변화 없음)
  • gamma=1.5 — 약간 어두움 (밝기 ~70% 감소)
  • gamma=2.5 — 어두움 (밝기 ~40% 감소)
  • gamma=4.0 — 매우 어두움 (밝기 ~25% 감소)

감마 보정 공식 (카메라 underexposure 시뮬레이션):
  output = (input/255)^gamma * 255      (gamma > 1 → 더 어두움)

각 darkened 버전에 3 조건 측정:
  ① Baseline       — darkened → YOLO
  ② 조건부          — should_enhance(darkened) True → Zero-DCE → YOLO
  ③ 무조건 적용     — darkened → Zero-DCE 강제 → YOLO

기대 결과:
  • gamma 1.0: Baseline ≈ Zero-DCE (이미 측정)
  • gamma 1.5 ~ 4.0: Baseline AP 감소, Zero-DCE 가 복원해야 정상
  • 만약 Zero-DCE 가 복원 못 함 → 시스템 가정에 결함

실행:
  python scripts/eval_safety_synthetic_dark.py
  python scripts/eval_safety_synthetic_dark.py --gammas 1.0 2.0 3.0
  python scripts/eval_safety_synthetic_dark.py --gate-brightness 70   # 원본 gate 사용
"""
import argparse
import json
import os
import sys
import time
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


# ─── 평가 함수 (eval_safety.py 와 동일) ────────────────────────────

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
    preds_sorted = sorted(preds, key=lambda x: -x[1])
    n_gt = sum(len(b) for b in gts_by_img.values())
    if n_gt == 0:
        return {"ap": 0.0, "n_gt": 0, "n_pred": len(preds), "tp": 0, "fp": len(preds), "fn": 0}

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
    filtered = [(i, s, b) for i, s, b in preds if s >= conf_thr]
    s = compute_ap(filtered, gts_by_img, iou_thr)
    p = s["tp"] / max(s["tp"] + s["fp"], 1)
    r = s["tp"] / max(s["n_gt"], 1)
    f1 = 2 * p * r / max(p + r, 1e-9)
    return {**s, "precision": p, "recall": r, "f1": f1}


# ─── Darkening ─────────────────────────────────────────────────────

def apply_gamma_darkening(img: np.ndarray, gamma: float) -> np.ndarray:
    """감마 보정으로 darkening. gamma > 1 = 더 어두움.
       카메라 underexposure 시뮬레이션 (선형 multiplication 보다 더 사실적)."""
    if gamma <= 0:
        raise ValueError(f"gamma must be > 0, got {gamma}")
    if abs(gamma - 1.0) < 1e-6:
        return img.copy()
    f = img.astype(np.float32) / 255.0
    darkened = np.power(f, gamma) * 255.0
    return np.clip(darkened, 0, 255).astype(np.uint8)


# ─── 메인 ───────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=str(DATASET_DEFAULT))
    ap.add_argument("--split", default="test", choices=["test", "valid"])
    ap.add_argument("--gammas", nargs="+", type=float,
                    default=[1.0, 1.5, 2.5, 4.0],
                    help="darkening 감마 값들 (>1 = 더 어두움). 예: --gammas 1.0 2.0 3.0")
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--conf", type=float, default=0.20)
    # 게이트 설정 (기본은 우리가 찾은 최적값)
    ap.add_argument("--gate-brightness", type=float, default=60.0,
                    help="밝기 트리거 임계값 (기본 60, 우리 분석 결과 최적)")
    ap.add_argument("--gate-contrast", type=float, default=30.0)
    ap.add_argument("--gate-noise", type=float, default=999.0,
                    help="노이즈 트리거 임계값 (기본 999=비활성, 우리 분석 결과 결함)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--max", type=int, default=0)
    ap.add_argument("--out", default=str(WIN / "scripts" / "eval_safety_synthetic_dark_result.json"))
    args = ap.parse_args()

    # CONFIG override
    import config as safety_config
    safety_config.CONFIG.detection.device = args.device
    safety_config.CONFIG.enhancer.brightness_trigger = args.gate_brightness
    safety_config.CONFIG.enhancer.contrast_trigger = args.gate_contrast
    safety_config.CONFIG.enhancer.noise_trigger = args.gate_noise

    from services.detection_service import DetectionService
    from services.enhancer_service import LowLightEnhancerService
    from utils.image_quality import quality_report

    img_dir = Path(args.dataset) / args.split / "images"
    lbl_dir = Path(args.dataset) / args.split / "labels"
    images = sorted(list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png")))
    if args.max:
        images = images[: args.max]

    print(f"dataset       : {args.dataset}")
    print(f"split         : {args.split}")
    print(f"images        : {len(images)}")
    print(f"gammas        : {args.gammas}  (>1 = 더 어두움)")
    print(f"gate (최적)    : brightness<{args.gate_brightness} OR "
          f"contrast<{args.gate_contrast} OR noise>{args.gate_noise}")
    print(f"IoU thr       : {args.iou} · conf thr (P/R/F1): {args.conf}")
    print()

    print("loading YOLO + Zero-DCE ...")
    detector = DetectionService()
    enhancer = LowLightEnhancerService(device=args.device)

    # GT 로드 (감마 무관 — 같은 이미지 좌표)
    gts_by_img = {}
    img_data = {}
    for path in images:
        img_id = path.stem
        img = cv2.imread(str(path))
        H, W = img.shape[:2]
        img_data[img_id] = (img, W, H, path)

        lbl = lbl_dir / (path.stem + ".txt")
        boxes = []
        if lbl.exists():
            for line in lbl.read_text().strip().splitlines():
                if not line:
                    continue
                parts = line.split()
                cx, cy, bw, bh = map(float, parts[1:5])
                boxes.append(yolo_to_xyxy(cx, cy, bw, bh, W, H))
        gts_by_img[img_id] = boxes

    total_gt = sum(len(b) for b in gts_by_img.values())
    print(f"total GT helmet boxes: {total_gt}")

    # ── 감마별 outer loop ──
    all_results = {}
    for g_idx, gamma in enumerate(args.gammas):
        print(f"\n{'═' * 70}")
        print(f"  [{g_idx+1}/{len(args.gammas)}]  gamma = {gamma}  "
              f"({'원본' if abs(gamma-1.0) < 1e-6 else '어둡게 ↓' if gamma > 1 else '밝게 ↑'})")
        print(f"{'═' * 70}")

        preds_baseline = []
        preds_gated    = []
        preds_forced   = []
        lat_baseline = []
        lat_gated    = []
        lat_forced   = []
        triggered_count = 0
        bright_arr = []

        t0_g = time.time()
        log_every = max(1, len(images) // 8)

        for i, path in enumerate(images):
            img_id = path.stem
            img_orig, W, H, _ = img_data[img_id]

            # ── darkening 적용 ──
            img = apply_gamma_darkening(img_orig, gamma)

            # 품질 측정 (게이트 결정용)
            q = quality_report(img)
            bright_arr.append(q["brightness"])
            will_trigger = enhancer.should_enhance(img)
            if will_trigger:
                triggered_count += 1

            # ① Baseline
            t1 = time.perf_counter()
            dets_bl = detector.infer(img)
            lat_baseline.append((time.perf_counter() - t1) * 1000)
            for d in dets_bl:
                if d.class_name in ("helmet", "Hardhat"):
                    preds_baseline.append((img_id, d.score, list(d.bbox)))

            # ③ 무조건 (먼저 enhance + YOLO)
            t1 = time.perf_counter()
            img_enh = enhancer.enhance(img)
            t_enh = (time.perf_counter() - t1) * 1000
            t1 = time.perf_counter()
            dets_force = detector.infer(img_enh)
            t_yolo = (time.perf_counter() - t1) * 1000
            lat_forced.append(t_enh + t_yolo)
            for d in dets_force:
                if d.class_name in ("helmet", "Hardhat"):
                    preds_forced.append((img_id, d.score, list(d.bbox)))

            # ② 조건부 (트리거 여부에 따라 분기, 위 결과 재사용)
            if will_trigger:
                lat_gated.append(t_enh + t_yolo)
                for d in dets_force:
                    if d.class_name in ("helmet", "Hardhat"):
                        preds_gated.append((img_id, d.score, list(d.bbox)))
            else:
                lat_gated.append(lat_baseline[-1])
                for d in dets_bl:
                    if d.class_name in ("helmet", "Hardhat"):
                        preds_gated.append((img_id, d.score, list(d.bbox)))

            if (i + 1) % log_every == 0:
                dt = time.time() - t0_g
                eta = dt / (i + 1) * (len(images) - i - 1)
                print(f"  {i+1}/{len(images)} · elapsed {dt:.0f}s · ETA {eta:.0f}s", flush=True)

        # ── 지표 계산 ──
        ap_b = compute_ap(preds_baseline, gts_by_img, args.iou)
        ap_g = compute_ap(preds_gated,    gts_by_img, args.iou)
        ap_f = compute_ap(preds_forced,   gts_by_img, args.iou)
        pr_b = threshold_metrics(preds_baseline, gts_by_img, args.conf, args.iou)
        pr_g = threshold_metrics(preds_gated,    gts_by_img, args.conf, args.iou)
        pr_f = threshold_metrics(preds_forced,   gts_by_img, args.conf, args.iou)

        bright_mean = float(np.mean(bright_arr))
        all_results[f"gamma_{gamma}"] = {
            "gamma": gamma,
            "darkened_brightness_mean": bright_mean,
            "triggered": triggered_count,
            "baseline": {**ap_b, **{k: pr_b[k] for k in ("precision", "recall", "f1")},
                         "latency_ms_mean": float(np.mean(lat_baseline))},
            "gated":    {**ap_g, **{k: pr_g[k] for k in ("precision", "recall", "f1")},
                         "latency_ms_mean": float(np.mean(lat_gated))},
            "forced":   {**ap_f, **{k: pr_f[k] for k in ("precision", "recall", "f1")},
                         "latency_ms_mean": float(np.mean(lat_forced))},
        }

        # ── 출력 ──
        print()
        print(f"  darkening 후 평균 밝기: {bright_mean:.1f} (원본 약 140) · "
              f"게이트 트리거: {triggered_count}/{len(images)}")
        print()
        print(f"  {'조건':<14}  {'AP@0.5':>8}  {'P':>6}  {'R':>6}  {'F1':>6}  "
              f"{'TP':>4} {'FP':>4} {'FN':>4}  {'lat(ms)':>9}")
        print("  " + "-" * 80)
        for tag, st, lat in (
            ("① Baseline",   {**ap_b, **pr_b}, lat_baseline),
            ("② 조건부",     {**ap_g, **pr_g}, lat_gated),
            ("③ 무조건",     {**ap_f, **pr_f}, lat_forced),
        ):
            mlat = np.mean(lat)
            print(f"  {tag:<14}  {st['ap']:>8.4f}  "
                  f"{st['precision']:>6.3f}  {st['recall']:>6.3f}  {st['f1']:>6.3f}  "
                  f"{st['tp']:>4} {st['fp']:>4} {st['fn']:>4}  {mlat:>9.1f}")
        print()
        print(f"  Δ AP (② − ①) = {ap_g['ap'] - ap_b['ap']:+.4f}   "
              f"← Zero-DCE 가 darkening 손실을 회복하는가?")
        print(f"  Δ AP (③ − ①) = {ap_f['ap'] - ap_b['ap']:+.4f}")

    # ── 최종 요약 표 ──
    print(f"\n{'═' * 70}")
    print("  📊 감마별 요약 (Zero-DCE 의 저조도 효과 검증)")
    print(f"{'═' * 70}")
    print(f"  {'gamma':>6}  {'avg brightness':>15}  {'① Baseline':>12}  "
          f"{'② 조건부':>12}  {'③ 무조건':>12}  {'② − ①':>10}")
    print("  " + "-" * 80)
    for gamma in args.gammas:
        r = all_results[f"gamma_{gamma}"]
        ap_b = r["baseline"]["ap"]
        ap_g = r["gated"]["ap"]
        ap_f = r["forced"]["ap"]
        print(f"  {gamma:>6.2f}  {r['darkened_brightness_mean']:>15.1f}  "
              f"{ap_b:>12.4f}  {ap_g:>12.4f}  {ap_f:>12.4f}  "
              f"{ap_g - ap_b:>+10.4f}")

    print()
    print("  해석:")
    print("    • gamma 1.0 = 원본")
    print("    • gamma 가 클수록 더 어두움 (gamma 4.0 = 매우 어두움)")
    print("    • 'Δ ② − ①' 양수 ⇒ Zero-DCE 가 darkening 손실을 회복")
    print("    • 'Δ ② − ①' 음수 ⇒ Zero-DCE 가 도움 안 됨 (이 모델 domain 에 부적합)")

    Path(args.out).write_text(json.dumps({
        "dataset": str(args.dataset),
        "split": args.split,
        "n_images": len(images),
        "n_gt_helmet": total_gt,
        "gammas": args.gammas,
        "gate": {
            "brightness": args.gate_brightness,
            "contrast":   args.gate_contrast,
            "noise":      args.gate_noise,
        },
        "results": all_results,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n  saved → {args.out}")


if __name__ == "__main__":
    main()
