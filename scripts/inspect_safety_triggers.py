"""
Zero-DCE 게이트가 어떤 sample 에 트리거됐는지 진단.

각 이미지의 brightness / contrast / noise 측정값과
현재 임계값 (brightness < 70, contrast < 35, noise > 0.02) 위반 여부를 표시.

목적:
  • 트리거된 sample 의 실제 품질 분포 확인
  • 임계값이 너무 관대한지 (정상 sample 도 trigger) / 적절한지 판단
  • 더 엄격한 임계값 후보 도출

실행:
  python scripts/inspect_safety_triggers.py
  python scripts/inspect_safety_triggers.py --only-triggered      # 트리거된 것만 보기
  python scripts/inspect_safety_triggers.py --sort brightness     # 밝기 오름차순 정렬
"""
import argparse
import json
import os
import sys
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=str(DATASET_DEFAULT))
    ap.add_argument("--split", default="test", choices=["test", "valid", "train"])
    ap.add_argument("--brightness", type=float, default=70.0)
    ap.add_argument("--contrast",   type=float, default=35.0)
    ap.add_argument("--noise",      type=float, default=0.02)
    ap.add_argument("--only-triggered", action="store_true")
    ap.add_argument("--sort", default="brightness",
                    choices=["brightness", "contrast", "noise", "name"])
    ap.add_argument("--out", default=str(WIN / "scripts" / "inspect_safety_triggers.json"))
    args = ap.parse_args()

    from utils.image_quality import quality_report

    img_dir = Path(args.dataset) / args.split / "images"
    images = sorted(list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png")))
    print(f"dataset    : {args.dataset}")
    print(f"split      : {args.split}")
    print(f"images     : {len(images)}")
    print(f"임계값      : brightness < {args.brightness}  OR  contrast < {args.contrast}  OR  noise > {args.noise}")
    print()

    records = []
    for path in images:
        img = cv2.imread(str(path))
        q = quality_report(img)
        b, c, n = float(q["brightness"]), float(q["contrast"]), float(q["noise"])

        # 각 임계값 위반 여부
        v_b = b < args.brightness
        v_c = c < args.contrast
        v_n = n > args.noise
        triggered = v_b or v_c or v_n

        # 어떤 임계값이 발동됐는지 표시
        why = []
        if v_b: why.append(f"B<{args.brightness:g}")
        if v_c: why.append(f"C<{args.contrast:g}")
        if v_n: why.append(f"N>{args.noise:g}")

        records.append({
            "img_id": path.stem,
            "brightness": b,
            "contrast":   c,
            "noise":      n,
            "triggered":  triggered,
            "violates":   why,
        })

    # 통계
    b_arr = np.array([r["brightness"] for r in records])
    c_arr = np.array([r["contrast"]   for r in records])
    n_arr = np.array([r["noise"]      for r in records])
    trig  = [r for r in records if r["triggered"]]
    nontrig = [r for r in records if not r["triggered"]]

    print("┌─ 전체 분포 ──────────────────────────────────────────────")
    print(f"│  brightness    mean={b_arr.mean():6.2f}  median={np.median(b_arr):6.2f}  "
          f"min={b_arr.min():6.2f}  max={b_arr.max():6.2f}  p5={np.percentile(b_arr,5):6.2f}  p25={np.percentile(b_arr,25):6.2f}")
    print(f"│  contrast      mean={c_arr.mean():6.2f}  median={np.median(c_arr):6.2f}  "
          f"min={c_arr.min():6.2f}  max={c_arr.max():6.2f}  p5={np.percentile(c_arr,5):6.2f}  p25={np.percentile(c_arr,25):6.2f}")
    print(f"│  noise         mean={n_arr.mean():.5f}  median={np.median(n_arr):.5f}  "
          f"min={n_arr.min():.5f}  max={n_arr.max():.5f}  p75={np.percentile(n_arr,75):.5f}  p95={np.percentile(n_arr,95):.5f}")
    print("└──────────────────────────────────────────────────────────")
    print(f"  triggered     : {len(trig)} / {len(records)}  ({len(trig)/len(records)*100:.1f}%)")
    print(f"  non-triggered : {len(nontrig)} / {len(records)}")

    # 트리거된 sample 의 품질 분포 (정말 어두운가?)
    if trig:
        tb = np.array([r["brightness"] for r in trig])
        tc = np.array([r["contrast"]   for r in trig])
        tn = np.array([r["noise"]      for r in trig])
        print()
        print("┌─ 트리거된 sample 의 분포 ───────────────────────────────")
        print(f"│  brightness    mean={tb.mean():6.2f}  min={tb.min():6.2f}  max={tb.max():6.2f}")
        print(f"│  contrast      mean={tc.mean():6.2f}  min={tc.min():6.2f}  max={tc.max():6.2f}")
        print(f"│  noise         mean={tn.mean():.5f}  min={tn.min():.5f}  max={tn.max():.5f}")
        print("└──────────────────────────────────────────────────────────")

        # 어떤 임계값에 의해 트리거됐는지 breakdown
        b_only = sum(1 for r in trig if r["violates"] == ["B<%g" % args.brightness])
        c_only = sum(1 for r in trig if r["violates"] == ["C<%g" % args.contrast])
        n_only = sum(1 for r in trig if r["violates"] == ["N>%g" % args.noise])
        multi  = sum(1 for r in trig if len(r["violates"]) > 1)
        print()
        print(f"  trigger 원인 breakdown:")
        print(f"    brightness 만   : {b_only}")
        print(f"    contrast 만     : {c_only}")
        print(f"    noise 만        : {n_only}")
        print(f"    복수 임계값     : {multi}")

    # ── 임계값 후보 시뮬레이션 (안 바꾸고 다양한 값 적용 시 몇 장이 trigger 되는지) ──
    print()
    print("┌─ 임계값 후보 시뮬레이션 ────────────────────────────────")
    print(f"│  {'brightness':>10}  {'contrast':>9}  {'noise':>7}  {'trig (n/88)':>12}  {'%':>5}")
    candidates = [
        (70, 35, 0.020),  # 현재
        (60, 30, 0.025),
        (50, 25, 0.030),
        (40, 20, 0.040),
        (30, 15, 0.050),
    ]
    for thr_b, thr_c, thr_n in candidates:
        n_trig = sum(1 for r in records
                     if r["brightness"] < thr_b or r["contrast"] < thr_c or r["noise"] > thr_n)
        marker = "  ← 현재" if (thr_b, thr_c, thr_n) == (70, 35, 0.020) else ""
        print(f"│  {thr_b:>10}  {thr_c:>9}  {thr_n:>7}  {n_trig:>12}  {n_trig/len(records)*100:>4.1f}%{marker}")
    print("└──────────────────────────────────────────────────────────")

    # 정렬 후 출력
    if args.sort == "brightness": records.sort(key=lambda r: r["brightness"])
    elif args.sort == "contrast": records.sort(key=lambda r: r["contrast"])
    elif args.sort == "noise":    records.sort(key=lambda r: -r["noise"])
    else: records.sort(key=lambda r: r["img_id"])

    print()
    print(f"=== 개별 sample ({len(records)}장, sort={args.sort}) ===")
    print(f"  {'img_id':<40}  {'B':>6}  {'C':>6}  {'N':>7}  {'trig':>5}  violates")
    print("  " + "-" * 95)
    for r in records:
        if args.only_triggered and not r["triggered"]:
            continue
        flag = "★" if r["triggered"] else ""
        why = ",".join(r["violates"]) if r["violates"] else "-"
        # img_id 너무 길면 자름
        display_id = r["img_id"][:40] if len(r["img_id"]) <= 40 else r["img_id"][:37] + "..."
        print(f"  {display_id:<40}  {r['brightness']:>6.2f}  {r['contrast']:>6.2f}  "
              f"{r['noise']:>7.5f}  {flag:>5}  {why}")

    Path(args.out).write_text(json.dumps({
        "thresholds": {"brightness": args.brightness, "contrast": args.contrast, "noise": args.noise},
        "n_total": len(images),
        "n_triggered": len(trig),
        "records": records,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n  saved → {args.out}")


if __name__ == "__main__":
    main()
