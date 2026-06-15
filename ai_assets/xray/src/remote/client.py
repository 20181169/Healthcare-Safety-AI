"""
구급차 / 공공 보건의료 센터 단말 클라이언트.

휴대형 X-ray 장비로 촬영한 영상을 원격 진단 지원 서버에 전송하고,
판독 보조 결과(질병별 확률, Top-N 의심 소견, Grad-CAM 영상)를 수신한다.

콘솔 텍스트 형태의 'AI 1차 판독 리포트' 를 출력하여 응급 의료진의
빠른 의사결정을 돕는 것을 목표로 한다.
"""

import argparse
import base64
import json
import os
from typing import Dict

import requests


def send_for_diagnosis(server_url: str, image_path: str, threshold: float) -> Dict:
    with open(image_path, "rb") as f:
        files = {"image": (os.path.basename(image_path), f, "image/png")}
        data = {"threshold": str(threshold)}
        resp = requests.post(f"{server_url}/predict", files=files, data=data, timeout=30)
    resp.raise_for_status()
    return resp.json()


def request_gradcam(server_url: str, image_path: str, class_name: str, save_to: str) -> Dict:
    with open(image_path, "rb") as f:
        files = {"image": (os.path.basename(image_path), f, "image/png")}
        data = {"class": class_name}
        resp = requests.post(f"{server_url}/predict_cam", files=files, data=data, timeout=30)
    resp.raise_for_status()
    payload = resp.json()
    with open(save_to, "wb") as out:
        out.write(base64.b64decode(payload["gradcam_png_base64"]))
    return payload


def print_report(result: Dict, top_k: int = 5):
    preds = result["predictions"]
    ranked = sorted(preds.items(), key=lambda kv: kv[1]["prob"], reverse=True)

    print("=" * 60)
    print("      [AI 판독 보조 리포트 - 소아 흉부 X-ray]")
    print("=" * 60)
    print(f"추론 소요 시간 : {result['elapsed_ms']} ms")
    print(f"저장 파일      : {result['image_saved_as']}")
    print("-" * 60)
    print(f"Top-{top_k} 의심 소견 (확률 내림차순):")
    for name, info in ranked[:top_k]:
        flag = " ●" if info["positive"] else "  "
        print(f"  {flag} {name:<22s}  prob = {info['prob']*100:6.2f} %")
    print("-" * 60)

    positives = result["positive_findings"]
    if positives:
        print(f"⚠️  임계값 이상 양성 소견: {', '.join(positives)}")
        print("   → 원격지 영상의학과 전문의에게 즉시 전송 권장")
    else:
        print("✓ 임계값 이상 양성 소견 없음 (단, 임상 판단 우선)")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", type=str, default="http://localhost:8080")
    parser.add_argument("--image", type=str, required=True, help="휴대형 X-ray 촬영 파일")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--cam_for", type=str, default=None,
                        help="특정 클래스에 대한 Grad-CAM 영상도 요청 (예: Pneumonia)")
    parser.add_argument("--cam_out", type=str, default="./gradcam.png")
    args = parser.parse_args()

    print(f"[전송] {args.image} → {args.server}/predict")
    result = send_for_diagnosis(args.server, args.image, args.threshold)
    print_report(result)

    if args.cam_for:
        print(f"[Grad-CAM 요청] class={args.cam_for}")
        info = request_gradcam(args.server, args.image, args.cam_for, args.cam_out)
        print(f"  · {args.cam_for}: prob = {info['prob']*100:.2f}%")
        print(f"  · heatmap 저장 → {args.cam_out}")

    # 로컬 로그 (JSON) 저장 - 추후 원격지 EMR 시스템과 연동 가능
    log_path = os.path.splitext(args.image)[0] + "_aireport.json"
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"[로그] {log_path}")


if __name__ == "__main__":
    main()
