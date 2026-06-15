"""
원격 진단 지원 서버.

[시나리오]
구급차/공공 보건의료 센터에 배치된 휴대형 X-ray 장비에서 촬영된 영상을
HTTP 로 업로드 → 서버에서 Multi-Label 분류 모델이 즉시 판독 보조 결과(확률+
Grad-CAM)를 반환 → 출동 의료진 또는 원격지 영상의학과 전문의의 의사결정을 지원.

엔드포인트:
  POST /predict        : 영상 업로드 + 추론 결과 JSON
  POST /predict_cam    : 영상 + 클래스 이름 → Grad-CAM PNG (base64)
  GET  /healthz        : 헬스체크
"""

import argparse
import base64
import io
import os
import time
import uuid

import numpy as np
from flask import Flask, jsonify, request
from PIL import Image

from src.data import DISEASE_LABELS
from src.classifier import XrayDiagnosticEngine


app = Flask(__name__)
ENGINE: XrayDiagnosticEngine = None
UPLOAD_DIR = "./uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


def _save_uploaded_image() -> str:
    if "image" not in request.files:
        raise ValueError("'image' 필드에 이미지를 첨부해주세요.")
    f = request.files["image"]
    fname = f"{int(time.time()*1000)}_{uuid.uuid4().hex[:8]}.png"
    path = os.path.join(UPLOAD_DIR, fname)
    f.save(path)
    return path


def _overlay_heatmap(image_path: str, cam: np.ndarray) -> bytes:
    """원본 위에 Grad-CAM heatmap 을 합성하여 PNG 바이트로 반환."""
    img = Image.open(image_path).convert("L").resize((cam.shape[1], cam.shape[0]))
    arr = np.asarray(img, dtype=np.float32) / 255.0

    # 단순 red-channel overlay (외부 의존성 최소화)
    heat = (cam * 255).astype(np.uint8)
    base = (arr * 255).astype(np.uint8)
    rgb = np.stack([
        np.clip(base.astype(np.int32) + heat.astype(np.int32) // 2, 0, 255).astype(np.uint8),
        base,
        base,
    ], axis=-1)

    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="PNG")
    return buf.getvalue()


@app.route("/healthz", methods=["GET"])
def healthz():
    return jsonify({"status": "ok", "labels": DISEASE_LABELS})


@app.route("/predict", methods=["POST"])
def predict():
    try:
        threshold = float(request.form.get("threshold", 0.5))
        path = _save_uploaded_image()
        t0 = time.time()
        result = ENGINE.predict(path, threshold=threshold)
        elapsed_ms = int((time.time() - t0) * 1000)

        positives = [k for k, v in result.items() if v["positive"]]
        return jsonify({
            "elapsed_ms": elapsed_ms,
            "predictions": result,
            "positive_findings": positives,
            "image_saved_as": os.path.basename(path),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/predict_cam", methods=["POST"])
def predict_cam():
    try:
        cls_name = request.form.get("class")
        if cls_name not in DISEASE_LABELS:
            return jsonify({"error": f"'class' 는 {DISEASE_LABELS} 중 하나"}), 400
        idx = DISEASE_LABELS.index(cls_name)

        path = _save_uploaded_image()
        cam, prob = ENGINE.gradcam(path, class_index=idx)
        png_bytes = _overlay_heatmap(path, cam)
        b64 = base64.b64encode(png_bytes).decode("ascii")
        return jsonify({
            "class": cls_name,
            "prob": prob,
            "gradcam_png_base64": b64,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400


def main():
    global ENGINE
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True,
                        help="train_classifier.py 로 저장한 best.pt 경로")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()

    ENGINE = XrayDiagnosticEngine(ckpt_path=args.ckpt)
    print(f"[서버] http://{args.host}:{args.port} 에서 대기 중")
    app.run(host=args.host, port=args.port, threaded=True)


if __name__ == "__main__":
    main()
