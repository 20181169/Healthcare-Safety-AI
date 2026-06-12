"""
AI 추론 서비스 (Django 앱 단위 단일 인스턴스).

체크포인트가 존재하면 실제 모델로, 없으면 demo 모드로 동작.
"""

import os
import random
import time
from typing import Dict, Optional, Tuple

import numpy as np

from src.data import DISEASE_LABELS


class AIService:
    def __init__(self, ckpt_path: Optional[str], supervised_labels: Optional[list] = None):
        self.ckpt_path = ckpt_path
        self.engine = None
        self.demo = True
        # 실제 학습 신호를 받은 라벨만 신뢰 (나머지는 추론 시 마스킹)
        self.supervised_labels = (
            [s.strip() for s in supervised_labels if s.strip()]
            if supervised_labels else list(DISEASE_LABELS)
        )
        self._try_load()

    def _try_load(self):
        if not self.ckpt_path or not os.path.isfile(self.ckpt_path):
            print(f"[AI] 체크포인트 없음 → demo 모드: {self.ckpt_path}")
            return
        try:
            from src.classifier import XrayDiagnosticEngine
            self.engine = XrayDiagnosticEngine(ckpt_path=self.ckpt_path)
            self.demo = False
            print(f"[AI] 체크포인트 로드: {self.ckpt_path}")
        except Exception as e:  # pragma: no cover
            print(f"[AI] 로드 실패 → demo: {e}")

    def predict(self, image_path: str, threshold: float = 0.5) -> Tuple[Dict, int]:
        t0 = time.time()
        if self.engine:
            preds = self.engine.predict(image_path, threshold=threshold)
            result = {k: float(v["prob"]) for k, v in preds.items()}
        else:
            result = self._demo_predict(image_path)

        # 비감독 라벨은 0.0 으로 마스킹 → UI/통계에서 가짜 양성 차단
        if set(self.supervised_labels) != set(result.keys()):
            for name in list(result.keys()):
                if name not in self.supervised_labels:
                    result[name] = 0.0

        return result, int((time.time() - t0) * 1000)

    def gradcam(self, image_path: str, class_name: str, save_to: str) -> Tuple[str, float]:
        if class_name not in DISEASE_LABELS:
            raise ValueError(f"알 수 없는 클래스: {class_name}")
        idx = DISEASE_LABELS.index(class_name)
        if self.engine:
            cam, prob = self.engine.gradcam(image_path, class_index=idx)
        else:
            cam, prob = self._demo_gradcam(image_path)
        self._save_overlay(image_path, cam, save_to)
        return save_to, float(prob)

    def _demo_predict(self, image_path: str) -> Dict[str, float]:
        seed = abs(hash(os.path.basename(image_path))) % (2 ** 31)
        rng = random.Random(seed)
        positives = rng.sample(DISEASE_LABELS, k=rng.randint(1, 3))
        probs = {}
        for name in DISEASE_LABELS:
            probs[name] = round(
                rng.uniform(0.55, 0.95) if name in positives else rng.uniform(0.02, 0.35),
                4,
            )
        return probs

    def _demo_gradcam(self, image_path: str):
        from PIL import Image
        img = Image.open(image_path).convert("L")
        w, h = img.size
        seed = abs(hash(os.path.basename(image_path) + "cam")) % (2 ** 31)
        rng = np.random.default_rng(seed)
        cx, cy = rng.uniform(0.3, 0.7) * w, rng.uniform(0.3, 0.7) * h
        ys, xs = np.mgrid[0:h, 0:w]
        sigma = min(w, h) * 0.15
        cam = np.exp(-((xs - cx) ** 2 + (ys - cy) ** 2) / (2 * sigma ** 2))
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam, float(rng.uniform(0.6, 0.9))

    def _save_overlay(self, image_path: str, cam: np.ndarray, save_to: str):
        from PIL import Image
        img = Image.open(image_path).convert("L").resize((cam.shape[1], cam.shape[0]))
        arr = np.asarray(img, dtype=np.float32) / 255.0
        heat = (cam * 255).astype(np.uint8)
        base = (arr * 255).astype(np.uint8)
        rgb = np.stack([
            np.clip(base.astype(np.int32) + heat.astype(np.int32) // 2, 0, 255).astype(np.uint8),
            np.clip(base.astype(np.int32) - heat.astype(np.int32) // 4, 0, 255).astype(np.uint8),
            np.clip(base.astype(np.int32) - heat.astype(np.int32) // 4, 0, 255).astype(np.uint8),
        ], axis=-1)
        os.makedirs(os.path.dirname(save_to), exist_ok=True)
        Image.fromarray(rgb).save(save_to, format="PNG")
