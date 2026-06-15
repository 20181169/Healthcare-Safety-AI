"""YOLO 탐지 서비스 - 사람/머리/안전모 1차 탐지
   - Ultralytics YOLO API 사용
   - 결과는 dataclass 리스트로 정규화하여 후단 서비스에 전달
"""
from dataclasses import dataclass
from typing import List
import numpy as np

from config import CONFIG
from utils.logger import get_logger


@dataclass
class Detection:
    bbox: tuple           # (x1, y1, x2, y2)
    score: float
    class_id: int
    class_name: str

    @property
    def area(self) -> int:
        x1, y1, x2, y2 = self.bbox
        return max(0, x2 - x1) * max(0, y2 - y1)


class DetectionService:
    def __init__(self):
        self.cfg = CONFIG.detection
        self.logger = get_logger("DetectionService", CONFIG.log_level)
        self.model = self._load_model()

    def _load_model(self):
        try:
            from ultralytics import YOLO
            model = YOLO(self.cfg.weights)
            self.logger.info(f"YOLO loaded: {self.cfg.weights} (device={self.cfg.device})")
            # 모델 자체가 들고 있는 클래스명을 우선 사용 (COCO/커스텀 모두 호환)
            if hasattr(model, "names") and model.names:
                self._names = model.names
            else:
                self._names = {i: n for i, n in enumerate(self.cfg.class_names)}
            return model
        except Exception as e:
            self.logger.error(f"YOLO load failed: {e}")
            raise

    def infer(self, image: np.ndarray) -> List[Detection]:
        results = self.model.predict(
            source=image,
            conf=self.cfg.conf_threshold,
            iou=self.cfg.iou_threshold,
            imgsz=self.cfg.img_size,
            device=self.cfg.device,
            verbose=False,
        )
        detections: List[Detection] = []
        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int).tolist()
                cls = int(box.cls[0].item())
                conf = float(box.conf[0].item())
                name = self._names.get(cls, str(cls)) if isinstance(self._names, dict) else str(cls)
                detections.append(Detection((x1, y1, x2, y2), conf, cls, name))
        return detections

    @staticmethod
    def filter_heads(dets: List[Detection]) -> List[Detection]:
        return [d for d in dets if d.class_name in ("head", "person")]
