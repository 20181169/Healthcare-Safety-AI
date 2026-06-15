"""프라이버시 비식별 처리 서비스
   - YOLO 박스의 종류(안전모/머리/사람) 별로 실제 얼굴 위치를 다르게 추정해 블러/모자이크 적용.
   - 결과 영상 저장/송출 직전에 호출되어 개인정보(얼굴) 노출을 차단.
"""
from typing import Iterable, Tuple
import cv2
import numpy as np

from config import CONFIG
from utils.logger import get_logger


def _norm(name: str) -> str:
    return (name or "").lower().replace("-", "").replace("_", "").replace(" ", "")


def estimate_face_box(bbox: Tuple[int, int, int, int],
                      yolo_class: str,
                      img_shape: Tuple[int, int]) -> Tuple[int, int, int, int]:
    """탐지 박스 + 클래스명을 기반으로 얼굴이 있을 만한 영역을 추정.

       - Hardhat / Helmet  : 박스 *아래쪽* 에 얼굴이 있음 → 박스 하단부터 박스 높이만큼 아래로 확장
       - NO-Hardhat / Head : 박스가 머리 자체 → 박스 하단 65% (이마 가림)
       - Person 등 전신    : 박스 상단 25% 가 머리
       - 그 외             : 박스 상단 50% (안전 기본값)
    """
    x1, y1, x2, y2 = bbox
    H, W = img_shape[:2]
    w = max(1, x2 - x1)
    h = max(1, y2 - y1)
    cls = _norm(yolo_class)

    if "nohardhat" in cls or "nohelmet" in cls or "head" in cls:
        # 머리(맨머리) → 박스의 아래쪽 일부가 얼굴
        fx1, fx2 = x1, x2
        fy1 = y1 + int(h * 0.20)
        fy2 = y1 + int(h * 0.95)
    elif "hardhat" in cls or "helmet" in cls:
        # 안전모 → 박스 자체는 모자만 가리키므로 박스 아래로 확장
        fx1 = x1 - int(w * 0.10)
        fx2 = x2 + int(w * 0.10)
        fy1 = y1 + int(h * 0.55)             # 박스 중간부터 (모자 챙 아래)
        fy2 = y2 + int(h * 1.30)             # 박스 아래로 1.3배 확장
    elif "person" in cls:
        fx1 = x1 + int(w * 0.20)
        fx2 = x2 - int(w * 0.20)
        fy1 = y1
        fy2 = y1 + int(h * 0.25)
    else:
        # 클래스 불명 - 박스 전체 블러 (가장 안전)
        fx1, fy1, fx2, fy2 = x1, y1, x2, y2

    # 이미지 경계 클램프
    fx1 = max(0, min(W - 1, fx1))
    fx2 = max(0, min(W,     fx2))
    fy1 = max(0, min(H - 1, fy1))
    fy2 = max(0, min(H,     fy2))
    if fx2 <= fx1: fx2 = min(W, fx1 + 1)
    if fy2 <= fy1: fy2 = min(H, fy1 + 1)
    return fx1, fy1, fx2, fy2


class PrivacyService:
    def __init__(self):
        self.cfg = CONFIG.privacy
        self.logger = get_logger("PrivacyService", CONFIG.log_level)

    def _gaussian_blur(self, roi: np.ndarray) -> np.ndarray:
        k = self.cfg.blur_kernel | 1
        return cv2.GaussianBlur(roi, (k, k), 0)

    def _pixelate(self, roi: np.ndarray) -> np.ndarray:
        h, w = roi.shape[:2]
        n = max(1, self.cfg.pixelate_blocks)
        small = cv2.resize(roi, (n, n), interpolation=cv2.INTER_LINEAR)
        return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)

    def anonymize(self,
                  image: np.ndarray,
                  bboxes: Iterable[Tuple[int, int, int, int]],
                  classes: Iterable[str] | None = None,
                  method: str = "blur") -> np.ndarray:
        """bboxes 와 (선택) 해당 클래스명을 받아 얼굴 영역에 블러/모자이크 적용.

           classes 가 주어지지 않으면 모든 박스를 'head' 로 가정 (하단 65%).
        """
        if not self.cfg.enable_face_blur:
            return image
        out = image.copy()
        bboxes = list(bboxes)
        classes = list(classes) if classes is not None else ["head"] * len(bboxes)
        if len(classes) < len(bboxes):
            classes += ["head"] * (len(bboxes) - len(classes))

        for bbox, cls in zip(bboxes, classes):
            fx1, fy1, fx2, fy2 = estimate_face_box(bbox, cls, out.shape)
            roi = out[fy1:fy2, fx1:fx2]
            if roi.size == 0:
                continue
            out[fy1:fy2, fx1:fx2] = (self._pixelate(roi)
                                     if method == "pixelate"
                                     else self._gaussian_blur(roi))
        return out
