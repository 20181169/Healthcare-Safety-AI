"""안전모 탐지 통합 파이프라인
   Frame ─► [Enhancer] ─► [Detector(YOLO)] ─► [SR (ROI 조건부)] ─►
        [Head ROI 이중분류] ─► [융합 판정] ─► [Privacy] ─► 결과
"""
from dataclasses import dataclass, asdict
from typing import List
import time
import cv2
import numpy as np

from config import CONFIG
from services.enhancer_service import LowLightEnhancerService
from services.detection_service import DetectionService, Detection
from services.sr_service import SuperResolutionService
from services.classifier_service import HeadROIClassifierService, HeadClassResult
from services.privacy_service import PrivacyService
from utils.logger import get_logger


@dataclass
class PipelineResult:
    bbox: tuple
    helmet: bool
    correct: bool
    yolo_score: float
    yolo_class: str = ""
    cls_presence: float = 0.0
    cls_correct: float = 0.0
    decision: str = ""

    def to_dict(self):
        return asdict(self)


class HelmetSafetyPipeline:
    """MSA 서비스들을 조합하는 오케스트레이터.
       각 서비스는 독립 컨테이너로 분리 배포 가능하도록 설계."""

    def __init__(self):
        self.logger = get_logger("Pipeline", CONFIG.log_level)
        self.enhancer = LowLightEnhancerService()
        self.detector = DetectionService()
        self.sr = SuperResolutionService()
        self.classifier = HeadROIClassifierService()
        self.privacy = PrivacyService()

    @staticmethod
    def _fuse(det: Detection, cls: HeadClassResult,
              trust_yolo_when_classifier_untrained: bool = True) -> PipelineResult:
        """YOLO 라벨과 분류기 결과 융합.
           - 다양한 데이터셋의 클래스명(Hardhat / NO-Hardhat / hat / helmet 등)을 정규화
           - 분류기가 학습 안 된 경우(P/C ≈ 0.5) YOLO 결과를 신뢰
        """
        # 클래스명 정규화: 소문자 + 구분자 통일
        name = det.class_name.lower().replace("-", "_").replace(" ", "_")
        is_helmet_label = name in (
            "helmet", "hardhat", "hard_hat", "with_helmet", "helmet_on",
        )
        is_no_helmet_label = name in (
            "no_helmet", "no_hardhat", "nohelmet", "without_helmet", "head",
        )

        classifier_untrained = abs(cls.score_presence - 0.5) < 0.05

        if classifier_untrained and trust_yolo_when_classifier_untrained:
            # 분류기 신호가 무의미하므로 YOLO 라벨에만 의존
            if is_helmet_label:
                helmet = det.score > 0.4
            elif is_no_helmet_label:
                helmet = False
            else:
                helmet = False  # 'person' 등은 불확실 → 미착용으로 간주
        else:
            helmet = cls.helmet_present
            if is_helmet_label:
                helmet = helmet or (det.score > 0.5)
            elif is_no_helmet_label:
                helmet = helmet and (det.score < 0.7)

        correct = helmet and (cls.helmet_correct or classifier_untrained)

        if not helmet:
            decision = "NO_HELMET"
        elif not correct:
            decision = "INCORRECT_WEAR"
        else:
            decision = "OK"

        return PipelineResult(
            bbox=det.bbox,
            helmet=helmet,
            correct=correct,
            yolo_score=det.score,
            yolo_class=det.class_name,
            cls_presence=cls.score_presence,
            cls_correct=cls.score_correct,
            decision=decision,
        )

    def process(self, frame: np.ndarray, anonymize: bool = True) -> tuple[np.ndarray, List[PipelineResult]]:
        t0 = time.time()

        # 1) 저조도 보정 (조건부)
        enhanced, did_enhance = self.enhancer(frame)

        # 2) 1차 탐지
        detections = self.detector.infer(enhanced)
        heads = self.detector.filter_heads(detections) or detections

        # 3) Head ROI 단위로 SR + 이중분류
        results: List[PipelineResult] = []
        sr_applied_count = 0
        for i, det in enumerate(heads):
            roi = self.classifier.crop_head(enhanced, det.bbox)
            roi_sr, sr_used = self.sr(roi)
            if sr_used:
                sr_applied_count += 1
                self.logger.debug(f"  SR applied to ROI {i}: {roi.shape} -> {roi_sr.shape}")
            cls = self.classifier.classify(roi_sr)
            results.append(self._fuse(det, cls))
        self.logger.info(f"  SR applied to {sr_applied_count}/{len(heads)} ROIs")

        # 4) 결과 시각화 + 비식별 (클래스별로 얼굴 위치 다르게 추정)
        vis = self._render(enhanced, results)
        if anonymize:
            vis = self.privacy.anonymize(
                vis,
                bboxes=[r.bbox for r in results],
                classes=[r.yolo_class for r in results],
            )

        self.logger.info(
            f"frame processed in {(time.time() - t0) * 1000:.1f}ms "
            f"(enhance={did_enhance}, n_targets={len(results)})"
        )
        return vis, results

    @staticmethod
    def _render(image: np.ndarray, results: List[PipelineResult]) -> np.ndarray:
        palette = {"OK": (0, 200, 0), "NO_HELMET": (0, 0, 255), "INCORRECT_WEAR": (0, 165, 255)}
        out = image.copy()
        for r in results:
            x1, y1, x2, y2 = r.bbox
            color = palette.get(r.decision, (200, 200, 200))
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
            label = f"{r.decision} [{r.yolo_class}:{r.yolo_score:.2f}]"
            cv2.putText(out, label, (x1, max(0, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        return out
