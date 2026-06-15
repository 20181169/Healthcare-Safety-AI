"""전역 설정 모듈 - MSA 서비스 간 공유 파라미터 관리
   가중치 경로는 항상 프로젝트 루트(수상관리/) 기준 절대 경로로 해석되어,
   어디서 실행하든(CLI / FastAPI / Django) 동일하게 동작.
"""
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple


# 프로젝트 루트 = 이 파일이 있는 폴더 (수상관리/)
PROJECT_ROOT = Path(__file__).resolve().parent
WEIGHTS_DIR = PROJECT_ROOT / "weights"


def _w(rel: str) -> str:
    """weights/ 아래 상대경로를 절대경로 문자열로 변환"""
    return str(WEIGHTS_DIR / rel)


@dataclass
class DetectionConfig:
    weights: str = field(default_factory=lambda: _w("yolov5s_helmet.pt"))
    conf_threshold: float = 0.20
    iou_threshold: float = 0.45
    img_size: int = 416     # 시연 성능 우선 (정확도 우선시 640)
    device: str = os.environ.get("SAFETY_DEVICE", "cpu")
    class_names: Tuple[str, ...] = ("person", "helmet", "no_helmet", "head")


@dataclass
class ClassifierConfig:
    weights: str = field(default_factory=lambda: _w("head_roi_dual.pt"))
    input_size: Tuple[int, int] = (96, 96)
    threshold: float = 0.5
    head_expand_ratio: float = 0.25


@dataclass
class EnhancerConfig:
    zero_dce_weights: str = field(default_factory=lambda: _w("zero_dce_plus.pth"))
    brightness_trigger: float = 70.0      # 평균 밝기 임계
    contrast_trigger: float = 35.0        # 대비(표준편차) 임계
    noise_trigger: float = 0.020          # 노이즈 분산 임계


@dataclass
class SRConfig:
    model_name: str = "RealESRGAN_x4plus"
    weights: str = field(default_factory=lambda: _w("RealESRGAN_x4plus.pth"))
    scale: int = 4
    sharpness_trigger: float = 80.0       # Laplacian 분산 임계
    min_face_pixels: int = 64             # 얼굴/머리 ROI 최소 픽셀


@dataclass
class PrivacyConfig:
    enable_face_blur: bool = True
    blur_kernel: int = 31
    pixelate_blocks: int = 12


@dataclass
class SystemConfig:
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    classifier: ClassifierConfig = field(default_factory=ClassifierConfig)
    enhancer: EnhancerConfig = field(default_factory=EnhancerConfig)
    sr: SRConfig = field(default_factory=SRConfig)
    privacy: PrivacyConfig = field(default_factory=PrivacyConfig)
    max_fps: int = 30
    log_level: str = "WARNING"   # 시연 시 매 프레임 로그 억제 (디버깅은 "INFO")


CONFIG = SystemConfig()
