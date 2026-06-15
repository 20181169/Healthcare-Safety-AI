"""초해상도(SR) 서비스 - Real-ESRGAN / SRGAN
   - 작은 ROI(머리/얼굴) 또는 흐릿한 영역에만 조건부 적용
   - 전체 프레임이 아닌 ROI 단위로 처리하여 비용 절감
"""
import cv2
import numpy as np

from config import CONFIG
from utils.image_quality import measure_sharpness
from utils.logger import get_logger


class SuperResolutionService:
    def __init__(self):
        self.cfg = CONFIG.sr
        self.logger = get_logger("SRService", CONFIG.log_level)
        self.upsampler = self._build_realesrgan()

    def _build_realesrgan(self):
        """Real-ESRGAN 가중치를 통한 SR 모델 초기화"""
        try:
            from basicsr.archs.rrdbnet_arch import RRDBNet
            from realesrgan import RealESRGANer

            model = RRDBNet(
                num_in_ch=3, num_out_ch=3, num_feat=64,
                num_block=23, num_grow_ch=32, scale=self.cfg.scale,
            )
            upsampler = RealESRGANer(
                scale=self.cfg.scale,
                model_path=self.cfg.weights,
                model=model,
                tile=256,
                tile_pad=10,
                pre_pad=0,
                half=True,
            )
            self.logger.info(f"Real-ESRGAN loaded: {self.cfg.weights}")
            return upsampler
        except Exception as e:
            self.logger.warning(f"Real-ESRGAN init failed ({e}) - falling back to bicubic")
            return None

    def should_upscale(self, roi: np.ndarray) -> bool:
        h, w = roi.shape[:2]
        if min(h, w) < self.cfg.min_face_pixels:
            return True
        return measure_sharpness(roi) < self.cfg.sharpness_trigger

    def upscale(self, roi: np.ndarray) -> np.ndarray:
        if self.upsampler is not None:
            try:
                out, _ = self.upsampler.enhance(roi, outscale=self.cfg.scale)
                return out
            except Exception as e:
                self.logger.warning(f"SR inference failed ({e}) - bicubic fallback")
        h, w = roi.shape[:2]
        return cv2.resize(roi, (w * self.cfg.scale, h * self.cfg.scale),
                          interpolation=cv2.INTER_CUBIC)

    def __call__(self, roi: np.ndarray) -> tuple[np.ndarray, bool]:
        if roi.size == 0:
            return roi, False
        if self.should_upscale(roi):
            return self.upscale(roi), True
        return roi, False
