"""저조도 보정 서비스 - Zero-DCE++ 조건부 적용
   - 입력 프레임의 밝기/대비/노이즈를 측정하여 임계치 미달 시에만 보정 수행
   - 불필요한 연산을 줄여 실시간성 확보
"""
import cv2
import numpy as np
import torch

from config import CONFIG
from models.zero_dce import ZeroDCEPlus
from utils.image_quality import quality_report
from utils.logger import get_logger


class LowLightEnhancerService:
    def __init__(self, device: str = None):
        self.cfg = CONFIG.enhancer
        self.device = device or CONFIG.detection.device
        self.logger = get_logger("EnhancerService", CONFIG.log_level)
        self.model = ZeroDCEPlus().to(self.device).eval()
        self._load_weights()

    def _load_weights(self):
        try:
            state = torch.load(self.cfg.zero_dce_weights, map_location=self.device)
            # 공식 가중치는 평평한 state_dict — 키 prefix가 다르면 정규화
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            missing, unexpected = self.model.load_state_dict(state, strict=False)
            total = len(self.model.state_dict())
            loaded = total - len(missing)
            self.logger.info(
                f"Zero-DCE++ weights: {loaded}/{total} matched "
                f"(missing={len(missing)}, unexpected={len(unexpected)})"
            )
            if loaded == 0:
                self.logger.error(
                    "Zero-DCE++ 가중치 키가 전혀 매칭되지 않았습니다. "
                    "모델 구조가 가중치와 일치하는지 확인하세요."
                )
                if unexpected[:3]:
                    self.logger.error(f"  unexpected sample: {unexpected[:3]}")
                if missing[:3]:
                    self.logger.error(f"  missing sample   : {missing[:3]}")
        except FileNotFoundError:
            self.logger.warning("Zero-DCE++ weights not found - using random init (demo)")

    def should_enhance(self, image: np.ndarray) -> bool:
        """조건부 트리거 - 어둡거나 대비 낮거나 노이즈가 강할 때만 True"""
        q = quality_report(image)
        return (
            q["brightness"] < self.cfg.brightness_trigger
            or q["contrast"] < self.cfg.contrast_trigger
            or q["noise"] > self.cfg.noise_trigger
        )

    @torch.no_grad()
    def enhance(self, image: np.ndarray) -> np.ndarray:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).to(self.device)
        out = self.model(tensor).squeeze(0).permute(1, 2, 0).cpu().numpy()
        out = np.clip(out * 255.0, 0, 255).astype(np.uint8)
        return cv2.cvtColor(out, cv2.COLOR_RGB2BGR)

    def __call__(self, image: np.ndarray) -> tuple[np.ndarray, bool]:
        if self.should_enhance(image):
            self.logger.debug("low-light condition detected - applying Zero-DCE++")
            return self.enhance(image), True
        return image, False
