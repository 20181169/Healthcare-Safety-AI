"""Head ROI 이중분류 추론 서비스
   - YOLO가 놓치거나 애매한 케이스를 보강하기 위해
     머리 영역 crop → 안전모 착용 여부 + 올바른 착용 여부를 분류
"""
from dataclasses import dataclass
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms

from config import CONFIG
from models.head_classifier import HeadDualClassifier
from utils.logger import get_logger


@dataclass
class HeadClassResult:
    helmet_present: bool
    helmet_correct: bool
    score_presence: float
    score_correct: float


class HeadROIClassifierService:
    def __init__(self, device: str = None):
        self.cfg = CONFIG.classifier
        self.device = device or CONFIG.detection.device
        self.logger = get_logger("ClassifierService", CONFIG.log_level)
        self.model = HeadDualClassifier(pretrained=False).to(self.device).eval()
        self._load_weights()
        self.tf = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(self.cfg.input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    def _load_weights(self):
        try:
            state = torch.load(self.cfg.weights, map_location=self.device)
            self.model.load_state_dict(state, strict=False)
            self.logger.info(f"Head classifier weights loaded: {self.cfg.weights}")
        except FileNotFoundError:
            self.logger.warning("Classifier weights not found - using random init (demo)")

    @staticmethod
    def expand_bbox(bbox, img_shape, ratio: float):
        x1, y1, x2, y2 = bbox
        H, W = img_shape[:2]
        w, h = x2 - x1, y2 - y1
        dx, dy = int(w * ratio), int(h * ratio)
        return (max(0, x1 - dx), max(0, y1 - dy),
                min(W, x2 + dx), min(H, y2 + dy))

    def crop_head(self, image: np.ndarray, bbox) -> np.ndarray:
        x1, y1, x2, y2 = self.expand_bbox(bbox, image.shape, self.cfg.head_expand_ratio)
        return image[y1:y2, x1:x2].copy()

    @torch.no_grad()
    def classify(self, head_roi: np.ndarray) -> HeadClassResult:
        if head_roi.size == 0:
            return HeadClassResult(False, False, 0.0, 0.0)
        rgb = cv2.cvtColor(head_roi, cv2.COLOR_BGR2RGB)
        x = self.tf(rgb).unsqueeze(0).to(self.device)
        out = self.model(x)
        p_pres = F.softmax(out["presence"], dim=1)[0, 1].item()
        p_corr = F.softmax(out["correctness"], dim=1)[0, 1].item()
        return HeadClassResult(
            helmet_present=p_pres >= self.cfg.threshold,
            helmet_correct=p_corr >= self.cfg.threshold,
            score_presence=p_pres,
            score_correct=p_corr,
        )
