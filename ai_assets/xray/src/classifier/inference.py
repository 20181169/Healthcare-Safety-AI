"""
단일 X-ray 영상에 대한 추론 모듈.
Grad-CAM 으로 판독 보조 시각화도 함께 제공한다.
"""

from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from src.data import DISEASE_LABELS, build_transform
from src.classifier.model import MultiLabelXrayClassifier


class XrayDiagnosticEngine:
    """
    판독 보조 엔진. 모델 가중치를 한 번만 로드하고,
    여러 영상에 대해 추론 + Grad-CAM 을 수행.
    """

    def __init__(self, ckpt_path: str, image_size: int = 224, device: str = None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.transform = build_transform(image_size, train=False)
        self.model = MultiLabelXrayClassifier(pretrained=False).to(self.device).eval()
        state = torch.load(ckpt_path, map_location=self.device)
        self.model.load_state_dict(state)

        # Grad-CAM 용 hook
        # 주의: features.norm5 다음에 오는 inplace ReLU 가 torch 2.x 의 autograd
        # 검사와 충돌하므로 마지막 dense block 의 출력을 대상으로 잡는다.
        self._activations = None
        self._gradients = None
        target_layer = self.model.backbone.features.denseblock4
        target_layer.register_forward_hook(self._fwd_hook)
        target_layer.register_full_backward_hook(self._bwd_hook)

    def _fwd_hook(self, _module, _inp, out):
        self._activations = out.detach()

    def _bwd_hook(self, _module, _grad_in, grad_out):
        self._gradients = grad_out[0].detach()

    def _load_image(self, image_path: str) -> torch.Tensor:
        img = Image.open(image_path).convert("L")
        return self.transform(img).unsqueeze(0).to(self.device)

    def predict(self, image_path: str, threshold: float = 0.5) -> Dict[str, dict]:
        """질병별 확률과 양/음성 라벨을 반환."""
        x = self._load_image(image_path)
        with torch.no_grad():
            probs = self.model.predict_proba(x).cpu().numpy()[0]
        result = {}
        for i, name in enumerate(DISEASE_LABELS):
            result[name] = {
                "prob": float(probs[i]),
                "positive": bool(probs[i] >= threshold),
            }
        return result

    def gradcam(self, image_path: str, class_index: int) -> Tuple[np.ndarray, float]:
        """
        지정된 클래스에 대한 Grad-CAM heatmap (HxW, 0~1) 반환.
        - 휴대형 X-ray 영상에서 의심 영역을 시각화 → 응급 의료진 의사결정 보조.
        """
        x = self._load_image(image_path).requires_grad_(True)
        logits = self.model(x)
        prob = torch.sigmoid(logits)[0, class_index].item()

        self.model.zero_grad(set_to_none=True)
        logits[0, class_index].backward()

        weights = self._gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self._activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)
        cam = cam[0, 0].cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam, prob
