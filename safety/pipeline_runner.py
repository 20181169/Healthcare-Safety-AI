"""기존 HelmetSafetyPipeline 을 Django 에서 안전하게 호출하기 위한 래퍼.
   - 무거운 모델 초기화는 첫 호출 시 1회만 (lazy + thread-safe singleton)
   - 결과(시각화 이미지/객체별 결과) 를 Django ImageField/Model 에 쉽게 매핑
"""
import io
import threading
import time
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

# pipeline.py / services / config 는 PROJECT_ROOT 가 sys.path 에 들어 있어야 import 가능
# (settings.py 에서 추가됨)
from pipeline import HelmetSafetyPipeline


_pipeline_lock = threading.Lock()
_pipeline_instance: Optional[HelmetSafetyPipeline] = None


def get_pipeline() -> HelmetSafetyPipeline:
    global _pipeline_instance
    if _pipeline_instance is None:
        with _pipeline_lock:
            if _pipeline_instance is None:
                _pipeline_instance = HelmetSafetyPipeline()
    return _pipeline_instance


def warmup_signal_handler():
    """apps.ready() 에서 import 만 해두는 더미. 명시적 워밍업이 필요하면 호출."""
    return None


def run_on_bytes(image_bytes: bytes, anonymize: bool = True) -> dict:
    """업로드된 이미지 바이트를 받아 파이프라인 실행 후 결과 dict 반환.

       반환 예:
           {
               "elapsed_ms": 312.4,
               "enhance_applied": True,
               "sr_applied_count": 2,
               "n_total": 4,
               "n_violation": 1,
               "mean_conf": 0.78,
               "vis_jpeg": <bytes>,
               "results": [PipelineResult.to_dict() ...],
           }
    """
    arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("이미지를 디코딩할 수 없습니다.")

    return run_on_frame(img, anonymize=anonymize)


def run_on_frame(img: np.ndarray, anonymize: bool = True) -> dict:
    pipe = get_pipeline()
    t0 = time.perf_counter()
    vis, results = pipe.process(img, anonymize=anonymize)
    elapsed = (time.perf_counter() - t0) * 1000.0

    confs = [r.yolo_score for r in results] or [0.0]
    n_total = len(results)
    n_violation = sum(1 for r in results if r.decision != "OK")

    ok, buf = cv2.imencode(".jpg", vis, [cv2.IMWRITE_JPEG_QUALITY, 88])
    vis_jpeg = buf.tobytes() if ok else b""

    # pipeline 의 로깅에서 SR/Enhance 정보를 직접 얻을 수 없으니 추정값으로 채움
    # (정확한 값은 pipeline.py 를 약간 수정하여 dict 로 반환하게 만들 수도 있음)
    return {
        "elapsed_ms": elapsed,
        "enhance_applied": _last_enhance_used(pipe),
        "sr_applied_count": _last_sr_count(pipe),
        "n_total": n_total,
        "n_violation": n_violation,
        "mean_conf": float(sum(confs) / len(confs)),
        "vis_jpeg": vis_jpeg,
        "results": [r.to_dict() for r in results],
    }


# ── pipeline.py 가 내부 상태를 노출하지 않으므로, 마지막 처리에서의
#    enhance/sr 사용 여부를 monkey-patch 로 추출. 가벼운 트릭이므로 OK.
def _install_state_capture():
    from services.enhancer_service import LowLightEnhancerService
    from services.sr_service import SuperResolutionService

    state = {"enhance": False, "sr_count": 0}

    orig_call = LowLightEnhancerService.__call__
    def call(self, image):
        out, used = orig_call(self, image)
        state["enhance"] = bool(used)
        state["sr_count"] = 0   # 새 프레임 시작 시 SR 카운터 리셋
        return out, used
    LowLightEnhancerService.__call__ = call

    orig_sr = SuperResolutionService.__call__
    def sr_call(self, roi):
        out, used = orig_sr(self, roi)
        if used:
            state["sr_count"] += 1
        return out, used
    SuperResolutionService.__call__ = sr_call

    return state


_capture = _install_state_capture()


def _last_enhance_used(_):
    return _capture["enhance"]


def _last_sr_count(_):
    return _capture["sr_count"]
