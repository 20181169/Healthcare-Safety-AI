import logging

from django.apps import AppConfig
from django.conf import settings


logger = logging.getLogger(__name__)


class SafetyConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "safety"
    verbose_name = "공사현장 안전관리"

    def ready(self):
        # ── ① Zero-DCE 게이트 튜닝 override ──────────────────────────────
        # 원본 config (수상관리/config.py) 의 게이트 임계값 (B<70, C<35, N>0.02) 은
        # scripts/eval_safety.py + inspect_safety_triggers.py 분석 결과 결함 발견:
        #   • noise > 0.02 트리거된 13장 중 9장이 평균 밝기 170 의 밝은 sample
        #   • 저조도 보정 모델(Zero-DCE) 을 밝은 sample 에 적용 → AP −0.0141
        # 튜닝 결과 (B<60, C<30, noise 비활성) 적용 시:
        #   • 트리거 13 → 1 (정확한 sample 만 선별)
        #   • AP −0.0141 → +0.0004 (Baseline 도달)
        # Django 부팅 시 settings 의 값으로 CONFIG.enhancer override.
        try:
            import config as safety_config  # 수상관리/config.py (SAFETY_ROOT 가 sys.path 에)
            safety_config.CONFIG.enhancer.brightness_trigger = settings.SAFETY_GATE_BRIGHTNESS
            safety_config.CONFIG.enhancer.contrast_trigger   = settings.SAFETY_GATE_CONTRAST
            safety_config.CONFIG.enhancer.noise_trigger      = settings.SAFETY_GATE_NOISE
            logger.info(
                "safety gate tuned: brightness<%s OR contrast<%s OR noise>%s "
                "(원본 70/35/0.02 결함 — 분석 결과로 override)",
                settings.SAFETY_GATE_BRIGHTNESS,
                settings.SAFETY_GATE_CONTRAST,
                settings.SAFETY_GATE_NOISE,
            )
        except Exception as exc:
            logger.warning(
                "safety gate override 실패 — SAFETY_ROOT/config.py 를 찾을 수 없음 (%s: %s)",
                type(exc).__name__, exc,
            )

        # ── ② 파이프라인 lazy 로드 트리거 ──────────────────────────────
        # 파이프라인은 첫 요청 시 lazy 로드된다 (마이그레이션/admin 만 쓰는 경우 GPU 점유 X).
        # 단, monkey-patch (_install_state_capture) 는 모듈 임포트 시 실행되므로 여기서 한 번 트리거.
        try:
            from .pipeline_runner import warmup_signal_handler  # noqa: F401
        except Exception as exc:
            logger.warning(
                "safety pipeline import 실패 — SAFETY_ROOT(수상관리/) 가 PYTHONPATH 에 있고 "
                "weights/ 가 준비되어 있는지 확인하세요. (%s: %s)",
                type(exc).__name__, exc,
            )
