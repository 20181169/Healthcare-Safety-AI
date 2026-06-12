import logging

from django.apps import AppConfig


logger = logging.getLogger(__name__)


class SafetyConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "safety"
    verbose_name = "공사현장 안전관리"

    def ready(self):
        # 파이프라인은 첫 요청 시 lazy 로드된다 (마이그레이션/admin 만 쓰는 경우 GPU 점유 X).
        # 단, monkey-patch (_install_state_capture) 는 모듈 임포트 시 실행되므로 여기서 한 번 트리거.
        # SAFETY_ROOT 외부 의존성이 없는 환경에서는 import 가 실패할 수 있어 방어적으로 처리.
        try:
            from .pipeline_runner import warmup_signal_handler  # noqa: F401
        except Exception as exc:
            logger.warning(
                "safety pipeline import 실패 — SAFETY_ROOT(수상관리/) 가 PYTHONPATH 에 있고 "
                "weights/ 가 준비되어 있는지 확인하세요. (%s: %s)",
                type(exc).__name__, exc,
            )
