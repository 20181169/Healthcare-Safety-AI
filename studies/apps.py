import logging

from django.apps import AppConfig
from django.conf import settings


logger = logging.getLogger(__name__)


class StudiesConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "studies"
    verbose_name = "검사"
    ai_service = None  # ready() 에서 단일 인스턴스로 로드

    def ready(self):
        # XRAY_ROOT (수상관리_소아엑스레이/) 가 sys.path 에 없거나 src 패키지가
        # 깨졌을 수 있으므로 import 실패에 방어적으로 대응 — 다른 앱(diagnosis, safety) 은 그대로 사용 가능.
        try:
            from .ai_service import AIService
            StudiesConfig.ai_service = AIService(
                ckpt_path=settings.CLASSIFIER_CKPT,
                supervised_labels=getattr(settings, "SUPERVISED_LABELS", None),
            )
            self.ai_service = StudiesConfig.ai_service
        except Exception as exc:
            logger.warning(
                "studies AIService 로드 실패 — XRAY_ROOT(수상관리_소아엑스레이/) 가 PYTHONPATH 에 "
                "있는지 확인하세요. (%s: %s)",
                type(exc).__name__, exc,
            )
