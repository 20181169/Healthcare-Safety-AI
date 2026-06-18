"""템플릿 글로벌 컨텍스트 — 통합 헬스케어/안전 AI."""

from django.apps import apps
from django.conf import settings


def global_flags(request):
    try:
        ai = apps.get_app_config("studies").ai_service
        xray_demo = ai.demo if ai else True
    except Exception:
        xray_demo = True
    return {
        "AI_DEMO_MODE": xray_demo,         # 기존 템플릿 호환 (소아 X-ray)
        "XRAY_DEMO_MODE": xray_demo,
        # 방문자 트래킹 — 환경변수 비어있으면 템플릿에서 렌더링 안 됨
        "GA_MEASUREMENT_ID":  getattr(settings, "GA_MEASUREMENT_ID",  ""),
        "CLARITY_PROJECT_ID": getattr(settings, "CLARITY_PROJECT_ID", ""),
    }
