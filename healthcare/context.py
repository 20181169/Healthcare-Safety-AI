"""템플릿 글로벌 컨텍스트 — 통합 헬스케어/안전 AI."""

from django.apps import apps


def global_flags(request):
    try:
        ai = apps.get_app_config("studies").ai_service
        xray_demo = ai.demo if ai else True
    except Exception:
        xray_demo = True
    return {
        "AI_DEMO_MODE": xray_demo,         # 기존 템플릿 호환 (소아 X-ray)
        "XRAY_DEMO_MODE": xray_demo,
    }
