"""JSON API."""

import os
import uuid

from django.apps import apps
from django.conf import settings
from django.contrib.auth.decorators import login_required
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_GET, require_POST


def _ai():
    return apps.get_app_config("studies").ai_service


@require_GET
def healthz(request):
    ai = _ai()
    return JsonResponse({"status": "ok", "mode": "demo" if ai.demo else "production"})


@csrf_exempt
@login_required
@require_POST
def inference(request):
    """업로드 미리보기용 임시 추론 (DB 저장 X)."""
    file = request.FILES.get("image")
    if not file:
        return JsonResponse({"error": "이미지가 필요합니다."}, status=400)

    ext = file.name.rsplit(".", 1)[-1].lower() if "." in file.name else "png"
    fname = f"tmp_{uuid.uuid4().hex}.{ext}"
    tmp_dir = os.path.join(settings.MEDIA_ROOT, "xray")
    os.makedirs(tmp_dir, exist_ok=True)
    path = os.path.join(tmp_dir, fname)
    with open(path, "wb") as out:
        for chunk in file.chunks():
            out.write(chunk)

    ai = _ai()
    threshold = float(request.POST.get("threshold", settings.DEFAULT_THRESHOLD))
    probs, ms = ai.predict(path, threshold=threshold)
    ranked = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)
    positives = sorted([k for k, v in probs.items() if v >= threshold])

    return JsonResponse({
        "mode": "demo" if ai.demo else "production",
        "elapsed_ms": ms,
        "threshold": threshold,
        "probabilities": probs,
        "ranked": [{"name": k, "prob": v} for k, v in ranked],
        "positives": positives,
        "preview_url": settings.MEDIA_URL + f"xray/{fname}",
    })
