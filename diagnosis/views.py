"""뷰 — 업로드, 진단 처리, 결과/이력 표시."""
import logging
import uuid
from pathlib import Path

from django.conf import settings
from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.shortcuts import get_object_or_404, redirect, render
from django.urls import reverse
from django.utils import timezone
from django.views.decorators.http import require_http_methods

from .forms import UploadForm
from .models import Recording


def _DiagnosisPipeline():
    """audio_sr → torch/torchaudio/librosa 가 미설치된 환경에서도 페이지 라우팅이
    살아남도록 lazy import. 실제 추론 호출 시점에만 모듈을 로드."""
    from .pipeline import DiagnosisPipeline  # noqa: WPS433
    return DiagnosisPipeline


logger = logging.getLogger(__name__)


@login_required
@require_http_methods(["GET", "POST"])
def upload_view(request):
    """업로드 폼 + 최근 이력 표시."""
    if request.method == "POST":
        form = UploadForm(request.POST, request.FILES)
        if form.is_valid():
            rec = form.save()
            return redirect("diagnosis:process", pk=rec.pk)
    else:
        form = UploadForm()

    history = Recording.objects.exclude(predicted_label="")[:10]
    return render(request, "diagnosis/upload.html",
                  {"form": form, "history": history})


@login_required
@require_http_methods(["GET"])
def process_view(request, pk: int):
    """업로드된 녹음에 대해 SR 복원 + 진단을 동기 실행."""
    rec = get_object_or_404(Recording, pk=pk)

    if rec.predicted_label:
        return redirect("diagnosis:result", pk=rec.pk)

    try:
        pipeline = _DiagnosisPipeline().get()
        restored_rel = f"restored/{uuid.uuid4().hex}.wav"
        result = pipeline.run(
            upload_path=rec.original_wav.path,
            restored_rel_path=restored_rel,
        )

        rec.restored_wav.name = result.restored_path
        rec.waveform_png.name = result.waveform_png
        rec.spectrogram_png.name = result.spectrogram_png
        rec.spectrum_png.name = result.spectrum_png
        rec.predicted_label = result.predicted_label
        rec.confidence = result.confidence
        rec.probabilities = result.probabilities
        rec.snr_in_db = None if result.snr_in_db != result.snr_in_db else result.snr_in_db
        rec.snr_out_db = None if result.snr_out_db != result.snr_out_db else result.snr_out_db
        rec.processed_at = timezone.now()
        rec.save()

        if result.demo_mode:
            messages.warning(
                request,
                "데모 모드: 학습된 체크포인트가 없어 무작위 가중치로 동작했습니다. "
                "checkpoints/sr/best.pt 와 checkpoints/cls/best_sr.pt 를 배치해 주세요.",
            )
    except Exception as e:
        logger.exception("inference failed")
        rec.error = f"{type(e).__name__}: {e}"
        rec.save()
        messages.error(request, f"진단 처리 중 오류가 발생했습니다: {e}")

    return redirect("diagnosis:result", pk=rec.pk)


@login_required
@require_http_methods(["GET"])
def result_view(request, pk: int):
    rec = get_object_or_404(Recording, pk=pk)

    # 확률을 정렬하여 템플릿에서 막대그래프로 그리기 좋게 변환
    probs = []
    if rec.probabilities:
        for label, p in sorted(rec.probabilities.items(), key=lambda kv: -kv[1]):
            probs.append({
                "label": label,
                "label_kor": dict(Recording.LABEL_CHOICES).get(label, label),
                "prob": p,
                "percent": round(p * 100, 2),
            })
    return render(request, "diagnosis/result.html", {"rec": rec, "probs": probs})


@login_required
@require_http_methods(["GET"])
def history_view(request):
    qs = Recording.objects.all()[:100]
    return render(request, "diagnosis/history.html", {"recordings": qs})


@require_http_methods(["GET"])
def about_view(request):
    """포트폴리오용 기술 노트 — 비로그인도 열람 가능."""
    return render(request, "diagnosis/about.html")
