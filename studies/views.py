import json
import os
from datetime import datetime

from django.apps import apps
from django.conf import settings
from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.http import HttpResponseForbidden
from django.shortcuts import get_object_or_404, redirect, render
from django.utils import timezone
from django.views.decorators.http import require_POST

from patients.models import Patient

from .forms import ConfirmDiagnosisForm, StudyUploadForm
from .models import Diagnosis, Study


def _ai():
    return apps.get_app_config("studies").ai_service


def _disease_labels():
    """xray_ai 의 DISEASE_LABELS 는 pandas 등 무거운 ML 스택을 끌어오므로 lazy import.
    requirements-xray.txt 미설치 환경에서도 페이지 라우팅은 살아남도록 한다."""
    try:
        from src.data import DISEASE_LABELS  # noqa: WPS433
        return DISEASE_LABELS
    except Exception:
        return []


@login_required
def study_list(request):
    status = request.GET.get("status")
    qs = Study.objects.select_related("patient", "uploader").all()
    if status:
        qs = qs.filter(status=status)
    return render(request, "studies/list.html", {"studies": qs[:200], "status": status})


@login_required
def study_new(request):
    patient_id = request.GET.get("patient_id")
    patient = Patient.objects.filter(pk=patient_id).first() if patient_id else None

    if request.method == "POST":
        form = StudyUploadForm(request.POST, request.FILES)
        if form.is_valid():
            study = form.save(commit=False)
            study.uploader = request.user
            study.status = Study.STATUS_UPLOADED
            study.save()
            _run_inference(study)
            return redirect("studies:detail", pk=study.pk)
    else:
        initial = {"patient": patient} if patient else {}
        form = StudyUploadForm(initial=initial)

    patients = Patient.objects.all()[:50]
    return render(request, "studies/new.html",
                  {"form": form, "patient": patient, "patients": patients})


def _run_inference(study: Study):
    ai = _ai()
    threshold = settings.DEFAULT_THRESHOLD
    image_path = study.image.path

    probs, ms = ai.predict(image_path, threshold=threshold)
    positives = [k for k, v in probs.items() if v >= threshold]
    top_name, top_prob = max(probs.items(), key=lambda kv: kv[1])

    cam_filename = f"cam_{study.pk}_{top_name}.png"
    cam_path = os.path.join(settings.MEDIA_ROOT, "gradcam", cam_filename)
    try:
        ai.gradcam(image_path, class_name=top_name, save_to=cam_path)
        cam_relpath = f"gradcam/{cam_filename}"
    except Exception as e:
        print(f"[GradCAM 실패] {e}")
        cam_relpath = None

    Diagnosis.objects.create(
        study=study,
        predictions_json=json.dumps(probs),
        positive_findings_json=json.dumps(positives),
        top_finding=top_name,
        top_prob=top_prob,
        threshold=threshold,
        inference_ms=ms,
        gradcam=cam_relpath or "",
        gradcam_class=top_name if cam_relpath else "",
        is_demo=ai.demo,
    )
    study.status = Study.STATUS_AI
    study.save(update_fields=["status"])


@login_required
def study_detail(request, pk):
    study = get_object_or_404(Study.objects.select_related("patient", "uploader"), pk=pk)
    diagnosis = getattr(study, "diagnosis", None)
    return render(request, "studies/detail.html", {
        "study": study, "diagnosis": diagnosis,
        "labels": _disease_labels(),
    })


@login_required
@require_POST
def study_rerun_cam(request, pk):
    study = get_object_or_404(Study, pk=pk)
    cls_name = request.POST.get("class")
    if cls_name not in _disease_labels():
        messages.error(request, "올바른 질환을 선택해주세요.")
        return redirect("studies:detail", pk=pk)

    ai = _ai()
    cam_filename = f"cam_{study.pk}_{cls_name}.png"
    cam_path = os.path.join(settings.MEDIA_ROOT, "gradcam", cam_filename)
    ai.gradcam(study.image.path, class_name=cls_name, save_to=cam_path)

    diag = study.diagnosis
    diag.gradcam = f"gradcam/{cam_filename}"
    diag.gradcam_class = cls_name
    diag.save(update_fields=["gradcam", "gradcam_class"])
    return redirect("studies:detail", pk=pk)


@login_required
@require_POST
def study_claim(request, pk):
    study = get_object_or_404(Study, pk=pk)
    if not request.user.is_radiologist:
        return HttpResponseForbidden("전문의만 검토를 시작할 수 있습니다.")
    if study.status == Study.STATUS_AI:
        study.status = Study.STATUS_REVIEW
        study.save(update_fields=["status"])
    return redirect("studies:detail", pk=pk)


@login_required
@require_POST
def study_confirm(request, pk):
    study = get_object_or_404(Study, pk=pk)
    if not request.user.is_radiologist:
        return HttpResponseForbidden("전문의만 확정할 수 있습니다.")
    diag = study.diagnosis

    final = request.POST.getlist("final_findings")
    diag.final_findings_json = json.dumps(final)
    diag.radiologist_note = request.POST.get("radiologist_note", "").strip()
    diag.severity = request.POST.get("severity", "")
    diag.confirmed_by = request.user
    diag.confirmed_at = timezone.now()
    diag.save()
    study.status = Study.STATUS_CONFIRMED
    study.save(update_fields=["status"])

    messages.success(request, "최종 판독 의견이 저장되었습니다.")
    return redirect("studies:detail", pk=pk)


@login_required
def study_report(request, pk):
    study = get_object_or_404(Study.objects.select_related("patient", "uploader"), pk=pk)
    diagnosis = getattr(study, "diagnosis", None)
    return render(request, "studies/report.html",
                  {"study": study, "diagnosis": diagnosis})
