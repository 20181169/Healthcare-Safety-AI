import json

from django.conf import settings
from django.db import models

from patients.models import Patient


class Study(models.Model):
    STATUS_UPLOADED = "uploaded"
    STATUS_AI = "ai_analyzed"
    STATUS_REVIEW = "under_review"
    STATUS_CONFIRMED = "confirmed"
    STATUS_ARCHIVED = "archived"
    STATUS_CHOICES = [
        (STATUS_UPLOADED, "업로드 완료"),
        (STATUS_AI, "AI 분석 완료"),
        (STATUS_REVIEW, "전문의 검토 중"),
        (STATUS_CONFIRMED, "전문의 확정"),
        (STATUS_ARCHIVED, "보관"),
    ]
    STATUS_COLOR = {
        STATUS_UPLOADED: "gray", STATUS_AI: "blue",
        STATUS_REVIEW: "amber", STATUS_CONFIRMED: "emerald",
        STATUS_ARCHIVED: "slate",
    }

    patient = models.ForeignKey(Patient, on_delete=models.CASCADE, related_name="studies")
    uploader = models.ForeignKey(
        settings.AUTH_USER_MODEL, on_delete=models.PROTECT, related_name="studies_uploaded",
    )
    image = models.ImageField("X-ray 영상", upload_to="xray/")
    captured_at = models.DateTimeField("촬영시각", auto_now_add=True)
    location = models.CharField("촬영 위치", max_length=120, blank=True)
    clinical_note = models.TextField("임상 메모", blank=True)
    status = models.CharField("상태", max_length=20, choices=STATUS_CHOICES, default=STATUS_UPLOADED)

    class Meta:
        ordering = ["-captured_at"]
        verbose_name = "검사"
        verbose_name_plural = "검사"

    def __str__(self):
        return f"검사 #{self.pk} ({self.patient.name})"

    @property
    def status_kor(self):
        return dict(self.STATUS_CHOICES).get(self.status, self.status)

    @property
    def status_color(self):
        return self.STATUS_COLOR.get(self.status, "gray")


class Diagnosis(models.Model):
    SEVERITY_CHOICES = [
        ("mild", "경증"), ("moderate", "중등도"),
        ("severe", "중증"), ("critical", "위중"),
    ]

    study = models.OneToOneField(Study, on_delete=models.CASCADE, related_name="diagnosis")

    predictions_json = models.TextField()  # {질환명: prob}
    positive_findings_json = models.TextField()  # ["Pneumonia", ...]
    top_finding = models.CharField(max_length=40, blank=True)
    top_prob = models.FloatField(default=0.0)
    threshold = models.FloatField(default=0.5)
    inference_ms = models.IntegerField(default=0)
    gradcam = models.ImageField(upload_to="gradcam/", null=True, blank=True)
    gradcam_class = models.CharField(max_length=40, blank=True)
    is_demo = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)

    confirmed_by = models.ForeignKey(
        settings.AUTH_USER_MODEL, on_delete=models.SET_NULL,
        null=True, blank=True, related_name="diagnoses_confirmed",
    )
    confirmed_at = models.DateTimeField(null=True, blank=True)
    final_findings_json = models.TextField(blank=True, default="")
    radiologist_note = models.TextField(blank=True)
    severity = models.CharField(max_length=10, choices=SEVERITY_CHOICES, blank=True)

    class Meta:
        verbose_name = "AI 판독"
        verbose_name_plural = "AI 판독"

    @property
    def predictions(self) -> dict:
        return json.loads(self.predictions_json)

    @property
    def positive_findings(self) -> list:
        return json.loads(self.positive_findings_json)

    @property
    def final_findings(self) -> list:
        return json.loads(self.final_findings_json) if self.final_findings_json else []

    @property
    def ranked(self) -> list:
        return sorted(self.predictions.items(), key=lambda kv: kv[1], reverse=True)
