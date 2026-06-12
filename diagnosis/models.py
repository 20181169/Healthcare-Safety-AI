"""심폐음 측정/진단 기록."""
from django.db import models


class Recording(models.Model):
    LABEL_CHOICES = [
        ("normal", "정상"),
        ("crackle", "수포음 (Crackle)"),
        ("wheeze", "천명음 (Wheeze)"),
        ("both", "수포음+천명음 동반"),
        ("", "미진단"),
    ]
    PART_CHOICES = [("heart", "심음"), ("lung", "폐음")]

    patient_name = models.CharField("이름/식별자", max_length=64, blank=True)
    body_part = models.CharField("측정 부위", max_length=8, choices=PART_CHOICES,
                                 default="lung")
    uploaded_at = models.DateTimeField("업로드 시각", auto_now_add=True)

    original_wav = models.FileField("원본(LR) 측정본", upload_to="uploads/")
    restored_wav = models.FileField("SR 복원본", upload_to="restored/", blank=True, null=True)

    waveform_png = models.FileField("파형 비교", upload_to="visuals/", blank=True, null=True)
    spectrogram_png = models.FileField("스펙트로그램 비교", upload_to="visuals/", blank=True, null=True)
    spectrum_png = models.FileField("평균 스펙트럼", upload_to="visuals/", blank=True, null=True)

    predicted_label = models.CharField("진단 라벨", max_length=16,
                                       choices=LABEL_CHOICES, default="", blank=True)
    confidence = models.FloatField("Top-1 확신도", default=0.0)
    probabilities = models.JSONField("클래스별 확률", default=dict, blank=True)

    snr_in_db = models.FloatField("입력 SNR(dB)", null=True, blank=True)
    snr_out_db = models.FloatField("복원 SNR(dB)", null=True, blank=True)
    processed_at = models.DateTimeField("처리 완료 시각", null=True, blank=True)
    error = models.TextField("오류 메시지", blank=True)

    class Meta:
        ordering = ["-uploaded_at"]

    def __str__(self) -> str:
        who = self.patient_name or f"#{self.pk}"
        return f"{who} · {self.get_body_part_display()} · {self.get_predicted_label_display() or '대기'}"

    @property
    def label_kor(self) -> str:
        return self.get_predicted_label_display() or "—"
