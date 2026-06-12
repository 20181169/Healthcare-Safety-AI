from datetime import date

from django.conf import settings
from django.db import models


class Patient(models.Model):
    SEX_CHOICES = [("M", "남"), ("F", "여")]

    patient_code = models.CharField("환자코드", max_length=30, unique=True)
    name = models.CharField("이름", max_length=80)
    birth_date = models.DateField("생년월일", null=True, blank=True)
    sex = models.CharField("성별", max_length=1, choices=SEX_CHOICES, blank=True)
    guardian_phone = models.CharField("보호자 연락처", max_length=20, blank=True)
    notes = models.TextField("메모", blank=True)
    created_by = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.SET_NULL,
        null=True,
        related_name="patients_registered",
    )
    created_at = models.DateTimeField("등록일", auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]
        verbose_name = "환자"
        verbose_name_plural = "환자"

    def __str__(self):
        return f"{self.name} ({self.patient_code})"

    @property
    def age_years(self):
        if not self.birth_date:
            return None
        today = date.today()
        return today.year - self.birth_date.year - (
            (today.month, today.day) < (self.birth_date.month, self.birth_date.day)
        )
