from django.contrib import admin

from .models import Diagnosis, Study


@admin.register(Study)
class StudyAdmin(admin.ModelAdmin):
    list_display = ("id", "patient", "uploader", "status", "captured_at")
    list_filter = ("status",)
    search_fields = ("patient__name", "patient__patient_code")


@admin.register(Diagnosis)
class DiagnosisAdmin(admin.ModelAdmin):
    list_display = ("id", "study", "top_finding", "top_prob",
                    "is_demo", "confirmed_by", "confirmed_at")
    list_filter = ("is_demo", "severity")
