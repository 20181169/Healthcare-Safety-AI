from django.contrib import admin
from .models import Recording


@admin.register(Recording)
class RecordingAdmin(admin.ModelAdmin):
    list_display = ("id", "patient_name", "body_part", "predicted_label",
                    "confidence", "uploaded_at")
    list_filter = ("body_part", "predicted_label")
    search_fields = ("patient_name",)
    readonly_fields = ("uploaded_at", "processed_at", "snr_in_db", "snr_out_db",
                       "probabilities", "error")
