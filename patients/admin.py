from django.contrib import admin

from .models import Patient


@admin.register(Patient)
class PatientAdmin(admin.ModelAdmin):
    list_display = ("patient_code", "name", "sex", "birth_date", "created_at")
    list_filter = ("sex",)
    search_fields = ("patient_code", "name")
