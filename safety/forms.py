from django import forms
from .models import Camera, DetectionEvent


_INPUT = ("w-full border border-slate-300 rounded-lg px-3 py-2 text-sm "
          "focus:border-brand-500 focus:ring-1 focus:ring-brand-500 focus:outline-none "
          "transition")


class UploadForm(forms.ModelForm):
    anonymize = forms.BooleanField(
        label="얼굴 비식별 처리(블러)", required=False, initial=True,
    )

    class Meta:
        model = DetectionEvent
        fields = ["camera", "original_image"]
        labels = {
            "camera": "카메라 선택 (선택사항)",
            "original_image": "분석할 이미지",
        }
        widgets = {
            "camera": forms.Select(attrs={"class": _INPUT}),
            "original_image": forms.ClearableFileInput(
                attrs={"accept": "image/*"},
            ),
        }
