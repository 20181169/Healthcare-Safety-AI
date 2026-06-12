from django import forms
from .models import Recording


_INPUT = ("w-full border border-slate-300 rounded-lg px-3 py-2 text-sm "
          "focus:border-brand-500 focus:ring-1 focus:ring-brand-500 focus:outline-none "
          "transition")


class UploadForm(forms.ModelForm):
    class Meta:
        model = Recording
        fields = ("patient_name", "body_part", "original_wav")
        widgets = {
            "patient_name": forms.TextInput(attrs={
                "class": _INPUT,
                "placeholder": "예: 환자123 (선택)",
            }),
            "body_part": forms.Select(attrs={"class": _INPUT}),
            "original_wav": forms.FileInput(attrs={
                "accept": "audio/wav,audio/x-wav,audio/wave,.wav",
            }),
        }
        labels = {
            "patient_name": "이름 / 식별자",
            "body_part": "측정 부위",
            "original_wav": "심폐음 측정본 (.wav)",
        }

    def clean_original_wav(self):
        f = self.cleaned_data["original_wav"]
        if not f.name.lower().endswith(".wav"):
            raise forms.ValidationError("WAV 형식만 업로드 가능합니다.")
        if f.size > 25 * 1024 * 1024:
            raise forms.ValidationError("파일이 너무 큽니다 (최대 25MB).")
        return f
