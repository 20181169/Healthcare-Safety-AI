from django import forms

from .models import Study


_input = ("w-full border border-slate-300 rounded-lg px-3 py-2 text-sm "
          "focus:border-brand-500 focus:ring-1 focus:ring-brand-500 focus:outline-none transition")


class StudyUploadForm(forms.ModelForm):
    class Meta:
        model = Study
        fields = ["patient", "image", "location", "clinical_note"]
        widgets = {
            "patient": forms.Select(attrs={"class": _input}),
            "image": forms.ClearableFileInput(attrs={
                "id": "image-input", "class": "hidden",
                "accept": "image/*,.dcm",
            }),
            "location": forms.TextInput(attrs={"class": _input,
                "placeholder": "○○구급차 / ○○보건소"}),
            "clinical_note": forms.TextInput(attrs={"class": _input,
                "placeholder": "발열, 기침 3일째 등"}),
        }


class ConfirmDiagnosisForm(forms.Form):
    final_findings = forms.MultipleChoiceField(
        widget=forms.CheckboxSelectMultiple, required=False,
    )
    severity = forms.ChoiceField(
        required=False,
        choices=[("", "선택")] + [
            ("mild", "경증"), ("moderate", "중등도"),
            ("severe", "중증"), ("critical", "위중"),
        ],
        widget=forms.Select(attrs={"class": _input}),
    )
    radiologist_note = forms.CharField(
        required=False,
        widget=forms.Textarea(attrs={"class": _input, "rows": 4,
            "placeholder": "추가 권고사항, 후속 검사, 응급실 이송 권고 등"}),
    )
