from django import forms

from .models import Patient


_input = ("w-full border border-slate-300 rounded-lg px-3 py-2 text-sm "
          "focus:border-brand-500 focus:ring-1 focus:ring-brand-500 focus:outline-none transition")


class PatientForm(forms.ModelForm):
    class Meta:
        model = Patient
        fields = ["patient_code", "name", "sex", "birth_date", "guardian_phone", "notes"]
        widgets = {
            "patient_code": forms.TextInput(attrs={"class": _input + " font-mono"}),
            "name": forms.TextInput(attrs={"class": _input}),
            "sex": forms.Select(attrs={"class": _input}),
            "birth_date": forms.DateInput(attrs={"class": _input, "type": "date"}),
            "guardian_phone": forms.TextInput(attrs={"class": _input}),
            "notes": forms.Textarea(attrs={"class": _input, "rows": 3}),
        }
