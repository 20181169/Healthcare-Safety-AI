from django import forms
from django.contrib.auth.forms import AuthenticationForm

from .models import User


_INPUT = ("w-full border border-slate-300 rounded-lg px-3 py-2 text-sm "
          "focus:border-brand-500 focus:ring-1 focus:ring-brand-500 focus:outline-none "
          "transition")


class LoginForm(AuthenticationForm):
    username = forms.EmailField(
        label="이메일",
        widget=forms.EmailInput(attrs={
            "class": _INPUT, "autofocus": True, "placeholder": "you@hospital.kr",
        }),
    )
    password = forms.CharField(
        label="비밀번호",
        widget=forms.PasswordInput(attrs={"class": _INPUT, "placeholder": "••••••••"}),
    )


class SignupForm(forms.ModelForm):
    password = forms.CharField(
        label="비밀번호", min_length=8,
        widget=forms.PasswordInput(attrs={"class": _INPUT, "placeholder": "8자 이상"}),
    )
    password2 = forms.CharField(
        label="비밀번호 확인",
        widget=forms.PasswordInput(attrs={"class": _INPUT}),
    )

    class Meta:
        model = User
        fields = ["email", "name", "role", "affiliation", "license_no"]
        widgets = {
            "email": forms.EmailInput(attrs={"class": _INPUT, "placeholder": "you@hospital.kr"}),
            "name": forms.TextInput(attrs={"class": _INPUT, "placeholder": "홍길동"}),
            "role": forms.Select(attrs={"class": _INPUT}),
            "affiliation": forms.TextInput(attrs={
                "class": _INPUT,
                "placeholder": "○○구급대 / ○○병원 영상의학과",
            }),
            "license_no": forms.TextInput(attrs={
                "class": _INPUT, "placeholder": "의료진만 입력",
            }),
        }

    def clean(self):
        cleaned = super().clean()
        if cleaned.get("password") != cleaned.get("password2"):
            self.add_error("password2", "비밀번호가 일치하지 않습니다.")
        role = cleaned.get("role")
        if role in ("clinician", "radiologist") and not cleaned.get("license_no"):
            self.add_error("license_no", "의료진은 면허번호를 입력해야 합니다.")
        return cleaned

    def save(self, commit=True):
        user = super().save(commit=False)
        user.set_password(self.cleaned_data["password"])
        if commit:
            user.save()
        return user
