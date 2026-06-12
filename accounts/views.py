from django.contrib import messages
from django.contrib.auth import login, logout
from django.contrib.auth.decorators import login_required
from django.contrib.auth.views import LoginView
from django.shortcuts import redirect, render
from django.urls import reverse_lazy

from .forms import LoginForm, SignupForm


class CustomLoginView(LoginView):
    template_name = "registration/login.html"
    authentication_form = LoginForm
    redirect_authenticated_user = True

    def form_valid(self, form):
        user = form.get_user()
        messages.success(self.request, f"환영합니다, {user.name} {user.role_kor}님.")
        return super().form_valid(form)


def signup_view(request):
    if request.user.is_authenticated:
        return redirect("dashboard:home")
    if request.method == "POST":
        form = SignupForm(request.POST)
        if form.is_valid():
            user = form.save()
            login(request, user)
            messages.success(request, "회원가입이 완료되었습니다.")
            return redirect("dashboard:home")
    else:
        form = SignupForm()
    return render(request, "registration/signup.html", {"form": form})


@login_required
def logout_view(request):
    logout(request)
    messages.info(request, "로그아웃되었습니다.")
    return redirect("accounts:login")
