"""프로젝트 루트 뷰 — 통합 랜딩 페이지."""

from django.shortcuts import redirect, render


def landing(request):
    """비로그인: 통합 소개 화면. 로그인 상태면 대시보드로."""
    if request.user.is_authenticated:
        return redirect("dashboard:home")
    return render(request, "landing.html")
