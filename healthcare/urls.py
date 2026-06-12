"""루트 URL 설정 — 통합 헬스케어/안전 AI."""

from django.conf import settings
from django.conf.urls.static import static
from django.contrib import admin
from django.urls import include, path

from . import views


urlpatterns = [
    path("admin/", admin.site.urls),
    path("", views.landing, name="landing"),
    path("accounts/", include("accounts.urls", namespace="accounts")),
    path("dashboard/", include("dashboard.urls", namespace="dashboard")),
    path("patients/", include("patients.urls", namespace="patients")),
    path("studies/", include("studies.urls", namespace="studies")),
    path("api/", include("studies.api_urls", namespace="api")),
    path("diagnosis/", include("diagnosis.urls", namespace="diagnosis")),
    path("safety/", include("safety.urls", namespace="safety")),
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
