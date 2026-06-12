from django.urls import path

from . import api_views


app_name = "api"

urlpatterns = [
    path("healthz/", api_views.healthz, name="healthz"),
    path("inference/", api_views.inference, name="inference"),
]
