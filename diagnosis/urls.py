from django.urls import path
from . import views


app_name = "diagnosis"

urlpatterns = [
    path("", views.upload_view, name="upload"),
    path("process/<int:pk>/", views.process_view, name="process"),
    path("result/<int:pk>/", views.result_view, name="result"),
    path("history/", views.history_view, name="history"),
    path("about/", views.about_view, name="about"),
]
