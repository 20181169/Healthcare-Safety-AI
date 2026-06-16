from django.urls import path
from . import views

app_name = "safety"

urlpatterns = [
    path("", views.dashboard, name="dashboard"),
    path("upload/", views.upload, name="upload"),
    path("history/", views.history, name="history"),
    path("event/<int:pk>/", views.event_detail, name="event_detail"),
    path("live/", views.live, name="live"),
    path("live/snapshot/<int:camera_id>/", views.live_snapshot, name="live_snapshot"),
    path("api/quick-infer/", views.quick_infer, name="quick_infer"),
    path("about/", views.about_view, name="about"),
]
