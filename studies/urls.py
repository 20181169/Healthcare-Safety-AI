from django.urls import path

from . import views


app_name = "studies"

urlpatterns = [
    path("", views.study_list, name="list"),
    path("new/", views.study_new, name="new"),
    path("<int:pk>/", views.study_detail, name="detail"),
    path("<int:pk>/rerun_cam/", views.study_rerun_cam, name="rerun_cam"),
    path("<int:pk>/claim/", views.study_claim, name="claim"),
    path("<int:pk>/confirm/", views.study_confirm, name="confirm"),
    path("<int:pk>/report/", views.study_report, name="report"),
]
