from django.urls import path

from . import views


app_name = "patients"

urlpatterns = [
    path("", views.patient_list, name="list"),
    path("new/", views.patient_new, name="new"),
    path("<int:pk>/", views.patient_detail, name="detail"),
]
