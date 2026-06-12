from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.db.models import Q
from django.shortcuts import get_object_or_404, redirect, render

from .forms import PatientForm
from .models import Patient


@login_required
def patient_list(request):
    q = request.GET.get("q", "").strip()
    qs = Patient.objects.all()
    if q:
        qs = qs.filter(Q(name__icontains=q) | Q(patient_code__icontains=q))
    return render(request, "patients/list.html", {"patients": qs[:200], "q": q})


@login_required
def patient_new(request):
    if request.method == "POST":
        form = PatientForm(request.POST)
        if form.is_valid():
            patient = form.save(commit=False)
            patient.created_by = request.user
            patient.save()
            messages.success(request, "환자 정보가 등록되었습니다.")
            return redirect("patients:detail", pk=patient.pk)
    else:
        form = PatientForm()
    return render(request, "patients/new.html", {"form": form})


@login_required
def patient_detail(request, pk):
    patient = get_object_or_404(Patient, pk=pk)
    studies = patient.studies.all()
    return render(request, "patients/detail.html", {"patient": patient, "studies": studies})
