from datetime import timedelta

from django.contrib.auth.decorators import login_required
from django.shortcuts import render
from django.utils import timezone

from patients.models import Patient
from studies.models import Diagnosis, Study


@login_required
def home(request):
    now = timezone.now()
    since7 = now - timedelta(days=7)
    since30 = now - timedelta(days=30)

    stats = {
        "patients_total": Patient.objects.count(),
        "studies_total": Study.objects.count(),
        "studies_7d": Study.objects.filter(captured_at__gte=since7).count(),
        "pending_review": Study.objects.filter(
            status__in=[Study.STATUS_AI, Study.STATUS_REVIEW]
        ).count(),
        "confirmed_7d": Study.objects.filter(
            status=Study.STATUS_CONFIRMED, captured_at__gte=since7
        ).count(),
    }

    my_studies = (
        Study.objects.filter(uploader=request.user)
        .select_related("patient")[:8]
    )
    review_queue = (
        Study.objects.filter(status__in=[Study.STATUS_AI, Study.STATUS_REVIEW])
        .select_related("patient", "diagnosis")[:8]
    )

    bucket = {}
    for d in Diagnosis.objects.filter(study__captured_at__gte=since30):
        for f in d.positive_findings:
            bucket[f] = bucket.get(f, 0) + 1
    finding_counts = sorted(bucket.items(), key=lambda kv: kv[1], reverse=True)[:8]

    return render(request, "dashboard/home.html", {
        "stats": stats,
        "my_studies": my_studies,
        "review_queue": review_queue,
        "finding_counts": finding_counts,
    })
