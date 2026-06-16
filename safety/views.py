"""안전관리 웹 화면 뷰
   - dashboard    : 요약 통계 + 최근 위반 목록
   - upload       : 이미지 업로드 → 파이프라인 실행 → 결과 저장/조회
   - live         : 카메라 선택 후 MJPEG 실시간 스트림
   - history      : 위반 이력 조회 (날짜/카메라/판정 종류 필터)
   - event_detail : 한 이벤트의 상세 (객체 단위 위반 표 + 결과 이미지)
   - quick_infer  : (API) 멀티파트 업로드 즉시 추론 JSON 반환
"""
import io
import json
import threading
import time
from datetime import timedelta

import cv2
import numpy as np
from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.core.files.base import ContentFile
from django.db.models import Count, Avg, Q
from django.http import (HttpResponse, JsonResponse,
                         StreamingHttpResponse, Http404)
from django.shortcuts import get_object_or_404, redirect, render
from django.utils import timezone
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST

from .forms import UploadForm
from .models import Camera, DetectionEvent, ViolationLog


def _pipeline_runner():
    """basicsr / YOLO 가중치 의존성이 없는 환경에서도 페이지 라우팅이 살아남도록
    실제 추론이 필요한 뷰에서만 lazy import 한다."""
    from . import pipeline_runner  # noqa: WPS433
    return pipeline_runner


# ---------- helpers ----------

def _save_event_from_result(event: DetectionEvent, run: dict):
    """run_on_bytes 결과를 DetectionEvent 와 ViolationLog 로 저장"""
    event.n_total = run["n_total"]
    event.n_violation = run["n_violation"]
    event.mean_conf = run["mean_conf"]
    event.elapsed_ms = run["elapsed_ms"]
    event.enhance_applied = run["enhance_applied"]
    event.sr_applied_count = run["sr_applied_count"]
    event.result_image.save(
        f"result_{event.id or 'tmp'}.jpg",
        ContentFile(run["vis_jpeg"]),
        save=False,
    )
    event.save()

    # 객체별 위반 로그
    ViolationLog.objects.filter(event=event).delete()
    bulk = []
    for r in run["results"]:
        x1, y1, x2, y2 = r["bbox"]
        bulk.append(ViolationLog(
            event=event,
            decision=r["decision"],
            yolo_class=r.get("yolo_class", ""),
            yolo_score=r.get("yolo_score", 0.0),
            cls_presence=r.get("cls_presence", 0.0),
            cls_correct=r.get("cls_correct", 0.0),
            bbox_x1=int(x1), bbox_y1=int(y1),
            bbox_x2=int(x2), bbox_y2=int(y2),
        ))
    ViolationLog.objects.bulk_create(bulk)


# ---------- 화면 ----------

def dashboard(request):
    today = timezone.now().date()
    week_ago = today - timedelta(days=7)
    qs_today = DetectionEvent.objects.filter(created_at__date=today)
    qs_week = DetectionEvent.objects.filter(created_at__date__gte=week_ago)

    cards = {
        "events_today": qs_today.count(),
        "violations_today": qs_today.aggregate(s=Count("violations",
            filter=Q(violations__decision__in=["NO_HELMET", "INCORRECT_WEAR"])))["s"] or 0,
        "events_week": qs_week.count(),
        "mean_conf_week": qs_week.aggregate(a=Avg("mean_conf"))["a"] or 0.0,
        "active_cameras": Camera.objects.filter(is_active=True).count(),
    }

    # 최근 위반 5건
    recent_violations = (ViolationLog.objects
                         .filter(decision__in=["NO_HELMET", "INCORRECT_WEAR"])
                         .select_related("event", "event__camera")
                         .order_by("-created_at")[:5])

    # 카메라별 오늘 통계
    cam_stats = []
    for cam in Camera.objects.filter(is_active=True):
        c_qs = qs_today.filter(camera=cam)
        cam_stats.append({
            "camera": cam,
            "events": c_qs.count(),
            "violations": ViolationLog.objects.filter(
                event__camera=cam, created_at__date=today,
                decision__in=["NO_HELMET", "INCORRECT_WEAR"]).count(),
        })

    return render(request, "safety/dashboard.html", {
        "cards": cards,
        "recent_violations": recent_violations,
        "cam_stats": cam_stats,
    })


def upload(request):
    if request.method == "POST":
        form = UploadForm(request.POST, request.FILES)
        if form.is_valid():
            event: DetectionEvent = form.save(commit=False)
            event.source = "upload"
            event.save()

            event.original_image.open("rb")
            image_bytes = event.original_image.read()
            event.original_image.close()

            try:
                run = _pipeline_runner().run_on_bytes(
                    image_bytes,
                    anonymize=form.cleaned_data.get("anonymize", True),
                )
            except Exception as e:
                event.delete()
                messages.error(request, f"추론 실패: {e}")
                return redirect("safety:upload")

            _save_event_from_result(event, run)
            messages.success(
                request,
                f"분석 완료 ({event.elapsed_ms:.0f}ms) - "
                f"탐지 {event.n_total}, 위반 {event.n_violation}",
            )
            return redirect("safety:event_detail", pk=event.pk)
    else:
        form = UploadForm()
    return render(request, "safety/upload.html", {"form": form})


def history(request):
    qs = (DetectionEvent.objects
          .select_related("camera")
          .prefetch_related("violations")
          .order_by("-created_at"))

    camera_id = request.GET.get("camera")
    if camera_id:
        qs = qs.filter(camera_id=camera_id)
    date_from = request.GET.get("from")
    if date_from:
        qs = qs.filter(created_at__date__gte=date_from)
    date_to = request.GET.get("to")
    if date_to:
        qs = qs.filter(created_at__date__lte=date_to)
    only_violation = request.GET.get("violation")
    if only_violation:
        qs = qs.filter(n_violation__gt=0)

    return render(request, "safety/history.html", {
        "events": qs[:200],
        "cameras": Camera.objects.all(),
        "filters": {
            "camera": camera_id, "from": date_from,
            "to": date_to, "violation": only_violation,
        },
    })


def event_detail(request, pk: int):
    event = get_object_or_404(
        DetectionEvent.objects.select_related("camera")
                              .prefetch_related("violations"),
        pk=pk,
    )
    return render(request, "safety/event_detail.html", {"event": event})


def live(request):
    cameras = Camera.objects.filter(is_active=True)
    return render(request, "safety/live.html", {"cameras": cameras})


# ─────────── 라이브 스트림 ───────────

_PALETTE = {"OK": (0, 200, 0),
            "NO_HELMET": (0, 0, 255),
            "INCORRECT_WEAR": (0, 165, 255)}

# PrivacyService 는 매 프레임마다 새로 만들면 비효율 → 모듈 전역 캐시
_privacy_singleton = None
def _privacy():
    global _privacy_singleton
    if _privacy_singleton is None:
        from services.privacy_service import PrivacyService
        _privacy_singleton = PrivacyService()
    return _privacy_singleton


def _scale_results(results: list, sx: float, sy: float) -> list:
    """축소된 입력으로 추론한 결과의 박스 좌표를 원본 크기로 복원."""
    out = []
    for r in results:
        x1, y1, x2, y2 = r["bbox"]
        nr = dict(r)
        nr["bbox"] = (int(x1 * sx), int(y1 * sy),
                      int(x2 * sx), int(y2 * sy))
        out.append(nr)
    return out


def _overlay(frame: np.ndarray, results: list, anonymize: bool) -> bytes:
    """캐시된 추론 결과를 새 프레임에 가볍게 덧그려 JPEG 송출"""
    out = frame.copy()
    bboxes, classes = [], []
    for r in results:
        x1, y1, x2, y2 = r["bbox"]
        color = _PALETTE.get(r["decision"], (200, 200, 200))
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        cv2.putText(out,
                    f"{r['decision']} [{r.get('yolo_class','')}:{r.get('yolo_score',0):.2f}]",
                    (x1, max(0, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        bboxes.append((x1, y1, x2, y2))
        classes.append(r.get("yolo_class", ""))
    if anonymize and bboxes:
        out = _privacy().anonymize(out, bboxes=bboxes, classes=classes)
    ok, buf = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return buf.tobytes() if ok else b""


def live_stream(request, camera_id: int):
    """비동기 추론 라이브 스트림.

       동작 구조:
         - 메인 루프: 캡처 → 박스 오버레이 → JPEG 송출 (가볍고 빠름)
         - 백그라운드 스레드: 가장 최근 프레임을 작은 사이즈로 추론하여 박스만 갱신
         - 박스 TTL(ttl 초) 지나면 캐시 자동 무효화 → 박스 잔상 제거

       URL 파라미터 (모두 선택):
         fps=2          송출 FPS (기본 2, 포트폴리오 시연용 저부하)
         infer_size=224 추론 입력 폭(px). 작을수록 빠름. (기본 224)
         infer_interval=2.0  백그라운드 추론 최소 간격(초). 기본 2.0
         ttl=3.0        박스 캐시 유효 시간(초). 지나면 박스 사라짐. (기본 3.0)
         anonymize=0|1  얼굴 비식별 (기본 1)
    """
    cam = get_object_or_404(Camera, pk=camera_id)
    source = cam.source
    src = int(source) if source.isdigit() else source
    target_fps = max(1, min(30, int(request.GET.get("fps", 2))))
    infer_size = max(160, min(1280, int(request.GET.get("infer_size", 224))))
    infer_interval = max(0.1, min(10.0, float(request.GET.get("infer_interval", 2.0))))
    ttl = float(request.GET.get("ttl", 3.0))
    anonymize = request.GET.get("anonymize", "1") != "0"
    # frame_skip: 1 이면 모든 프레임 사용, 3 이면 3프레임 중 1개만 = 영상 3배속
    frame_skip = max(1, int(request.GET.get("skip", 3)))
    # 추론 모드:
    #   "bg"   = 백그라운드 스레드 (영상 부드러움, 박스 약간 지연) - 기본
    #   "sync" = 매 프레임 동기 추론 (영상 느림, 박스 즉시 표시)
    #   "off"  = 추론 완전 비활성 (원본 영상만)
    mode = request.GET.get("mode", "bg").lower()
    if mode not in ("bg", "sync", "off"):
        mode = "bg"
    # 하위호환: infer=0 → off
    if request.GET.get("infer") == "0":
        mode = "off"

    state = {
        "frame": None,       # 가장 최근 캡처 프레임 (BGR)
        "frame_shape": None, # (H, W)
        "results": [],       # 가장 최근 추론 결과 (원본 좌표)
        "ts": 0.0,           # 추론 완료 시각
        "stop": False,
    }
    lock = threading.Lock()

    def infer_loop():
        # 백그라운드: 저부하 시연을 위해 일정 간격마다 최신 프레임만 추론.
        last_infer = 0.0
        while not state["stop"]:
            now = time.perf_counter()
            if now - last_infer < infer_interval:
                time.sleep(min(0.05, infer_interval - (now - last_infer)))
                continue
            with lock:
                frame = state["frame"]
            if frame is None:
                time.sleep(0.02)
                continue
            try:
                H, W = frame.shape[:2]
                # 입력 축소로 추론 가속
                if W > infer_size:
                    scale = infer_size / W
                    small = cv2.resize(frame, (infer_size, int(H * scale)),
                                       interpolation=cv2.INTER_AREA)
                else:
                    small = frame
                    scale = 1.0
                run = _pipeline_runner().run_on_frame(small, anonymize=False)
                # 박스 좌표를 원본 크기로 복원
                inv = 1.0 / scale
                results = _scale_results(run["results"], inv, inv)
                with lock:
                    state["results"] = results
                    state["ts"] = time.time()
                last_infer = time.perf_counter()
            except Exception:
                # 추론 실패해도 송출은 계속
                last_infer = time.perf_counter()
                time.sleep(0.05)

    def generator():
        cap = cv2.VideoCapture(src)
        if not cap.isOpened():
            placeholder = np.full((360, 640, 3), 30, dtype=np.uint8)
            cv2.putText(placeholder, f"cannot open: {source}", (20, 180),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            ok, buf = cv2.imencode(".jpg", placeholder)
            if ok:
                yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                       + buf.tobytes() + b"\r\n")
            return

        # 백그라운드 모드일 때만 스레드 시작
        if mode == "bg":
            threading.Thread(target=infer_loop, daemon=True).start()

        interval = 1.0 / target_fps
        try:
            while True:
                t0 = time.perf_counter()
                # 빨리감기 효과: frame_skip-1 프레임을 디코딩 없이 빠르게 건너뜀
                for _ in range(frame_skip - 1):
                    if not cap.grab():
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        break
                ok, frame = cap.read()
                if not ok:
                    # 파일이면 처음으로 되감기 (시연 루프)
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ok, frame = cap.read()
                    if not ok:
                        break

                if mode == "sync":
                    # 동기 모드 - 매 프레임 즉시 추론 (영상 느려지지만 박스 즉시 동기)
                    # 추론 입력 축소로 그나마 속도 확보
                    H, W = frame.shape[:2]
                    if W > infer_size:
                        scale = infer_size / W
                        small = cv2.resize(frame, (infer_size, int(H * scale)),
                                           interpolation=cv2.INTER_AREA)
                    else:
                        small = frame
                        scale = 1.0
                    run = _pipeline_runner().run_on_frame(small, anonymize=False)
                    inv = 1.0 / scale
                    results = _scale_results(run["results"], inv, inv)
                elif mode == "bg":
                    with lock:
                        state["frame"] = frame
                        age = time.time() - state["ts"]
                        results = state["results"] if age <= ttl else []
                else:  # off
                    results = []

                jpeg = _overlay(frame, results, anonymize=anonymize and mode != "off")
                yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                       + jpeg + b"\r\n")

                dt = time.perf_counter() - t0
                if dt < interval:
                    time.sleep(interval - dt)
        finally:
            state["stop"] = True
            cap.release()

    return StreamingHttpResponse(
        generator(),
        content_type="multipart/x-mixed-replace; boundary=frame",
    )


@csrf_exempt
@require_POST
def quick_infer(request):
    """간이 JSON API - 외부 시스템 연동용
       POST multipart/form-data with 'file'
    """
    f = request.FILES.get("file")
    if f is None:
        return JsonResponse({"error": "file required"}, status=400)
    anonymize = request.POST.get("anonymize", "true").lower() == "true"
    try:
        run = _pipeline_runner().run_on_bytes(f.read(), anonymize=anonymize)
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)
    return JsonResponse({
        "elapsed_ms": run["elapsed_ms"],
        "n_total": run["n_total"],
        "n_violation": run["n_violation"],
        "mean_conf": run["mean_conf"],
        "enhance_applied": run["enhance_applied"],
        "sr_applied_count": run["sr_applied_count"],
        "results": run["results"],
    })


def about_view(request):
    """포트폴리오용 기술 노트 — 비로그인도 열람 가능."""
    return render(request, "safety/about.html")
