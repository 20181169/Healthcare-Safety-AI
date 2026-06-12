"""안전관리 도메인 모델
   - Camera         : 현장 카메라 (위치/스트림 URL)
   - DetectionEvent : 한 번의 추론에 대한 메타 (이미지/영상 분석 단위)
   - ViolationLog   : 안전모 미착용/잘못 착용이 탐지된 개별 객체 단위 기록
"""
from django.db import models
from django.utils import timezone


class Camera(models.Model):
    name = models.CharField("카메라 명", max_length=100)
    location = models.CharField("설치 위치", max_length=200, blank=True)
    stream_url = models.CharField(
        "스트림 URL (선택)",
        max_length=300, blank=True,
        help_text="0 = 로컬 웹캠, 또는 RTSP/HTTP URL. 파일을 업로드하면 이 값은 무시됩니다.",
        default="",
    )
    video_file = models.FileField(
        "영상 파일 (선택)",
        upload_to="cameras/",
        blank=True, null=True,
        help_text="mp4/avi 등 시연용 영상을 업로드하면 이 파일이 우선 사용됩니다.",
    )
    is_active = models.BooleanField("운영 여부", default=True)
    created_at = models.DateTimeField(auto_now_add=True)

    @property
    def source(self) -> str:
        """실제 OpenCV VideoCapture 에 넘겨줄 소스 (업로드 파일 > URL > 0 순)."""
        if self.video_file:
            return self.video_file.path
        return (self.stream_url or "0").strip()

    class Meta:
        verbose_name = "카메라"
        verbose_name_plural = "카메라"
        ordering = ["name"]

    def __str__(self):
        return f"{self.name} ({self.location})"


class DetectionEvent(models.Model):
    """한 번의 이미지/영상 분석 결과"""
    SOURCE_CHOICES = [
        ("upload", "이미지 업로드"),
        ("live", "실시간 스트림"),
        ("api", "외부 API"),
    ]
    camera = models.ForeignKey(
        Camera, on_delete=models.SET_NULL, null=True, blank=True,
        related_name="events", verbose_name="카메라",
    )
    source = models.CharField("소스", max_length=20, choices=SOURCE_CHOICES, default="upload")
    original_image = models.ImageField("원본", upload_to="originals/%Y/%m/%d/")
    result_image = models.ImageField("결과", upload_to="results/%Y/%m/%d/", blank=True, null=True)
    n_total = models.PositiveIntegerField("탐지 객체 수", default=0)
    n_violation = models.PositiveIntegerField("위반 수", default=0)
    mean_conf = models.FloatField("평균 신뢰도", default=0.0)
    elapsed_ms = models.FloatField("처리 시간(ms)", default=0.0)
    enhance_applied = models.BooleanField("Zero-DCE++ 적용", default=False)
    sr_applied_count = models.PositiveIntegerField("SR 적용 ROI 수", default=0)
    created_at = models.DateTimeField("발생 시각", default=timezone.now, db_index=True)

    class Meta:
        verbose_name = "탐지 이벤트"
        verbose_name_plural = "탐지 이벤트"
        ordering = ["-created_at"]

    def __str__(self):
        return f"#{self.id} {self.get_source_display()} {self.created_at:%Y-%m-%d %H:%M}"

    @property
    def has_violation(self) -> bool:
        return self.n_violation > 0


class ViolationLog(models.Model):
    """개별 객체 단위 위반 기록 - 안전모 미착용 / 잘못 착용"""
    KIND_CHOICES = [
        ("OK", "정상"),
        ("NO_HELMET", "미착용"),
        ("INCORRECT_WEAR", "올바르지 않은 착용"),
    ]
    event = models.ForeignKey(
        DetectionEvent, on_delete=models.CASCADE,
        related_name="violations", verbose_name="이벤트",
    )
    decision = models.CharField("판정", max_length=20, choices=KIND_CHOICES)
    yolo_class = models.CharField("YOLO 클래스", max_length=40, blank=True)
    yolo_score = models.FloatField("YOLO 신뢰도", default=0.0)
    cls_presence = models.FloatField("착용 확률", default=0.0)
    cls_correct = models.FloatField("올바른 착용 확률", default=0.0)
    bbox_x1 = models.IntegerField()
    bbox_y1 = models.IntegerField()
    bbox_x2 = models.IntegerField()
    bbox_y2 = models.IntegerField()
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        verbose_name = "위반 로그"
        verbose_name_plural = "위반 로그"
        ordering = ["-created_at"]

    def __str__(self):
        return f"{self.get_decision_display()} ({self.yolo_class} {self.yolo_score:.2f})"
