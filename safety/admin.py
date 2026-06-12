from django.contrib import admin
from .models import Camera, DetectionEvent, ViolationLog


@admin.register(Camera)
class CameraAdmin(admin.ModelAdmin):
    list_display = ("name", "location", "stream_url", "video_file",
                    "is_active", "created_at")
    list_filter = ("is_active",)
    search_fields = ("name", "location")
    fields = ("name", "location", "video_file", "stream_url", "is_active")


class ViolationInline(admin.TabularInline):
    model = ViolationLog
    extra = 0
    readonly_fields = ("decision", "yolo_class", "yolo_score",
                       "cls_presence", "cls_correct",
                       "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2", "created_at")


@admin.register(DetectionEvent)
class DetectionEventAdmin(admin.ModelAdmin):
    list_display = ("id", "created_at", "camera", "source",
                    "n_total", "n_violation", "mean_conf",
                    "enhance_applied", "sr_applied_count", "elapsed_ms")
    list_filter = ("source", "camera", "enhance_applied")
    date_hierarchy = "created_at"
    inlines = [ViolationInline]
    readonly_fields = ("original_image", "result_image", "created_at")


@admin.register(ViolationLog)
class ViolationLogAdmin(admin.ModelAdmin):
    list_display = ("id", "created_at", "event", "decision",
                    "yolo_class", "yolo_score")
    list_filter = ("decision",)
    date_hierarchy = "created_at"
