"""
통합 헬스케어/안전 AI Django 설정.

세 개의 독립 프로젝트(수상관리: 공사현장 안전, 수상관리_심폐음: 심폐음 진단,
수상관리_소아엑스레이: 소아 X-ray 판독)를 하나로 합친 통합 웹앱.
"""
import os
import sys
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent           # .../win
ASSETS_DIR = BASE_DIR / "ai_assets"

# 배포 서버에는 외부 한글 경로 프로젝트가 없으므로, 필요한 데모 자산만
# win/ai_assets 아래에 모아두고 기본 경로로 사용한다.
SAFETY_ROOT = Path(os.environ.get("SAFETY_ROOT", str(ASSETS_DIR / "safety")))
XRAY_ROOT = Path(os.environ.get("XRAY_ROOT", str(ASSETS_DIR / "xray")))

# 안전관리 파이프라인 (pipeline.py, services, models, utils, config) 임포트용
if SAFETY_ROOT.is_dir() and str(SAFETY_ROOT) not in sys.path:
    sys.path.insert(0, str(SAFETY_ROOT))

# 소아 X-ray src 임포트용 (from src.classifier / from src.data)
if XRAY_ROOT.is_dir() and str(XRAY_ROOT) not in sys.path:
    sys.path.append(str(XRAY_ROOT))

# win/ 자체도 (audio_sr import 대비)
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))


SECRET_KEY = os.environ.get(
    "DJANGO_SECRET_KEY",
    "django-insecure-CHANGE-ME-IN-PRODUCTION-aaaaaaaaaaaaaaaaaaaaaaaaaa",
)
DEBUG = os.environ.get("DJANGO_DEBUG", "1") == "1"
ALLOWED_HOSTS = ["*"] if DEBUG else os.environ.get("DJANGO_ALLOWED_HOSTS", "").split(",")

# 외부 도메인(ngrok 등)에서 업로드 POST 가능
CSRF_TRUSTED_ORIGINS = [
    "https://*.ngrok-free.app",
    "https://*.ngrok.app",
    "https://*.ngrok.io",
]


INSTALLED_APPS = [
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
    "django.contrib.humanize",
    # 자체 앱
    "accounts.apps.AccountsConfig",
    "dashboard.apps.DashboardConfig",
    "patients.apps.PatientsConfig",
    "studies.apps.StudiesConfig",
    "diagnosis.apps.DiagnosisConfig",
    "safety.apps.SafetyConfig",
]

MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
]

ROOT_URLCONF = "healthcare.urls"

TEMPLATES = [{
    "BACKEND": "django.template.backends.django.DjangoTemplates",
    "DIRS": [BASE_DIR / "templates"],
    "APP_DIRS": True,
    "OPTIONS": {
        "context_processors": [
            "django.template.context_processors.debug",
            "django.template.context_processors.request",
            "django.contrib.auth.context_processors.auth",
            "django.contrib.messages.context_processors.messages",
            "healthcare.context.global_flags",
        ],
    },
}]

WSGI_APPLICATION = "healthcare.wsgi.application"

DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.sqlite3",
        "NAME": BASE_DIR / "db.sqlite3",
    }
}

AUTH_USER_MODEL = "accounts.User"
LOGIN_URL = "accounts:login"
LOGIN_REDIRECT_URL = "dashboard:home"
LOGOUT_REDIRECT_URL = "accounts:login"

AUTH_PASSWORD_VALIDATORS = [
    {"NAME": "django.contrib.auth.password_validation.MinimumLengthValidator",
     "OPTIONS": {"min_length": 8}},
    {"NAME": "django.contrib.auth.password_validation.CommonPasswordValidator"},
]

LANGUAGE_CODE = "ko-kr"
TIME_ZONE = "Asia/Seoul"
USE_I18N = True
USE_TZ = True

STATIC_URL = "static/"
STATICFILES_DIRS = [BASE_DIR / "static"]
STATIC_ROOT = BASE_DIR / "staticfiles"

MEDIA_URL = "media/"
MEDIA_ROOT = BASE_DIR / "media"

DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"

MESSAGE_TAGS = {
    10: "info",      # DEBUG
    20: "info",      # INFO
    25: "success",   # SUCCESS
    30: "warning",   # WARNING
    40: "danger",    # ERROR
}

# 업로드 크기 (영상/이미지/오디오)
DATA_UPLOAD_MAX_MEMORY_SIZE = 50 * 1024 * 1024
FILE_UPLOAD_MAX_MEMORY_SIZE = 50 * 1024 * 1024


# ───────── 소아 X-ray (studies 앱) ─────────
CLASSIFIER_CKPT = os.environ.get(
    "CLASSIFIER_CKPT", str(XRAY_ROOT / "outputs" / "classifier" / "best.pt")
)
DEFAULT_THRESHOLD = float(os.environ.get("DEFAULT_THRESHOLD", "0.5"))
SUPERVISED_LABELS = os.environ.get("SUPERVISED_LABELS", "Pneumonia").split(",")
ALLOWED_XRAY_EXTENSIONS = {"png", "jpg", "jpeg", "dcm"}


# ───────── 심폐음 진단 (diagnosis 앱) ─────────
HEART_LUNG_ROOT = Path(
    os.environ.get("HEART_LUNG_ROOT", str(ASSETS_DIR / "heartlung"))
)
SR_CKPT = os.environ.get("SR_CKPT", str(HEART_LUNG_ROOT / "checkpoints" / "sr" / "best.pt"))
CLS_CKPT = os.environ.get("CLS_CKPT", str(HEART_LUNG_ROOT / "checkpoints" / "cls" / "best_sr.pt"))
INFERENCE_DEVICE = os.environ.get("INFERENCE_DEVICE", "auto")  # auto/cpu/cuda
INFERENCE_SEGMENT_SEC = float(os.environ.get("INFERENCE_SEGMENT_SEC", "6.0"))
