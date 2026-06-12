# 통합 헬스케어·안전 AI 플랫폼

세 개의 독립 Django 프로젝트를 **하나의 통합 Django 프로젝트**로 합쳤습니다.

| 원본 프로젝트            | 합쳐진 앱                                     | 기능                                                 |
|--------------------------|----------------------------------------------|------------------------------------------------------|
| 수상관리 (공사현장 안전) | `safety`                                     | YOLO + Zero-DCE++ + Real-ESRGAN 안전모 탐지         |
| 수상관리_심폐음          | `diagnosis` (+ 패키지 `audio_sr`)            | 청진음 Audio Super-Resolution + 4-class 분류         |
| 수상관리_소아엑스레이    | `accounts`, `patients`, `studies`, `dashboard` | 소아 X-ray 다중 라벨 분류 + Grad-CAM + 전문의 확정 |

단일 로그인 계정으로 세 가지 시스템을 모두 사용할 수 있습니다.

## 디렉토리 구조

```
win/
├── manage.py
├── requirements.txt
├── README.md
├── healthcare/              # Django 프로젝트 패키지
│   ├── settings.py          # 통합 설정 (sys.path 도 여기서 조정)
│   ├── urls.py              # 루트 URL — 3개 시스템 include
│   ├── views.py             # 랜딩 페이지
│   ├── context.py           # 글로벌 컨텍스트 (DEMO 모드 배지)
│   ├── wsgi.py / asgi.py
│   └── __init__.py
├── accounts/                # 커스텀 User (email 로그인) + 로그인/회원가입
├── dashboard/               # 통합 대시보드 (3개 시스템 진입)
├── patients/                # X-ray 환자 마스터
├── studies/                 # X-ray 검사 업로드/판독/확정 + API
├── diagnosis/               # 심폐음 측정 업로드 → SR → 진단
├── safety/                  # 현장 안전 — 이미지/실시간 안전모 탐지
├── audio_sr/                # 심폐음 ML 패키지 (원본 수상관리_심폐음/src 를 복사)
├── templates/
│   ├── base.html            # 통합 네비게이션 셸
│   ├── landing.html         # 비로그인 랜딩
│   ├── dashboard/ patients/ studies/ registration/
├── static/css/  static/js/
└── media/                   # 업로드 파일
```

### 외부 디렉토리 의존성

크기가 큰 ML 가중치와 사전 학습 코드는 원본 위치에서 그대로 참조합니다
(`settings.py` 가 `sys.path` 를 자동으로 추가).

| 환경변수         | 기본 경로                                            | 용도                                                  |
|------------------|------------------------------------------------------|-------------------------------------------------------|
| `SAFETY_ROOT`    | `../수상관리`                                        | `pipeline.py`, `services/`, `models/`, `weights/`     |
| `XRAY_ROOT`      | `../수상관리_소아엑스레이`                           | `src/classifier`, `src/data`, `outputs/classifier`    |
| `HEART_LUNG_ROOT`| `../수상관리_심폐음`                                 | `checkpoints/sr`, `checkpoints/cls`                   |

세 폴더가 `win/` 과 같은 부모 디렉토리(`Desktop/`)에 있다면 추가 설정 없이 동작합니다.

## 설치 & 실행

의존성은 4개 파일로 나뉘어 있습니다 — **코어만 설치하면 일단 서버는 뜨고**,
각 시스템 ML 스택은 필요할 때 따로 깔면 됩니다.

| 파일                       | 용도                                                     |
|----------------------------|----------------------------------------------------------|
| `requirements.txt`         | Django + 공통 (필수)                                     |
| `requirements-xray.txt`    | 소아 X-ray 분류기 (`studies` 앱)                         |
| `requirements-audio.txt`   | 심폐음 SR + 분류 (`diagnosis` 앱)                        |
| `requirements-safety.txt`  | 현장 안전 YOLO/SR/Zero-DCE++ (`safety` 앱) — 빌드 까다로움 |

### 1. 가상환경 + 코어 (필수)

```powershell
cd C:\Users\user\Desktop\win
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip

# Django + 공통 의존성
pip install -r requirements.txt

# DB 초기화 & 슈퍼유저
python manage.py migrate
python manage.py createsuperuser   # email 로 로그인

# 개발 서버
python manage.py runserver
```

`http://localhost:8000/` 접속 — 이 시점에서 X-ray/심폐음/안전 모두 **DEMO 모드**로 동작합니다.

### 2. torch 설치 (X-ray/심폐음 실제 추론에 필요)

GPU 환경에 맞춰 PyTorch 휠을 먼저 설치합니다:

```powershell
# CUDA 12.x
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
# CPU 전용
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### 3. 사용할 시스템의 추가 의존성

```powershell
pip install -r requirements-xray.txt        # 소아 X-ray
pip install -r requirements-audio.txt       # 심폐음
pip install -r requirements-safety.txt      # 현장 안전 (아래 주의)
```

### 4. 현장 안전 (`safety`) — basicsr 빌드 문제

`basicsr==1.4.2` 는 **Python 3.12+ 에서 빌드 실패** (`KeyError: '__version__'`) 합니다.
다음 중 하나를 선택하세요:

* **A. Python 3.10 / 3.11 venv 사용** — `pip install -r requirements-safety.txt` 그대로.
* **B. Python 3.12+ 에서 우회**:
  ```powershell
  pip install "setuptools<69" wheel
  pip install basicsr==1.4.2 --no-build-isolation
  pip install realesrgan==0.3.0 --no-deps
  pip install ultralytics opencv-python scikit-image albumentations
  ```
* **C. SR 단계 생략**: `pip install ultralytics scikit-image albumentations` 만 설치.
  `SAFETY_ROOT/weights/RealESRGAN_x4plus.pth` 가 없으면 파이프라인이 SR 을 건너뜁니다.

`requirements-safety.txt` 상단에 동일한 안내가 들어있습니다.

## 주요 URL

| 경로            | 화면                                              |
|-----------------|---------------------------------------------------|
| `/`             | 랜딩 (비로그인) / 대시보드 (로그인)               |
| `/accounts/`    | 로그인·회원가입                                    |
| `/dashboard/`   | 통합 대시보드 (3개 시스템 진입 카드 + X-ray 통계) |
| `/patients/`    | 환자 목록·상세·등록                                |
| `/studies/`     | X-ray 검사 업로드·AI 추론·전문의 확정              |
| `/api/`         | X-ray 추론 JSON API (`/api/healthz/`, `/api/inference/`) |
| `/diagnosis/`   | 심폐음 측정 업로드·SR 복원·분류 결과               |
| `/safety/`      | 현장 안전 대시보드·이미지 분석·실시간 스트림·이력 |
| `/admin/`       | Django Admin                                       |

## 합치면서 적용된 변경점

* **단일 인증 통합**: `safety`, `diagnosis` 의 모든 뷰에 `@login_required` 추가.
  (원본 두 프로젝트는 로그인 없이 동작했음)
* **템플릿 통합 셸**: `safety/base.html`, `diagnosis/base.html` 이 프로젝트
  `base.html` 을 상속하도록 변경. 자식 템플릿의 `{% block content %}` 는
  `{% block safety_content %}` / `{% block diagnosis_content %}` 로 재명명.
* **`src` 네임스페이스 충돌 해결**: 두 ML 프로젝트가 모두 `src/` 패키지를
  쓰고 있었으므로 심폐음 측을 `audio_sr/` 로 복사·임포트만 갱신.
  소아 X-ray 측은 원본 위치를 `sys.path` 에 추가하여 그대로 사용.
* **`AUTH_USER_MODEL`**: pediaxray 의 커스텀 `accounts.User` (email 로그인,
  의료진 역할/면허번호)를 통합 사용자 모델로 채택.
* **단일 SQLite DB**: `win/db.sqlite3` 한 파일에 6개 앱 마이그레이션 통합.

## 데모 모드

체크포인트 파일이 없으면 각 모듈이 자동으로 데모 모드로 동작합니다.

* **X-ray**: `XRAY_ROOT/outputs/classifier/best.pt` 없으면 의사난수 기반 데모.
  네비게이션 우측에 `X-ray DEMO` 배지 표시.
* **심폐음**: `HEART_LUNG_ROOT/checkpoints/{sr,cls}/best*.pt` 없으면 무작위
  가중치 모델 사용. 결과 화면에 경고 메시지.
* **안전**: `SAFETY_ROOT/weights/*.pt` 가 없으면 파이프라인 import 시점에
  에러. 데모 시연 전 가중치 배치 필수.
