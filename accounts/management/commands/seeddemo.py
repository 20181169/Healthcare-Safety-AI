"""데모 계정 + 환자 시드.  python manage.py seeddemo"""

from datetime import date

from django.core.management.base import BaseCommand

from accounts.models import User
from patients.models import Patient


DEMO_USERS = [
    {"email": "medic@demo.kr",  "password": "demopass123",
     "name": "김응급",  "role": "paramedic",
     "affiliation": "○○소방서 119구급대"},
    {"email": "doctor@demo.kr", "password": "demopass123",
     "name": "박임상",  "role": "clinician",
     "affiliation": "○○보건의료센터", "license_no": "MD-12345"},
    {"email": "rad@demo.kr",    "password": "demopass123",
     "name": "이영상",  "role": "radiologist",
     "affiliation": "고려대학교 안암병원 영상의학과", "license_no": "MD-99001"},
    {"email": "admin@demo.kr",  "password": "demopass123",
     "name": "관리자",  "role": "admin", "affiliation": "운영팀"},
]

DEMO_PATIENTS = [
    ("PED-2026-0001", "김민준", "M", date(2020, 3, 12)),
    ("PED-2026-0002", "이서윤", "F", date(2018, 7, 1)),
    ("PED-2026-0003", "박지호", "M", date(2022, 11, 24)),
    ("PED-2026-0004", "최하은", "F", date(2019, 5, 30)),
]


class Command(BaseCommand):
    help = "데모 사용자/환자를 생성합니다."

    def handle(self, *args, **options):
        for u in DEMO_USERS:
            user, created = User.objects.get_or_create(
                email=u["email"],
                defaults={
                    "name": u["name"], "role": u["role"],
                    "affiliation": u.get("affiliation", ""),
                    "license_no": u.get("license_no", ""),
                    "is_staff": u["role"] == "admin",
                    "is_superuser": u["role"] == "admin",
                },
            )
            if created:
                user.set_password(u["password"])
                user.save()
                self.stdout.write(self.style.SUCCESS(f"[user] 생성 {u['email']}"))

        medic = User.objects.filter(email="medic@demo.kr").first()
        for code, name, sex, bd in DEMO_PATIENTS:
            _, created = Patient.objects.get_or_create(
                patient_code=code,
                defaults={
                    "name": name, "sex": sex, "birth_date": bd,
                    "guardian_phone": "010-0000-0000",
                    "notes": "시연용 가상 환자",
                    "created_by": medic,
                },
            )
            if created:
                self.stdout.write(self.style.SUCCESS(f"[patient] 생성 {code} {name}"))

        self.stdout.write("")
        self.stdout.write(self.style.SUCCESS("데모 시드 완료"))
        self.stdout.write("  로그인: medic@demo.kr / doctor@demo.kr / rad@demo.kr / admin@demo.kr")
        self.stdout.write("  비밀번호: demopass123")
