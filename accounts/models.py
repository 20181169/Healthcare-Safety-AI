from django.contrib.auth.models import AbstractUser
from django.db import models

from .managers import UserManager


class User(AbstractUser):
    ROLE_PARAMEDIC = "paramedic"
    ROLE_CLINICIAN = "clinician"
    ROLE_RADIOLOGIST = "radiologist"
    ROLE_ADMIN = "admin"
    ROLE_CHOICES = [
        (ROLE_PARAMEDIC, "응급구조사"),
        (ROLE_CLINICIAN, "임상의"),
        (ROLE_RADIOLOGIST, "영상의학과 전문의"),
        (ROLE_ADMIN, "관리자"),
    ]

    # username 필드 제거하고 email 사용
    username = None
    email = models.EmailField("이메일", unique=True)
    name = models.CharField("이름", max_length=80)
    role = models.CharField("역할", max_length=20, choices=ROLE_CHOICES, default=ROLE_PARAMEDIC)
    affiliation = models.CharField("소속", max_length=120, blank=True)
    license_no = models.CharField("면허번호", max_length=40, blank=True)

    USERNAME_FIELD = "email"
    REQUIRED_FIELDS = ["name"]

    objects = UserManager()

    class Meta:
        verbose_name = "사용자"
        verbose_name_plural = "사용자"

    def __str__(self):
        return f"{self.name} <{self.email}>"

    @property
    def role_kor(self) -> str:
        return dict(self.ROLE_CHOICES).get(self.role, self.role)

    @property
    def is_radiologist(self) -> bool:
        return self.role in (self.ROLE_RADIOLOGIST, self.ROLE_ADMIN)
