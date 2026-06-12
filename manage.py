#!/usr/bin/env python
"""Django management entrypoint — 통합 헬스케어/안전 AI."""
import os
import sys


def main():
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "healthcare.settings")
    try:
        from django.core.management import execute_from_command_line
    except ImportError as exc:
        raise ImportError(
            "Django 가 설치되어 있지 않습니다. requirements.txt 를 확인하세요."
        ) from exc
    execute_from_command_line(sys.argv)


if __name__ == "__main__":
    main()
