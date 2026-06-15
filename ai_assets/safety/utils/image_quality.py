"""이미지 품질 측정 - 조건부 보정/SR 트리거 판단용"""
import cv2
import numpy as np


def measure_brightness(image: np.ndarray) -> float:
    """V 채널 평균값으로 밝기 측정 (0~255)"""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    return float(np.mean(hsv[:, :, 2]))


def measure_contrast(image: np.ndarray) -> float:
    """그레이스케일 표준편차 기반 대비"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return float(np.std(gray))


def measure_noise(image: np.ndarray) -> float:
    """라플라시안 기반 노이즈 추정 (정규화)"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    H, W = gray.shape
    M = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype=np.float32)
    conv = np.abs(cv2.filter2D(gray, -1, M))
    sigma = np.sum(conv) / (6.0 * (H - 2) * (W - 2))
    return float(sigma)


def measure_sharpness(image: np.ndarray) -> float:
    """라플라시안 분산으로 선명도 추정 - 값이 작을수록 흐릿함"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def quality_report(image: np.ndarray) -> dict:
    return {
        "brightness": measure_brightness(image),
        "contrast": measure_contrast(image),
        "noise": measure_noise(image),
        "sharpness": measure_sharpness(image),
    }
