"""심폐음 데이터셋 — SR 학습용 페어 / 분류용 라벨링."""
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from ..utils.audio import (
    HR_SR, LR_SR, load_wave, random_crop,
    simulate_lr_from_hr, peak_normalize,
)


LABELS = ["normal", "crackle", "wheeze", "both"]


class SRDataset(Dataset):
    """HR wav → (LR, HR) 페어 생성. 실제 대회에서는 측정본 페어가 주어졌지만
    본 코드는 단일 HR 음원으로부터 가정환경 측정을 시뮬레이션."""

    def __init__(self, csv_path: str, segment_sec: float = 4.0,
                 augment: bool = True):
        self.df = pd.read_csv(csv_path)
        self.segment = int(segment_sec * HR_SR)
        self.augment = augment

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        hr = load_wave(row["filepath"], target_sr=HR_SR)

        if self.augment:
            hr = hr * np.random.uniform(0.7, 1.0)
            shift = np.random.randint(-HR_SR // 10, HR_SR // 10)
            hr = np.roll(hr, shift)

        hr = random_crop(hr, self.segment)
        lr = simulate_lr_from_hr(hr, hr_sr=HR_SR, lr_sr=LR_SR,
                                 add_noise=self.augment)

        hr = peak_normalize(hr)
        lr = peak_normalize(lr)
        return (
            torch.from_numpy(lr).float().unsqueeze(0),
            torch.from_numpy(hr).float().unsqueeze(0),
        )


class ClsDataset(Dataset):
    """4-class 분류용. lr_input=True 면 LR 시뮬레이션 후 반환."""

    def __init__(self, csv_path: str, segment_sec: float = 6.0,
                 augment: bool = True, lr_input: bool = True):
        self.df = pd.read_csv(csv_path)
        self.segment = int(segment_sec * HR_SR)
        self.augment = augment
        self.lr_input = lr_input
        self.label2idx = {l: i for i, l in enumerate(LABELS)}

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        wav = load_wave(row["filepath"], target_sr=HR_SR)

        if self.augment:
            wav = wav * np.random.uniform(0.7, 1.0)

        wav = random_crop(wav, self.segment)

        if self.lr_input:
            wav = simulate_lr_from_hr(wav, add_noise=self.augment)
            wav = peak_normalize(wav)

        label = self.label2idx[row["label"]]
        return torch.from_numpy(wav).float().unsqueeze(0), torch.tensor(label).long()
