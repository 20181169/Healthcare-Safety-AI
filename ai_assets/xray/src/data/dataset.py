"""
소아 X-ray 데이터셋 로더
- NIH ChestX-ray14 와 유사한 Multi-Label 포맷을 가정
- CSV: image_name, Atelectasis, Cardiomegaly, Effusion, ... (0/1)
"""

import os
from typing import List, Tuple

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


DISEASE_LABELS: List[str] = [
    "Atelectasis",
    "Cardiomegaly",
    "Effusion",
    "Infiltration",
    "Mass",
    "Nodule",
    "Pneumonia",
    "Pneumothorax",
    "Consolidation",
    "Edema",
    "Emphysema",
    "Fibrosis",
    "Pleural_Thickening",
    "Hernia",
]


def build_transform(image_size: int, train: bool) -> transforms.Compose:
    if train:
        return transforms.Compose([
            transforms.Resize((image_size + 16, image_size + 16)),
            transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=5),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ])
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])


class PediatricXrayDataset(Dataset):
    """Multi-Label 분류용 데이터셋."""

    def __init__(
        self,
        csv_path: str,
        image_dir: str,
        image_size: int = 224,
        train: bool = True,
        labels: List[str] = None,
    ):
        self.df = pd.read_csv(csv_path)
        self.image_dir = image_dir
        self.labels = labels or DISEASE_LABELS
        self.transform = build_transform(image_size, train)

        missing = [c for c in self.labels if c not in self.df.columns]
        if missing:
            raise ValueError(f"CSV에 다음 라벨 컬럼이 없습니다: {missing}")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        row = self.df.iloc[idx]
        path = os.path.join(self.image_dir, row["image_name"])
        image = Image.open(path).convert("L")
        x = self.transform(image)
        y = torch.tensor(row[self.labels].values.astype("float32"))
        return x, y


class UnlabeledXrayDataset(Dataset):
    """PGGAN 학습용 (라벨 불필요). 폴더 내 모든 X-ray 이미지를 사용."""

    def __init__(self, image_dir: str, image_size: int):
        self.paths = [
            os.path.join(image_dir, f)
            for f in os.listdir(image_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.Grayscale(num_output_channels=1),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ])

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.transform(Image.open(self.paths[idx]).convert("L"))
