"""분류기 학습 루프 — baseline(LR 직접) / sr(SR 복원 후) 비교."""
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, confusion_matrix
from tqdm import tqdm

from ..data import ClsDataset, LABELS
from ..models import HeartLungCNN, load_sr_model
from ..utils.metrics import icbhi_score


def _class_weights(csv_path: str, n_classes: int) -> torch.Tensor:
    df = pd.read_csv(csv_path)
    label2idx = {l: i for i, l in enumerate(LABELS)}
    counts = np.zeros(n_classes, dtype=np.float64)
    for lab in df["label"]:
        counts[label2idx[lab]] += 1
    w = counts.sum() / (n_classes * np.clip(counts, 1, None))
    return torch.tensor(w, dtype=torch.float32)


class ClsTrainer:
    def __init__(self, cfg, mode: str = "sr", sr_ckpt: str = None,
                 resume: bool = True):
        assert mode in ("baseline", "sr")
        self.cfg = cfg
        self.mode = mode
        self.resume = resume
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        os.makedirs(cfg.train.out_dir, exist_ok=True)

        self.train_loader = DataLoader(
            ClsDataset(cfg.data.train_csv, segment_sec=cfg.audio.segment_sec,
                       augment=True, lr_input=True),
            batch_size=cfg.train.batch_size, shuffle=True,
            num_workers=cfg.train.num_workers, pin_memory=True, drop_last=True,
        )
        self.val_loader = DataLoader(
            ClsDataset(cfg.data.val_csv, segment_sec=cfg.audio.segment_sec,
                       augment=False, lr_input=True),
            batch_size=cfg.train.batch_size, shuffle=False,
            num_workers=cfg.train.num_workers, pin_memory=True,
        )

        self.sr_model = None
        if mode == "sr":
            assert sr_ckpt, "mode=sr 일 때 sr_ckpt 필요"
            self.sr_model, _ = load_sr_model(sr_ckpt, device=self.device)
            self.sr_model.eval()
            for p in self.sr_model.parameters():
                p.requires_grad_(False)

        self.model = HeartLungCNN(
            n_classes=cfg.model.n_classes,
            sr=cfg.audio.hr_sr,
            n_mels=cfg.model.n_mels,
            n_fft=cfg.model.n_fft,
            hop=cfg.model.hop,
            fmin=cfg.model.fmin,
            fmax=cfg.model.fmax,
        ).to(self.device)

        weight = _class_weights(cfg.data.train_csv, cfg.model.n_classes).to(self.device)
        self.criterion = nn.CrossEntropyLoss(
            weight=weight, label_smoothing=cfg.train.label_smoothing
        )
        self.opt = torch.optim.AdamW(
            self.model.parameters(),
            lr=cfg.train.lr, weight_decay=cfg.train.weight_decay,
        )
        self.sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.opt, T_max=cfg.train.epochs
        )

    def _maybe_sr(self, wave: torch.Tensor) -> torch.Tensor:
        if self.sr_model is None:
            return wave
        with torch.no_grad():
            return self.sr_model(wave)

    def _train_epoch(self) -> float:
        self.model.train()
        total = 0.0
        for wave, label in tqdm(self.train_loader, desc="train", leave=False):
            wave = wave.to(self.device, non_blocking=True)
            label = label.to(self.device, non_blocking=True)

            wave = self._maybe_sr(wave)
            logits = self.model(wave)
            loss = self.criterion(logits, label)

            self.opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.train.grad_clip)
            self.opt.step()
            total += loss.item() * wave.size(0)
        return total / len(self.train_loader.dataset)

    @torch.no_grad()
    def _validate(self) -> dict:
        self.model.eval()
        preds, gts = [], []
        for wave, label in self.val_loader:
            wave = wave.to(self.device); label = label.to(self.device)
            wave = self._maybe_sr(wave)
            logits = self.model(wave)
            preds.append(logits.argmax(1).cpu().numpy())
            gts.append(label.cpu().numpy())
        preds = np.concatenate(preds); gts = np.concatenate(gts)
        cm = confusion_matrix(gts, preds, labels=list(range(self.cfg.model.n_classes)))
        return {
            "acc": float((preds == gts).mean()),
            "f1_macro": float(f1_score(gts, preds, average="macro", zero_division=0)),
            **icbhi_score(cm),
        }

    def fit(self):
        best_f1 = 0.0
        ckpt_path = os.path.join(self.cfg.train.out_dir, f"best_{self.mode}.pt")

        if self.resume and os.path.isfile(ckpt_path):
            state = torch.load(ckpt_path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(state["model"])
            best_f1 = float(state.get("f1_macro", 0.0))
            print(f"[resume] {ckpt_path} 로드  prev best macroF1={best_f1:.3f}")
            print(f"[note] optimizer/scheduler 는 새 학습으로 시작 (warm-start)")

        for ep in range(1, self.cfg.train.epochs + 1):
            t0 = time.time()
            tr_loss = self._train_epoch()
            m = self._validate()
            self.sched.step()

            print(f"[ep {ep:03d}] loss={tr_loss:.4f}  "
                  f"acc={m['acc']:.3f}  macroF1={m['f1_macro']:.3f}  "
                  f"ICBHI={m['icbhi']:.3f} (Sp={m['specificity']:.3f}/Sn={m['sensitivity']:.3f})  "
                  f"({time.time()-t0:.1f}s)")

            if m["f1_macro"] > best_f1:
                best_f1 = m["f1_macro"]
                torch.save({"model": self.model.state_dict(), "mode": self.mode, **m},
                           ckpt_path)
                print(f"  -> saved (macroF1={best_f1:.3f})")
        return ckpt_path
