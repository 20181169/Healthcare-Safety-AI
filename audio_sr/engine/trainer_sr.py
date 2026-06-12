"""Audio Super-Resolution 학습 루프."""
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..data import SRDataset
from ..models import AudioSRUNet
from ..losses import MultiResolutionSTFTLoss
from ..utils.metrics import snr_db, lsd_db


class SRTrainer:
    def __init__(self, cfg, resume: bool = True):
        self.cfg = cfg
        self.resume = resume
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        os.makedirs(cfg.train.out_dir, exist_ok=True)

        self.train_loader = DataLoader(
            SRDataset(cfg.data.train_csv, segment_sec=cfg.audio.segment_sec, augment=True),
            batch_size=cfg.train.batch_size, shuffle=True,
            num_workers=cfg.train.num_workers, pin_memory=True, drop_last=True,
        )
        self.val_loader = DataLoader(
            SRDataset(cfg.data.val_csv, segment_sec=cfg.audio.segment_sec, augment=False),
            batch_size=cfg.train.batch_size, shuffle=False,
            num_workers=cfg.train.num_workers, pin_memory=True,
        )

        self.model = AudioSRUNet(
            channels=cfg.model.channels, kernel=cfg.model.kernel
        ).to(self.device)
        self.opt = torch.optim.AdamW(
            self.model.parameters(),
            lr=cfg.train.lr, weight_decay=cfg.train.weight_decay,
        )
        self.sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.opt, T_max=cfg.train.epochs
        )
        self.mrstft = MultiResolutionSTFTLoss(
            n_ffts=cfg.loss.n_ffts, hops=cfg.loss.hops
        ).to(self.device)

    def _train_epoch(self) -> float:
        self.model.train()
        total = 0.0
        for lr_x, hr_x in tqdm(self.train_loader, desc="train", leave=False):
            lr_x = lr_x.to(self.device, non_blocking=True)
            hr_x = hr_x.to(self.device, non_blocking=True)

            pred = self.model(lr_x)
            loss = (
                self.cfg.loss.l1_weight * F.l1_loss(pred, hr_x)
                + self.cfg.loss.mrstft_weight * self.mrstft(pred, hr_x)
            )

            self.opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.train.grad_clip)
            self.opt.step()

            total += loss.item() * lr_x.size(0)
        return total / len(self.train_loader.dataset)

    @torch.no_grad()
    def _validate(self) -> tuple:
        self.model.eval()
        snr_in, snr_out, lsd_out, n = 0.0, 0.0, 0.0, 0
        for lr_x, hr_x in tqdm(self.val_loader, desc="val", leave=False):
            lr_x = lr_x.to(self.device); hr_x = hr_x.to(self.device)
            pred = self.model(lr_x)
            snr_in += snr_db(hr_x.squeeze(1), lr_x.squeeze(1)).sum().item()
            snr_out += snr_db(hr_x.squeeze(1), pred.squeeze(1)).sum().item()
            lsd_out += lsd_db(hr_x.squeeze(1), pred.squeeze(1)).item() * lr_x.size(0)
            n += lr_x.size(0)
        return snr_in / n, snr_out / n, lsd_out / n

    def fit(self):
        best_lsd = float("inf")
        ckpt_path = os.path.join(self.cfg.train.out_dir, "best.pt")

        if self.resume and os.path.isfile(ckpt_path):
            state = torch.load(ckpt_path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(state["model"])
            best_lsd = float(state.get("lsd", float("inf")))
            print(f"[resume] {ckpt_path} 로드  prev best LSD={best_lsd:.3f}")
            print(f"[note] optimizer/scheduler 는 새 학습으로 시작 (warm-start)")

        for ep in range(1, self.cfg.train.epochs + 1):
            t0 = time.time()
            tr_loss = self._train_epoch()
            snr_in, snr_out, lsd_out = self._validate()
            self.sched.step()

            print(f"[ep {ep:03d}] loss={tr_loss:.4f}  "
                  f"SNR(in→out)={snr_in:.2f}→{snr_out:.2f} dB  "
                  f"LSD={lsd_out:.3f}  ({time.time()-t0:.1f}s)")

            if lsd_out < best_lsd:
                best_lsd = lsd_out
                torch.save({"model": self.model.state_dict(), "lsd": lsd_out}, ckpt_path)
                print(f"  -> saved (LSD={lsd_out:.3f})")
        return ckpt_path
