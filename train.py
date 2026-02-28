from __future__ import annotations

import os
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Optional
from collections import Counter

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import torchvision.transforms as T
from torchvision import models
from torchvision.models import ViT_B_16_Weights


# --------------------------- CONFIG ---------------------------

@dataclass
class TrainConfig:
    # data
    images_root: Path = Path("/scratch/vmli3/cs370/data/")
    labels_csv: Path = Path("data/train.csv")
    filename_col: str = "filename"
    target_col: str = "elk_count"

    val_frac: float = 0.2
    seed: int = 42
    drop_missing_images: bool = True

    # training
    batch_size: int = 32
    num_workers: int = 4
    pin_memory: bool = True

    max_epochs: int = 24
    backbone_lr: float = 1e-5
    head_lr: float = 5e-4
    weight_decay: float = 0.0

    # output
    run_dir: Path = Path("/scratch/vmli3/cs370/model-checkpoints")
    log_path: Path = Path("train.log")

    # loss
    huber_beta: float = 1.0  # SmoothL1Loss(beta=...)


# ------------------------ UTILS / LOG -------------------------

def normalize_rel(p: str) -> str:
    """Normalize any gs:// or https:// prefixes to match local relative paths."""
    p = str(p).strip()
    p = p.replace("gs://public-datasets-lila/nacti-unzipped/", "")
    p = p.replace("https://storage.googleapis.com/public-datasets-lila/nacti-unzipped/", "")
    return p.lstrip("/")


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def log_line(cfg: TrainConfig, msg: str) -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    cfg.log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cfg.log_path, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def save_checkpoint(cfg: TrainConfig, model: nn.Module, epoch: int, best: bool, extra: dict) -> Path:
    cfg.run_dir.mkdir(parents=True, exist_ok=True)
    ckpt = {
        "epoch": epoch,
        "model_state": model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict(),
        "extra": extra,
    }
    name = "best.pt" if best else f"epoch_{epoch:03d}.pt"
    path = cfg.run_dir / name
    torch.save(ckpt, path)
    return path


# ---------------------- DATASET + LOADERS ---------------------

class ElkCountDataset(Dataset):
    def __init__(self, df: pd.DataFrame, images_root: Path, transform=None):
        self.df = df.reset_index(drop=True)
        self.images_root = images_root
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        rel = row["filename"]
        y = float(row["elk_count"])

        img_path = self.images_root / rel
        img = Image.open(img_path).convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        return img, torch.tensor(y, dtype=torch.float32)


def make_transforms(train: bool = True):
    """
    Robust across torchvision versions:
    - uses weights.transforms() for Resize/ToTensor/Normalize
    - applies light augmentation before that (on PIL images)
    """
    weights = ViT_B_16_Weights.IMAGENET1K_V1
    base = weights.transforms()  # includes Resize(224) + ToTensor + Normalize

    if not train:
        return base

    aug = T.Compose([
        T.RandomResizedCrop(224, scale=(0.8, 1.0)),
        T.RandomHorizontalFlip(p=0.5),
    ])

    # Apply aug first (PIL -> PIL), then base (PIL -> tensor normalized)
    return T.Compose([aug, base])


def load_labels(cfg: TrainConfig) -> pd.DataFrame:
    df = pd.read_csv(cfg.labels_csv)

    if cfg.filename_col not in df.columns or cfg.target_col not in df.columns:
        raise ValueError(
            f"labels_csv must have columns {cfg.filename_col!r}, {cfg.target_col!r}. "
            f"Found: {list(df.columns)}"
        )

    df = df[[cfg.filename_col, cfg.target_col]].copy()
    df[cfg.filename_col] = df[cfg.filename_col].astype(str).map(normalize_rel)
    df[cfg.target_col] = pd.to_numeric(df[cfg.target_col], errors="coerce").fillna(0).astype(float)

    df = df.rename(columns={cfg.filename_col: "filename", cfg.target_col: "elk_count"})
    df = df.drop_duplicates(subset=["filename"], keep="last").reset_index(drop=True)

    if cfg.drop_missing_images:
        exists_mask = df["filename"].map(lambda r: (cfg.images_root / r).exists())
        missing = int((~exists_mask).sum())
        if missing:
            print(f"[data] Dropping {missing} rows with missing images on disk.")
        df = df[exists_mask].copy().reset_index(drop=True)

    return df


def split_train_val(df: pd.DataFrame, val_frac: float, seed: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    n_val = int(round(len(df) * val_frac))
    val_df = df.iloc[:n_val].copy()
    train_df = df.iloc[n_val:].copy()
    return train_df, val_df


def get_train_val_loaders(cfg: TrainConfig) -> Tuple[DataLoader, DataLoader]:
    df = load_labels(cfg)
    train_df, val_df = split_train_val(df, cfg.val_frac, cfg.seed)

    train_ds = ElkCountDataset(train_df, cfg.images_root, transform=make_transforms(train=True))
    val_ds = ElkCountDataset(val_df, cfg.images_root, transform=make_transforms(train=False))

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        drop_last=False,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        drop_last=False,
    )

    return train_loader, val_loader


# ------------------------ MODEL (REG) -------------------------

class ElkCountViT(nn.Module):
    """
    Pretrained ViT-B/16 with a regression head outputting a single scalar.
    Keeps your original "resize to 224 if needed" behavior.
    """
    def __init__(self):
        super().__init__()
        weights = ViT_B_16_Weights.IMAGENET1K_V1
        self.vit = models.vit_b_16(weights=weights)

        in_features = self.vit.heads.head.in_features
        self.vit.heads.head = nn.Linear(in_features, 1)

    def forward(self, x):
        if x.shape[-1] != 224 or x.shape[-2] != 224:
            x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
        y = self.vit(x)          # (N, 1)
        return y.squeeze(1)      # (N,)


# -------------------- TRAIN / EVAL LOOPS ----------------------

def train_epoch(loader: DataLoader, model: nn.Module, criterion, optimizer, device: torch.device) -> float:
    model.train()
    losses = []

    for X, y in loader:
        X = X.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True).float()

        optimizer.zero_grad(set_to_none=True)
        pred = model(X)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

    return float(np.mean(losses)) if losses else float("nan")


@torch.no_grad()
def eval_epoch(loader: DataLoader, model: nn.Module, criterion, device: torch.device) -> Tuple[float, float, float]:
    model.eval()
    losses = []
    abs_sum = 0.0
    sq_sum = 0.0
    n = 0

    for X, y in loader:
        X = X.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True).float()

        pred = model(X)
        loss = criterion(pred, y).item()
        losses.append(loss)

        err = (pred - y)
        abs_sum += err.abs().sum().item()
        sq_sum += (err * err).sum().item()
        n += y.numel()

    mean_loss = float(np.mean(losses)) if losses else float("nan")
    mae = abs_sum / n if n else float("nan")
    rmse = (sq_sum / n) ** 0.5 if n else float("nan")
    return mean_loss, mae, rmse


# ------------------------------ MAIN --------------------------

def main():
    cfg = TrainConfig()

    # reset log
    if cfg.log_path.exists():
        cfg.log_path.unlink()
    cfg.run_dir.mkdir(parents=True, exist_ok=True)

    device = get_device()
    log_line(cfg, f"Device: {device}")

    train_loader, val_loader = get_train_val_loaders(cfg)
    log_line(cfg, f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

    model = ElkCountViT().to(device)
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        log_line(cfg, f"Using DataParallel across {torch.cuda.device_count()} GPUs")

    criterion = nn.SmoothL1Loss(beta=cfg.huber_beta)

    # Param groups: backbone (low LR) vs head (high LR)
    vit_params = []
    head_params = []
    for name, param in model.named_parameters():
        # DataParallel prefix can be "module."
        if "vit.heads.head" in name:
            head_params.append(param)
        else:
            vit_params.append(param)

    optimizer = torch.optim.Adam(
        [
            {"params": vit_params, "lr": cfg.backbone_lr},
            {"params": head_params, "lr": cfg.head_lr},
        ],
        weight_decay=cfg.weight_decay,
    )

    best_val_mae = float("inf")
    best_epoch = -1

    # initial eval
    tr_loss, tr_mae, tr_rmse = eval_epoch(train_loader, model, criterion, device)
    va_loss, va_mae, va_rmse = eval_epoch(val_loader, model, criterion, device)
    log_line(cfg, f"Epoch 0 | Train loss {tr_loss:.4f} MAE {tr_mae:.4f} RMSE {tr_rmse:.4f} | "
                  f"Val loss {va_loss:.4f} MAE {va_mae:.4f} RMSE {va_rmse:.4f}")

    for epoch in range(1, cfg.max_epochs + 1):
        train_loss = train_epoch(train_loader, model, criterion, optimizer, device)

        tr_loss, tr_mae, tr_rmse = eval_epoch(train_loader, model, criterion, device)
        va_loss, va_mae, va_rmse = eval_epoch(val_loader, model, criterion, device)

        log_line(cfg, f"Epoch {epoch} | Train loss {tr_loss:.4f} MAE {tr_mae:.4f} RMSE {tr_rmse:.4f} | "
                      f"Val loss {va_loss:.4f} MAE {va_mae:.4f} RMSE {va_rmse:.4f}")

        is_best = va_mae < best_val_mae
        if is_best:
            best_val_mae = va_mae
            best_epoch = epoch

        ckpt_path = save_checkpoint(
            cfg,
            model,
            epoch=epoch,
            best=is_best,
            extra={
                "val_mae": va_mae,
                "val_rmse": va_rmse,
                "val_loss": va_loss,
                "train_loss": tr_loss,
                "train_mae": tr_mae,
                "train_rmse": tr_rmse,
                "config": cfg.__dict__,
            },
        )
        if is_best:
            log_line(cfg, f"  -> new best checkpoint: {ckpt_path} (Val MAE={va_mae:.4f})")

    log_line(cfg, f"Done. Best epoch={best_epoch} best Val MAE={best_val_mae:.4f}")
    print("Saved checkpoints in:", cfg.run_dir.resolve())


if __name__ == "__main__":
    main()