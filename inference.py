#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from torchvision import models
from torchvision.models import ViT_B_16_Weights


# ---- Model matches your training.py ----
class ElkCountViT(nn.Module):
    def __init__(self):
        super().__init__()
        weights = ViT_B_16_Weights.IMAGENET1K_V1
        self.vit = models.vit_b_16(weights=weights)

        in_features = self.vit.heads.head.in_features
        self.vit.heads.head = nn.Linear(in_features, 1)

    def forward(self, x):
        if x.shape[-1] != 224 or x.shape[-2] != 224:
            x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
        y = self.vit(x)
        return y.squeeze(1)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_checkpoint(model: nn.Module, ckpt_path: Path, device: torch.device) -> None:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt["model_state"] if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt

    cleaned = {k.replace("module.", "", 1): v for k, v in state.items()}
    model.load_state_dict(cleaned, strict=True)

def preprocess_image(img_path: Path) -> torch.Tensor:
    # Same preprocessing as ViT pretrained weights
    weights = ViT_B_16_Weights.IMAGENET1K_V1
    tfm = weights.transforms()

    img = Image.open(img_path).convert("RGB")
    x = tfm(img)           # (3, 224, 224)
    return x.unsqueeze(0)  # (1, 3, 224, 224)


def safe_copy(src: Path, dst_dir: Path) -> Path:
    """
    Copy src into dst_dir. If a file with the same name exists, add a numeric suffix.
    Returns the final destination path.
    """
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / src.name

    if not dst.exists():
        shutil.copy2(str(src), str(dst))
        return dst

    stem, suffix = src.stem, src.suffix
    k = 1
    while True:
        candidate = dst_dir / f"{stem}_{k}{suffix}"
        if not candidate.exists():
            shutil.copy2(str(src), str(candidate))
            return candidate
        k += 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, help="Path to an image file (jpg/png).")
    ap.add_argument("--checkpoint", required=True, help="Path to model checkpoint (e.g., best.pt).")
    ap.add_argument("--round", action="store_true", help="Round prediction to nearest int.")
    ap.add_argument("--clamp0", action="store_true", help="Clamp negative predictions to 0.")
    ap.add_argument(
        "--copy-to",
        default=".",
        help="Directory to copy the image into after inference (default: current directory).",
    )
    args = ap.parse_args()

    img_path = Path(args.image)
    ckpt_path = Path(args.checkpoint)
    copy_dir = Path(args.copy_to)

    if not img_path.exists():
        raise FileNotFoundError(f"Image not found: {img_path}")
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    device = get_device()

    model = ElkCountViT().to(device)
    model.eval()
    load_checkpoint(model, ckpt_path, device)

    x = preprocess_image(img_path).to(device)

    with torch.no_grad():
        pred = model(x).item()

    if args.clamp0:
        pred = max(0.0, pred)
    if args.round:
        pred = int(round(pred))

    print(pred)

    copied_to = safe_copy(img_path, copy_dir)
    print(f"Copied image to: {copied_to.resolve()}")


if __name__ == "__main__":
    main()