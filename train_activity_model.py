# train_activity_model.py
"""
Train a CNN + LSTM model for human activity recognition:
    Classes: normal | running_panic | violence | suspicious

Architecture:
    EfficientNet-B0 (ImageNet pretrained) → spatial features per frame (1280-dim)
    → BiLSTM sequence head → softmax classification

Input:  Sequence of 16 frames (112 × 112 RGB)
Output: Activity class probabilities

Usage:
    python train_activity_model.py                # uses configs/detection_config.yaml
    python train_activity_model.py --epochs 80 --batch 4

Best model saved to: models/activity_classifier.pt
Plots saved to:      results/activity_training/
"""

from __future__ import annotations

import argparse
import os
import time
import random
from pathlib import Path
from typing import Optional

import numpy as np
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.models as models
import cv2
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for saving plots
import matplotlib.pyplot as plt

from utils.logger import get_logger

logger = get_logger("train_activity")

# ── Reproducibility ───────────────────────────────────────────────────────────

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# ── Class Labels ──────────────────────────────────────────────────────────────

CLASS_NAMES = ["normal", "violence"]
NUM_CLASSES = len(CLASS_NAMES)


# ── Dataset ───────────────────────────────────────────────────────────────────

class ActivityVideoDataset(Dataset):
    """
    Loads video clips as fixed-length frame sequences for classification.

    Expects directory layout:
        data_dir/
            normal/        ← video files (.mp4, .avi) or frame sub-folders
            running_panic/ ← ...
            violence/      ← ...
            suspicious/    ← ...

    If subdirs contain image frames directly, they are read as sequences.
    If subdirs contain video files, they are decoded on-the-fly.
    """

    VALID_VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv"}
    VALID_IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp"}

    def __init__(
        self,
        data_dir: str,
        class_names: list[str],
        seq_len: int = 16,
        img_size: int = 112,
        transform: Optional[T.Compose] = None,
        augment: bool = True,
    ):
        self.data_dir = data_dir
        self.class_names = class_names
        self.seq_len = seq_len
        self.img_size = img_size
        self.transform = transform
        self.augment = augment

        self.samples: list[tuple[str, int]] = []  # (path_to_clip, label_idx)
        self._build_sample_list()

    def _build_sample_list(self) -> None:
        for idx, cls_name in enumerate(self.class_names):
            cls_dir = os.path.join(self.data_dir, cls_name)
            if not os.path.exists(cls_dir):
                logger.warning("Class directory missing: %s (will be skipped)", cls_dir)
                continue

            videos_found = 0
            for root, dirs, files in os.walk(cls_dir):
                for fn in sorted(files):
                    ext = Path(fn).suffix.lower()
                    if ext in self.VALID_VIDEO_EXT:
                        self.samples.append((os.path.join(root, fn), idx))
                        videos_found += 1

            # Also support image-sequence subdirs (each subdir = one clip)
            for subdir in sorted(os.listdir(cls_dir)):
                full_sub = os.path.join(cls_dir, subdir)
                if not os.path.isdir(full_sub):
                    continue
                imgs = [
                    f for f in sorted(os.listdir(full_sub))
                    if Path(f).suffix.lower() in self.VALID_IMAGE_EXT
                ]
                if len(imgs) >= self.seq_len:
                    self.samples.append((full_sub, idx))

            logger.info("Class '%s': %d clips found", cls_name, videos_found)

        if not self.samples:
            logger.error(
                "No samples found in %s. "
                "Run dataset_downloader.py --task activity first.",
                self.data_dir,
            )

    def _load_video_frames(self, video_path: str) -> list[np.ndarray]:
        cap = cv2.VideoCapture(video_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            cap.release()
            return []

        # Pick seq_len evenly spaced frame indices
        indices = np.linspace(0, total - 1, self.seq_len, dtype=int)
        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, (self.img_size, self.img_size))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        cap.release()
        return frames

    def _load_image_sequence(self, seq_dir: str) -> list[np.ndarray]:
        imgs = sorted([
            f for f in os.listdir(seq_dir)
            if Path(f).suffix.lower() in self.VALID_IMAGE_EXT
        ])
        # Sample seq_len frames evenly
        indices = np.linspace(0, len(imgs) - 1, self.seq_len, dtype=int)
        frames = []
        for idx in indices:
            img_path = os.path.join(seq_dir, imgs[idx])
            img = cv2.imread(img_path)
            if img is None:
                continue
            img = cv2.resize(img, (self.img_size, self.img_size))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            frames.append(img)
        return frames

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        clip_path, label = self.samples[index]

        if os.path.isdir(clip_path):
            frames = self._load_image_sequence(clip_path)
        else:
            frames = self._load_video_frames(clip_path)

        # Pad or truncate to seq_len
        if len(frames) < self.seq_len:
            while len(frames) < self.seq_len:
                frames.append(frames[-1] if frames else np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8))
        frames = frames[: self.seq_len]

        # Convert to tensors and apply transforms
        tensor_frames = []
        for frame in frames:
            if self.transform:
                t = self.transform(frame)
            else:
                t = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
            tensor_frames.append(t)

        # Shape: (seq_len, C, H, W)
        clip_tensor = torch.stack(tensor_frames, dim=0)
        return clip_tensor, label


# ── Model ─────────────────────────────────────────────────────────────────────


class CNNLSTMClassifier(nn.Module):
    """
    Per-frame CNN feature extractor (EfficientNet-B0 pretrained on ImageNet)
    + Bidirectional LSTM sequence head → activity class probabilities.

    Why EfficientNet-B0 over MobileNetV3-Small:
        - Feature dim: 1280 vs 576  → richer representation for LSTM
        - Parameters: 5.3M vs 2.5M → stronger generalisation
        - ImageNet top-1: 77.1% vs 67.7% → better pretrained features
        - Still fits comfortably in 4GB VRAM at batch=8, seq=16

    Args:
        num_classes:     Number of activity classes.
        lstm_hidden:     LSTM hidden state size.
        lstm_layers:     Number of LSTM layers.
        dropout:         Dropout rate.
        freeze_backbone: Freeze CNN during warmup epochs.
    """

    def __init__(
        self,
        num_classes: int = 4,
        lstm_hidden: int = 256,
        lstm_layers: int = 2,
        dropout: float = 0.5,
        freeze_backbone: bool = True,
    ):
        super().__init__()

        # ── CNN backbone (MobileNetV3-Small) ──────────────────────────────────
        backbone = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
        self.cnn = nn.Sequential(*list(backbone.children())[:-1])  # → (B, 576, 1, 1)
        self.pool = nn.Identity()   # already pooled by backbone
        self.feature_dim = 576
        logger.info("Backbone: MobileNetV3-Small (feature_dim=576)")

        if freeze_backbone:
            for param in self.cnn.parameters():
                param.requires_grad = False

        # ── BiLSTM sequence head ───────────────────────────────────────────────
        self.lstm = nn.LSTM(
            input_size=self.feature_dim,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden * 2, 256),  # *2 because bidirectional
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def unfreeze_backbone(self) -> None:
        for param in self.cnn.parameters():
            param.requires_grad = True
        logger.info("EfficientNet-B0 backbone unfrozen for fine-tuning.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, C, H, W) — batch of frame sequences.
        Returns:
            Logits (B, num_classes).
        """
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)
        features = self.cnn(x)               # (B*T, 576, 1, 1)
        features = features.view(B, T, -1)   # (B, T, 576)

        lstm_out, _ = self.lstm(features)    # (B, T, hidden*2)
        out = lstm_out[:, -1, :]             # last timestep
        out = self.dropout(out)
        return self.classifier(out)          # (B, num_classes)


# ── Training helpers ──────────────────────────────────────────────────────────


def get_transforms(img_size: int, augment: bool) -> T.Compose:
    if augment:
        return T.Compose([
            T.ToPILImage(),
            T.Resize((img_size, img_size)),
            T.RandomHorizontalFlip(p=0.5),
            T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
            T.RandomRotation(degrees=10),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    else:
        return T.Compose([
            T.ToPILImage(),
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    scaler: Optional[torch.cuda.amp.GradScaler],
) -> tuple[float, float]:
    """Returns (avg_loss, accuracy)."""
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for clips, labels in loader:
        clips = clips.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        if scaler:
            with torch.cuda.amp.autocast():
                logits = model(clips)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(clips)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

        total_loss += loss.item() * clips.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += clips.size(0)

    avg_loss = total_loss / max(total, 1)
    acc = correct / max(total, 1)
    return avg_loss, acc


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    for clips, labels in loader:
        clips = clips.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(clips)
        loss = criterion(logits, labels)
        total_loss += loss.item() * clips.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += clips.size(0)

    return total_loss / max(total, 1), correct / max(total, 1)


def plot_history(
    history: dict[str, list],
    save_dir: str,
) -> None:
    """Save loss and accuracy curves."""
    os.makedirs(save_dir, exist_ok=True)
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Loss
    ax1.plot(epochs, history["train_loss"], label="Train Loss", color="cornflowerblue")
    ax1.plot(epochs, history["val_loss"], label="Val Loss", color="tomato")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Activity Model — Training Loss")
    ax1.legend()
    ax1.grid(alpha=0.3)

    # Accuracy
    ax2.plot(epochs, [a * 100 for a in history["train_acc"]], label="Train Acc", color="cornflowerblue")
    ax2.plot(epochs, [a * 100 for a in history["val_acc"]], label="Val Acc", color="tomato")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.set_title("Activity Model — Accuracy")
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(save_dir, "training_curves.png")
    plt.savefig(save_path, dpi=120)
    plt.close()
    logger.info("Training curves saved → %s", save_path)


# ── Main training ─────────────────────────────────────────────────────────────


def train(args: argparse.Namespace) -> None:
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    train_cfg = cfg["training"]["activity"]

    if args.epochs:
        train_cfg["epochs"] = args.epochs
    if args.batch:
        train_cfg["batch_size"] = args.batch
    if args.data:
        train_cfg["data_dir"] = args.data

    data_dir = train_cfg["data_dir"]
    seq_len  = train_cfg["sequence_length"]
    img_size = train_cfg["img_size"]
    epochs   = train_cfg["epochs"]
    batch    = train_cfg["batch_size"]
    save_dir = train_cfg["save_dir"]
    patience = train_cfg["patience"]

    device_str = cfg.get("device", "auto")
    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)
    logger.info("Using device: %s", device)

    # ── Datasets ──────────────────────────────────────────────────────────────
    train_ds = ActivityVideoDataset(
        data_dir=data_dir,
        class_names=CLASS_NAMES,
        seq_len=seq_len,
        img_size=img_size,
        transform=get_transforms(img_size, augment=True),
    )
    val_ds = ActivityVideoDataset(
        data_dir=data_dir,
        class_names=CLASS_NAMES,
        seq_len=seq_len,
        img_size=img_size,
        transform=get_transforms(img_size, augment=False),
    )

    if len(train_ds) == 0:
        logger.error("No training samples found. Check datasets/activity/ folder.")
        raise SystemExit(1)

    # 80/20 split
    n_total = len(train_ds)
    n_val = max(1, int(n_total * 0.2))
    n_train = n_total - n_val
    train_subset, val_subset = torch.utils.data.random_split(train_ds, [n_train, n_val])

    train_loader = DataLoader(
        train_subset, batch_size=batch, shuffle=True,
        num_workers=min(train_cfg["workers"], 4), pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_subset, batch_size=batch, shuffle=False,
        num_workers=min(train_cfg["workers"], 4), pin_memory=device.type == "cuda",
    )
    logger.info("Dataset: %d train | %d val samples", len(train_subset), len(val_subset))

    # ── Model ─────────────────────────────────────────────────────────────────
    model = CNNLSTMClassifier(
        num_classes=NUM_CLASSES,
        lstm_hidden=256,
        lstm_layers=2,
        dropout=0.5,
        freeze_backbone=True,   # Freeze initially, unfreeze after warmup
    ).to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=train_cfg["lr"],
        weight_decay=train_cfg["weight_decay"],
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    scaler = torch.cuda.amp.GradScaler() if device.type == "cuda" else None

    # ── Training loop ─────────────────────────────────────────────────────────
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs("models", exist_ok=True)

    best_val_acc = 0.0
    patience_counter = 0
    warmup_done = False
    warmup_epoch = 5  # Unfreeze backbone after 5 epochs
    dest_pt = "models/activity_classifier.pt"

    history: dict[str, list] = {
        "train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []
    }

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        # Unfreeze backbone after warmup epochs for fine-tuning
        if epoch == warmup_epoch and not warmup_done:
            model.unfreeze_backbone()
            # Reinit optimizer with all params now unfrozen
            optimizer = optim.AdamW(
                model.parameters(),
                lr=train_cfg["lr"] * 0.1,   # Lower LR for fine-tuning
                weight_decay=train_cfg["weight_decay"],
            )
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=epochs - warmup_epoch, eta_min=1e-7
            )
            warmup_done = True

        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device, scaler)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        elapsed = time.time() - t0
        logger.info(
            "Epoch %3d/%d | train_loss=%.4f acc=%.2f%% | val_loss=%.4f acc=%.2f%% | %.1fs",
            epoch, epochs,
            train_loss, train_acc * 100,
            val_loss, val_acc * 100,
            elapsed,
        )

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "class_names": CLASS_NAMES,
                "seq_len": seq_len,
                "img_size": img_size,
                "val_acc": val_acc,
            }, dest_pt)
            logger.info("  ↑ Best model saved (val_acc=%.2f%%)", val_acc * 100)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info("Early stopping at epoch %d (no improvement in %d epochs)", epoch, patience)
                break

    # Final plots
    plot_history(history, save_dir)

    print("\n" + "=" * 60)
    print(f"✓  Activity model training complete!")
    print(f"   Best val accuracy: {best_val_acc:.2%}")
    print(f"   Weights saved: {dest_pt}")
    print(f"   Plots saved:   {save_dir}/training_curves.png")
    print("=" * 60 + "\n")


# ── CLI ───────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train CNN+LSTM activity classifier (normal/running/violence/suspicious)."
    )
    parser.add_argument("--config", default="configs/detection_config.yaml")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--data", default=None, help="Override data_dir")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
