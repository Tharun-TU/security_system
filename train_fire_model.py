# train_fire_model.py
"""
Train a YOLOv8m model for fire and smoke detection.

Usage:
    python train_fire_model.py                     # use configs/detection_config.yaml
    python train_fire_model.py --epochs 80 --batch 8

After training, the best model is saved to:
    models/fire_detector.pt

Results (loss curves + mAP graphs) are saved to:
    results/fire_training/
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

import yaml
import torch
from ultralytics import YOLO

from utils.logger import get_logger

logger = get_logger("train_fire")


# ── Helpers ───────────────────────────────────────────────────────────────────


def load_config(config_path: str = "configs/detection_config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def verify_dataset(data_yaml: str) -> bool:
    """Check that the dataset YAML and image folders exist."""
    if not os.path.exists(data_yaml):
        logger.error(
            "Dataset YAML not found: %s\n"
            "Run: python dataset_downloader.py --task fire",
            data_yaml,
        )
        return False

    with open(data_yaml, "r") as f:
        ds = yaml.safe_load(f)

    base = ds.get("path", os.path.dirname(data_yaml))
    for split in ("train", "val"):
        split_path = ds.get(split, "")
        full = os.path.join(base, split_path)
        if not os.path.exists(full):
            logger.error("Missing split directory: %s", full)
            return False
    return True


def get_device(cfg: dict) -> str:
    device_cfg = cfg.get("device", "auto")
    if device_cfg == "auto":
        return "0" if torch.cuda.is_available() else "cpu"
    return device_cfg


def export_onnx(model_path: str, img_size: int = 640) -> None:
    """Optionally export best model to ONNX."""
    logger.info("Exporting model to ONNX...")
    model = YOLO(model_path)
    export_path = model.export(format="onnx", imgsz=img_size, simplify=True, opset=17)
    logger.info("ONNX model saved to: %s", export_path)


# ── Main training function ────────────────────────────────────────────────────


def train(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    train_cfg = cfg["training"]["fire"]

    # CLI overrides
    if args.epochs:
        train_cfg["epochs"] = args.epochs
    if args.batch:
        train_cfg["batch_size"] = args.batch
    if args.data:
        train_cfg["data_yaml"] = args.data

    data_yaml = train_cfg["data_yaml"]
    if not verify_dataset(data_yaml):
        raise SystemExit(1)

    device = get_device(cfg)
    logger.info("Using device: %s", "GPU" if device != "cpu" else "CPU")

    # Load base model (pretrained on COCO)
    logger.info("Loading base model: %s", train_cfg["base_model"])
    model = YOLO(train_cfg["base_model"])

    # ── Training ──────────────────────────────────────────────────────────────
    logger.info(
        "Starting fire/smoke training | epochs=%d | batch=%d | img=%d",
        train_cfg["epochs"],
        train_cfg["batch_size"],
        train_cfg["img_size"],
    )

    results = model.train(
        data=data_yaml,
        epochs=train_cfg["epochs"],
        batch=train_cfg["batch_size"],
        imgsz=train_cfg["img_size"],
        device=device,
        workers=train_cfg["workers"],
        optimizer=train_cfg["optimizer"],
        lr0=train_cfg["lr0"],
        lrf=train_cfg["lrf"],
        patience=train_cfg["patience"],
        project=train_cfg["save_dir"],
        name="fire_run",
        exist_ok=True,
        # --- Augmentation ---
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        flipud=0.1,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.2,
        scale=0.5,
        translate=0.1,
        # --- Regularization ---
        weight_decay=0.0005,
        warmup_epochs=3,
        cos_lr=True,
        # --- Output ---
        save=True,
        save_period=10,
        plots=True,
        verbose=True,
    )

    # ── Save best weights ──────────────────────────────────────────────────────
    os.makedirs("models", exist_ok=True)
    best_pt = Path(train_cfg["save_dir"]) / "fire_run" / "weights" / "best.pt"
    dest_pt = "models/fire_detector.pt"
    if best_pt.exists():
        shutil.copy2(str(best_pt), dest_pt)
        logger.info("✓ Best fire model saved → %s", dest_pt)
    else:
        logger.warning("best.pt not found at %s — check training logs", best_pt)

    # ── Validation metrics ─────────────────────────────────────────────────────
    logger.info("Running validation on best model...")
    val_model = YOLO(dest_pt) if os.path.exists(dest_pt) else model
    metrics = val_model.val(data=data_yaml, imgsz=train_cfg["img_size"], device=device, verbose=True)
    logger.info(
        "Validation results | mAP50=%.4f | mAP50-95=%.4f",
        metrics.box.map50,
        metrics.box.map,
    )

    # ── Optional ONNX export ───────────────────────────────────────────────────
    if cfg.get("onnx", {}).get("enabled", False) and os.path.exists(dest_pt):
        export_onnx(dest_pt, img_size=train_cfg["img_size"])

    print("\n" + "=" * 60)
    print(f"✓  Fire model training complete!")
    print(f"   Best weights: {dest_pt}")
    print(f"   mAP@0.5:      {metrics.box.map50:.4f}")
    print(f"   mAP@0.5:0.95: {metrics.box.map:.4f}")
    print(f"   Training plots: {train_cfg['save_dir']}/fire_run/")
    print("=" * 60 + "\n")


# ── CLI ───────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="Train YOLOv8 fire/smoke detector.")
    parser.add_argument("--config", default="configs/detection_config.yaml")
    parser.add_argument("--epochs", type=int, default=None, help="Override epoch count")
    parser.add_argument("--batch", type=int, default=None, help="Override batch size")
    parser.add_argument("--data", default=None, help="Override path to data.yaml")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
