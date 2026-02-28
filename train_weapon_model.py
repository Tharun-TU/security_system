# train_weapon_model.py
"""
Train a YOLOv8m model for weapon detection: gun + knife.

Usage:
    python train_weapon_model.py
    python train_weapon_model.py --epochs 100 --batch 8

Best model saved to:  models/weapon_detector.pt
Results + plots:      results/weapon_training/
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

logger = get_logger("train_weapon")


# ── Helpers ───────────────────────────────────────────────────────────────────


def load_config(config_path: str = "configs/detection_config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def verify_dataset(data_yaml: str) -> bool:
    if not os.path.exists(data_yaml):
        logger.error(
            "Dataset YAML not found: %s\n"
            "Run: python dataset_downloader.py --task weapon",
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


def compute_class_weights(data_yaml: str) -> list[float] | None:
    """
    Compute per-class weights inversely proportional to class frequency,
    used to address class imbalance in the weapon dataset.

    Returns:
        List of weights per class, or None if count fails.
    """
    import glob

    try:
        with open(data_yaml, "r") as f:
            ds = yaml.safe_load(f)
        base = ds.get("path", os.path.dirname(data_yaml))
        train_labels = os.path.join(base, ds.get("train", "images/train").replace("images", "labels"))
        counts: dict[int, int] = {}
        for lbl_file in glob.glob(os.path.join(train_labels, "*.txt")):
            with open(lbl_file) as f:
                for line in f:
                    cls = int(line.strip().split()[0])
                    counts[cls] = counts.get(cls, 0) + 1
        if not counts:
            return None
        total = sum(counts.values())
        nc = max(counts.keys()) + 1
        weights = [total / (nc * counts.get(i, 1)) for i in range(nc)]
        logger.info("Class weights: %s", {i: round(w, 3) for i, w in enumerate(weights)})
        return weights
    except Exception as e:
        logger.warning("Could not compute class weights: %s", e)
        return None


def export_onnx(model_path: str, img_size: int = 640) -> None:
    logger.info("Exporting weapon model to ONNX...")
    model = YOLO(model_path)
    export_path = model.export(format="onnx", imgsz=img_size, simplify=True, opset=17)
    logger.info("ONNX model saved to: %s", export_path)


# ── Main training function ────────────────────────────────────────────────────


def train(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    train_cfg = cfg["training"]["weapon"]

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

    # Class weighting to handle gun/knife imbalance
    class_weights = compute_class_weights(data_yaml)

    logger.info("Loading base model: %s", train_cfg["base_model"])
    model = YOLO(train_cfg["base_model"])

    logger.info(
        "Starting weapon training | epochs=%d | batch=%d | img=%d",
        train_cfg["epochs"],
        train_cfg["batch_size"],
        train_cfg["img_size"],
    )

    model.train(
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
        name="weapon_run",
        exist_ok=True,
        # --- Augmentation (more aggressive for small-object weapons) ---
        hsv_h=0.02,
        hsv_s=0.7,
        hsv_v=0.4,
        fliplr=0.5,
        flipud=0.0,
        mosaic=1.0,
        mixup=0.1,
        copy_paste=0.1,           # Copy-paste aug great for weapons on backgrounds
        scale=0.6,
        translate=0.1,
        shear=2.0,
        perspective=0.0005,
        # --- Training specifics ---
        weight_decay=0.0005,
        warmup_epochs=3,
        cos_lr=True,
        # NMS threshold
        iou=train_cfg.get("iou_threshold", 0.7),
        # Confidence threshold
        conf=cfg["models"]["weapon"]["conf_threshold"],
        # --- Output ---
        save=True,
        save_period=10,
        plots=True,
        verbose=True,
    )

    # Save best weights
    os.makedirs("models", exist_ok=True)
    best_pt = Path(train_cfg["save_dir"]) / "weapon_run" / "weights" / "best.pt"
    dest_pt = "models/weapon_detector.pt"
    if best_pt.exists():
        shutil.copy2(str(best_pt), dest_pt)
        logger.info("✓ Best weapon model saved → %s", dest_pt)
    else:
        logger.warning("best.pt not found at %s", best_pt)

    # Validation
    logger.info("Running validation on best model...")
    val_model = YOLO(dest_pt) if os.path.exists(dest_pt) else model
    metrics = val_model.val(
        data=data_yaml,
        imgsz=train_cfg["img_size"],
        device=device,
        verbose=True,
        conf=0.001,    # Low conf for val to get full P-R curve
        iou=0.6,
    )

    logger.info(
        "Validation | mAP50=%.4f | mAP50-95=%.4f",
        metrics.box.map50,
        metrics.box.map,
    )

    if cfg.get("onnx", {}).get("enabled", False) and os.path.exists(dest_pt):
        export_onnx(dest_pt, img_size=train_cfg["img_size"])

    print("\n" + "=" * 60)
    print(f"✓  Weapon model training complete!")
    print(f"   Best weights: {dest_pt}")
    print(f"   mAP@0.5:      {metrics.box.map50:.4f}")
    print(f"   mAP@0.5:0.95: {metrics.box.map:.4f}")
    print(f"   Plots: {train_cfg['save_dir']}/weapon_run/")
    print("=" * 60 + "\n")


# ── CLI ───────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="Train YOLOv8 weapon detector (gun + knife).")
    parser.add_argument("--config", default="configs/detection_config.yaml")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--data", default=None)
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
