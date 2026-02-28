# dataset_downloader.py
"""
Automatic dataset downloader for:
  - Fire / Smoke detection (Roboflow public dataset)
  - Weapon detection — Gun + Knife (Roboflow public dataset)
  - Human Activity recognition — Violence / Abnormal (direct download)

Usage:
    python dataset_downloader.py               # download all datasets
    python dataset_downloader.py --task fire   # download only fire dataset
    python dataset_downloader.py --task weapon
    python dataset_downloader.py --task activity

Datasets are saved in:
    datasets/fire/
    datasets/weapon/
    datasets/activity/
"""

from __future__ import annotations

import argparse
import os
import sys
import zipfile
import shutil
import hashlib
import time
from pathlib import Path

import requests
from tqdm import tqdm

from utils.logger import get_logger

logger = get_logger("dataset_downloader")

# ── Dataset registry ─────────────────────────────────────────────────────────
# Each entry: dict with "url", "dest", "format", optional "roboflow" dict

DATASETS = {
    "fire": {
        "name": "Fire and Smoke Detection (Roboflow Universe)",
        "dest": "datasets/fire",
        "fallback_instructions": (
            "Manual download: Go to "
            "https://universe.roboflow.com/fire-cctv/fire-smoke-detector-yolov8 "
            "→ Download → YOLOv8 format → extract to datasets/fire/"
        ),
        # Multiple Roboflow project slugs to try in order (public, well-known datasets)
        "roboflow_candidates": [
            {"workspace": "fire-rqbio", "project": "fire-and-smoke-yikzn", "version": 1},
            {"workspace": "fire-rqbio", "project": "fire-and-smoke-yikzn", "version": 2},
            {"workspace": "fire-rqbio", "project": "fire-and-smoke-yikzn", "version": 3},
        ],
        "data_yaml_path": "datasets/fire/data.yaml",
    },
    "weapon": {
        "name": "Weapon Detection — Gun + Knife (Roboflow Universe)",
        "dest": "datasets/weapon",
        "fallback_instructions": (
            "Manual download: Go to "
            "https://universe.roboflow.com/weapons-detection/weapon-detection-mkfzp "
            "→ Download → YOLOv8 format → extract to datasets/weapon/"
        ),
        "roboflow_candidates": [
            {"workspace": "www-oedzr", "project": "weapons-8fe0z", "version": 1},
            {"workspace": "www-oedzr", "project": "weapons-8fe0z", "version": 2},
            {"workspace": "www-oedzr", "project": "weapons-8fe0z", "version": 3},
            {"workspace": "weapons-detection", "project": "weapon-detection-gun-knife", "version": 1},
        ],
        "data_yaml_path": "datasets/weapon/data.yaml",
    },
    "activity": {
        "name": "Real World Violence / Abnormal Activity",
        "dest": "datasets/activity",
        # Real World Violence Situations Dataset — via gdown (Google Drive, reliable)
        # Source: https://github.com/mohamedmerzougui/Violence-Detection
        "gdown_sources": [
            {
                "label": "violence",
                # Real World Violence Situations Dataset (hosted on Google Drive)
                "gdrive_id": "1NRf_b9D8cWkJBHx-BNf8lRCatNFniPTu",
                "filename": "violence_videos.zip",
            },
        ],
        "data_yaml_path": "datasets/activity/data_info.yaml",
    },
}

# ── Helpers ───────────────────────────────────────────────────────────────────


def _download_file(url: str, dest_path: str, desc: str = "Downloading") -> bool:
    """Stream-download a file with a tqdm progress bar. Returns True on success."""
    try:
        resp = requests.get(url, stream=True, timeout=60)
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0))
        with open(dest_path, "wb") as f, tqdm(
            desc=desc,
            total=total,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            ncols=80,
        ) as bar:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
                bar.update(len(chunk))
        return True
    except Exception as e:
        logger.error("Download failed for %s: %s", url, e)
        return False


def _extract_zip(zip_path: str, extract_to: str) -> None:
    logger.info("Extracting %s → %s", zip_path, extract_to)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_to)


def _try_roboflow_download(ds_key: str, dest: str) -> bool:
    """
    Try to download via the Roboflow Python SDK, trying multiple project candidates.
    Requires: pip install roboflow  + ROBOFLOW_API_KEY env var
    """
    api_key = os.environ.get("ROBOFLOW_API_KEY", "")
    if not api_key:
        logger.warning(
            "ROBOFLOW_API_KEY not set. Skipping Roboflow API download for '%s'.", ds_key
        )
        return False
    try:
        from roboflow import Roboflow  # type: ignore
        rf = Roboflow(api_key=api_key)
    except Exception as e:
        logger.warning("Roboflow init failed: %s", e)
        return False

    candidates = DATASETS[ds_key].get("roboflow_candidates", [])
    if not candidates:
        # Legacy single-entry support
        info = DATASETS[ds_key].get("roboflow", {})
        if info:
            candidates = [info]

    for info in candidates:
        try:
            logger.info(
                "Trying Roboflow project: %s / %s v%s",
                info["workspace"], info["project"], info["version"]
            )
            proj = rf.workspace(info["workspace"]).project(info["project"])
            dataset = proj.version(info["version"]).download("yolov8", location=dest)
            logger.info("Roboflow download complete → %s", dataset.location)
            return True
        except Exception as e:
            logger.warning("  ↳ Failed (%s / %s): %s", info["workspace"], info["project"], e)
            continue

    logger.warning("All Roboflow candidates failed for '%s'.", ds_key)
    return False


def _write_data_yaml(dest: str, nc: int, names: list[str]) -> str:
    """Write a minimal YOLO data.yaml if one doesn't already exist."""
    yaml_path = os.path.join(dest, "data.yaml")
    if os.path.exists(yaml_path):
        return yaml_path
    content = (
        f"path: {os.path.abspath(dest)}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"nc: {nc}\n"
        f"names: {names}\n"
    )
    with open(yaml_path, "w") as f:
        f.write(content)
    logger.info("Wrote data.yaml → %s", yaml_path)
    return yaml_path


def _make_splits(source_dir: str, dest_dir: str, val_ratio: float = 0.2) -> None:
    """
    If images are in a flat directory (no train/val split), create the split.
    Copies images + labels into:
        dest_dir/images/train/
        dest_dir/images/val/
        dest_dir/labels/train/
        dest_dir/labels/val/
    """
    import random

    images_dir = os.path.join(source_dir, "images")
    labels_dir = os.path.join(source_dir, "labels")

    if not os.path.exists(images_dir):
        logger.warning("No images/ folder found in %s. Skipping split.", source_dir)
        return

    all_images = [
        f for f in os.listdir(images_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
    ]
    if not all_images:
        return

    random.shuffle(all_images)
    n_val = max(1, int(len(all_images) * val_ratio))
    val_set = set(all_images[:n_val])
    train_set = set(all_images[n_val:])

    for split_name, img_set in [("train", train_set), ("val", val_set)]:
        os.makedirs(os.path.join(dest_dir, "images", split_name), exist_ok=True)
        os.makedirs(os.path.join(dest_dir, "labels", split_name), exist_ok=True)
        for img_file in img_set:
            src_img = os.path.join(images_dir, img_file)
            dst_img = os.path.join(dest_dir, "images", split_name, img_file)
            shutil.copy2(src_img, dst_img)

            # Copy corresponding label if exists
            label_file = Path(img_file).stem + ".txt"
            src_lbl = os.path.join(labels_dir, label_file)
            if os.path.exists(src_lbl):
                dst_lbl = os.path.join(dest_dir, "labels", split_name, label_file)
                shutil.copy2(src_lbl, dst_lbl)

    logger.info(
        "Split complete: %d train / %d val images", len(train_set), len(val_set)
    )


# ── Dataset-specific downloaders ──────────────────────────────────────────────


def download_fire_dataset() -> bool:
    """Download fire/smoke dataset. Returns True on success."""
    ds = DATASETS["fire"]
    dest = ds["dest"]
    os.makedirs(dest, exist_ok=True)
    logger.info("=== Downloading Fire/Smoke Dataset ===")

    # 1. Try Roboflow SDK (tries multiple candidate projects)
    if _try_roboflow_download("fire", dest):
        # Write data.yaml if the downloaded dataset doesn't have one
        _write_data_yaml(dest, nc=2, names=["fire", "smoke"])
        logger.info("✓ Fire dataset ready at: %s", dest)
        return True

    # 2. All Roboflow candidates failed — show manual instructions
    logger.error(
        "Automatic fire dataset download failed.\n"
        "Please download manually:\n%s",
        ds["fallback_instructions"],
    )
    return False


def download_weapon_dataset() -> bool:
    """Download gun + knife weapon dataset. Returns True on success."""
    ds = DATASETS["weapon"]
    dest = ds["dest"]
    os.makedirs(dest, exist_ok=True)
    logger.info("=== Downloading Weapon Dataset (Gun + Knife) ===")

    # 1. Try Roboflow SDK
    if _try_roboflow_download("weapon", dest):
        logger.info("✓ Weapon dataset ready at: %s", dest)
        return True

    # 2. Direct URL fallback
    zip_path = os.path.join(dest, "weapon_dataset.zip")
    if not os.path.exists(zip_path):
        success = _download_file(ds["url"], zip_path, "Weapon dataset")
        if not success:
            logger.error(
                "Cannot download weapon dataset automatically.\n%s",
                ds["fallback_instructions"],
            )
            return False

    _extract_zip(zip_path, dest)
    os.remove(zip_path)
    _write_data_yaml(dest, nc=2, names=["gun", "knife"])

    logger.info("✓ Weapon dataset ready at: %s", dest)
    return True


def download_activity_dataset() -> bool:
    """
    Download violence / activity datasets via direct public HTTP URLs (no auth needed).

    Sources (all free, no login):
      - Hockey Fight Dataset: public violence video dataset (500 fight + 500 no-fight)
      - Surrogate normal clips via sample CCTV frames from public repos

    Organizes into:
        datasets/activity/violence/   ← violence video clips
        datasets/activity/normal/     ← normal (non-violent) video clips
    """
    dest = DATASETS["activity"]["dest"]
    os.makedirs(dest, exist_ok=True)
    logger.info("=== Downloading Activity / Violence Dataset ===")

    violence_dir = os.path.join(dest, "violence")
    normal_dir   = os.path.join(dest, "normal")
    os.makedirs(violence_dir, exist_ok=True)
    os.makedirs(normal_dir,   exist_ok=True)

    # ── Direct download sources (no API key required) ─────────────────────────
    # Hockey Fight Dataset — 500 fight clips + 500 non-fight clips
    # Hosted on Zenodo (open access research repository)
    DIRECT_SOURCES = [
        {
            "label": "violence",
            "url": "https://zenodo.org/record/7368820/files/HockeyFight_violence.zip?download=1",
            "desc": "Hockey Fight Dataset - Fight clips",
        },
        {
            "label": "normal",
            "url": "https://zenodo.org/record/7368820/files/HockeyFight_normal.zip?download=1",
            "desc": "Hockey Fight Dataset - Normal clips",
        },
    ]

    any_downloaded = False
    for src in DIRECT_SOURCES:
        label     = src["label"]
        label_dir = violence_dir if label == "violence" else normal_dir
        zip_path  = os.path.join(dest, f"{label}_direct.zip")

        if os.path.exists(label_dir) and len(os.listdir(label_dir)) > 0:
            logger.info("✓ %s clips already exist, skipping download.", label)
            any_downloaded = True
            continue

        logger.info("Downloading: %s", src["desc"])
        ok = _download_file(src["url"], zip_path, src["desc"])
        if not ok:
            logger.warning("  ↳ Direct download failed for %s", label)
            continue

        if zipfile.is_zipfile(zip_path):
            tmp = os.path.join(dest, f"_tmp_{label}")
            _extract_zip(zip_path, tmp)
            os.remove(zip_path)
            for root_dir, _, files in os.walk(tmp):
                for fn in files:
                    if Path(fn).suffix.lower() in {".mp4", ".avi", ".mov", ".mkv"}:
                        shutil.move(os.path.join(root_dir, fn), os.path.join(label_dir, fn))
            shutil.rmtree(tmp, ignore_errors=True)
            count = len(os.listdir(label_dir))
            logger.info("  ✓ Extracted %d clips → %s", count, label_dir)
            any_downloaded = True
        else:
            logger.warning("  ↳ Downloaded file is not a valid zip: %s", zip_path)
            if os.path.exists(zip_path):
                os.remove(zip_path)

    if not any_downloaded:
        logger.error(
            "All automatic downloads failed.\n"
            "Please manually download a violence dataset and place clips in:\n"
            "  Violence clips → %s\n"
            "  Normal clips   → %s\n\n"
            "Recommended free dataset (requires free Kaggle account):\n"
            "  https://www.kaggle.com/datasets/mohamedmerzougui/real-world-violence-situations\n"
            "After downloading, extract and copy .mp4 files to the folders above.",
            violence_dir, normal_dir,
        )
        _write_activity_yaml(dest)
        return False

    v_count = len([f for f in os.listdir(violence_dir) if os.path.isfile(os.path.join(violence_dir, f))])
    n_count = len([f for f in os.listdir(normal_dir)   if os.path.isfile(os.path.join(normal_dir, f))])
    logger.info("✓ Activity dataset ready: %d violence, %d normal clips", v_count, n_count)

    _write_activity_yaml(dest)
    return True




def _print_kaggle_setup_instructions() -> None:
    print("""
╔══════════════════════════════════════════════════════════════╗
║              KAGGLE API SETUP (one-time)                     ║
╠══════════════════════════════════════════════════════════════╣
║  1. Go to https://www.kaggle.com → Account → API             ║
║  2. Click "Create New API Token" → downloads kaggle.json     ║
║  3. Move kaggle.json to:  C:\\Users\\<you>\\.kaggle\\kaggle.json ║
║  4. Run: pip install kaggle                                   ║
║  5. Re-run: python dataset_downloader.py --task activity      ║
╚══════════════════════════════════════════════════════════════╝

  Dataset: https://www.kaggle.com/datasets/rutviknakum/real-life-violence-situations-dataset
  After downloading manually, extract and place videos in:
    datasets/activity/violence/   ← violence clips
    datasets/activity/normal/     ← non-violence clips
""")


def _write_activity_yaml(dest: str) -> None:
    info_yaml = os.path.join(dest, "data_info.yaml")
    with open(info_yaml, "w") as f:
        f.write(
            "# Activity dataset info\n"
            "classes:\n"
            "  0: normal\n"
            "  1: running_panic\n"
            "  2: violence\n"
            "  3: suspicious\n"
            f"root: {os.path.abspath(dest)}\n"
        )



# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download datasets for real-time activity detection system."
    )
    parser.add_argument(
        "--task",
        choices=["fire", "weapon", "activity", "all"],
        default="all",
        help="Which dataset to download (default: all)",
    )
    parser.add_argument(
        "--roboflow-key",
        default="",
        help="Roboflow API key (or set ROBOFLOW_API_KEY env var)",
    )
    args = parser.parse_args()

    if args.roboflow_key:
        os.environ["ROBOFLOW_API_KEY"] = args.roboflow_key

    results: dict[str, bool] = {}

    if args.task in ("fire", "all"):
        results["fire"] = download_fire_dataset()

    if args.task in ("weapon", "all"):
        results["weapon"] = download_weapon_dataset()

    if args.task in ("activity", "all"):
        results["activity"] = download_activity_dataset()

    print("\n" + "=" * 50)
    print("Dataset Download Summary:")
    for name, ok in results.items():
        status = "✓ OK" if ok else "✗ FAILED (see logs)"
        print(f"  {name:12s}: {status}")
    print("=" * 50)

    if not all(results.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
