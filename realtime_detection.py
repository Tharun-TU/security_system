# realtime_detection.py
"""
Real-Time Unusual Activity Detection System
===========================================
Integrates:
  • YOLOv8 person detector + ByteTrack tracking
  • YOLOv8 fire/smoke detector (fine-tuned)
  • YOLOv8 weapon detector — gun + knife (fine-tuned)
  • CNN+LSTM activity classifier — running/violence/suspicious

Run:
    python realtime_detection.py                        # webcam (source=0)
    python realtime_detection.py --source 0             # explicit webcam
    python realtime_detection.py --source "rtsp://..."  # RTSP stream
    python realtime_detection.py --source video.mp4     # video file
    python realtime_detection.py --dry-run              # no models, test camera only

Press Q to quit.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import threading
import queue
from collections import deque
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
import yaml
from ultralytics import YOLO

from utils.logger import get_logger
from utils.alert import ThreatAlert
from utils.tracker import TrackManager
from utils.visualizer import Visualizer, COLORS
from utils.iou import is_overlapping, compute_containment, expand_box

logger = get_logger("realtime_detection")


# =============================================================================
# Activity Classifier Inference Wrapper
# =============================================================================

class ActivityClassifier:
    """
    Loads the CNN+LSTM model checkpoint and runs inference on frame sequences.
    Input:  List of (img_size × img_size) BGR numpy frames (len = seq_len).
    Output: (predicted_class_name, confidence)
    """

    def __init__(self, weights_path: str, device: torch.device, cfg: dict):
        import torchvision.transforms as T

        self.device = device
        self.cfg = cfg
        self.seq_len = cfg["models"]["activity"]["sequence_length"]
        self.img_size = cfg["models"]["activity"]["img_size"]
        self.conf_threshold = cfg["models"]["activity"]["conf_threshold"]
        self.class_names = cfg["models"]["activity"]["classes"]
        self.loaded = False

        if not os.path.exists(weights_path):
            logger.warning(
                "Activity model weights not found: %s\n"
                "Train first: python train_activity_model.py",
                weights_path,
            )
            return

        from train_activity_model import CNNLSTMClassifier

        ckpt = torch.load(weights_path, map_location=device)
        self.class_names = ckpt.get("class_names", self.class_names)
        self.seq_len = ckpt.get("seq_len", self.seq_len)
        self.img_size = ckpt.get("img_size", self.img_size)

        self.model = CNNLSTMClassifier(
            num_classes=len(self.class_names),
            lstm_hidden=256,
            lstm_layers=2,
            dropout=0.0,               # No dropout at inference
            freeze_backbone=False,
        ).to(device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()

        self.transform = T.Compose([
            T.ToPILImage(),
            T.Resize((self.img_size, self.img_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.loaded = True
        logger.info("Activity classifier loaded: %s", weights_path)

    @torch.no_grad()
    def predict(self, frames: list[np.ndarray]) -> tuple[str, float]:
        if not self.loaded or len(frames) < 2:
            return "normal", 0.0

        # Resample to seq_len
        if len(frames) >= self.seq_len:
            idxs = np.linspace(0, len(frames) - 1, self.seq_len, dtype=int)
            sampled = [frames[i] for i in idxs]
        else:
            sampled = list(frames)
            while len(sampled) < self.seq_len:
                sampled.append(sampled[-1])

        tensor_frames = []
        for f in sampled:
            rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
            t = self.transform(rgb)
            tensor_frames.append(t)

        clip = torch.stack(tensor_frames).unsqueeze(0).to(self.device)  # (1, T, C, H, W)

        with torch.amp.autocast("cuda", enabled=self.device.type == "cuda"):
            logits = self.model(clip)

        probs = torch.softmax(logits, dim=1)[0]
        conf, pred_idx = probs.max(dim=0)
        cls_name = self.class_names[pred_idx.item()]
        return cls_name, conf.item()


# =============================================================================
# Threaded Frame Capture
# =============================================================================

class ThreadedCapture:
    """
    Non-blocking camera frame reader.
    Reads frames in a background thread to avoid inference blocking the camera.
    """

    def __init__(self, source, width: int, height: int, buffer_size: int = 2):
        self._source = source
        # On Windows, try DirectShow backend first to avoid MSMF errors
        if isinstance(source, int):
            self._cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)
            if not self._cap.isOpened():
                self._cap = cv2.VideoCapture(source)   # fallback to default
        else:
            self._cap = cv2.VideoCapture(source)

        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open video source: {source}")

        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, buffer_size)

        self._q: queue.Queue = queue.Queue(maxsize=buffer_size)
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._reader_loop, daemon=True)
        self._thread.start()

    def _reader_loop(self) -> None:
        while not self._stop_event.is_set():
            ret, frame = self._cap.read()
            if not ret:
                break
            if self._q.full():
                try:
                    self._q.get_nowait()   # Drop oldest frame
                except queue.Empty:
                    pass
            self._q.put(frame)

    def read(self) -> tuple[bool, Optional[np.ndarray]]:
        try:
            frame = self._q.get(timeout=2.0)
            return True, frame
        except queue.Empty:
            return False, None

    def release(self) -> None:
        self._stop_event.set()
        self._thread.join(timeout=2.0)
        self._cap.release()

    @property
    def fps(self) -> float:
        return self._cap.get(cv2.CAP_PROP_FPS) or 30.0


# =============================================================================
# Detection Pipeline
# =============================================================================

class DetectionPipeline:
    """Orchestrates all models for real-time inference on incoming frames."""

    def __init__(self, cfg: dict, device: torch.device, dry_run: bool = False):
        self.cfg = cfg
        self.device = device
        self.dry_run = dry_run

        if not dry_run:
            self._load_models()

        self.alert_engine = ThreatAlert(
            screenshot_dir=cfg["alerts"]["screenshot_dir"],
            log_dir=cfg["alerts"]["log_dir"],
            debounce_seconds=cfg["alerts"]["debounce_seconds"],
            consecutive_required=cfg["alerts"]["consecutive_frames_required"],
        )
        self.vis = Visualizer()
        self._track_buffers: dict[int, deque] = {}
        self._seq_len = cfg["models"]["activity"]["sequence_length"]

    def _load_models(self) -> None:
        """Load all YOLOv8 + CNN+LSTM models."""
        logger.info("Loading person detector (YOLOv8m)...")
        self.person_model = YOLO(self.cfg["models"]["person"]["weights"])

        logger.info("Loading fire/smoke detector...")
        fire_weights = self.cfg["models"]["fire"]["weights"]
        if not os.path.exists(fire_weights):
            logger.warning("Fire weights not found (%s). Run: python train_fire_model.py", fire_weights)
            self.fire_model = None
        else:
            self.fire_model = YOLO(fire_weights)

        logger.info("Loading weapon detector...")
        weapon_weights = self.cfg["models"]["weapon"]["weights"]
        if not os.path.exists(weapon_weights):
            logger.warning("Weapon weights not found (%s). Run: python train_weapon_model.py", weapon_weights)
            self.weapon_model = None
        else:
            self.weapon_model = YOLO(weapon_weights)

        logger.info("Loading activity classifier (CNN+LSTM)...")
        self.activity_clf = ActivityClassifier(
            weights_path=self.cfg["models"]["activity"]["weights"],
            device=self.device,
            cfg=self.cfg,
        )

        logger.info("All models loaded.")

    def _detect_persons(self, frame: np.ndarray) -> list[dict]:
        """Run person detection + ByteTrack tracking."""
        cfg_p = self.cfg["models"]["person"]
        results = self.person_model.track(
            frame,
            persist=True,
            conf=cfg_p["conf_threshold"],
            iou=cfg_p["iou_threshold"],
            classes=[cfg_p["person_class_id"]],
            imgsz=cfg_p["img_size"],
            device=self.device,
            verbose=False,
            tracker="bytetrack.yaml",
        )
        persons = []
        if results and results[0].boxes is not None:
            for box in results[0].boxes:
                if box.id is None:
                    continue
                persons.append({
                    "bbox": tuple(box.xyxy[0].cpu().numpy().tolist()),
                    "confidence": float(box.conf.item()),
                    "track_id": int(box.id.item()),
                    "label": "person",
                })
        return persons

    def _detect_fire(self, frame: np.ndarray) -> list[dict]:
        if self.fire_model is None:
            return []
        cfg_f = self.cfg["models"]["fire"]
        results = self.fire_model(
            frame,
            conf=cfg_f["conf_threshold"],
            iou=cfg_f["iou_threshold"],
            imgsz=cfg_f["img_size"],
            device=self.device,
            verbose=False,
        )
        dets = []
        if results and results[0].boxes is not None:
            for box in results[0].boxes:
                cls_id = int(box.cls.item())
                cls_name = cfg_f["classes"][cls_id] if cls_id < len(cfg_f["classes"]) else "fire"
                dets.append({
                    "bbox": tuple(box.xyxy[0].cpu().numpy().tolist()),
                    "confidence": float(box.conf.item()),
                    "label": cls_name,
                })
        return dets

    def _detect_weapons(self, frame: np.ndarray) -> list[dict]:
        if self.weapon_model is None:
            return []
        cfg_w = self.cfg["models"]["weapon"]
        results = self.weapon_model(
            frame,
            conf=cfg_w["conf_threshold"],
            iou=cfg_w["iou_threshold"],
            imgsz=cfg_w["img_size"],
            device=self.device,
            verbose=False,
        )
        dets = []
        if results and results[0].boxes is not None:
            for box in results[0].boxes:
                cls_id = int(box.cls.item())
                cls_name = cfg_w["classes"][cls_id] if cls_id < len(cfg_w["classes"]) else "weapon"
                dets.append({
                    "bbox": tuple(box.xyxy[0].cpu().numpy().tolist()),
                    "confidence": float(box.conf.item()),
                    "label": cls_name,
                })
        return dets

    def _classify_activity(self, frame: np.ndarray, person: dict) -> tuple[str, float]:
        if self.dry_run or not hasattr(self, "activity_clf") or not self.activity_clf.loaded:
            return "normal", 0.0

        tid = person["track_id"]
        x1, y1, x2, y2 = [int(v) for v in person["bbox"]]
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        if x2 <= x1 or y2 <= y1:
            return "normal", 0.0

        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return "normal", 0.0

        if tid not in self._track_buffers:
            self._track_buffers[tid] = deque(maxlen=self._seq_len * 2)
        self._track_buffers[tid].append(crop)

        if len(self._track_buffers[tid]) >= self._seq_len:
            return self.activity_clf.predict(list(self._track_buffers[tid]))

        return "normal", 0.0

    def _check_threat_proximity(
        self,
        persons: list[dict],
        fire_dets: list[dict],
        weapon_dets: list[dict],
        expand_factor: float,
        frame_h: int,
        frame_w: int,
    ) -> list[dict]:
        threats: list[dict] = []
        for threat in fire_dets + weapon_dets:
            threat_box = threat["bbox"]
            near_person_id = None
            for person in persons:
                expanded_person_box = expand_box(
                    person["bbox"], factor=expand_factor,
                    frame_w=frame_w, frame_h=frame_h,
                )
                containment = compute_containment(threat_box, expanded_person_box)
                if containment > 0.1 or is_overlapping(threat_box, expanded_person_box, iou_threshold=0.02):
                    near_person_id = person.get("track_id")
                    break
            threats.append({
                "threat_type": threat["label"].upper(),
                "confidence":  threat["confidence"],
                "bbox":        threat_box,
                "track_id":    near_person_id,
                "near_person": near_person_id is not None,
            })
        return threats

    def process(self, frame: np.ndarray) -> tuple[np.ndarray, list[dict]]:
        """
        Run full detection pipeline on a single frame.
        Returns annotated_frame and list of active threat dicts.
        """
        h, w = frame.shape[:2]
        annotated = frame.copy()
        active_threats: list[dict] = []

        if self.dry_run:
            cv2.putText(annotated, "DRY RUN — no models loaded", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 165, 255), 2)
            return annotated, []

        persons    = self._detect_persons(frame)
        fire_dets  = self._detect_fire(frame)
        weapon_dets = self._detect_weapons(frame)

        activity_alerts: list[dict] = []
        for person in persons:
            act_class, act_conf = self._classify_activity(frame, person)
            if act_class != "normal" and act_conf >= self.cfg["models"]["activity"]["conf_threshold"]:
                activity_alerts.append({
                    "threat_type": act_class.upper().replace("_", " "),
                    "confidence": act_conf,
                    "bbox": person["bbox"],
                    "track_id": person["track_id"],
                    "near_person": True,
                })

        proximity_threats = self._check_threat_proximity(
            persons, fire_dets, weapon_dets,
            expand_factor=self.cfg["overlap"]["person_box_expand_factor"],
            frame_h=h, frame_w=w,
        )

        all_threats = proximity_threats + activity_alerts

        for p in persons:
            self.vis.draw_box(annotated, p["bbox"], "person", p["confidence"], color=(0, 200, 255))
            self.vis.draw_track_id(annotated, p["bbox"], p["track_id"])

        for f in fire_dets:
            self.vis.draw_box(annotated, f["bbox"], f["label"], f["confidence"])

        for w_det in weapon_dets:
            self.vis.draw_box(annotated, w_det["bbox"], w_det["label"], w_det["confidence"])

        for a in activity_alerts:
            self.vis.draw_box(annotated, a["bbox"], a["threat_type"], a["confidence"], color=(0, 128, 255))

        for threat in all_threats:
            fired = self.alert_engine.update(
                threat_type=threat["threat_type"],
                confidence=threat["confidence"],
                frame=annotated,
                track_id=threat.get("track_id"),
                bbox=threat.get("bbox"),
            )
            if fired:
                active_threats.append(threat)

        if active_threats:
            msg = " | ".join(t["threat_type"] for t in active_threats[:3])
            self.vis.draw_alert_banner(annotated, msg)

        return annotated, active_threats


# =============================================================================
# Main Loop
# =============================================================================

def run(args: argparse.Namespace) -> None:
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    if args.source is not None:
        src = args.source
        if src.isdigit():
            src = int(src)
        cfg["camera"]["source"] = src

    device_cfg = cfg.get("device", "auto")
    if device_cfg == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_cfg)
    logger.info("Running on device: %s", device)
    if device.type == "cuda":
        logger.info("GPU: %s | VRAM: %.1f GB", torch.cuda.get_device_name(0),
                    torch.cuda.get_device_properties(0).total_memory / 1e9)

    pipeline = DetectionPipeline(cfg=cfg, device=device, dry_run=args.dry_run)

    cam_cfg = cfg["camera"]
    logger.info("Opening camera source: %s", cam_cfg["source"])
    try:
        cap = ThreadedCapture(
            source=cam_cfg["source"],
            width=cam_cfg["width"],
            height=cam_cfg["height"],
            buffer_size=cam_cfg.get("buffer_size", 2),
        )
    except RuntimeError as e:
        logger.error("Camera error: %s", e)
        sys.exit(1)

    fps_target = cam_cfg.get("fps_target", 30)
    fps_counter = deque(maxlen=30)
    frame_count = 0

    logger.info("Starting detection loop. Press Q to quit.")
    os.makedirs("alerts", exist_ok=True)

    try:
        while True:
            t_frame_start = time.perf_counter()

            ret, frame = cap.read()
            if not ret or frame is None:
                logger.warning("No frame received — waiting...")
                time.sleep(0.05)
                continue

            frame_count += 1

            annotated, threats = pipeline.process(frame)

            elapsed = time.perf_counter() - t_frame_start
            fps_counter.append(1.0 / max(elapsed, 1e-9))
            fps = np.mean(fps_counter)
            Visualizer.draw_fps(annotated, fps)

            Visualizer.draw_info_panel(annotated, {
                "Device": str(device),
                "Frame": str(frame_count),
                "Alerts": str(len(threats)),
            }, start_y=annotated.shape[0] - 70, x=10)

            cv2.imshow("Real-Time Activity Detection | Q=Quit", annotated)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                logger.info("User quit.")
                break

            processing_ms = (time.perf_counter() - t_frame_start) * 1000
            target_ms = 1000.0 / fps_target
            sleep_ms = target_ms - processing_ms
            if sleep_ms > 1:
                time.sleep(sleep_ms / 1000.0)

    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt — stopping.")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        logger.info("Detection loop ended. Total frames processed: %d", frame_count)


# =============================================================================
# CLI Entry Point
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Real-Time Unusual Activity Detection from webcam / RTSP / video file."
    )
    parser.add_argument(
        "--source", default=None,
        help="Camera source: 0 (webcam), 'rtsp://...', or path to video file.",
    )
    parser.add_argument(
        "--config", default="configs/detection_config.yaml",
        help="Path to detection config YAML.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Open camera without loading models (for testing camera only).",
    )
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
