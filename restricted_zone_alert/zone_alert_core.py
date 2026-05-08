"""
Core detection and geometry utilities for the restricted-zone alert demo.

This module intentionally contains no window/UI loop. It owns model loading,
target extraction, zone intersection checks, and cross-platform alert sound.
"""

import os
import platform
import ssl
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


ssl._create_default_https_context = ssl._create_unverified_context

REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class ZoneAlertConfig:
    camera_index: int = 1
    camera_width: int = 1280
    camera_height: int = 720
    device: str = "mps"
    conf: float = 0.25
    iou: float = 0.45
    imgsz: int = 640
    alert_cooldown: float = 5.0
    model_path: str = str(REPO_ROOT / "yolov8x-seg.pt")


PERSON_CLASS = "person"
PET_CLASSES = {"cat", "dog", "bird"}
TARGET_CLASSES = {PERSON_CLASS} | PET_CLASSES


@dataclass
class TargetDetection:
    class_name: str
    box: tuple[int, int, int, int]
    mask: np.ndarray | None
    in_zone: bool = False


def play_alert_sound() -> None:
    system = platform.system()
    if system == "Darwin":
        os.system('say "Warning, restricted zone" &')
    elif system == "Windows":
        import winsound
        winsound.Beep(1000, 500)
    else:
        os.system('paplay /usr/share/sounds/alsa/Front_Left.wav &')


def open_camera(config: ZoneAlertConfig) -> cv2.VideoCapture:
    backend = cv2.CAP_AVFOUNDATION if platform.system() == "Darwin" else cv2.CAP_DSHOW
    cap = cv2.VideoCapture(config.camera_index, backend)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.camera_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.camera_height)
    return cap


def load_model(config: ZoneAlertConfig) -> YOLO:
    return YOLO(config.model_path)


def resize_mask(raw_mask: np.ndarray, frame_w: int, frame_h: int) -> np.ndarray:
    return cv2.resize(raw_mask, (frame_w, frame_h)) > 0.5


def build_zone_mask(points: list[tuple[int, int]], frame_w: int, frame_h: int) -> np.ndarray:
    mask = np.zeros((frame_h, frame_w), dtype=np.uint8)
    pts = np.array(points, dtype=np.int32)
    cv2.fillPoly(mask, [pts], 1)
    return mask.astype(bool)


def expand_zone_mask(zone_mask: np.ndarray, margin_px: int) -> np.ndarray:
    if margin_px <= 0:
        return zone_mask
    kernel_size = margin_px * 2 + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    expanded = cv2.dilate(zone_mask.astype(np.uint8), kernel, iterations=1)
    return expanded.astype(bool)


def mask_overlaps_zone(target_mask: np.ndarray, zone_mask: np.ndarray) -> bool:
    return bool(np.logical_and(target_mask, zone_mask).any())


def box_overlaps_zone(box: tuple[int, int, int, int], zone_mask: np.ndarray) -> bool:
    x1, y1, x2, y2 = box
    h, w = zone_mask.shape[:2]
    x1 = max(0, min(x1, w - 1))
    x2 = max(0, min(x2, w - 1))
    y1 = max(0, min(y1, h - 1))
    y2 = max(0, min(y2, h - 1))
    if x2 < x1 or y2 < y1:
        return False
    return bool(zone_mask[y1:y2 + 1, x1:x2 + 1].any())


def detect_targets(
    model: YOLO,
    frame: np.ndarray,
    config: ZoneAlertConfig,
    zone_mask: np.ndarray | None,
) -> list[TargetDetection]:
    frame_h, frame_w = frame.shape[:2]
    results = model.track(
        source=frame,
        persist=True,
        conf=config.conf,
        iou=config.iou,
        device=config.device,
        imgsz=config.imgsz,
        verbose=False,
    )

    r = results[0]
    detections: list[TargetDetection] = []

    if r.boxes is None:
        return detections

    has_masks = r.masks is not None
    masks_data = r.masks.data.cpu().numpy() if has_masks else None

    for i, box in enumerate(r.boxes):
        class_name = model.names[int(box.cls[0].item())]
        if class_name not in TARGET_CLASSES:
            continue

        xyxy = tuple(map(int, box.xyxy[0].tolist()))
        target_mask = None
        in_zone = False

        if has_masks and masks_data is not None and i < len(masks_data):
            target_mask = resize_mask(masks_data[i], frame_w, frame_h)

        if zone_mask is not None:
            if target_mask is not None:
                in_zone = mask_overlaps_zone(target_mask, zone_mask)
            else:
                in_zone = box_overlaps_zone(xyxy, zone_mask)

        detections.append(TargetDetection(class_name, xyxy, target_mask, in_zone))

    return detections
