"""
Core detection logic for object-contact alert monitoring.

This module owns model loading, camera capture, danger-class definitions, and
contact detection. It intentionally has no web/UI code.
"""

from __future__ import annotations

import os
import platform
import ssl
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO, YOLOWorld


ssl._create_default_https_context = ssl._create_unverified_context


REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class ObjectContactConfig:
    camera_index: int = 1
    camera_width: int = 1280
    camera_height: int = 720
    device: str = "mps"
    seg_conf: float = 0.25
    world_conf: float = 0.20
    iou: float = 0.45
    imgsz: int = 640
    world_skip_frames: int = 3
    alert_cooldown: float = 5.0
    seg_model_path: str = str(REPO_ROOT / "yolov8x-seg.pt")
    world_model_path: str = str(REPO_ROOT / "yolov8x-worldv2.pt")


PERSON_CLASS = "person"

COCO_DANGER_CLASSES: set[str] = {
    "knife",
    "scissors",
    "bottle",
    "wine glass",
    "fork",
    "microwave",
    "oven",
    "toaster",
}

CUSTOM_DANGER_CLASSES: list[str] = [
    "cigarette lighter",
    "box of matches",
    "prescription medicine bottle",
    "electrical extension cord",
    "plastic grocery bag",
    "wax candle",
    "clothes iron",
    "medical syringe",
    "household cleaning spray bottle",
]

DEFAULT_MONITORED_CLASSES = set(COCO_DANGER_CLASSES) | set(CUSTOM_DANGER_CLASSES)


@dataclass
class ContactDetection:
    class_name: str
    source: str
    box: tuple[int, int, int, int]
    mask: np.ndarray | None = None
    expanded_box: tuple[int, int, int, int] | None = None
    alert: bool = False


@dataclass
class FrameDetections:
    persons: list[ContactDetection]
    dangers: list[ContactDetection]
    alerts: list[str]
    cached_world: list[tuple[str, tuple[int, int, int, int]]]


def play_alert_sound() -> None:
    system = platform.system()
    if system == "Darwin":
        os.system('say "Warning, danger detected" &')
    elif system == "Windows":
        import winsound
        winsound.Beep(1000, 500)
    else:
        os.system('paplay /usr/share/sounds/alsa/Front_Left.wav &')


def open_camera(config: ObjectContactConfig) -> cv2.VideoCapture:
    backend = cv2.CAP_AVFOUNDATION if platform.system() == "Darwin" else cv2.CAP_DSHOW
    cap = cv2.VideoCapture(config.camera_index, backend)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.camera_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.camera_height)
    return cap


def load_models(config: ObjectContactConfig) -> tuple[YOLO, YOLOWorld]:
    seg_model = YOLO(config.seg_model_path)
    world_model = YOLOWorld(config.world_model_path)
    world_model.set_classes(CUSTOM_DANGER_CLASSES)
    return seg_model, world_model


def resize_mask(raw_mask: np.ndarray, frame_w: int, frame_h: int) -> np.ndarray:
    return cv2.resize(raw_mask, (frame_w, frame_h)) > 0.5


def masks_overlap(mask_a: np.ndarray, mask_b: np.ndarray) -> bool:
    return bool(np.logical_and(mask_a, mask_b).any())


def expand_box(
    box: tuple[int, int, int, int],
    margin: int,
    frame_w: int,
    frame_h: int,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    return (
        max(0, x1 - margin),
        max(0, y1 - margin),
        min(frame_w - 1, x2 + margin),
        min(frame_h - 1, y2 + margin),
    )


def boxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    return not (ax2 < bx1 or bx2 < ax1 or ay2 < by1 or by2 < ay1)


def mask_intersects_box(mask: np.ndarray, box: tuple[int, int, int, int]) -> bool:
    x1, y1, x2, y2 = box
    return bool(mask[y1:y2 + 1, x1:x2 + 1].any())


def detect_contacts(
    frame: np.ndarray,
    seg_model: YOLO,
    world_model: YOLOWorld,
    config: ObjectContactConfig,
    monitored_classes: set[str],
    proximity_px: int,
    frame_count: int,
    cached_world: list[tuple[str, tuple[int, int, int, int]]],
) -> FrameDetections:
    frame_h, frame_w = frame.shape[:2]

    seg_results = seg_model.track(
        source=frame,
        persist=True,
        conf=config.seg_conf,
        iou=config.iou,
        device=config.device,
        imgsz=config.imgsz,
        verbose=False,
    )
    r_seg = seg_results[0]

    persons: list[ContactDetection] = []
    coco_dangers: list[ContactDetection] = []

    if r_seg.boxes is not None:
        has_masks = r_seg.masks is not None
        masks_data = r_seg.masks.data.cpu().numpy() if has_masks else None

        for i, box in enumerate(r_seg.boxes):
            class_name = seg_model.names[int(box.cls[0].item())]
            xyxy = tuple(map(int, box.xyxy[0].tolist()))
            mask = None
            if has_masks and masks_data is not None and i < len(masks_data):
                mask = resize_mask(masks_data[i], frame_w, frame_h)

            if class_name == PERSON_CLASS:
                persons.append(ContactDetection(class_name, "seg", xyxy, mask=mask))
            elif class_name in COCO_DANGER_CLASSES and class_name in monitored_classes:
                coco_dangers.append(ContactDetection(class_name, "coco", xyxy, mask=mask))

    if frame_count % config.world_skip_frames == 0:
        cached_world = []
        world_results = world_model.predict(
            source=frame,
            conf=config.world_conf,
            iou=config.iou,
            device=config.device,
            imgsz=config.imgsz,
            verbose=False,
        )
        r_world = world_results[0]
        if r_world.boxes is not None:
            for box in r_world.boxes:
                class_name = world_model.names[int(box.cls[0].item())]
                if class_name not in monitored_classes:
                    continue
                xyxy = tuple(map(int, box.xyxy[0].tolist()))
                cached_world.append((class_name, xyxy))

    dangers: list[ContactDetection] = []
    alerts: list[str] = []

    for danger in coco_dangers:
        expanded = expand_box(danger.box, proximity_px, frame_w, frame_h)
        danger.expanded_box = expanded

        for person in persons:
            if danger.mask is not None and person.mask is not None:
                if masks_overlap(person.mask, danger.mask):
                    danger.alert = True
                    break
            elif person.mask is not None:
                if mask_intersects_box(person.mask, expanded):
                    danger.alert = True
                    break
            elif boxes_overlap(person.box, expanded):
                danger.alert = True
                break

        if danger.alert:
            alerts.append(danger.class_name)
        dangers.append(danger)

    for class_name, box in cached_world:
        expanded = expand_box(box, proximity_px, frame_w, frame_h)
        danger = ContactDetection(class_name, "world", box, expanded_box=expanded)

        for person in persons:
            if person.mask is not None:
                if mask_intersects_box(person.mask, expanded):
                    danger.alert = True
                    break
            elif boxes_overlap(person.box, expanded):
                danger.alert = True
                break

        if danger.alert:
            alerts.append(danger.class_name)
        dangers.append(danger)

    return FrameDetections(persons, dangers, list(dict.fromkeys(alerts)), cached_world)
