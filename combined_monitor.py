"""
Child Safety Monitor - Combined Web UI

Runs Feature 1 (Object Contact Alert) and Feature 2 (Restricted Zone Alert)
on a single camera feed with a unified browser interface.

Run:
    cd "/Users/shirch/Desktop/Objection Detection"
    /Users/shirch/vscode101/.venv/bin/python combined_monitor.py

Open:
    http://127.0.0.1:5004
"""

from __future__ import annotations

import os
import platform
import ssl
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, Response, jsonify, render_template_string, request
from ultralytics import YOLO, YOLOWorld

ssl._create_default_https_context = ssl._create_unverified_context


# Config

@dataclass(frozen=True)
class Config:
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
    seg_model_path: str = str(Path(__file__).parent / "yolov8x-seg.pt")
    world_model_path: str = str(Path(__file__).parent / "yolov8x-worldv2.pt")


# Class definitions

ZONE_TARGET_CLASSES: set[str] = {"person", "cat", "dog", "bird"}

COCO_DANGER_CLASSES: set[str] = {
    "knife", "scissors", "bottle", "wine glass",
    "fork", "microwave", "oven", "toaster",
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

ALL_DANGER_CLASSES: list[str] = sorted(COCO_DANGER_CLASSES) + CUSTOM_DANGER_CLASSES

CONTACT_BUFFER = {"small": 30, "medium": 60, "large": 100}
ZONE_BUFFER = {"small": 20, "medium": 50, "large": 90}
BUF_LABELS = {"small": "Small", "medium": "Medium", "large": "Large"}


# Detection dataclass

@dataclass
class Det:
    class_name: str
    source: str
    box: tuple[int, int, int, int]
    mask: np.ndarray | None = None
    expanded_box: tuple[int, int, int, int] | None = None
    contact_alert: bool = False
    in_zone: bool = False


# Web state

@dataclass
class WebState:
    view_mode: str = "debug"
    lock: threading.Lock = field(default_factory=threading.Lock)

    # Feature 1 - Contact Alert
    monitored_classes: set[str] = field(
        default_factory=lambda: set(COCO_DANGER_CLASSES) | set(CUSTOM_DANGER_CLASSES)
    )
    contact_proximity_px: int = CONTACT_BUFFER["medium"]
    contact_alerts: list[str] = field(default_factory=list)
    detected_dangers: list[str] = field(default_factory=list)
    detected_people: int = 0
    last_contact_alert_t: float = 0.0

    # Feature 2 - Zone Alert
    zone_points: list[tuple[int, int]] = field(default_factory=list)
    zone_locked: bool = False
    zone_mask: np.ndarray | None = None
    expanded_zone_mask: np.ndarray | None = None
    zone_proximity_px: int = ZONE_BUFFER["medium"]
    zone_alerts: list[str] = field(default_factory=list)
    detected_zone_count: int = 0
    last_zone_alert_t: float = 0.0
    frame_w: int = 0
    frame_h: int = 0


# Geometry and mask helpers

def resize_mask(raw: np.ndarray, w: int, h: int) -> np.ndarray:
    return cv2.resize(raw, (w, h)) > 0.5

def masks_overlap(a: np.ndarray, b: np.ndarray) -> bool:
    return bool(np.logical_and(a, b).any())

def mask_intersects_box(mask: np.ndarray, box: tuple) -> bool:
    x1, y1, x2, y2 = box
    return bool(mask[y1:y2 + 1, x1:x2 + 1].any())

def expand_box(box: tuple, margin: int, w: int, h: int) -> tuple:
    x1, y1, x2, y2 = box
    return (max(0, x1 - margin), max(0, y1 - margin),
            min(w - 1, x2 + margin), min(h - 1, y2 + margin))

def boxes_overlap(a: tuple, b: tuple) -> bool:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    return not (ax2 < bx1 or bx2 < ax1 or ay2 < by1 or by2 < ay1)

def build_zone_mask(points: list, w: int, h: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=np.uint8)
    if len(points) >= 3:
        cv2.fillPoly(mask, [np.array(points, dtype=np.int32)], 1)
    return mask.astype(bool)

def expand_zone_mask(mask: np.ndarray, px: int) -> np.ndarray:
    if px <= 0:
        return mask
    k = px * 2 + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    return cv2.dilate(mask.astype(np.uint8), kernel).astype(bool)


# Alert sound

def play_alert() -> None:
    if platform.system() == "Darwin":
        os.system('say "Warning, danger detected" &')
    elif platform.system() == "Windows":
        import winsound; winsound.Beep(1000, 500)
    else:
        os.system('paplay /usr/share/sounds/alsa/Front_Left.wav &')


# Runtime globals

CONFIG = Config()
STATE = WebState()
SEG_MODEL = None
WORLD_MODEL = None
CAP = None
FRAME_COUNT = 0
CACHED_WORLD: list[tuple[str, tuple]] = []


def ensure_runtime() -> None:
    global SEG_MODEL, WORLD_MODEL, CAP
    if SEG_MODEL is None:
        print("Loading YOLOv8x-seg...")
        SEG_MODEL = YOLO(CONFIG.seg_model_path)
    if WORLD_MODEL is None:
        print("Loading YOLOWorld...")
        WORLD_MODEL = YOLOWorld(CONFIG.world_model_path)
        WORLD_MODEL.set_classes(CUSTOM_DANGER_CLASSES)
    if CAP is None:
        backend = cv2.CAP_AVFOUNDATION if platform.system() == "Darwin" else cv2.CAP_DSHOW
        CAP = cv2.VideoCapture(CONFIG.camera_index, backend)
        CAP.set(cv2.CAP_PROP_FRAME_WIDTH, CONFIG.camera_width)
        CAP.set(cv2.CAP_PROP_FRAME_HEIGHT, CONFIG.camera_height)
        if not CAP.isOpened():
            raise RuntimeError("Cannot open camera.")
        print(f"Camera: {int(CAP.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(CAP.get(cv2.CAP_PROP_FRAME_HEIGHT))}")


# Per-frame detection

def process_frame(
    frame: np.ndarray,
    monitored: set[str],
    contact_px: int,
    expanded_zone: np.ndarray | None,
) -> tuple[list[Det], list[str], list[str]]:
    global CACHED_WORLD, FRAME_COUNT

    h, w = frame.shape[:2]

    # Single segmentation call extracts people, pets, and COCO danger items.
    seg_res = SEG_MODEL.track(
        source=frame, persist=True,
        conf=CONFIG.seg_conf, iou=CONFIG.iou,
        device=CONFIG.device, imgsz=CONFIG.imgsz, verbose=False,
    )
    r = seg_res[0]

    persons: list[Det] = []
    coco_dangers: list[Det] = []
    zone_targets: list[Det] = []
    all_dets: list[Det] = []

    if r.boxes is not None:
        has_masks = r.masks is not None
        masks_data = r.masks.data.cpu().numpy() if has_masks else None

        for i, box in enumerate(r.boxes):
            cls = SEG_MODEL.names[int(box.cls[0].item())]
            xyxy = tuple(map(int, box.xyxy[0].tolist()))
            mask = None
            if has_masks and masks_data is not None and i < len(masks_data):
                mask = resize_mask(masks_data[i], w, h)

            if cls == "person":
                det = Det(cls, "seg", xyxy, mask=mask)
                persons.append(det)
                zone_targets.append(det)
                all_dets.append(det)
            elif cls in ZONE_TARGET_CLASSES:
                det = Det(cls, "seg", xyxy, mask=mask)
                zone_targets.append(det)
                all_dets.append(det)

            if cls in COCO_DANGER_CLASSES and cls in monitored:
                det = Det(cls, "coco", xyxy, mask=mask)
                coco_dangers.append(det)
                all_dets.append(det)

    # YOLOWorld handles custom danger classes every N frames.
    if FRAME_COUNT % CONFIG.world_skip_frames == 0:
        CACHED_WORLD = []
        w_res = WORLD_MODEL.predict(
            source=frame, conf=CONFIG.world_conf, iou=CONFIG.iou,
            device=CONFIG.device, imgsz=CONFIG.imgsz, verbose=False,
        )
        r_w = w_res[0]
        if r_w.boxes is not None:
            for box in r_w.boxes:
                cls = WORLD_MODEL.names[int(box.cls[0].item())]
                if cls in monitored:
                    CACHED_WORLD.append((cls, tuple(map(int, box.xyxy[0].tolist()))))

    # Contact detection for COCO items, mask-based where possible.
    contact_alerts: list[str] = []
    for danger in coco_dangers:
        exp = expand_box(danger.box, contact_px, w, h)
        danger.expanded_box = exp
        for person in persons:
            if danger.mask is not None and person.mask is not None:
                if masks_overlap(person.mask, danger.mask):
                    danger.contact_alert = True
                    break
            elif person.mask is not None:
                if mask_intersects_box(person.mask, exp):
                    danger.contact_alert = True
                    break
            elif boxes_overlap(person.box, exp):
                danger.contact_alert = True
                break
        if danger.contact_alert:
            contact_alerts.append(danger.class_name)

    # Contact detection for YOLOWorld custom items, box-based.
    for cls, xyxy in CACHED_WORLD:
        exp = expand_box(xyxy, contact_px, w, h)
        det = Det(cls, "world", xyxy, expanded_box=exp)
        for person in persons:
            if person.mask is not None:
                if mask_intersects_box(person.mask, exp):
                    det.contact_alert = True
                    break
            elif boxes_overlap(person.box, exp):
                det.contact_alert = True
                break
        if det.contact_alert:
            contact_alerts.append(cls)
        all_dets.append(det)

    # Zone detection
    zone_alerts: list[str] = []
    if expanded_zone is not None:
        for target in zone_targets:
            if target.mask is not None:
                target.in_zone = masks_overlap(target.mask, expanded_zone)
            else:
                x1, y1, x2, y2 = target.box
                fh, fw = expanded_zone.shape[:2]
                roi = expanded_zone[max(0,y1):min(fh,y2+1), max(0,x1):min(fw,x2+1)]
                target.in_zone = bool(roi.any())
            if target.in_zone:
                zone_alerts.append(target.class_name)

    return all_dets, list(dict.fromkeys(contact_alerts)), list(dict.fromkeys(zone_alerts))


# Rendering

def overlay_mask(frame: np.ndarray, mask: np.ndarray,
                 color: np.ndarray, alpha: float = 0.35) -> None:
    frame[mask] = (frame[mask] * (1 - alpha) + color * alpha).astype(np.uint8)


def draw_label(frame: np.ndarray, box: tuple, label: str,
               color: tuple[int, int, int]) -> None:
    x1, y1, x2, y2 = box
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.62, 2)
    top = max(8, y1 - th - 8)
    bottom = top + th + 8
    cv2.rectangle(frame, (x1, top), (min(x1 + tw + 6, frame.shape[1] - 1), bottom), color, -1)
    cv2.putText(frame, label, (x1 + 3, bottom - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA)


def draw_zone_overlay(frame: np.ndarray, points: list, locked: bool) -> None:
    if not points:
        return
    pts = np.array(points, dtype=np.int32)
    if locked and len(points) >= 3:
        overlay = frame.copy()
        cv2.fillPoly(overlay, [pts], (0, 140, 255))
        cv2.addWeighted(overlay, 0.18, frame, 0.82, 0, frame)
        cv2.polylines(frame, [pts], True, (0, 140, 255), 2)
    else:
        cv2.polylines(frame, [pts], False, (0, 210, 255), 2)
    for pt in points:
        cv2.circle(frame, pt, 5, (0, 255, 255), -1)


def draw_zone_boundary(frame: np.ndarray, expanded_mask: np.ndarray | None) -> None:
    if expanded_mask is None:
        return
    contours, _ = cv2.findContours(
        expanded_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(frame, contours, -1, (0, 200, 255), 1)


def render_frame(
    frame: np.ndarray,
    all_dets: list[Det],
    zone_pts: list,
    zone_locked: bool,
    expanded_zone: np.ndarray | None,
    show_debug: bool,
    any_alert: bool,
) -> np.ndarray:
    out = frame.copy()

    # Zone geometry
    if show_debug:
        draw_zone_boundary(out, expanded_zone if zone_locked else None)
        draw_zone_overlay(out, zone_pts, zone_locked)

    for det in all_dets:
        if det.class_name in ZONE_TARGET_CLASSES:
            # Person and pet targets for zone alert.
            if not show_debug and not det.in_zone:
                continue
            if det.in_zone:
                color = (0, 0, 255)
                label = f"ZONE: {det.class_name}"
                mask_color = np.array([0, 0, 220], dtype=np.uint8)
            else:
                color = (0, 185, 80)
                label = det.class_name
                mask_color = np.array([220, 80, 50], dtype=np.uint8)
            if show_debug and det.mask is not None:
                overlay_mask(out, det.mask, mask_color, 0.30)
            draw_label(out, det.box, label, color)

        else:
            # Danger items for contact alert.
            if not show_debug and not det.contact_alert:
                continue
            if det.contact_alert:
                color = (0, 0, 255)
                label = f"DANGER: {det.class_name}"
                mask_color = np.array([0, 0, 220], dtype=np.uint8)
            elif det.source == "world":
                color = (0, 165, 255)
                label = det.class_name
                mask_color = np.array([70, 150, 230], dtype=np.uint8)
            else:
                color = (0, 190, 80)
                label = det.class_name
                mask_color = np.array([70, 150, 230], dtype=np.uint8)

            if show_debug and det.mask is not None:
                overlay_mask(out, det.mask, mask_color)
            if show_debug and det.expanded_box is not None and det.mask is None:
                ex1, ey1, ex2, ey2 = det.expanded_box
                cv2.rectangle(out, (ex1, ey1), (ex2, ey2), (0, 210, 255), 1)
            draw_label(out, det.box, label, color)

    if any_alert:
        cv2.rectangle(out, (0, 0), (out.shape[1] - 1, out.shape[0] - 1), (0, 0, 255), 8)

    return out


# Video stream

def video_frames():
    global FRAME_COUNT
    ensure_runtime()

    while True:
        ok, frame = CAP.read()
        if not ok:
            time.sleep(0.05)
            continue

        FRAME_COUNT += 1
        frame_h, frame_w = frame.shape[:2]
        now = time.time()

        with STATE.lock:
            monitored = set(STATE.monitored_classes)
            contact_px = STATE.contact_proximity_px
            zone_locked = STATE.zone_locked
            zone_pts = list(STATE.zone_points)
            zone_px = STATE.zone_proximity_px
            view_mode = STATE.view_mode
            STATE.frame_w = frame_w
            STATE.frame_h = frame_h

            # Build / rebuild zone masks when needed
            if zone_locked and STATE.zone_mask is None and len(zone_pts) >= 3:
                STATE.zone_mask = build_zone_mask(zone_pts, frame_w, frame_h)
                STATE.expanded_zone_mask = expand_zone_mask(STATE.zone_mask, zone_px)

            expanded_zone = STATE.expanded_zone_mask if zone_locked else None

        all_dets, contact_alerts, zone_alerts = process_frame(
            frame, monitored, contact_px, expanded_zone
        )

        with STATE.lock:
            STATE.detected_people = sum(1 for d in all_dets if d.class_name == "person")
            STATE.detected_dangers = list(dict.fromkeys(
                d.class_name for d in all_dets if d.source in {"coco", "world"}
            ))
            STATE.detected_zone_count = sum(1 for d in all_dets if d.class_name in ZONE_TARGET_CLASSES)
            STATE.contact_alerts = contact_alerts
            STATE.zone_alerts = zone_alerts

            fire_contact = bool(contact_alerts) and now - STATE.last_contact_alert_t > CONFIG.alert_cooldown
            fire_zone = bool(zone_alerts) and now - STATE.last_zone_alert_t > CONFIG.alert_cooldown
            if fire_contact:
                STATE.last_contact_alert_t = now
            if fire_zone:
                STATE.last_zone_alert_t = now

        if fire_contact or fire_zone:
            play_alert()

        out = render_frame(
            frame, all_dets, zone_pts, zone_locked, expanded_zone,
            view_mode == "debug", bool(contact_alerts or zone_alerts)
        )
        ok, buf = cv2.imencode(".jpg", out, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
        if not ok:
            continue
        yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n"


# Flask app

app = Flask(__name__)

PAGE_HTML = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Child Safety Monitor</title>
  <style>
    :root {
      --bg: #0f1115; --panel: #181b20; --panel2: #20242b;
      --line: #303640; --text: #f2f5f7; --muted: #aab3bf;
      --blue: #4f7cff; --green: #2f9d62; --red: #d64545; --amber: #d99a2b;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      min-height: 100vh; background: var(--bg); color: var(--text);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, sans-serif;
    }
    .app {
      min-height: 100vh;
      display: grid;
      grid-template-columns: minmax(520px, 1fr) 360px;
    }
    .stage {
      padding: 20px;
      display: flex; align-items: center; justify-content: center;
      background: #0f1115;
    }
    .video-shell {
      width: min(100%, 1280px);
      position: relative;
      border: 1px solid var(--line);
      background: #06070a;
      box-shadow: 0 18px 50px rgba(0,0,0,.35);
    }
    #video { display: block; width: 100%; height: auto; cursor: crosshair; }
    .top-badge {
      position: absolute; left: 14px; top: 14px;
      padding: 8px 12px; border-radius: 6px;
      background: rgba(18,22,27,.84);
      border: 1px solid rgba(255,255,255,.14);
      font-size: 13px; backdrop-filter: blur(6px);
    }
    .alarm .top-badge { background: rgba(160,30,30,.92); }

    /* Sidebar */
    .panel {
      border-left: 1px solid var(--line); background: var(--panel);
      display: flex; flex-direction: column; overflow-y: auto;
    }
    .header { padding: 18px 18px 14px; border-bottom: 1px solid var(--line); }
    .title { font-size: 18px; font-weight: 700; margin-bottom: 4px; }
    .subtitle { color: var(--muted); font-size: 13px; }

    /* Status card */
    .card {
      margin: 14px 14px 0;
      padding: 14px; border-radius: 8px;
      border: 1px solid var(--line); background: var(--panel2);
    }
    .row {
      display: flex; align-items: center; justify-content: space-between;
      gap: 8px; margin-bottom: 10px;
    }
    .row:last-child { margin-bottom: 0; }
    .lbl { color: var(--muted); font-size: 12px; }
    .val { font-size: 13px; font-weight: 600; text-align: right;
           max-width: 180px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
    .pill {
      display: inline-flex; align-items: center;
      height: 24px; padding: 0 9px; border-radius: 999px;
      font-size: 12px; font-weight: 700;
      background: rgba(47,157,98,.14); color: #7ee2aa;
      border: 1px solid rgba(126,226,170,.25);
    }
    .pill.alert { background: rgba(214,69,69,.15); color: #ff9b9b; border-color: rgba(255,155,155,.3); }

    /* Section headings */
    .section-head {
      display: flex; align-items: center; gap: 8px;
      margin: 14px 14px 0; padding: 10px 12px;
      background: var(--panel2); border-radius: 6px;
      border: 1px solid var(--line);
      font-size: 12px; font-weight: 700; letter-spacing: .04em;
      text-transform: uppercase; color: var(--muted);
    }
    .dot {
      width: 8px; height: 8px; border-radius: 50%;
      background: var(--green); flex-shrink: 0;
    }
    .dot.alert { background: var(--red); }

    /* Controls */
    .ctrl { padding: 10px 14px; display: grid; gap: 8px; }
    button {
      height: 38px; border: 1px solid var(--line); border-radius: 7px;
      background: #242a32; color: var(--text); font: inherit;
      font-size: 13px; font-weight: 650; cursor: pointer;
    }
    button:hover { background: #2b323c; }
    button.active, button.primary { background: var(--blue); border-color: var(--blue); color: #fff; }
    button.danger { background: rgba(214,69,69,.13); border-color: rgba(214,69,69,.38); color: #ffb1b1; }
    .seg2 { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }
    .seg3 { display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px; }
    .seg4 { display: grid; grid-template-columns: repeat(2, 1fr); gap: 8px; }

    /* Field block */
    .field { margin: 0 14px 0; padding: 12px 14px; border-radius: 8px;
             border: 1px solid var(--line); background: var(--panel2); }
    .field-head { display: flex; justify-content: space-between; gap: 8px;
                  margin-bottom: 8px; color: var(--muted); font-size: 12px; }

    /* Object checklist */
    .checks { display: grid; gap: 6px; max-height: 220px; overflow-y: auto; padding-right: 2px; }
    .check { display: flex; align-items: center; gap: 8px;
             min-height: 26px; font-size: 13px; }
    input[type="checkbox"] { width: 15px; height: 15px; accent-color: var(--blue); flex-shrink: 0; }

    .hint {
      margin: 10px 14px; padding: 10px 12px; border-radius: 6px;
      border: 1px solid rgba(217,154,43,.3); background: rgba(217,154,43,.08);
      color: #f0c77d; font-size: 12px; line-height: 1.5;
    }
    .footer {
      margin-top: auto; padding: 12px 14px 16px;
      color: var(--muted); font-size: 11px;
      border-top: 1px solid var(--line);
    }
    @media (max-width: 900px) {
      .app { grid-template-columns: 1fr; }
      .panel { border-left: 0; border-top: 1px solid var(--line); }
      .stage { padding: 10px; }
    }
  </style>
</head>
<body>
<main class="app">
  <section class="stage">
    <div id="shell" class="video-shell">
      <img id="video" src="/video_feed" alt="Live camera feed">
      <div id="badge" class="top-badge">Monitoring</div>
    </div>
  </section>

  <aside class="panel">
    <div class="header">
      <div class="title">Child Safety Monitor</div>
      <div class="subtitle">Object contact alert + restricted zone alert</div>
    </div>

    <!-- Overall status -->
    <div class="card">
      <div class="row"><span class="lbl">Zone alert</span><span id="zonePill" class="pill">MONITORING</span></div>
      <div class="row"><span class="lbl">Contact alert</span><span id="contactPill" class="pill">MONITORING</span></div>
      <div class="row"><span class="lbl">View</span><span id="viewVal" class="val">Debug</span></div>
      <div class="row"><span class="lbl">People</span><span id="peopleVal" class="val">0</span></div>
    </div>

    <!-- View toggle -->
    <div class="ctrl">
      <div class="seg2">
        <button id="debugBtn" class="active" onclick="setView('debug')">Debug</button>
        <button id="cleanBtn" onclick="setView('clean')">Clean</button>
      </div>
    </div>

    <!-- Zone Alert section -->
    <div class="section-head">
      <div id="zoneDot" class="dot"></div>
      Zone Alert
      <span id="zoneStatus" style="margin-left:auto;font-size:11px;font-weight:400;text-transform:none;color:var(--muted)">No zone</span>
    </div>

    <div class="ctrl">
      <div class="seg4">
        <button class="primary" onclick="lockZone()">Lock</button>
        <button onclick="undoPoint()">Undo</button>
        <button onclick="clearZone()" class="danger" style="grid-column:span 2">Clear Zone</button>
      </div>
    </div>

    <div class="field" style="margin:0 14px 0">
      <div class="field-head"><span>Zone buffer</span><strong id="zoneBufVal">Medium</strong></div>
      <div class="seg3">
        <button id="zs" onclick="setZoneBuf('small')">Small</button>
        <button id="zm" class="active" onclick="setZoneBuf('medium')">Medium</button>
        <button id="zl" onclick="setZoneBuf('large')">Large</button>
      </div>
    </div>

    <!-- Contact Alert section -->
    <div class="section-head" style="margin-top:14px">
      <div id="contactDot" class="dot"></div>
      Contact Alert
      <span id="contactStatus" style="margin-left:auto;font-size:11px;font-weight:400;text-transform:none;color:var(--muted)">—</span>
    </div>

    <div class="field" style="margin:10px 14px 0">
      <div class="field-head"><span>Contact buffer</span><strong id="cBufVal">Medium</strong></div>
      <div class="seg3">
        <button id="cs" onclick="setContactBuf('small')">Small</button>
        <button id="cm" class="active" onclick="setContactBuf('medium')">Medium</button>
        <button id="cl" onclick="setContactBuf('large')">Large</button>
      </div>
    </div>

    <div class="field" style="margin:10px 14px 0">
      <div class="field-head">
        <span>Monitored objects</span>
        <strong id="selCount">0/0</strong>
      </div>
      <div id="checks" class="checks"></div>
    </div>

    <p class="hint">Click the video to draw a zone. Debug view shows masks and boxes. Clean view shows alerts only.</p>
    <div class="footer">Port 5004 · COCO model + YOLOWorld · Apple MPS</div>
  </aside>
</main>

<script>
  const shell = document.getElementById("shell");
  const video = document.getElementById("video");

  async function post(url, data = {}) {
    const r = await fetch(url, { method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(data) });
    return r.json();
  }

  video.addEventListener("click", async (e) => {
    const rect = video.getBoundingClientRect();
    const x = Math.round((e.clientX - rect.left) * (video.naturalWidth || rect.width)  / rect.width);
    const y = Math.round((e.clientY - rect.top)  * (video.naturalHeight || rect.height) / rect.height);
    await post("/api/zone/add", { x, y });
    await refresh();
  });

  async function lockZone()        { await post("/api/zone/lock");          await refresh(); }
  async function undoPoint()       { await post("/api/zone/undo");          await refresh(); }
  async function clearZone()       { await post("/api/zone/clear");         await refresh(); }
  async function setView(m)        { await post("/api/view", { mode: m });  await refresh(); }
  async function setZoneBuf(s)     { await post("/api/zone/buffer", { size: s });    await refresh(); }
  async function setContactBuf(s)  { await post("/api/contact/buffer", { size: s }); await refresh(); }
  async function setMonitored(n,e) { await post("/api/monitored", { name: n, enabled: e }); await refresh(); }

  function renderChecks(all, monitored) {
    const box = document.getElementById("checks");
    if (!box.childElementCount) {
      for (const item of all) {
        const id = "c-" + item.replace(/[^a-z0-9]/gi, "-");
        const row = document.createElement("label");
        row.className = "check";
        row.innerHTML = `<input id="${id}" type="checkbox" checked><span>${item}</span>`;
        box.appendChild(row);
        row.querySelector("input").addEventListener("change", ev =>
          setMonitored(item, ev.target.checked));
      }
    }
    for (const item of all) {
      const el = document.getElementById("c-" + item.replace(/[^a-z0-9]/gi, "-"));
      if (el) el.checked = monitored.includes(item);
    }
    document.getElementById("selCount").textContent =
      `${monitored.length}/${all.length}`;
  }

  async function refresh() {
    const s = await (await fetch("/api/status")).json();

    const zAlert   = s.zone_alerts.length > 0;
    const cAlert   = s.contact_alerts.length > 0;
    const anyAlert = zAlert || cAlert;

    shell.classList.toggle("alarm", anyAlert);

    // Zone section
    const zonePill = document.getElementById("zonePill");
    zonePill.textContent = zAlert ? "ALERT" : "MONITORING";
    zonePill.className   = "pill" + (zAlert ? " alert" : "");
    document.getElementById("zoneDot").className = "dot" + (zAlert ? " alert" : "");
    document.getElementById("zoneStatus").textContent =
      s.zone_locked ? `Locked · ${s.zone_points} pts` :
      s.zone_points > 0 ? `Drawing · ${s.zone_points} pts` : "No zone";
    document.getElementById("zoneBufVal").textContent = s.zone_buf_label;
    for (const k of ["s","m","l"]) {
      const sizes = { s:"small", m:"medium", l:"large" };
      document.getElementById("z"+k).classList.toggle("active", s.zone_buf === sizes[k]);
    }

    // Contact section
    const cPill = document.getElementById("contactPill");
    cPill.textContent = cAlert ? "ALERT" : "MONITORING";
    cPill.className   = "pill" + (cAlert ? " alert" : "");
    document.getElementById("contactDot").className = "dot" + (cAlert ? " alert" : "");
    document.getElementById("contactStatus").textContent =
      cAlert ? s.contact_alerts.join(", ") : "—";
    document.getElementById("cBufVal").textContent = s.contact_buf_label;
    for (const k of ["s","m","l"]) {
      const sizes = { s:"small", m:"medium", l:"large" };
      document.getElementById("c"+k).classList.toggle("active", s.contact_buf === sizes[k]);
    }

    // View / people
    document.getElementById("viewVal").textContent = s.view_mode === "debug" ? "Debug" : "Clean";
    document.getElementById("peopleVal").textContent = s.people;
    document.getElementById("debugBtn").classList.toggle("active", s.view_mode === "debug");
    document.getElementById("cleanBtn").classList.toggle("active", s.view_mode === "clean");

    // Badge
    const msgs = [];
    if (zAlert) msgs.push("Zone: " + s.zone_alerts.join(", "));
    if (cAlert) msgs.push("Contact: " + s.contact_alerts.join(", "));
    document.getElementById("badge").textContent =
      msgs.length ? "WARNING — " + msgs.join("  |  ") :
      s.zone_locked ? "Zone locked · monitoring" : "Click video to draw a zone";

    renderChecks(s.all_classes, s.monitored_classes);
  }

  setInterval(refresh, 700);
  refresh();
</script>
</body>
</html>
"""


@app.route("/")
def index():
    return render_template_string(PAGE_HTML)


@app.route("/video_feed")
def video_feed():
    return Response(video_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.post("/api/view")
def set_view():
    mode = request.get_json(force=True).get("mode", "debug")
    if mode not in {"debug", "clean"}:
        return jsonify(ok=False), 400
    with STATE.lock:
        STATE.view_mode = mode
    return jsonify(ok=True)


@app.post("/api/zone/add")
def zone_add():
    data = request.get_json(force=True)
    with STATE.lock:
        if not STATE.zone_locked:
            STATE.zone_points.append((int(data["x"]), int(data["y"])))
            STATE.zone_mask = None
            STATE.expanded_zone_mask = None
    return jsonify(ok=True)


@app.post("/api/zone/lock")
def zone_lock():
    with STATE.lock:
        if len(STATE.zone_points) >= 3:
            STATE.zone_locked = True
            STATE.zone_mask = None
            STATE.expanded_zone_mask = None
    return jsonify(ok=True)


@app.post("/api/zone/undo")
def zone_undo():
    with STATE.lock:
        if not STATE.zone_locked and STATE.zone_points:
            STATE.zone_points.pop()
            STATE.zone_mask = None
            STATE.expanded_zone_mask = None
    return jsonify(ok=True)


@app.post("/api/zone/clear")
def zone_clear():
    with STATE.lock:
        STATE.zone_points.clear()
        STATE.zone_locked = False
        STATE.zone_mask = None
        STATE.expanded_zone_mask = None
        STATE.zone_alerts = []
    return jsonify(ok=True)


@app.post("/api/zone/buffer")
def zone_buffer():
    size = request.get_json(force=True).get("size", "medium")
    if size not in ZONE_BUFFER:
        return jsonify(ok=False), 400
    with STATE.lock:
        STATE.zone_proximity_px = ZONE_BUFFER[size]
        if STATE.zone_mask is not None:
            STATE.expanded_zone_mask = expand_zone_mask(STATE.zone_mask, STATE.zone_proximity_px)
    return jsonify(ok=True)


@app.post("/api/contact/buffer")
def contact_buffer():
    size = request.get_json(force=True).get("size", "medium")
    if size not in CONTACT_BUFFER:
        return jsonify(ok=False), 400
    with STATE.lock:
        STATE.contact_proximity_px = CONTACT_BUFFER[size]
    return jsonify(ok=True)


@app.post("/api/monitored")
def set_monitored():
    data = request.get_json(force=True)
    name = data.get("name")
    enabled = bool(data.get("enabled", True))
    if name not in set(ALL_DANGER_CLASSES):
        return jsonify(ok=False), 400
    with STATE.lock:
        if enabled:
            STATE.monitored_classes.add(name)
        else:
            STATE.monitored_classes.discard(name)
    return jsonify(ok=True)


@app.get("/api/status")
def status():
    with STATE.lock:
        zone_buf  = next((k for k, v in ZONE_BUFFER.items()    if v == STATE.zone_proximity_px),    "medium")
        c_buf     = next((k for k, v in CONTACT_BUFFER.items() if v == STATE.contact_proximity_px), "medium")
        payload = {
            "view_mode": STATE.view_mode,
            "people": STATE.detected_people,
            # zone
            "zone_points": len(STATE.zone_points),
            "zone_locked": STATE.zone_locked,
            "zone_alerts": list(dict.fromkeys(STATE.zone_alerts)),
            "zone_buf": zone_buf,
            "zone_buf_label": BUF_LABELS[zone_buf],
            # contact
            "contact_alerts": list(dict.fromkeys(STATE.contact_alerts)),
            "detected_dangers": list(dict.fromkeys(STATE.detected_dangers)),
            "contact_buf": c_buf,
            "contact_buf_label": BUF_LABELS[c_buf],
            "monitored_classes": sorted(STATE.monitored_classes),
            "all_classes": ALL_DANGER_CLASSES,
        }
    return jsonify(payload)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5004, debug=False, threaded=True)
