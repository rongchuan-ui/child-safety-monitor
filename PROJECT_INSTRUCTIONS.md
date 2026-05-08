# Child Safety Monitoring System - Project Instructions

## Project Overview

This project is a local real-time AI monitoring demo for home safety. It reads a webcam feed, detects people and dangerous objects, then triggers alerts based on user-defined browser controls.

The current recommended app is the combined monitor:

```bash
python combined_monitor.py
```

Open:

```text
http://127.0.0.1:5004
```

For phone-camera capture, use:

```bash
python mobile_camera_monitor.py
```

Open the HTTPS LAN URL printed by the server on the phone.

## Core Features

### Feature 1 - Object Contact Alert

The user selects dangerous objects from a checklist. If a person is near or touching a selected object, the system triggers an alert.

- Input: dangerous object checklist in the web UI
- Trigger: person mask overlaps a danger mask or intersects an expanded danger box
- Alert: red video border, status update, and system voice alert
- Combined UI: `combined_monitor.py`, port `5004`
- Standalone UI: `object_contact_alert/app/object_contact_web.py`, port `5003`

### Feature 2 - Restricted Zone Alert

The user draws a polygon zone directly on the live video. If a person or pet enters the zone or its buffer, the system triggers an alert.

- Input: clicked polygon points on the video
- Trigger: target mask overlaps the expanded zone mask
- Alert: red video border, status update, and system voice alert
- Combined UI: `combined_monitor.py`, port `5004`
- Standalone UI: `restricted_zone_alert/zone_alert_web.py`, port `5002`

Clean mode hides model overlays and the drawn zone. Debug mode shows masks, boxes, labels, zones, and buffer boundaries.

## Technology Stack

| Component | Current implementation |
|---|---|
| Camera capture | OpenCV (`cv2`) or phone browser `getUserMedia()` |
| Web UI | Flask + MJPEG stream or HTTPS frame upload |
| Primary detection | `yolov8x-seg` for COCO classes and masks |
| Custom class detection | `YOLOWorld` with `yolov8x-worldv2.pt` |
| Hardware acceleration | Apple MPS by default (`device="mps"`) |
| Language | Python 3.10+ |

## Model Files

The project expects these files in the repository root:

```text
yolov8x-seg.pt
yolov8x-worldv2.pt
```

They are larger than GitHub's normal 100 MB file limit. Use Git LFS, GitHub Release assets, or external download links. See `MODEL_WEIGHTS.md`.

## Detection Logic

### Object Contact Alert

COCO danger items from `yolov8x-seg` can have masks:

```text
Priority 1: person mask + danger mask -> masks_overlap()
Priority 2: person mask only          -> mask_intersects_box(person_mask, expanded_danger_box)
Priority 3: no masks                  -> boxes_overlap(person_box, expanded_danger_box)
```

YOLOWorld custom danger items only have boxes:

```text
Priority 1: person mask available -> mask_intersects_box(person_mask, expanded_danger_box)
Priority 2: no person mask        -> boxes_overlap(person_box, expanded_danger_box)
```

Contact buffer values:

```text
Small = 30 px
Medium = 60 px
Large = 100 px
```

### Restricted Zone Alert

```text
1. User clicks polygon points on the video.
2. Points become a binary zone mask with cv2.fillPoly.
3. The selected buffer expands the mask with cv2.dilate.
4. YOLOv8x-seg detects person / cat / dog / bird.
5. Target masks are checked against the expanded zone mask.
6. Overlap triggers the zone alert.
```

Zone buffer values:

```text
Small = 20 px
Medium = 50 px
Large = 90 px
```

## Repository Structure

```text
Objection Detection/
├── README.md
├── DEPLOYMENT.md
├── MODEL_WEIGHTS.md
├── combined_monitor.py
├── mobile_camera_monitor.py
├── yolov8x-seg.pt
├── yolov8x-worldv2.pt
├── PROJECT_INSTRUCTIONS.md
├── ROADMAP.md
├── object_contact_alert/
│   ├── README.md
│   ├── app/
│   │   ├── object_contact_core.py
│   │   ├── object_contact_web.py
│   │   └── object_contact_core.ipynb
│   └── benchmark/
│       ├── benchmark.py
│       └── benchmark_annotated.ipynb
├── restricted_zone_alert/
│   ├── README.md
│   ├── zone_alert_core.py
│   └── zone_alert_web.py
└── Fundamental_model_demos/
    ├── learn_yolov8x_seg.ipynb
    └── learn_yoloworld.ipynb
```

## Current Status

Done:

- Combined web UI in `combined_monitor.py`
- Phone-camera web UI in `mobile_camera_monitor.py`
- Standalone object contact UI
- Standalone restricted zone UI
- Dual-model object contact detection
- Polygon zone drawing and mask-based zone detection
- Debug and Clean display modes
- Alert cooldown and system voice alert
- Benchmark script for object classes

Not done:

- Full benchmark run in a real environment
- Custom image collection
- Grounded-SAM auto-annotation workflow
- YOLO11x-seg fine-tuning
- SAHI integration for distant small objects

## How Others Should Run It

See the root deployment guide:

```text
DEPLOYMENT.md
```
