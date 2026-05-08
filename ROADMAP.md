# Child Safety Monitor - Development Roadmap

## Strategy Overview: Three-Phase Approach

```
Phase 1 (Done)              Phase 2 (Data)              Phase 3 (Production)
────────────────            ────────────────            ────────────────────
Dual-model hybrid           Grounded-SAM                Fine-tuned YOLO11x-seg
YOLOv8x-seg                 Auto-annotation             Single unified model
+ YOLOWorld                 + human review              + SAHI inference
Combined web UI shipped     Dataset building            Pixel-accurate contact
~65–75% mAP est.            ~500+ imgs/class            ~85–90% mAP target
```

---

## Phase 1 — Working System with Open-Vocabulary Detection ✅

**Goal:** Ship both features as functional browser-based tools using existing models. No training required.

### Feature 1 — Object Contact Alert

| Task | Status |
|---|---|
| 1.1 Load `yolov8x-seg` + `YOLOWorld` dual-model hybrid | ✅ Done |
| 1.2 `model.set_classes([...])` with 9 custom danger classes | ✅ Done |
| 1.3 Three-tier contact detection: mask overlap → mask×box → box×box | ✅ Done |
| 1.4 Configurable `monitored_classes` per request | ✅ Done |
| 1.5 Flask web UI: MJPEG stream, debug/clean view, object checklist, alert buffer | ✅ Done |
| 1.6 Alert cooldown (5s) + macOS voice alert | ✅ Done |
| 1.7 Benchmark tool: per-class detection rate + false positive rate across 3 distances | ✅ Done |
| 1.8 Run benchmark: record actual detection rates, identify classes that need training | ⏳ Pending |
| 1.9 Combined monitor UI: object contact + restricted zone in one page | ✅ Done |
| 1.10 Mobile camera UI: phone browser captures frames, sends to server over HTTPS | ✅ Done |

### Feature 2 — Restricted Zone Alert

| Task | Status |
|---|---|
| 2.1 YOLOv8x-seg tracking for person / cat / dog / bird | ✅ Done |
| 2.2 Polygon zone drawing via browser clicks | ✅ Done |
| 2.3 Zone mask + morphological dilation for alert buffer | ✅ Done |
| 2.4 Flask web UI: zone draw/lock/undo/clear, debug/clean view, alert buffer | ✅ Done |
| 2.5 Alert cooldown (5s) + macOS voice alert | ✅ Done |

### Phase 1 Deliverables
- `combined_monitor.py` — recommended combined UI, runs on http://127.0.0.1:5004
- `mobile_camera_monitor.py` — phone camera UI, runs on https://\<lan-ip\>:5443
- `object_contact_alert/app/object_contact_web.py` — runs on http://127.0.0.1:5003
- `restricted_zone_alert/zone_alert_web.py` — runs on http://127.0.0.1:5002
- `object_contact_alert/benchmark/benchmark.py` — benchmark tool ready to run
- `Makefile` — shortcuts: `make install`, `make run`, `make run-mobile`, `make run-object`, `make run-zone`, `make check`, `make clean`
- `DEPLOYMENT.md` — full setup and sharing guide including phone camera mode and Git LFS instructions
- `MODEL_WEIGHTS.md` — instructions for handling large `.pt` files on GitHub

### Known Limitations (carried into Phase 2 & 3)
- YOLOWorld zero-shot accuracy for visually ambiguous items (pill bottle vs. bottle, battery vs. coin) is ~50–60%. These are the primary Phase 2 training targets.
- Custom classes (YOLOWorld) use bbox proximity only — no pixel masks until Phase 3.
- `imgsz=640` — raise to `1280` for better distant-object detection at ~2× slower inference.
- Model weights are larger than GitHub's regular 100 MB file limit. Use Git LFS or release assets when sharing the repository.

---

## Phase 2 — Custom Dataset with Grounded-SAM

**Goal:** Build a high-quality labeled segmentation dataset for the 9 custom (non-COCO) classes. Only build data for classes where Phase 1 benchmark shows mAP < 70%.

### 2.1 Identify Which Classes Need Training

- [ ] Run Task 1.8 benchmark for 1–2 weeks in real environment
- [ ] Log per-class detection rate at near / mid / far distance
- [ ] Prioritise classes scoring below 70%: expected to be `lighter`, `pill bottle`, `button battery`, `power cord`
- [ ] Skip classes where YOLOWorld already performs well (≥ 70%)

### 2.2 Image Collection — Distance Diversity is Critical

> ⚠️ **Most important rule:** Every class must have images at three distance bands. A model trained only on close-up shots will fail at far-range detection.

- [ ] Near (~0.5–1 m): 30% of images per class
- [ ] Mid (~1.5–3 m): 40% of images per class
- [ ] Far (~4–6 m, item is small in frame): 30% of images per class
- [ ] Vary environment, lighting, angle, and background
- [ ] Target: 100–200 raw images per class

### 2.3 Auto-Annotation with Grounded-SAM

```bash
pip install groundingdino-py segment-anything-2

python scripts/annotate_with_grounded_sam.py \
    --images data/raw_images/ \
    --classes "lighter, pill bottle, button battery, power cord" \
    --output data/annotations/ \
    --format yolo-seg
```

- [ ] Run auto-annotation on all collected images
- [ ] Human review: ~20–30% of auto-masks will be wrong — budget ~1–2 min per image
- [ ] Export to YOLO segmentation format (normalised polygon `.txt` files)

### 2.4 Augmentation with Albumentations

```python
transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
    A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=15, p=0.4),
    A.GaussNoise(var_limit=(5, 25), p=0.3),
    A.Blur(blur_limit=3, p=0.2),
    A.Rotate(limit=15, p=0.3),
    A.RandomScale(scale_limit=0.3, p=0.4),
], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))
```

- [ ] Apply pipeline; target 500+ images per class after augmentation
- [ ] Verify all augmented masks remain valid
- [ ] Split 80% train / 20% val — split by original image, not augmented copies

### 2.5 dataset.yaml

```yaml
path: data/
train: images/train
val:   images/val
nc: 9
names:
  - cigarette_lighter
  - box_of_matches
  - prescription_medicine_bottle
  - electrical_extension_cord
  - plastic_grocery_bag
  - wax_candle
  - clothes_iron
  - medical_syringe
  - household_cleaning_spray
```

### Phase 2 Deliverable
`data/` folder with YOLO segmentation format dataset, human-reviewed, split into train/val, ready for Phase 3 training.

---

## Phase 3 — Fine-Tune YOLO11x-seg and Finalize

**Goal:** Replace YOLOWorld with a fine-tuned `yolo11x-seg` model. Integrate SAHI for far-distance detection.

### 3.1 Model Fine-Tuning

```python
from ultralytics import YOLO

model = YOLO("yolo11x-seg.pt")
model.train(
    data="data/dataset.yaml",
    epochs=50,
    imgsz=640,
    batch=8,
    device="mps",
    pretrained=True,   # CRITICAL: build on COCO backbone
    freeze=10,         # freeze backbone layers 0–9
    augment=True,
    patience=15,
    plots=True,
)
```

- [ ] Download `yolo11x-seg.pt` (auto on first run)
- [ ] Run training (~2–4 hours on M2/M3 Pro)
- [ ] Evaluate per-class mAP50 on validation set; target ≥ 75% per class
- [ ] Compare against Phase 1 YOLOWorld baseline

### 3.2 SAHI Integration

SAHI tiles each frame into overlapping 320×320 patches, runs YOLO on each tile, then merges results — the most effective technique for detecting small/distant objects.

```python
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

detection_model = AutoDetectionModel.from_pretrained(
    model_type="ultralytics",
    model_path="models/yolo11x-seg-custom.pt",
    confidence_threshold=0.25,
    device="mps",
)
result = get_sliced_prediction(frame, detection_model,
    slice_height=320, slice_width=320,
    overlap_height_ratio=0.2, overlap_width_ratio=0.2)
```

- [ ] Install SAHI and integrate into the detection loop
- [ ] Test: object at 5 m — confirm detection improves vs. non-SAHI baseline
- [ ] Note: SAHI adds ~100–200ms latency per frame; acceptable for monitoring

### 3.3 Replace YOLOWorld with Fine-Tuned Model

```python
# Phase 1 (dual model):
seg_model   = YOLO("yolov8x-seg.pt")
world_model = YOLOWorld("yolov8x-worldv2.pt")

# Phase 3 (single model, covers everything):
model = YOLO("models/yolo11x-seg-custom.pt")
```

- [ ] Swap model in `object_contact_core.py`; remove YOLOWorld import and inference block
- [ ] All contact detection now uses pixel-level mask overlap (no bbox fallback needed)
- [ ] Set `imgsz=1280` in config for better far-distance performance

### Phase 3 Deliverable
Single fine-tuned model replaces the dual-model setup. SAHI enabled. Both web UIs continue to work without UI changes.

---

## Timeline Estimate

| Phase | Estimated Duration | Status |
|---|---|---|
| Phase 1 — Feature 1 web UI | 2–3 days | ✅ Done |
| Phase 1 — Feature 2 web UI | 2–3 days | ✅ Done |
| Phase 1 — Combined web UI | 1 day | ✅ Done |
| Phase 1 — Mobile camera UI | 1 day | ✅ Done |
| Phase 1 — Benchmark tool | 1 day | ✅ Done |
| Phase 1 — Run benchmark (Task 1.8) | 1–2 weeks real-world use | ⏳ Not started |
| Phase 2 — Image collection | 1–2 weeks | ⏳ Not started |
| Phase 2 — Annotation + augmentation | 2–3 days | ⏳ Not started |
| Phase 3 — Fine-tuning | 1 day + compute time | ⏳ Not started |
| Phase 3 — SAHI + model swap | 1 day | ⏳ Not started |

---

## Decision Log

| Date | Decision | Rationale |
|---|---|---|
| April 2026 | Use YOLOWorld first, custom training later | Faster to validate; avoid premature optimisation |
| April 2026 | Drop ControlNet augmentation | Too complex and unstable; Albumentations is more reliable |
| April 2026 | Focus on children only (not pets) in v1 | Simpler scope; pets can be added later with same pipeline |
| April 2026 | Migrate from Jupyter to .py scripts | OpenCV GUI crashes in Jupyter on macOS |
| April 2026 | Dual-model hybrid (seg + YOLOWorld) for Phase 1 | Full scene masks + open-vocab custom classes |
| April 2026 | Fine-tune on custom classes only, keep COCO model separate | Avoids 20 GB COCO download; dual-model handles this cleanly |
| April 2026 | Use YOLO11x-seg instead of YOLOv8x-seg for Phase 3 | Newer model, better small-object detection, same API |
| April 2026 | Add SAHI for far-distance detection | Most effective technique for small/distant objects |
| April 2026 | Mandatory human review of Grounded-SAM masks | ~20–30% of auto-masks are incorrect |
| May 2026 | Flask web UI for both features | Browser-based UI is more flexible than OpenCV window |
| May 2026 | Reorganised folder structure | `object_contact_alert/`, `restricted_zone_alert/`, `Fundamental_model_demos/` |
| May 2026 | Add `combined_monitor.py` | One browser page demonstrates both safety features together |
| May 2026 | Add `mobile_camera_monitor.py` | Mobile browsers block `getUserMedia()` on plain HTTP LAN pages; HTTPS on port 5443 is required |
| May 2026 | Add `Makefile`, `DEPLOYMENT.md`, `MODEL_WEIGHTS.md`, `requirements.txt` | Standardise setup commands and document GitHub upload constraints for large model weights |
| May 2026 | Document GitHub deployment constraints | `.pt` weights require Git LFS, release assets, or external downloads |
