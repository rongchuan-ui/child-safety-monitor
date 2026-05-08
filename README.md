# Child Safety Monitor

Real-time AI camera monitoring for home safety. The project detects two kinds of risks from a webcam feed:

- **Object contact alert:** a person gets close to or touches selected dangerous objects.
- **Restricted zone alert:** a person or pet enters a user-drawn restricted area.

The recommended entry point is the combined web UI:

```bash
python combined_monitor.py
```

To use a phone as the camera, run:

```bash
python mobile_camera_monitor.py
```

Then open the printed HTTPS URL on the phone, for example:

```text
https://192.168.1.23:5443
```

## Demo Behavior

In the browser UI you can:

- View a live camera stream.
- Switch between Debug and Clean modes.
- Draw and lock a restricted zone by clicking on the video.
- Select dangerous object classes to monitor.
- Set separate buffer distances for zone alerts and contact alerts.
- See a red video border and hear a system voice alert when a rule is triggered.

Clean mode hides model overlays and the drawn zone. Debug mode shows masks, boxes, labels, zone geometry, and buffer boundaries.

## Project Structure

```text
Objection Detection/
├── README.md
├── DEPLOYMENT.md
├── MODEL_WEIGHTS.md
├── combined_monitor.py              # Recommended combined UI, port 5004
├── mobile_camera_monitor.py         # Phone browser camera UI, HTTPS port 5443
├── yolov8x-seg.pt                   # YOLO segmentation weights; use Git LFS or download separately
├── yolov8x-worldv2.pt               # YOLOWorld weights; use Git LFS or download separately
├── PROJECT_INSTRUCTIONS.md
├── ROADMAP.md
├── object_contact_alert/
│   ├── README.md
│   ├── app/
│   │   ├── object_contact_core.py
│   │   ├── object_contact_web.py    # Standalone object contact UI, port 5003
│   │   └── object_contact_core.ipynb
│   └── benchmark/
│       ├── benchmark.py
│       └── benchmark_annotated.ipynb
├── restricted_zone_alert/
│   ├── README.md
│   ├── zone_alert_core.py
│   └── zone_alert_web.py            # Standalone restricted zone UI, port 5002
└── Fundamental_model_demos/
    ├── learn_yolov8x_seg.ipynb
    └── learn_yoloworld.ipynb
```

## Quick Start

1. Clone the repository.

```bash
git clone https://github.com/<your-user>/<your-repo>.git
cd "<your-repo>"
```

2. Create and activate a virtual environment.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

3. Install dependencies.

```bash
pip install -r requirements.txt
```

4. Add model weights to the project root.

Required files:

```text
yolov8x-seg.pt
yolov8x-worldv2.pt
```

These files are large. GitHub blocks regular files larger than 100 MB, so use Git LFS or provide download instructions in your release notes.

See [MODEL_WEIGHTS.md](MODEL_WEIGHTS.md) for the model weight options.

5. Run the combined monitor.

```bash
python combined_monitor.py
```

Open:

```text
http://127.0.0.1:5004
```

Optional Makefile commands:

```bash
make install
make run
make run-mobile
make check
make clean
```

## Phone Camera Mode

Use this mode when you want the phone camera to provide the video feed and operate the UI on the phone.

1. Put the computer and phone on the same Wi-Fi.
2. Run:

```bash
python mobile_camera_monitor.py
```

3. The terminal prints an address like:

```text
Open on phone: https://192.168.1.23:5443
```

4. Open that URL on the phone.
5. Accept the local HTTPS certificate warning.
6. Allow camera permission in the phone browser.

Mobile browsers block camera access on plain HTTP LAN pages, so this mode uses HTTPS on port `5443`.

## Standalone Feature UIs

Object contact alert only:

```bash
cd object_contact_alert/app
python object_contact_web.py
```

Open `http://127.0.0.1:5003`.

Restricted zone alert only:

```bash
cd restricted_zone_alert
python zone_alert_web.py
```

Open `http://127.0.0.1:5002`.

## GitHub Upload Notes

Do not commit `.DS_Store`, `__pycache__/`, or virtual environments.

For model weights, choose one:

- **Recommended:** use Git LFS for `.pt` files.
- **Alternative:** do not commit weights; add links in `MODEL_WEIGHTS.md` or GitHub Releases.

Git LFS setup:

```bash
git lfs install
git lfs track "*.pt"
git add .gitattributes
git add yolov8x-seg.pt yolov8x-worldv2.pt
git commit -m "Add model weights with Git LFS"
```

See [DEPLOYMENT.md](DEPLOYMENT.md) for the full deployment and sharing checklist.
See [MODEL_WEIGHTS.md](MODEL_WEIGHTS.md) for large model file handling.

## Notes

- The default device is Apple MPS (`device="mps"`). On machines without Apple Silicon/MPS, change the config in the scripts to `device="cpu"` or a CUDA device.
- Camera index is configured in each script. If the camera does not open, try changing `camera_index` from `1` to `0`.
- This is a local demo app. Do not expose the Flask development server directly to the public internet.
