# Deployment Guide

This project can run in two local demo modes:

- Computer webcam mode: the server reads the computer camera with OpenCV.
- Phone camera mode: the phone browser captures camera frames and sends them to the server.

## Requirements

- Python 3.10 or newer
- Webcam access for computer webcam mode
- Same Wi-Fi network for phone camera mode
- macOS Apple Silicon is recommended for `device="mps"`
- Model weights in the project root:
  - `yolov8x-seg.pt`
  - `yolov8x-worldv2.pt`

## Local Setup

```bash
git clone https://github.com/<your-user>/<your-repo>.git
cd "<your-repo>"

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Add the two `.pt` model files to the project root.

See [MODEL_WEIGHTS.md](MODEL_WEIGHTS.md) for Git LFS and release-asset options.

Run:

```bash
python combined_monitor.py
```

Open:

```text
http://127.0.0.1:5004
```

## Phone Camera Setup

Run:

```bash
python mobile_camera_monitor.py
```

The terminal prints the computer's LAN URL:

```text
Open on phone: https://<computer-lan-ip>:5443
```

On the phone:

1. Connect to the same Wi-Fi as the computer.
2. Open the printed HTTPS URL.
3. Accept the local certificate warning.
4. Allow camera permission.
5. Use the page controls to switch camera, draw zones, lock zones, select objects, and change buffers.

Why HTTPS: mobile browsers do not allow `getUserMedia()` camera access from plain HTTP pages served over a LAN IP.

## How To Show The Effect

1. Allow camera access when the OS asks.
2. Open `http://127.0.0.1:5004`.
3. Keep the page in Debug mode first.
4. Click at least 3 points on the video to draw a restricted zone.
5. Click **Lock**.
6. Move a person or pet into the zone to trigger the zone alert.
7. Place a monitored object such as a bottle, scissors, or knife in view.
8. Move a person close to the object to trigger the contact alert.
9. Switch to Clean mode to show the normal camera feed. Clean mode hides the drawn zone and model overlays unless an alert happens.

## GitHub Model Weight Options

GitHub blocks normal files larger than 100 MB. The model files in this project are larger than that.

The repository includes `.gitattributes` configured for Git LFS:

```text
*.pt filter=lfs diff=lfs merge=lfs -text
```

### Option A: Git LFS

```bash
git lfs install
git lfs track "*.pt"
git add .gitattributes
git add yolov8x-seg.pt yolov8x-worldv2.pt
git commit -m "Track model weights with Git LFS"
```

After cloning, users with Git LFS installed can pull the weights:

```bash
git lfs pull
```

### Option B: Release Assets

Do not commit the `.pt` files. Upload them as GitHub Release assets and tell users to place them in the project root.

### Option C: External Download

Do not commit the `.pt` files. Provide a download link in the README and tell users to place them in the project root.

## Common Fixes

Port already in use:

```bash
lsof -i :5004
kill <PID>
```

Phone camera port already in use:

```bash
lsof -i :5443
kill <PID>
```

Camera does not open:

- Allow camera permission in system settings.
- Change `camera_index` in `combined_monitor.py` from `1` to `0`.
- Close other apps using the camera.

Phone browser cannot open camera:

- Confirm the URL starts with `https://`, not `http://`.
- Accept the browser certificate warning.
- Allow camera permission for the site.
- Make sure phone and computer are on the same Wi-Fi.

No MPS device:

- Change `device: str = "mps"` to `device: str = "cpu"` in the script config.

Model file not found:

- Confirm `yolov8x-seg.pt` and `yolov8x-worldv2.pt` are in the project root, next to `combined_monitor.py`.

## Production Note

This is a Flask development server for local demos. For a real deployment, place it behind a proper WSGI server and secure camera/network access.
