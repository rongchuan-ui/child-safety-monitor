# Model Weights

This project expects two model files in the repository root:

```text
yolov8x-seg.pt
yolov8x-worldv2.pt
```

They are intentionally kept at the root because the combined app and standalone feature apps all resolve weights from that location.

## GitHub Upload

Both files are larger than GitHub's normal 100 MB file limit. Use one of these options:

### Option A: Git LFS

```bash
git lfs install
git lfs track "*.pt"
git add .gitattributes
git add yolov8x-seg.pt yolov8x-worldv2.pt
git commit -m "Add model weights with Git LFS"
```

After cloning:

```bash
git lfs pull
```

### Option B: Release Assets

Upload the `.pt` files as GitHub Release assets and ask users to download them into the project root.

### Option C: External Link

Host the weights externally and document the download links in the README or release notes.

## Verification

Before running the app, the root folder should include:

```text
combined_monitor.py
mobile_camera_monitor.py
yolov8x-seg.pt
yolov8x-worldv2.pt
```
