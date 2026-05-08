# Restricted Zone Alert

This feature triggers an alert when a person or pet enters a restricted area drawn on the camera feed.

For the full demo with both features together, run the combined monitor from the repository root:

```bash
python combined_monitor.py
```

Open:

```text
http://127.0.0.1:5004
```

## Standalone Run

Use this when you only want to test the restricted zone feature.

```bash
cd restricted_zone_alert
python zone_alert_web.py
```

Open:

```text
http://127.0.0.1:5002
```

## File Map

```text
restricted_zone_alert/
├── README.md
├── zone_alert_core.py   # model, camera, zone mask, and proximity logic
└── zone_alert_web.py    # standalone Flask UI, port 5002
```

## Web UI Features

- Click directly on the video to draw a polygon zone.
- Lock, undo, and clear the zone.
- Debug view shows masks, labels, detection boxes, zone boundary, and buffer boundary.
- Clean view hides model overlays and the drawn zone unless an alert is active.
- Small / Medium / Large zone buffer selector.

## How It Works

```text
1. User clicks points on the video.
2. The points become a polygon restricted zone.
3. The polygon becomes a binary zone mask with cv2.fillPoly.
4. The selected alert buffer expands the mask with cv2.dilate.
5. YOLOv8x-seg detects person / cat / dog / bird.
6. Target masks are checked against the expanded zone mask.
7. Overlap triggers the alarm.
```

Important:

```text
YOLO does not detect the zone.
YOLO only detects person/pet targets.
The zone is user-defined geometry.
```

## Buffer Values

```text
Small = 20 px
Medium = 50 px
Large = 90 px
```

## Deployment

See the root deployment guide:

```text
../DEPLOYMENT.md
```
