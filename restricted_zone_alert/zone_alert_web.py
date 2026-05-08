"""
Web UI for restricted-zone monitoring.

Run:
    /Users/shirch/vscode101/.venv/bin/python zone_alert_web.py

Open:
    http://127.0.0.1:5002
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field

import cv2
import numpy as np
from flask import Flask, Response, jsonify, render_template_string, request

from zone_alert_core import (
    TARGET_CLASSES,
    ZoneAlertConfig,
    build_zone_mask,
    detect_targets,
    expand_zone_mask,
    load_model,
    open_camera,
    play_alert_sound,
)


app = Flask(__name__)


@dataclass
class WebState:
    zone_points: list[tuple[int, int]] = field(default_factory=list)
    zone_locked: bool = False
    zone_mask: np.ndarray | None = None
    expanded_zone_mask: np.ndarray | None = None
    proximity_px: int = 40
    view_mode: str = "debug"
    frame_w: int = 0
    frame_h: int = 0
    alert_classes: list[str] = field(default_factory=list)
    detected_count: int = 0
    last_alert_time: float = 0.0
    lock: threading.Lock = field(default_factory=threading.Lock)


CONFIG = ZoneAlertConfig()
STATE = WebState()
MODEL = None
CAP = None

BUFFER_PRESETS = {
    "small": 20,
    "medium": 50,
    "large": 90,
}
BUFFER_LABELS = {
    "small": "Small",
    "medium": "Medium",
    "large": "Large",
}


PAGE_HTML = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Restricted Zone Monitor</title>
  <style>
    :root {
      --bg: #101216;
      --panel: #181b20;
      --panel-2: #20242b;
      --line: #303640;
      --text: #f2f5f7;
      --muted: #aab3bf;
      --green: #2f9d62;
      --red: #d64545;
      --amber: #d99a2b;
      --blue: #4f7cff;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100vh;
      background: var(--bg);
      color: var(--text);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    .app {
      min-height: 100vh;
      display: grid;
      grid-template-columns: minmax(520px, 1fr) 340px;
      gap: 0;
    }
    .stage {
      padding: 22px;
      display: flex;
      align-items: center;
      justify-content: center;
      background:
        linear-gradient(180deg, rgba(255,255,255,.03), rgba(255,255,255,0)),
        #0f1115;
    }
    .video-shell {
      width: min(100%, 1280px);
      position: relative;
      border: 1px solid var(--line);
      background: #06070a;
      box-shadow: 0 18px 50px rgba(0,0,0,.35);
    }
    #video {
      display: block;
      width: 100%;
      height: auto;
      cursor: crosshair;
    }
    .top-badge {
      position: absolute;
      left: 14px;
      top: 14px;
      padding: 9px 12px;
      border-radius: 6px;
      background: rgba(18, 22, 27, .82);
      border: 1px solid rgba(255,255,255,.14);
      font-size: 14px;
      color: var(--text);
      backdrop-filter: blur(6px);
    }
    .alarm .top-badge {
      background: rgba(160, 30, 30, .9);
    }
    .panel {
      border-left: 1px solid var(--line);
      background: var(--panel);
      display: flex;
      flex-direction: column;
      min-width: 0;
    }
    .header {
      padding: 22px 22px 18px;
      border-bottom: 1px solid var(--line);
    }
    .title {
      margin: 0 0 8px;
      font-size: 22px;
      line-height: 1.2;
      font-weight: 680;
      letter-spacing: 0;
    }
    .subtitle {
      margin: 0;
      color: var(--muted);
      font-size: 14px;
      line-height: 1.45;
    }
    .status-card {
      margin: 18px 18px 0;
      padding: 16px;
      border: 1px solid var(--line);
      background: var(--panel-2);
      border-radius: 8px;
    }
    .status-row {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      margin-bottom: 12px;
    }
    .status-row:last-child { margin-bottom: 0; }
    .label { color: var(--muted); font-size: 13px; }
    .value { font-size: 14px; font-weight: 620; text-align: right; }
    .pill {
      display: inline-flex;
      align-items: center;
      height: 28px;
      padding: 0 10px;
      border-radius: 999px;
      background: rgba(47,157,98,.14);
      color: #7ee2aa;
      border: 1px solid rgba(126,226,170,.25);
      font-weight: 700;
      font-size: 13px;
    }
    .pill.alert {
      background: rgba(214,69,69,.15);
      color: #ff9b9b;
      border-color: rgba(255,155,155,.3);
    }
    .controls {
      padding: 18px;
      display: grid;
      gap: 10px;
    }
    .field {
      margin: 0 18px 6px;
      padding: 14px 16px;
      border-radius: 8px;
      border: 1px solid var(--line);
      background: var(--panel-2);
    }
    .field-head {
      display: flex;
      justify-content: space-between;
      gap: 12px;
      margin-bottom: 10px;
      color: var(--muted);
      font-size: 13px;
    }
    .preset-row {
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 8px;
    }
    button.preset {
      height: 36px;
      font-size: 13px;
      font-weight: 650;
    }
    button.preset.active {
      background: var(--blue);
      border-color: var(--blue);
      color: #fff;
    }
    button {
      height: 42px;
      border: 1px solid var(--line);
      border-radius: 7px;
      background: #242a32;
      color: var(--text);
      font: inherit;
      font-weight: 650;
      cursor: pointer;
    }
    button:hover { background: #2b323c; }
    button.primary {
      background: var(--blue);
      border-color: var(--blue);
      color: white;
    }
    button.danger {
      background: rgba(214,69,69,.13);
      border-color: rgba(214,69,69,.38);
      color: #ffb1b1;
    }
    .segmented {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 8px;
    }
    .hint {
      margin: 0 18px 18px;
      padding: 14px 16px;
      border-radius: 8px;
      border: 1px solid rgba(217,154,43,.35);
      background: rgba(217,154,43,.10);
      color: #f0c77d;
      font-size: 13px;
      line-height: 1.45;
    }
    .footer {
      margin-top: auto;
      padding: 16px 18px 20px;
      color: var(--muted);
      font-size: 12px;
      border-top: 1px solid var(--line);
    }
    @media (max-width: 900px) {
      .app { grid-template-columns: 1fr; }
      .panel { border-left: 0; border-top: 1px solid var(--line); }
      .stage { padding: 12px; }
    }
  </style>
</head>
<body>
  <main class="app">
    <section class="stage">
      <div id="videoShell" class="video-shell">
        <img id="video" src="/video_feed" alt="Live camera feed">
        <div id="badge" class="top-badge">Click video to draw a restricted zone</div>
      </div>
    </section>
    <aside class="panel">
      <div class="header">
        <h1 class="title">Restricted Zone Monitor</h1>
        <p class="subtitle">Draw a zone on the video. People or pets entering it will trigger an alert.</p>
      </div>

      <div class="status-card">
        <div class="status-row"><span class="label">Status</span><span id="statusPill" class="pill">MONITORING</span></div>
        <div class="status-row"><span class="label">View</span><span id="viewValue" class="value">Debug</span></div>
        <div class="status-row"><span class="label">Zone</span><span id="zoneValue" class="value">Drawing</span></div>
        <div class="status-row"><span class="label">Detected</span><span id="detectedValue" class="value">0</span></div>
        <div class="status-row"><span class="label">Inside</span><span id="insideValue" class="value">None</span></div>
      </div>

      <div class="controls">
        <button class="primary" onclick="lockZone()">Lock Zone</button>
        <div class="segmented">
          <button onclick="setView('debug')">Debug</button>
          <button onclick="setView('clean')">Clean</button>
        </div>
        <button onclick="undoPoint()">Undo Point</button>
        <button class="danger" onclick="clearZone()">Clear Zone</button>
      </div>

      <div class="field">
        <div class="field-head">
          <span>Alert buffer</span>
          <strong id="bufferValue">Medium</strong>
        </div>
        <div class="preset-row">
          <button id="bufferSmall" class="preset" onclick="setBuffer('small')">Small</button>
          <button id="bufferMedium" class="preset active" onclick="setBuffer('medium')">Medium</button>
          <button id="bufferLarge" class="preset" onclick="setBuffer('large')">Large</button>
        </div>
      </div>

      <p class="hint">Alert buffer controls how early the alarm starts outside the zone. Medium is recommended.</p>
      <div class="footer">Targets: person, cat, dog, bird</div>
    </aside>
  </main>

  <script>
    const video = document.getElementById("video");
    const shell = document.getElementById("videoShell");

    async function postJSON(url, data = {}) {
      const res = await fetch(url, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(data)
      });
      return res.json();
    }

    video.addEventListener("click", async (event) => {
      const rect = video.getBoundingClientRect();
      const naturalW = video.naturalWidth || rect.width;
      const naturalH = video.naturalHeight || rect.height;
      const x = Math.round((event.clientX - rect.left) * naturalW / rect.width);
      const y = Math.round((event.clientY - rect.top) * naturalH / rect.height);
      await postJSON("/api/zone/add", { x, y });
      await refreshStatus();
    });

    async function lockZone() { await postJSON("/api/zone/lock"); await refreshStatus(); }
    async function undoPoint() { await postJSON("/api/zone/undo"); await refreshStatus(); }
    async function clearZone() { await postJSON("/api/zone/clear"); await refreshStatus(); }
    async function setView(mode) { await postJSON("/api/view", { mode }); await refreshStatus(); }
    async function setBuffer(size) { await postJSON("/api/buffer", { size }); await refreshStatus(); }

    async function refreshStatus() {
      const res = await fetch("/api/status");
      const s = await res.json();
      const alerting = s.alert_classes.length > 0;
      shell.classList.toggle("alarm", alerting);
      document.getElementById("statusPill").textContent = alerting ? "ALERT" : "MONITORING";
      document.getElementById("statusPill").classList.toggle("alert", alerting);
      document.getElementById("viewValue").textContent = s.view_mode === "debug" ? "Debug" : "Clean";
      document.getElementById("zoneValue").textContent = s.zone_locked ? `Locked (${s.zone_points} pts)` : `Drawing (${s.zone_points} pts)`;
      document.getElementById("detectedValue").textContent = s.detected_count;
      document.getElementById("insideValue").textContent = alerting ? s.alert_classes.join(", ") : "None";
      document.getElementById("bufferValue").textContent = s.buffer_label;
      for (const key of ["Small", "Medium", "Large"]) {
        document.getElementById(`buffer${key}`).classList.toggle("active", s.buffer_size === key.toLowerCase());
      }
      document.getElementById("badge").textContent = alerting
        ? `WARNING: ${s.alert_classes.join(", ")} entered restricted zone`
        : (s.zone_locked ? "Zone locked" : "Click video to draw a restricted zone");
    }

    setInterval(refreshStatus, 700);
    refreshStatus();
  </script>
</body>
</html>
"""


def ensure_runtime() -> None:
    global MODEL, CAP
    if MODEL is None:
        print("Loading YOLO model...")
        MODEL = load_model(CONFIG)
    if CAP is None:
        CAP = open_camera(CONFIG)
        if not CAP.isOpened():
            raise RuntimeError("Cannot open camera. Check index and permissions.")
        actual_w = int(CAP.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(CAP.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"Camera: {actual_w} x {actual_h}")


def draw_zone(frame: np.ndarray, points: list[tuple[int, int]], locked: bool) -> None:
    if not points:
        return
    pts = np.array(points, dtype=np.int32)
    if locked and len(points) >= 3:
        overlay = frame.copy()
        cv2.fillPoly(overlay, [pts], (0, 165, 255))
        cv2.addWeighted(overlay, 0.25, frame, 0.75, 0, frame)
        cv2.polylines(frame, [pts], True, (0, 140, 255), 3)
    else:
        cv2.polylines(frame, [pts], False, (0, 210, 255), 2)
    for idx, point in enumerate(points):
        cv2.circle(frame, point, 5, (0, 255, 255), -1)
        cv2.putText(frame, str(idx + 1), (point[0] + 7, point[1] - 7),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)


def draw_proximity_boundary(frame: np.ndarray, expanded_mask: np.ndarray | None) -> None:
    if expanded_mask is None:
        return
    contours, _ = cv2.findContours(expanded_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(frame, contours, -1, (0, 210, 255), 1)


def overlay_target_mask(frame: np.ndarray, mask: np.ndarray, alert: bool) -> None:
    color = np.array([0, 0, 220], dtype=np.uint8) if alert else np.array([220, 85, 55], dtype=np.uint8)
    frame[mask] = (frame[mask] * 0.62 + color * 0.38).astype(np.uint8)


def draw_detection(frame: np.ndarray, detection) -> None:
    x1, y1, x2, y2 = detection.box
    color = (0, 0, 255) if detection.in_zone else (0, 190, 80)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    label = f"ZONE ALERT: {detection.class_name}" if detection.in_zone else detection.class_name
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.62, 2)
    top = max(8, y1 - th - 8)
    bottom = top + th + 8
    cv2.rectangle(frame, (x1, top), (min(x1 + tw + 6, frame.shape[1] - 1), bottom), color, -1)
    cv2.putText(frame, label, (x1 + 3, bottom - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA)


def render_frame(frame: np.ndarray, detections) -> np.ndarray:
    with STATE.lock:
        points = list(STATE.zone_points)
        locked = STATE.zone_locked
        view_mode = STATE.view_mode
        alert_classes = list(STATE.alert_classes)
        expanded_zone_mask = STATE.expanded_zone_mask.copy() if STATE.expanded_zone_mask is not None else None

    output = frame.copy()
    show_debug = view_mode == "debug" or not locked

    if show_debug:
        for detection in detections:
            if detection.mask is not None:
                overlay_target_mask(output, detection.mask, detection.in_zone)
            draw_detection(output, detection)
        draw_proximity_boundary(output, expanded_zone_mask if locked else None)
        draw_zone(output, points, locked)

    if alert_classes:
        cv2.rectangle(output, (0, 0), (output.shape[1] - 1, output.shape[0] - 1), (0, 0, 255), 8)

    return output


def video_frames():
    ensure_runtime()
    while True:
        ok, frame = CAP.read()
        if not ok:
            time.sleep(0.05)
            continue

        frame_h, frame_w = frame.shape[:2]
        with STATE.lock:
            STATE.frame_w = frame_w
            STATE.frame_h = frame_h
            if STATE.zone_locked and STATE.zone_mask is None:
                STATE.zone_mask = build_zone_mask(STATE.zone_points, frame_w, frame_h)
                STATE.expanded_zone_mask = expand_zone_mask(STATE.zone_mask, STATE.proximity_px)
            if STATE.zone_locked and STATE.zone_mask is not None and STATE.expanded_zone_mask is None:
                STATE.expanded_zone_mask = expand_zone_mask(STATE.zone_mask, STATE.proximity_px)
            zone_mask = STATE.expanded_zone_mask if STATE.zone_locked else None

        detections = detect_targets(MODEL, frame, CONFIG, zone_mask)
        alert_classes = [d.class_name for d in detections if d.in_zone]

        with STATE.lock:
            STATE.detected_count = len(detections)
            STATE.alert_classes = alert_classes
            should_alert = bool(alert_classes) and time.time() - STATE.last_alert_time > CONFIG.alert_cooldown
            if should_alert:
                STATE.last_alert_time = time.time()

        if should_alert:
            play_alert_sound()
            print(f"[ALERT] Restricted zone entered by: {', '.join(sorted(set(alert_classes)))}")

        output = render_frame(frame, detections)
        ok, buffer = cv2.imencode(".jpg", output, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
        if not ok:
            continue
        yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"


@app.route("/")
def index():
    return render_template_string(PAGE_HTML)


@app.route("/video_feed")
def video_feed():
    return Response(video_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.post("/api/zone/add")
def add_zone_point():
    data = request.get_json(force=True)
    x = int(data.get("x", 0))
    y = int(data.get("y", 0))
    with STATE.lock:
        if not STATE.zone_locked:
            STATE.zone_points.append((x, y))
            STATE.zone_mask = None
            STATE.expanded_zone_mask = None
            print(f"Added zone point {len(STATE.zone_points)}: {(x, y)}")
    return jsonify(ok=True)


@app.post("/api/zone/lock")
def lock_zone():
    with STATE.lock:
        if len(STATE.zone_points) >= 3:
            STATE.zone_locked = True
            STATE.zone_mask = None
            STATE.expanded_zone_mask = None
    return jsonify(ok=True)


@app.post("/api/zone/undo")
def undo_zone_point():
    with STATE.lock:
        if not STATE.zone_locked and STATE.zone_points:
            STATE.zone_points.pop()
            STATE.zone_mask = None
            STATE.expanded_zone_mask = None
    return jsonify(ok=True)


@app.post("/api/zone/clear")
def clear_zone():
    with STATE.lock:
        STATE.zone_points.clear()
        STATE.zone_locked = False
        STATE.zone_mask = None
        STATE.expanded_zone_mask = None
        STATE.view_mode = "debug"
        STATE.alert_classes = []
    return jsonify(ok=True)


@app.post("/api/view")
def set_view():
    data = request.get_json(force=True)
    mode = data.get("mode", "debug")
    if mode not in {"debug", "clean"}:
        return jsonify(ok=False, error="Invalid view mode"), 400
    with STATE.lock:
        if STATE.zone_locked or mode == "debug":
            STATE.view_mode = mode
    return jsonify(ok=True)


@app.post("/api/buffer")
def set_buffer():
    data = request.get_json(force=True)
    size = data.get("size", "medium")
    if size not in BUFFER_PRESETS:
        return jsonify(ok=False, error="Invalid buffer size"), 400
    with STATE.lock:
        STATE.proximity_px = BUFFER_PRESETS[size]
        if STATE.zone_mask is not None:
            STATE.expanded_zone_mask = expand_zone_mask(STATE.zone_mask, STATE.proximity_px)
    return jsonify(ok=True)


@app.get("/api/status")
def status():
    with STATE.lock:
        buffer_size = next((k for k, v in BUFFER_PRESETS.items() if v == STATE.proximity_px), "custom")
        payload = {
            "zone_points": len(STATE.zone_points),
            "zone_locked": STATE.zone_locked,
            "view_mode": STATE.view_mode,
            "detected_count": STATE.detected_count,
            "alert_classes": list(dict.fromkeys(STATE.alert_classes)),
            "proximity_px": STATE.proximity_px,
            "buffer_size": buffer_size,
            "buffer_label": BUFFER_LABELS.get(buffer_size, "Custom"),
            "frame_w": STATE.frame_w,
            "frame_h": STATE.frame_h,
        }
    return jsonify(payload)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5002, debug=False, threaded=True)
