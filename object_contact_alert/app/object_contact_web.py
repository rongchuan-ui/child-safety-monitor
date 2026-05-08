"""
Web UI for Feature 1 - object contact alert.

Run:
    /Users/shirch/vscode101/.venv/bin/python object_contact_web.py

Open:
    http://127.0.0.1:5003
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field

import cv2
import numpy as np
from flask import Flask, Response, jsonify, render_template_string, request

from object_contact_core import (
    COCO_DANGER_CLASSES,
    CUSTOM_DANGER_CLASSES,
    DEFAULT_MONITORED_CLASSES,
    ContactDetection,
    ObjectContactConfig,
    detect_contacts,
    load_models,
    open_camera,
    play_alert_sound,
)


app = Flask(__name__)


BUFFER_PRESETS = {
    "small": 30,
    "medium": 60,
    "large": 100,
}

BUFFER_LABELS = {
    "small": "Small",
    "medium": "Medium",
    "large": "Large",
}


@dataclass
class WebState:
    view_mode: str = "debug"
    monitored_classes: set[str] = field(default_factory=lambda: set(DEFAULT_MONITORED_CLASSES))
    proximity_px: int = BUFFER_PRESETS["medium"]
    alert_classes: list[str] = field(default_factory=list)
    detected_dangers: list[str] = field(default_factory=list)
    detected_people: int = 0
    last_alert_time: float = 0.0
    lock: threading.Lock = field(default_factory=threading.Lock)


CONFIG = ObjectContactConfig()
STATE = WebState()
SEG_MODEL = None
WORLD_MODEL = None
CAP = None
FRAME_COUNT = 0
CACHED_WORLD: list[tuple[str, tuple[int, int, int, int]]] = []


PAGE_HTML = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Object Contact Monitor</title>
  <style>
    :root {
      --bg: #0f1115;
      --panel: #181b20;
      --panel-2: #20242b;
      --line: #303640;
      --text: #f2f5f7;
      --muted: #aab3bf;
      --green: #2f9d62;
      --red: #d64545;
      --orange: #df9b2d;
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
      grid-template-columns: minmax(560px, 1fr) 380px;
    }
    .stage {
      padding: 22px;
      display: flex;
      align-items: center;
      justify-content: center;
      background: #0f1115;
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
    }
    .top-badge {
      position: absolute;
      left: 14px;
      top: 14px;
      padding: 9px 12px;
      border-radius: 6px;
      background: rgba(18, 22, 27, .84);
      border: 1px solid rgba(255,255,255,.14);
      font-size: 14px;
      color: var(--text);
      backdrop-filter: blur(6px);
    }
    .alarm .top-badge { background: rgba(160, 30, 30, .92); }
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
    .status-card, .field {
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
    .value { font-size: 14px; font-weight: 620; text-align: right; max-width: 210px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
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
    button.active, button.primary {
      background: var(--blue);
      border-color: var(--blue);
      color: white;
    }
    .segmented, .preset-row {
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 8px;
    }
    .segmented.two { grid-template-columns: 1fr 1fr; }
    .field-head {
      display: flex;
      justify-content: space-between;
      gap: 12px;
      margin-bottom: 10px;
      color: var(--muted);
      font-size: 13px;
    }
    .checks {
      display: grid;
      gap: 8px;
      max-height: 250px;
      overflow: auto;
      padding-right: 4px;
    }
    .check {
      display: flex;
      align-items: center;
      gap: 8px;
      min-height: 28px;
      color: var(--text);
      font-size: 13px;
    }
    input[type="checkbox"] { width: 16px; height: 16px; accent-color: var(--blue); }
    .hint {
      margin: 18px;
      padding: 14px 16px;
      border-radius: 8px;
      border: 1px solid rgba(223,155,45,.35);
      background: rgba(223,155,45,.10);
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
    @media (max-width: 980px) {
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
        <div id="badge" class="top-badge">Monitoring for object contact</div>
      </div>
    </section>
    <aside class="panel">
      <div class="header">
        <h1 class="title">Object Contact Monitor</h1>
        <p class="subtitle">Detect when a person gets close to selected dangerous objects.</p>
      </div>

      <div class="status-card">
        <div class="status-row"><span class="label">Status</span><span id="statusPill" class="pill">MONITORING</span></div>
        <div class="status-row"><span class="label">View</span><span id="viewValue" class="value">Debug</span></div>
        <div class="status-row"><span class="label">People</span><span id="peopleValue" class="value">0</span></div>
        <div class="status-row"><span class="label">Danger items</span><span id="dangerValue" class="value">None</span></div>
        <div class="status-row"><span class="label">Alert</span><span id="alertValue" class="value">None</span></div>
      </div>

      <div class="controls">
        <div class="segmented two">
          <button id="debugBtn" class="active" onclick="setView('debug')">Debug</button>
          <button id="cleanBtn" onclick="setView('clean')">Clean</button>
        </div>
      </div>

      <div class="field">
        <div class="field-head">
          <span>Alert buffer</span>
          <strong id="bufferValue">Medium</strong>
        </div>
        <div class="preset-row">
          <button id="bufferSmall" onclick="setBuffer('small')">Small</button>
          <button id="bufferMedium" class="active" onclick="setBuffer('medium')">Medium</button>
          <button id="bufferLarge" onclick="setBuffer('large')">Large</button>
        </div>
      </div>

      <div class="field">
        <div class="field-head">
          <span>Monitored objects</span>
          <strong id="selectedCount">0</strong>
        </div>
        <div id="checks" class="checks"></div>
      </div>

      <p class="hint">Debug view shows masks, labels, boxes, and object buffer boxes. Clean view hides model overlays unless an alarm triggers.</p>
      <div class="footer">COCO objects use masks. YOLOWorld objects use expanded boxes.</div>
    </aside>
  </main>

  <script>
    const shell = document.getElementById("videoShell");

    async function postJSON(url, data = {}) {
      const res = await fetch(url, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(data)
      });
      return res.json();
    }

    async function setView(mode) { await postJSON("/api/view", { mode }); await refreshStatus(); }
    async function setBuffer(size) { await postJSON("/api/buffer", { size }); await refreshStatus(); }
    async function setMonitored(name, enabled) { await postJSON("/api/monitored", { name, enabled }); await refreshStatus(); }

    function renderChecks(s) {
      const box = document.getElementById("checks");
      if (box.childElementCount) return;
      for (const item of s.all_classes) {
        const id = `item-${item.replace(/[^a-z0-9]/gi, "-")}`;
        const row = document.createElement("label");
        row.className = "check";
        row.innerHTML = `<input id="${id}" type="checkbox" checked> <span>${item}</span>`;
        box.appendChild(row);
        row.querySelector("input").addEventListener("change", (event) => setMonitored(item, event.target.checked));
      }
    }

    async function refreshStatus() {
      const res = await fetch("/api/status");
      const s = await res.json();
      renderChecks(s);
      const alerting = s.alert_classes.length > 0;
      shell.classList.toggle("alarm", alerting);
      document.getElementById("statusPill").textContent = alerting ? "ALERT" : "MONITORING";
      document.getElementById("statusPill").classList.toggle("alert", alerting);
      document.getElementById("viewValue").textContent = s.view_mode === "debug" ? "Debug" : "Clean";
      document.getElementById("peopleValue").textContent = s.people_count;
      document.getElementById("dangerValue").textContent = s.detected_dangers.length ? s.detected_dangers.join(", ") : "None";
      document.getElementById("alertValue").textContent = alerting ? s.alert_classes.join(", ") : "None";
      document.getElementById("bufferValue").textContent = s.buffer_label;
      document.getElementById("selectedCount").textContent = `${s.monitored_classes.length}/${s.all_classes.length}`;
      for (const key of ["Small", "Medium", "Large"]) {
        document.getElementById(`buffer${key}`).classList.toggle("active", s.buffer_size === key.toLowerCase());
      }
      document.getElementById("debugBtn").classList.toggle("active", s.view_mode === "debug");
      document.getElementById("cleanBtn").classList.toggle("active", s.view_mode === "clean");
      for (const item of s.all_classes) {
        const id = `item-${item.replace(/[^a-z0-9]/gi, "-")}`;
        const input = document.getElementById(id);
        if (input) input.checked = s.monitored_classes.includes(item);
      }
      document.getElementById("badge").textContent = alerting
        ? `WARNING: person near ${s.alert_classes.join(", ")}`
        : "Monitoring for object contact";
    }

    setInterval(refreshStatus, 700);
    refreshStatus();
  </script>
</body>
</html>
"""


def ensure_runtime() -> None:
    global SEG_MODEL, WORLD_MODEL, CAP
    if SEG_MODEL is None or WORLD_MODEL is None:
        print("Loading YOLOv8x-seg and YOLOWorld...")
        SEG_MODEL, WORLD_MODEL = load_models(CONFIG)
    if CAP is None:
        CAP = open_camera(CONFIG)
        if not CAP.isOpened():
            raise RuntimeError("Cannot open camera. Check index and permissions.")
        actual_w = int(CAP.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(CAP.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"Camera: {actual_w} x {actual_h}")


def overlay_mask(frame: np.ndarray, mask: np.ndarray, color: np.ndarray, alpha: float = 0.36) -> None:
    frame[mask] = (frame[mask] * (1.0 - alpha) + color * alpha).astype(np.uint8)


def draw_label(frame: np.ndarray, box: tuple[int, int, int, int], label: str, color: tuple[int, int, int]) -> None:
    x1, y1, x2, y2 = box
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.62, 2)
    top = max(8, y1 - th - 8)
    bottom = top + th + 8
    cv2.rectangle(frame, (x1, top), (min(x1 + tw + 6, frame.shape[1] - 1), bottom), color, -1)
    cv2.putText(frame, label, (x1 + 3, bottom - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA)


def draw_detection(frame: np.ndarray, detection: ContactDetection, show_debug: bool) -> None:
    if not show_debug and not detection.alert:
        return

    if detection.alert:
        color = (0, 0, 255)
        label = f"DANGER: {detection.class_name}"
    elif detection.source == "world":
        color = (0, 165, 255)
        label = detection.class_name
    else:
        color = (0, 190, 80)
        label = detection.class_name

    if show_debug and detection.mask is not None:
        mask_color = np.array([0, 0, 220], dtype=np.uint8) if detection.alert else np.array([70, 150, 230], dtype=np.uint8)
        overlay_mask(frame, detection.mask, mask_color)

    if show_debug and detection.expanded_box is not None and detection.mask is None:
        ex1, ey1, ex2, ey2 = detection.expanded_box
        cv2.rectangle(frame, (ex1, ey1), (ex2, ey2), (0, 210, 255), 1)

    draw_label(frame, detection.box, label, color)


def render_frame(frame: np.ndarray, detections) -> np.ndarray:
    with STATE.lock:
        view_mode = STATE.view_mode
        alerts = list(STATE.alert_classes)

    output = frame.copy()
    show_debug = view_mode == "debug"

    for person in detections.persons:
        if show_debug:
            if person.mask is not None:
                overlay_mask(output, person.mask, np.array([220, 80, 50], dtype=np.uint8), 0.30)
            draw_label(output, person.box, "person", (0, 185, 80))

    for danger in detections.dangers:
        draw_detection(output, danger, show_debug)

    if alerts:
        cv2.rectangle(output, (0, 0), (output.shape[1] - 1, output.shape[0] - 1), (0, 0, 255), 8)

    return output


def video_frames():
    global FRAME_COUNT, CACHED_WORLD
    ensure_runtime()
    while True:
        ok, frame = CAP.read()
        if not ok:
            time.sleep(0.05)
            continue

        FRAME_COUNT += 1
        with STATE.lock:
            monitored = set(STATE.monitored_classes)
            proximity_px = STATE.proximity_px

        detections = detect_contacts(
            frame,
            SEG_MODEL,
            WORLD_MODEL,
            CONFIG,
            monitored,
            proximity_px,
            FRAME_COUNT,
            CACHED_WORLD,
        )
        CACHED_WORLD = detections.cached_world

        with STATE.lock:
            STATE.detected_people = len(detections.persons)
            STATE.detected_dangers = [d.class_name for d in detections.dangers]
            STATE.alert_classes = detections.alerts
            should_alert = bool(detections.alerts) and time.time() - STATE.last_alert_time > CONFIG.alert_cooldown
            if should_alert:
                STATE.last_alert_time = time.time()

        if should_alert:
            play_alert_sound()
            print(f"[ALERT] Person near: {', '.join(detections.alerts)}")

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


@app.post("/api/view")
def set_view():
    data = request.get_json(force=True)
    mode = data.get("mode", "debug")
    if mode not in {"debug", "clean"}:
        return jsonify(ok=False, error="Invalid view mode"), 400
    with STATE.lock:
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
    return jsonify(ok=True)


@app.post("/api/monitored")
def set_monitored():
    data = request.get_json(force=True)
    name = data.get("name")
    enabled = bool(data.get("enabled", True))
    all_classes = COCO_DANGER_CLASSES | set(CUSTOM_DANGER_CLASSES)
    if name not in all_classes:
        return jsonify(ok=False, error="Unknown class"), 400
    with STATE.lock:
        if enabled:
            STATE.monitored_classes.add(name)
        else:
            STATE.monitored_classes.discard(name)
    return jsonify(ok=True)


@app.get("/api/status")
def status():
    all_classes = sorted(COCO_DANGER_CLASSES) + CUSTOM_DANGER_CLASSES
    with STATE.lock:
        buffer_size = next((k for k, v in BUFFER_PRESETS.items() if v == STATE.proximity_px), "custom")
        payload = {
            "view_mode": STATE.view_mode,
            "people_count": STATE.detected_people,
            "detected_dangers": list(dict.fromkeys(STATE.detected_dangers)),
            "alert_classes": list(dict.fromkeys(STATE.alert_classes)),
            "monitored_classes": sorted(STATE.monitored_classes),
            "all_classes": all_classes,
            "buffer_size": buffer_size,
            "buffer_label": BUFFER_LABELS.get(buffer_size, "Custom"),
            "proximity_px": STATE.proximity_px,
        }
    return jsonify(payload)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5003, debug=False, threaded=True)
