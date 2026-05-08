"""
Child Safety Monitor - Mobile Camera Web UI

The phone browser captures camera frames and sends them to this Flask server.
The server runs the same YOLO logic as combined_monitor.py and returns an
annotated frame plus alert status.

Run:
    cd "/Users/shirch/Desktop/Objection Detection"
    python mobile_camera_monitor.py

Open on the phone, using the computer's LAN IP:
    https://<computer-lan-ip>:5443

The phone and computer must be on the same Wi-Fi. HTTPS is required because
mobile browsers block camera access on plain HTTP LAN pages.
"""

from __future__ import annotations

import base64
import socket
import ssl
import subprocess
import threading
import time
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, jsonify, render_template_string, request
from ultralytics import YOLO, YOLOWorld

import combined_monitor as core


ssl._create_default_https_context = ssl._create_unverified_context


app = Flask(__name__)
PROCESS_LOCK = threading.Lock()


PAGE_HTML = """
<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
  <title>Mobile Safety Monitor</title>
  <style>
    :root {
      --bg: #0f1115;
      --panel: #181b20;
      --panel2: #20242b;
      --line: #303640;
      --text: #f2f5f7;
      --muted: #aab3bf;
      --blue: #4f7cff;
      --red: #d64545;
      --green: #2f9d62;
      --amber: #d99a2b;
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
      grid-template-rows: auto 1fr auto;
    }
    .bar {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 10px;
      padding: 10px 12px;
      border-bottom: 1px solid var(--line);
      background: var(--panel);
      position: sticky;
      top: 0;
      z-index: 5;
    }
    .title {
      font-size: 15px;
      font-weight: 750;
      white-space: nowrap;
    }
    .pill {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      min-width: 82px;
      height: 28px;
      padding: 0 10px;
      border-radius: 999px;
      border: 1px solid rgba(126,226,170,.25);
      background: rgba(47,157,98,.14);
      color: #7ee2aa;
      font-size: 12px;
      font-weight: 800;
    }
    .pill.alert {
      border-color: rgba(255,155,155,.34);
      background: rgba(214,69,69,.18);
      color: #ff9b9b;
    }
    .stage {
      display: flex;
      align-items: center;
      justify-content: center;
      padding: 10px;
      min-height: 0;
    }
    .video-shell {
      position: relative;
      width: 100%;
      max-width: 1100px;
      border: 1px solid var(--line);
      background: #05070a;
    }
    .video-shell.alarm {
      border-color: var(--red);
      box-shadow: 0 0 0 4px rgba(214, 69, 69, .35);
    }
    #processed {
      display: block;
      width: 100%;
      height: auto;
      min-height: 240px;
      object-fit: contain;
      cursor: crosshair;
      background: #05070a;
    }
    #camera, #captureCanvas { display: none; }
    .badge {
      position: absolute;
      left: 10px;
      top: 10px;
      max-width: calc(100% - 20px);
      padding: 8px 10px;
      border-radius: 6px;
      border: 1px solid rgba(255,255,255,.14);
      background: rgba(18,22,27,.86);
      color: var(--text);
      font-size: 12px;
      line-height: 1.35;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }
    .controls {
      border-top: 1px solid var(--line);
      background: var(--panel);
      padding: 10px;
      display: grid;
      gap: 10px;
    }
    .grid2, .grid3 {
      display: grid;
      gap: 8px;
    }
    .grid2 { grid-template-columns: repeat(2, minmax(0, 1fr)); }
    .grid3 { grid-template-columns: repeat(3, minmax(0, 1fr)); }
    button {
      height: 38px;
      border: 1px solid var(--line);
      border-radius: 7px;
      background: #242a32;
      color: var(--text);
      font: inherit;
      font-size: 13px;
      font-weight: 700;
    }
    button.active, button.primary {
      background: var(--blue);
      border-color: var(--blue);
      color: white;
    }
    button.danger {
      background: rgba(214,69,69,.13);
      border-color: rgba(214,69,69,.38);
      color: #ffb1b1;
    }
    .section {
      border: 1px solid var(--line);
      background: var(--panel2);
      border-radius: 8px;
      padding: 10px;
      display: grid;
      gap: 8px;
    }
    .head {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 10px;
      color: var(--muted);
      font-size: 12px;
      font-weight: 750;
    }
    .checks {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 7px;
      max-height: 130px;
      overflow: auto;
    }
    .check {
      display: flex;
      align-items: center;
      gap: 6px;
      min-width: 0;
      font-size: 12px;
      color: var(--text);
    }
    .check span {
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }
    input[type="checkbox"] {
      width: 15px;
      height: 15px;
      accent-color: var(--blue);
      flex: 0 0 auto;
    }
    .status {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 8px;
      color: var(--muted);
      font-size: 12px;
    }
    .status strong {
      color: var(--text);
      display: block;
      margin-top: 2px;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }
    @media (min-width: 880px) {
      .app {
        grid-template-columns: minmax(520px, 1fr) 360px;
        grid-template-rows: auto 1fr;
      }
      .bar { grid-column: 1 / -1; }
      .controls {
        border-top: 0;
        border-left: 1px solid var(--line);
        align-content: start;
        overflow: auto;
      }
      .stage { padding: 18px; }
      .checks { grid-template-columns: 1fr; max-height: 260px; }
    }
  </style>
</head>
<body>
  <main class="app">
    <header class="bar">
      <div class="title">手机摄像头监控</div>
      <span id="statusPill" class="pill">启动中</span>
    </header>

    <section class="stage">
      <div id="shell" class="video-shell">
        <img id="processed" alt="processed camera frame">
        <div id="badge" class="badge">等待手机摄像头权限</div>
      </div>
      <video id="camera" autoplay playsinline muted></video>
      <canvas id="captureCanvas"></canvas>
    </section>

    <aside class="controls">
      <div class="section">
        <div class="head"><span>视图</span><span id="fpsValue">0 fps</span></div>
        <div class="grid2">
          <button id="debugBtn" class="active" onclick="setView('debug')">Debug</button>
          <button id="cleanBtn" onclick="setView('clean')">Clean</button>
        </div>
        <div class="grid2">
          <button onclick="startCamera('environment')" class="primary">后置摄像头</button>
          <button onclick="startCamera('user')">前置摄像头</button>
        </div>
      </div>

      <div class="section">
        <div class="head"><span>Zone Alert</span><span id="zoneStatus">No zone</span></div>
        <div class="grid2">
          <button class="primary" onclick="lockZone()">Lock</button>
          <button onclick="undoPoint()">Undo</button>
          <button class="danger" onclick="clearZone()" style="grid-column:span 2">Clear Zone</button>
        </div>
        <div class="head"><span>Zone buffer</span><strong id="zoneBufVal">Medium</strong></div>
        <div class="grid3">
          <button id="zs" onclick="setZoneBuf('small')">Small</button>
          <button id="zm" class="active" onclick="setZoneBuf('medium')">Medium</button>
          <button id="zl" onclick="setZoneBuf('large')">Large</button>
        </div>
      </div>

      <div class="section">
        <div class="head"><span>Contact Alert</span><span id="contactStatus">None</span></div>
        <div class="head"><span>Contact buffer</span><strong id="contactBufVal">Medium</strong></div>
        <div class="grid3">
          <button id="cs" onclick="setContactBuf('small')">Small</button>
          <button id="cm" class="active" onclick="setContactBuf('medium')">Medium</button>
          <button id="cl" onclick="setContactBuf('large')">Large</button>
        </div>
      </div>

      <div class="section">
        <div class="head"><span>Monitored objects</span><span id="selectedCount">0/0</span></div>
        <div id="checks" class="checks"></div>
      </div>

      <div class="section">
        <div class="status">
          <div>People<strong id="peopleValue">0</strong></div>
          <div>Dangers<strong id="dangerValue">None</strong></div>
          <div>Zone alert<strong id="zoneAlertValue">None</strong></div>
          <div>Contact alert<strong id="contactAlertValue">None</strong></div>
        </div>
      </div>
    </aside>
  </main>

  <script>
    const video = document.getElementById("camera");
    const canvas = document.getElementById("captureCanvas");
    const processed = document.getElementById("processed");
    const shell = document.getElementById("shell");
    const badge = document.getElementById("badge");
    let stream = null;
    let busy = false;
    let started = false;
    let lastFrameTime = performance.now();
    let targetWidth = 960;

    async function postJSON(url, data = {}) {
      const res = await fetch(url, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(data)
      });
      return res.json();
    }

    async function startCamera(facingMode = "environment") {
      if (stream) {
        for (const track of stream.getTracks()) track.stop();
      }
      try {
        stream = await navigator.mediaDevices.getUserMedia({
          video: {
            facingMode: { ideal: facingMode },
            width: { ideal: 1280 },
            height: { ideal: 720 }
          },
          audio: false
        });
        video.srcObject = stream;
        await video.play();
        started = true;
        badge.textContent = "摄像头已连接";
        sendLoop();
      } catch (err) {
        badge.textContent = "无法打开手机摄像头，请确认使用 HTTPS 并允许权限";
        console.error(err);
      }
    }

    async function sendLoop() {
      if (!started || busy || video.readyState < 2) {
        setTimeout(sendLoop, 180);
        return;
      }
      busy = true;
      try {
        const scale = Math.min(1, targetWidth / video.videoWidth);
        const w = Math.max(320, Math.round(video.videoWidth * scale));
        const h = Math.max(240, Math.round(video.videoHeight * scale));
        canvas.width = w;
        canvas.height = h;
        const ctx = canvas.getContext("2d", { alpha: false });
        ctx.drawImage(video, 0, 0, w, h);
        const blob = await new Promise(resolve => canvas.toBlob(resolve, "image/jpeg", 0.72));
        const form = new FormData();
        form.append("frame", blob, "frame.jpg");
        const res = await fetch("/api/frame", { method: "POST", body: form });
        if (res.ok) {
          const payload = await res.json();
          processed.src = "data:image/jpeg;base64," + payload.image;
          updateStatus(payload.status);
          const now = performance.now();
          const fps = 1000 / Math.max(1, now - lastFrameTime);
          lastFrameTime = now;
          document.getElementById("fpsValue").textContent = fps.toFixed(1) + " fps";
        } else {
          const payload = await res.json().catch(() => ({}));
          badge.textContent = payload.error || "后端暂时无法处理帧";
        }
      } catch (err) {
        console.error(err);
        badge.textContent = "传输帧失败";
      } finally {
        busy = false;
        setTimeout(sendLoop, 120);
      }
    }

    processed.addEventListener("click", async (event) => {
      const rect = processed.getBoundingClientRect();
      const naturalW = processed.naturalWidth || canvas.width || rect.width;
      const naturalH = processed.naturalHeight || canvas.height || rect.height;
      const x = Math.round((event.clientX - rect.left) * naturalW / rect.width);
      const y = Math.round((event.clientY - rect.top) * naturalH / rect.height);
      await postJSON("/api/zone/add", { x, y });
    });

    async function setView(mode) { updateStatus(await postJSON("/api/view", { mode })); }
    async function lockZone() { updateStatus(await postJSON("/api/zone/lock")); }
    async function undoPoint() { updateStatus(await postJSON("/api/zone/undo")); }
    async function clearZone() { updateStatus(await postJSON("/api/zone/clear")); }
    async function setZoneBuf(size) { updateStatus(await postJSON("/api/zone/buffer", { size })); }
    async function setContactBuf(size) { updateStatus(await postJSON("/api/contact/buffer", { size })); }
    async function setMonitored(name, enabled) {
      updateStatus(await postJSON("/api/monitored", { name, enabled }));
    }

    function renderChecks(status) {
      const box = document.getElementById("checks");
      if (!box.childElementCount) {
        for (const item of status.all_classes) {
          const id = "c-" + item.replace(/[^a-z0-9]/gi, "-");
          const row = document.createElement("label");
          row.className = "check";
          row.innerHTML = `<input id="${id}" type="checkbox"><span title="${item}">${item}</span>`;
          box.appendChild(row);
          row.querySelector("input").addEventListener("change", ev => setMonitored(item, ev.target.checked));
        }
      }
      for (const item of status.all_classes) {
        const input = document.getElementById("c-" + item.replace(/[^a-z0-9]/gi, "-"));
        if (input) input.checked = status.monitored_classes.includes(item);
      }
      document.getElementById("selectedCount").textContent =
        `${status.monitored_classes.length}/${status.all_classes.length}`;
    }

    function toggleButtons(prefix, value) {
      const map = { s: "small", m: "medium", l: "large" };
      for (const key of Object.keys(map)) {
        document.getElementById(prefix + key).classList.toggle("active", value === map[key]);
      }
    }

    function updateStatus(status) {
      if (!status || !status.all_classes) return;
      const zoneAlert = status.zone_alerts.length > 0;
      const contactAlert = status.contact_alerts.length > 0;
      const anyAlert = zoneAlert || contactAlert;
      shell.classList.toggle("alarm", anyAlert);
      const pill = document.getElementById("statusPill");
      pill.textContent = anyAlert ? "ALERT" : "MONITORING";
      pill.className = "pill" + (anyAlert ? " alert" : "");

      document.getElementById("debugBtn").classList.toggle("active", status.view_mode === "debug");
      document.getElementById("cleanBtn").classList.toggle("active", status.view_mode === "clean");
      document.getElementById("zoneStatus").textContent = status.zone_locked
        ? `Locked · ${status.zone_points} pts`
        : (status.zone_points ? `Drawing · ${status.zone_points} pts` : "No zone");
      document.getElementById("zoneBufVal").textContent = status.zone_buf_label;
      document.getElementById("contactBufVal").textContent = status.contact_buf_label;
      toggleButtons("z", status.zone_buf);
      toggleButtons("c", status.contact_buf);

      document.getElementById("peopleValue").textContent = status.people;
      document.getElementById("dangerValue").textContent =
        status.detected_dangers.length ? status.detected_dangers.join(", ") : "None";
      document.getElementById("zoneAlertValue").textContent =
        zoneAlert ? status.zone_alerts.join(", ") : "None";
      document.getElementById("contactAlertValue").textContent =
        contactAlert ? status.contact_alerts.join(", ") : "None";
      document.getElementById("contactStatus").textContent =
        contactAlert ? status.contact_alerts.join(", ") : "None";

      const messages = [];
      if (zoneAlert) messages.push("Zone: " + status.zone_alerts.join(", "));
      if (contactAlert) messages.push("Contact: " + status.contact_alerts.join(", "));
      badge.textContent = messages.length ? "WARNING - " + messages.join(" | ") : "手机摄像头已连接";
      renderChecks(status);
    }

    async function initialStatus() {
      const res = await fetch("/api/status");
      updateStatus(await res.json());
    }

    initialStatus();
    startCamera("environment");
  </script>
</body>
</html>
"""


def local_ip() -> str:
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        try:
            sock.connect(("8.8.8.8", 80))
            return sock.getsockname()[0]
        except OSError:
            return "127.0.0.1"


def ensure_local_cert(ip: str) -> tuple[str, str]:
    cert_dir = Path(__file__).resolve().parent / ".certs"
    cert_dir.mkdir(exist_ok=True)
    cert_path = cert_dir / "mobile_camera.crt"
    key_path = cert_dir / "mobile_camera.key"
    config_path = cert_dir / "mobile_camera_openssl.cnf"

    config_path.write_text(
        "\n".join([
            "[req]",
            "distinguished_name=req_distinguished_name",
            "x509_extensions=v3_req",
            "prompt=no",
            "",
            "[req_distinguished_name]",
            "CN=Mobile Camera Monitor",
            "",
            "[v3_req]",
            f"subjectAltName=DNS:localhost,IP:127.0.0.1,IP:{ip}",
            "",
        ]),
        encoding="utf-8",
    )

    subprocess.run(
        [
            "openssl",
            "req",
            "-x509",
            "-newkey",
            "rsa:2048",
            "-nodes",
            "-days",
            "30",
            "-keyout",
            str(key_path),
            "-out",
            str(cert_path),
            "-config",
            str(config_path),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    return str(cert_path), str(key_path)


def ensure_models() -> None:
    if core.SEG_MODEL is None:
        print("Loading YOLOv8x-seg...")
        core.SEG_MODEL = YOLO(core.CONFIG.seg_model_path)
    if core.WORLD_MODEL is None:
        print("Loading YOLOWorld...")
        core.WORLD_MODEL = YOLOWorld(core.CONFIG.world_model_path)
        core.WORLD_MODEL.set_classes(core.CUSTOM_DANGER_CLASSES)


def status_payload() -> dict:
    with core.STATE.lock:
        zone_buf = next(
            (key for key, value in core.ZONE_BUFFER.items() if value == core.STATE.zone_proximity_px),
            "medium",
        )
        contact_buf = next(
            (key for key, value in core.CONTACT_BUFFER.items() if value == core.STATE.contact_proximity_px),
            "medium",
        )
        return {
            "view_mode": core.STATE.view_mode,
            "people": core.STATE.detected_people,
            "zone_points": len(core.STATE.zone_points),
            "zone_locked": core.STATE.zone_locked,
            "zone_alerts": list(dict.fromkeys(core.STATE.zone_alerts)),
            "zone_buf": zone_buf,
            "zone_buf_label": core.BUF_LABELS[zone_buf],
            "contact_alerts": list(dict.fromkeys(core.STATE.contact_alerts)),
            "detected_dangers": list(dict.fromkeys(core.STATE.detected_dangers)),
            "contact_buf": contact_buf,
            "contact_buf_label": core.BUF_LABELS[contact_buf],
            "monitored_classes": sorted(core.STATE.monitored_classes),
            "all_classes": core.ALL_DANGER_CLASSES,
        }


def process_uploaded_frame(frame: np.ndarray) -> tuple[np.ndarray, dict]:
    ensure_models()

    now = time.time()
    frame_h, frame_w = frame.shape[:2]

    with core.STATE.lock:
        monitored = set(core.STATE.monitored_classes)
        contact_px = core.STATE.contact_proximity_px
        zone_locked = core.STATE.zone_locked
        zone_pts = list(core.STATE.zone_points)
        zone_px = core.STATE.zone_proximity_px
        view_mode = core.STATE.view_mode
        core.STATE.frame_w = frame_w
        core.STATE.frame_h = frame_h

        if zone_locked and core.STATE.zone_mask is None and len(zone_pts) >= 3:
            core.STATE.zone_mask = core.build_zone_mask(zone_pts, frame_w, frame_h)
            core.STATE.expanded_zone_mask = core.expand_zone_mask(core.STATE.zone_mask, zone_px)

        expanded_zone = core.STATE.expanded_zone_mask if zone_locked else None

    core.FRAME_COUNT += 1
    all_dets, contact_alerts, zone_alerts = core.process_frame(
        frame,
        monitored,
        contact_px,
        expanded_zone,
    )

    with core.STATE.lock:
        core.STATE.detected_people = sum(1 for det in all_dets if det.class_name == "person")
        core.STATE.detected_dangers = list(dict.fromkeys(
            det.class_name for det in all_dets if det.source in {"coco", "world"}
        ))
        core.STATE.detected_zone_count = sum(
            1 for det in all_dets if det.class_name in core.ZONE_TARGET_CLASSES
        )
        core.STATE.contact_alerts = contact_alerts
        core.STATE.zone_alerts = zone_alerts

        fire_contact = (
            bool(contact_alerts)
            and now - core.STATE.last_contact_alert_t > core.CONFIG.alert_cooldown
        )
        fire_zone = (
            bool(zone_alerts)
            and now - core.STATE.last_zone_alert_t > core.CONFIG.alert_cooldown
        )
        if fire_contact:
            core.STATE.last_contact_alert_t = now
        if fire_zone:
            core.STATE.last_zone_alert_t = now

    if fire_contact or fire_zone:
        core.play_alert()

    output = core.render_frame(
        frame,
        all_dets,
        zone_pts,
        zone_locked,
        expanded_zone,
        view_mode == "debug",
        bool(contact_alerts or zone_alerts),
    )
    return output, status_payload()


@app.route("/")
def index():
    return render_template_string(PAGE_HTML)


@app.get("/api/status")
def status():
    return jsonify(status_payload())


@app.post("/api/frame")
def frame():
    uploaded = request.files.get("frame")
    if uploaded is None:
        return jsonify(error="Missing frame"), 400

    if not PROCESS_LOCK.acquire(blocking=False):
        return jsonify(error="Model is still processing previous frame"), 429

    try:
        data = np.frombuffer(uploaded.read(), dtype=np.uint8)
        frame_bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if frame_bgr is None:
            return jsonify(error="Invalid image frame"), 400

        output, payload = process_uploaded_frame(frame_bgr)
        ok, buffer = cv2.imencode(".jpg", output, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
        if not ok:
            return jsonify(error="Failed to encode output frame"), 500

        image_b64 = base64.b64encode(buffer).decode("ascii")
        return jsonify(status=payload, image=image_b64)
    finally:
        PROCESS_LOCK.release()


@app.post("/api/view")
def set_view():
    mode = request.get_json(force=True).get("mode", "debug")
    if mode not in {"debug", "clean"}:
        return jsonify(error="Invalid view mode"), 400
    with core.STATE.lock:
        core.STATE.view_mode = mode
    return jsonify(status_payload())


@app.post("/api/zone/add")
def zone_add():
    data = request.get_json(force=True)
    with core.STATE.lock:
        if not core.STATE.zone_locked:
            core.STATE.zone_points.append((int(data["x"]), int(data["y"])))
            core.STATE.zone_mask = None
            core.STATE.expanded_zone_mask = None
    return jsonify(status_payload())


@app.post("/api/zone/lock")
def zone_lock():
    with core.STATE.lock:
        if len(core.STATE.zone_points) >= 3:
            core.STATE.zone_locked = True
            core.STATE.zone_mask = None
            core.STATE.expanded_zone_mask = None
    return jsonify(status_payload())


@app.post("/api/zone/undo")
def zone_undo():
    with core.STATE.lock:
        if not core.STATE.zone_locked and core.STATE.zone_points:
            core.STATE.zone_points.pop()
            core.STATE.zone_mask = None
            core.STATE.expanded_zone_mask = None
    return jsonify(status_payload())


@app.post("/api/zone/clear")
def zone_clear():
    with core.STATE.lock:
        core.STATE.zone_points.clear()
        core.STATE.zone_locked = False
        core.STATE.zone_mask = None
        core.STATE.expanded_zone_mask = None
        core.STATE.zone_alerts = []
    return jsonify(status_payload())


@app.post("/api/zone/buffer")
def zone_buffer():
    size = request.get_json(force=True).get("size", "medium")
    if size not in core.ZONE_BUFFER:
        return jsonify(error="Invalid buffer size"), 400
    with core.STATE.lock:
        core.STATE.zone_proximity_px = core.ZONE_BUFFER[size]
        if core.STATE.zone_mask is not None:
            core.STATE.expanded_zone_mask = core.expand_zone_mask(
                core.STATE.zone_mask,
                core.STATE.zone_proximity_px,
            )
    return jsonify(status_payload())


@app.post("/api/contact/buffer")
def contact_buffer():
    size = request.get_json(force=True).get("size", "medium")
    if size not in core.CONTACT_BUFFER:
        return jsonify(error="Invalid buffer size"), 400
    with core.STATE.lock:
        core.STATE.contact_proximity_px = core.CONTACT_BUFFER[size]
    return jsonify(status_payload())


@app.post("/api/monitored")
def set_monitored():
    data = request.get_json(force=True)
    name = data.get("name")
    enabled = bool(data.get("enabled", True))
    if name not in set(core.ALL_DANGER_CLASSES):
        return jsonify(error="Unknown class"), 400
    with core.STATE.lock:
        if enabled:
            core.STATE.monitored_classes.add(name)
        else:
            core.STATE.monitored_classes.discard(name)
    return jsonify(status_payload())


if __name__ == "__main__":
    ip = local_ip()
    cert_file, key_file = ensure_local_cert(ip)
    print(f"Open on phone: https://{ip}:5443")
    print("If the browser shows a certificate warning, accept it for this local demo.")
    app.run(
        host="0.0.0.0",
        port=5443,
        debug=False,
        threaded=True,
        ssl_context=(cert_file, key_file),
    )
