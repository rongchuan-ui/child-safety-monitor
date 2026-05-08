"""
Child Safety Monitor — Benchmark Tool
======================================
Measures detection rate (recall) and false positive rate for all 16 dangerous
item classes across 3 distances. Results are saved incrementally so you can
stop and resume at any time.

How it works:
    For each class × distance combination, you place the object in front of the
    camera and press SPACE to record for 30 seconds. The script counts how many
    frames the model detects the object and reports the detection rate.
    A separate false-positive test records 60 seconds with NO objects present.

Usage:
    python benchmark.py            # run full benchmark
    python benchmark.py --report   # print report from saved results only
    python benchmark.py --fp       # run false-positive test only

Results saved to:  benchmark_results.json
Final report:      benchmark_report.md

Controls during recording:
    SPACE    →  start recording (when in "Ready" state)
    S        →  skip current test
    Q / ESC  →  quit (progress is saved)
"""

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import cv2
import sys
import json
import time
import platform
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime
from ultralytics import YOLO, YOLOWorld

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

CAMERA_INDEX  = 0
CAMERA_WIDTH  = 1280
CAMERA_HEIGHT = 720
DEVICE        = "mps"
IMGSZ         = 640

SEG_CONF   = 0.25
WORLD_CONF = 0.20
IOU        = 0.45

RECORD_SECONDS   = 30   # seconds to record per test
FP_TEST_SECONDS  = 60   # seconds for false-positive test (no objects present)
WORLD_SKIP       = 3    # run YOLOWorld every N frames

RESULTS_FILE = Path("benchmark_results.json")
REPORT_FILE  = Path("benchmark_report.md")

# ─────────────────────────────────────────────
# CLASSES TO BENCHMARK
# ─────────────────────────────────────────────

# COCO classes — detected by yolov8x-seg
COCO_CLASSES: set[str] = {
    "knife", "scissors", "bottle", "wine glass",
    "fork", "microwave", "oven", "toaster",
}

# Custom classes — detected by YOLOWorld
CUSTOM_CLASSES: list[str] = [
    "lighter",
    "matches",
    "pill bottle",
    "medicine bottle",
    "button battery",
    "power cord",
    "extension cord",
    "plastic bag",
    "candle",
    "iron",
    "needle",
    "syringe",
    "cleaning spray bottle",
]

ALL_DANGER_CLASSES = sorted(COCO_CLASSES) + CUSTOM_CLASSES

DISTANCES = ["near", "mid", "far"]
DISTANCE_LABELS = {
    "near": "NEAR  (0.5 – 1 m)",
    "mid":  "MID   (1.5 – 3 m)",
    "far":  "FAR   (4 – 6 m)  ",
}

# ─────────────────────────────────────────────
# STATE MACHINE
# ─────────────────────────────────────────────

IDLE      = "idle"       # waiting for SPACE
RECORDING = "recording"  # counting frames
DONE      = "done"       # showing result, moving on

# ─────────────────────────────────────────────
# RESULTS STORE
# ─────────────────────────────────────────────

def load_results() -> dict:
    if RESULTS_FILE.exists():
        with open(RESULTS_FILE) as f:
            return json.load(f)
    return {}


def save_results(results: dict) -> None:
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)


def is_done(results: dict, class_name: str, distance: str) -> bool:
    return class_name in results and distance in results.get(class_name, {})


def record_result(results: dict, class_name: str, distance: str,
                  total: int, detected: int) -> None:
    if class_name not in results:
        results[class_name] = {}
    dr = round(detected / total, 4) if total > 0 else 0.0
    results[class_name][distance] = {
        "total_frames":    total,
        "detected_frames": detected,
        "detection_rate":  dr,
        "timestamp":       datetime.now().isoformat(timespec="seconds"),
    }
    save_results(results)


def record_fp_result(results: dict, fp_counts: dict[str, int],
                     total_frames: int, duration_s: float) -> None:
    """Store false-positive counts (collected with no objects present)."""
    results["_false_positive_test"] = {
        "total_frames":    total_frames,
        "duration_seconds": round(duration_s, 1),
        "fp_per_class":    fp_counts,
        "fp_rate_per_min": {
            k: round(v / duration_s * 60, 2)
            for k, v in fp_counts.items()
        },
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    save_results(results)


# ─────────────────────────────────────────────
# OVERLAY DRAWING
# ─────────────────────────────────────────────

STATUS_H = 80   # height of status bar

def draw_status_bar(frame: np.ndarray, lines: list[tuple[str, tuple]]) -> None:
    """Draw a dark bar at the top with coloured text lines."""
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (w, STATUS_H), (20, 20, 20), -1)
    for i, (text, colour) in enumerate(lines):
        y = 22 + i * 26
        cv2.putText(frame, text, (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, colour, 2, cv2.LINE_AA)


def draw_progress_bar(frame: np.ndarray, progress: float,
                      colour: tuple = (0, 200, 80)) -> None:
    """Draw a progress bar at the bottom of the frame."""
    h, w = frame.shape[:2]
    bar_h = 12
    filled = int(w * min(progress, 1.0))
    cv2.rectangle(frame, (0, h - bar_h), (w, h), (50, 50, 50), -1)
    cv2.rectangle(frame, (0, h - bar_h), (filled, h), colour, -1)


def draw_detection_highlight(frame: np.ndarray, box: tuple,
                              label: str, detected: bool) -> None:
    x1, y1, x2, y2 = box
    colour = (0, 255, 0) if detected else (100, 100, 100)
    cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)
    cv2.putText(frame, label, (x1, max(y1 - 6, 14)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 2, cv2.LINE_AA)


# ─────────────────────────────────────────────
# DETECTION HELPERS
# ─────────────────────────────────────────────

def detect_classes_in_frame(frame, seg_model, world_model,
                             frame_count: int,
                             cached_world: list) -> tuple[set[str], list, list]:
    """
    Run inference and return:
        detected_set  — set of class names found this frame
        seg_boxes     — list of (class_name, xyxy) from seg model
        world_boxes   — list of (class_name, xyxy) from YOLOWorld (cached)
    """
    detected: set[str] = set()
    seg_boxes:   list[tuple[str, tuple]] = []
    world_boxes: list[tuple[str, tuple]] = []

    # Primary: YOLOv8x-seg
    seg_res = seg_model.predict(
        source=frame, conf=SEG_CONF, iou=IOU,
        device=DEVICE, imgsz=IMGSZ, verbose=False,
    )
    if seg_res and seg_res[0].boxes is not None:
        for box in seg_res[0].boxes:
            cls_name = seg_model.names[int(box.cls[0].item())]
            xyxy = tuple(map(int, box.xyxy[0].tolist()))
            seg_boxes.append((cls_name, xyxy))
            detected.add(cls_name)

    # Secondary: YOLOWorld (every WORLD_SKIP frames)
    if frame_count % WORLD_SKIP == 0:
        w_res = world_model.predict(
            source=frame, conf=WORLD_CONF, iou=IOU,
            device=DEVICE, imgsz=IMGSZ, verbose=False,
        )
        cached_world.clear()
        if w_res and w_res[0].boxes is not None:
            for box in w_res[0].boxes:
                cls_name = world_model.names[int(box.cls[0].item())]
                xyxy = tuple(map(int, box.xyxy[0].tolist()))
                cached_world.append((cls_name, xyxy))

    for cls_name, xyxy in cached_world:
        world_boxes.append((cls_name, xyxy))
        detected.add(cls_name)

    return detected, seg_boxes, world_boxes


# ─────────────────────────────────────────────
# DETECTION RATE TEST — one class, one distance
# ─────────────────────────────────────────────

def run_single_test(cap, seg_model, world_model,
                    class_name: str, distance: str,
                    test_num: int, total_tests: int) -> tuple[int, int] | None:
    """
    Returns (total_frames, detected_frames) or None if skipped/quit.
    """
    window = "Benchmark"
    state = IDLE

    total_frames    = 0
    detected_frames = 0
    record_start    = 0.0
    cached_world:   list = []
    frame_count     = 0

    dist_label = DISTANCE_LABELS[distance]
    source_label = "COCO model" if class_name in COCO_CLASSES else "YOLOWorld"

    print(f"\n{'─'*55}")
    print(f"  Test {test_num}/{total_tests}")
    print(f"  Class   : {class_name.upper()}")
    print(f"  Distance: {dist_label}")
    print(f"  Model   : {source_label}")
    print(f"  → Place the object at {dist_label.strip()} and press SPACE")
    print(f"  → Press S to skip this test")
    print(f"{'─'*55}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        frame_h, frame_w = frame.shape[:2]
        now = time.time()

        detected_set, seg_boxes, world_boxes = detect_classes_in_frame(
            frame, seg_model, world_model, frame_count, cached_world
        )

        target_found = class_name in detected_set
        annotated = frame.copy()

        # Draw all boxes, highlight target
        for (cn, box) in seg_boxes + world_boxes:
            draw_detection_highlight(annotated, box, cn, cn == class_name)

        # State machine
        if state == IDLE:
            elapsed_pct = 0.0
            colour_bar = (180, 180, 0)
            lines = [
                (f"[{test_num}/{total_tests}] Testing: {class_name.upper()}  |  {dist_label}",
                 (255, 255, 255)),
                (f"SPACE = start recording  |  S = skip  |  Q = quit",
                 (160, 160, 160)),
            ]

        elif state == RECORDING:
            elapsed     = now - record_start
            remaining   = max(0, RECORD_SECONDS - elapsed)
            elapsed_pct = elapsed / RECORD_SECONDS

            total_frames    += 1
            if target_found:
                detected_frames += 1

            dr_so_far = detected_frames / total_frames if total_frames else 0
            colour_rec = (0, 0, 220)  # red bar while recording
            colour_bar = colour_rec

            lines = [
                (f"RECORDING  {class_name.upper()}  |  {dist_label}",
                 (80, 80, 255) if not target_found else (80, 220, 80)),
                (f"{remaining:.1f}s left  |  Detected: {detected_frames}/{total_frames}"
                 f"  ({dr_so_far*100:.1f}%)",
                 (255, 255, 255)),
            ]

            if elapsed >= RECORD_SECONDS:
                state = DONE

        elif state == DONE:
            dr = detected_frames / total_frames if total_frames else 0
            colour_bar = (0, 200, 80)
            status_str = "✓ PASS (≥70%)" if dr >= 0.70 else "✗ FAIL (<70%) → needs training"
            colour_st  = (80, 220, 80) if dr >= 0.70 else (80, 80, 255)
            lines = [
                (f"DONE  {class_name.upper()}  |  {dist_label}",
                 (255, 255, 255)),
                (f"Detection rate: {dr*100:.1f}%  |  {status_str}",
                 colour_st),
            ]
            elapsed_pct = 1.0

        draw_status_bar(annotated, lines)
        draw_progress_bar(annotated, elapsed_pct,
                          colour=(0, 0, 200) if state == RECORDING else (0, 200, 80))

        cv2.imshow(window, annotated)

        key = cv2.waitKey(1) & 0xFF
        if key in [ord("q"), ord("Q"), 27]:
            return None   # quit signal
        elif key == ord(" ") and state == IDLE:
            state        = RECORDING
            record_start = time.time()
            total_frames    = 0
            detected_frames = 0
            print(f"  ► Recording…")
        elif key in [ord("s"), ord("S")]:
            print(f"  ⏭  Skipped.")
            return (-1, -1)   # skip signal
        elif state == DONE:
            # Auto-advance after 2 seconds
            if not hasattr(run_single_test, "_done_time"):
                run_single_test._done_time = now
            elif now - run_single_test._done_time > 2.0:
                del run_single_test._done_time
                return (total_frames, detected_frames)
            # Any key also advances
            if key != 255:
                if hasattr(run_single_test, "_done_time"):
                    del run_single_test._done_time
                return (total_frames, detected_frames)

    return None


# ─────────────────────────────────────────────
# FALSE POSITIVE TEST
# ─────────────────────────────────────────────

def run_fp_test(cap, seg_model, world_model) -> dict[str, int] | None:
    """
    Run for FP_TEST_SECONDS with NO objects present.
    Returns dict of class_name → number of frames with a false positive.
    """
    window   = "Benchmark"
    state    = IDLE
    fp_counts: dict[str, int] = {c: 0 for c in ALL_DANGER_CLASSES}
    total_frames = 0
    record_start = 0.0
    cached_world: list = []
    frame_count  = 0

    print(f"\n{'─'*55}")
    print(f"  FALSE POSITIVE TEST")
    print(f"  → Remove ALL dangerous objects from camera view")
    print(f"  → Record for {FP_TEST_SECONDS} seconds")
    print(f"  → Press SPACE when view is clear")
    print(f"{'─'*55}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        now = time.time()

        detected_set, seg_boxes, world_boxes = detect_classes_in_frame(
            frame, seg_model, world_model, frame_count, cached_world
        )

        annotated = frame.copy()
        for cn, box in seg_boxes + world_boxes:
            if cn in ALL_DANGER_CLASSES:
                draw_detection_highlight(annotated, box, f"FP: {cn}", True)

        if state == IDLE:
            lines = [
                ("FALSE POSITIVE TEST — Remove all objects, then press SPACE",
                 (255, 200, 50)),
                ("S = skip  |  Q = quit", (160, 160, 160)),
            ]
            draw_status_bar(annotated, lines)
            draw_progress_bar(annotated, 0.0, (180, 180, 0))

        elif state == RECORDING:
            elapsed   = now - record_start
            remaining = max(0, FP_TEST_SECONDS - elapsed)
            pct       = elapsed / FP_TEST_SECONDS
            total_frames += 1

            for cn in detected_set:
                if cn in fp_counts:
                    fp_counts[cn] += 1

            any_fp = bool(detected_set & set(ALL_DANGER_CLASSES))
            lines = [
                (f"RECORDING FP TEST  |  {remaining:.0f}s remaining", (255, 200, 50)),
                (f"Frames: {total_frames}  |  "
                 f"{'⚠ FALSE POSITIVE DETECTED' if any_fp else 'Clean ✓'}",
                 (80, 80, 255) if any_fp else (80, 220, 80)),
            ]
            draw_status_bar(annotated, lines)
            draw_progress_bar(annotated, pct, (0, 0, 200))

            if elapsed >= FP_TEST_SECONDS:
                return fp_counts, total_frames, elapsed

        cv2.imshow(window, annotated)

        key = cv2.waitKey(1) & 0xFF
        if key in [ord("q"), ord("Q"), 27]:
            return None
        elif key == ord(" ") and state == IDLE:
            state        = RECORDING
            record_start = time.time()
            total_frames = 0
            print("  ► Recording FP test…")
        elif key in [ord("s"), ord("S")]:
            print("  ⏭  FP test skipped.")
            return {}, 0, 0

    return None


# ─────────────────────────────────────────────
# REPORT GENERATION
# ─────────────────────────────────────────────

def generate_report(results: dict) -> str:
    """Generate a markdown report table from saved results."""
    lines = []
    lines.append("# Benchmark Report — Child Safety Monitor")
    lines.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append(f"\nRecord duration per test: {RECORD_SECONDS}s  "
                 f"| Pass threshold: ≥ 70% detection rate\n")

    lines.append("## Detection Rate by Class and Distance\n")
    header = "| Class | Source | Near DR | Mid DR | Far DR | Avg DR | Status |"
    sep    = "|---|---|---|---|---|---|---|"
    lines.append(header)
    lines.append(sep)

    needs_training = []
    for cls in ALL_DANGER_CLASSES:
        source = "COCO seg" if cls in COCO_CLASSES else "YOLOWorld"
        drs = {}
        for dist in DISTANCES:
            entry = results.get(cls, {}).get(dist)
            drs[dist] = entry["detection_rate"] if entry else None

        values = [v for v in drs.values() if v is not None]
        avg = sum(values) / len(values) if values else None

        def fmt(v):
            if v is None:
                return "—"
            pct = v * 100
            if pct >= 85:
                return f"**{pct:.0f}%** ✅"
            elif pct >= 70:
                return f"{pct:.0f}% 🟡"
            else:
                return f"**{pct:.0f}%** 🔴"

        avg_str = fmt(avg) if avg is not None else "—"
        status  = ("✅ Pass" if avg is not None and avg >= 0.70
                   else ("🔴 Needs training" if avg is not None else "⬜ Not tested"))

        if avg is not None and avg < 0.70:
            needs_training.append(cls)

        row = (f"| {cls} | {source} | {fmt(drs['near'])} | "
               f"{fmt(drs['mid'])} | {fmt(drs['far'])} | {avg_str} | {status} |")
        lines.append(row)

    # False positive section
    fp_data = results.get("_false_positive_test")
    if fp_data:
        lines.append("\n## False Positive Rate\n")
        lines.append(f"Test duration: {fp_data.get('duration_seconds', '?')}s  "
                     f"| Total frames: {fp_data.get('total_frames', '?')}\n")
        lines.append("| Class | FP Frames | FP / min |")
        lines.append("|---|---|---|")
        fp_per_class = fp_data.get("fp_per_class", {})
        fp_per_min   = fp_data.get("fp_rate_per_min", {})
        for cls in ALL_DANGER_CLASSES:
            count = fp_per_class.get(cls, 0)
            rate  = fp_per_min.get(cls, 0.0)
            flag  = " ⚠️" if rate > 2.0 else ""
            lines.append(f"| {cls} | {count} | {rate:.1f}{flag} |")

    # Summary
    lines.append("\n## Summary\n")
    if needs_training:
        lines.append(f"**{len(needs_training)} class(es) need custom training (avg DR < 70%):**\n")
        for cls in needs_training:
            lines.append(f"- `{cls}`")
    else:
        lines.append("All tested classes meet the 70% detection rate threshold. ✅")

    lines.append("\n\n---")
    lines.append("*Re-run `python benchmark.py` to fill in untested slots.*")

    return "\n".join(lines)


def print_console_summary(results: dict) -> None:
    """Print a quick summary table to the terminal."""
    print("\n" + "═" * 60)
    print("  BENCHMARK SUMMARY")
    print("═" * 60)
    print(f"  {'Class':<25} {'Near':>7} {'Mid':>7} {'Far':>7} {'Avg':>7}")
    print("  " + "─" * 56)

    needs_training = []
    for cls in ALL_DANGER_CLASSES:
        drs = {}
        for dist in DISTANCES:
            entry = results.get(cls, {}).get(dist)
            drs[dist] = entry["detection_rate"] if entry else None

        values = [v for v in drs.values() if v is not None]
        avg    = sum(values) / len(values) if values else None

        def fmt(v):
            return f"{v*100:.0f}%" if v is not None else " —  "

        flag = ""
        if avg is not None and avg < 0.70:
            flag = " ◄ NEEDS TRAINING"
            needs_training.append(cls)

        print(f"  {cls:<25} {fmt(drs['near']):>7} {fmt(drs['mid']):>7}"
              f" {fmt(drs['far']):>7} {fmt(avg):>7}{flag}")

    print("═" * 60)
    if needs_training:
        print(f"\n  Classes below 70%: {', '.join(needs_training)}")
    print()


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", action="store_true",
                        help="Print report from saved results without running benchmark")
    parser.add_argument("--fp", action="store_true",
                        help="Run false-positive test only")
    args = parser.parse_args()

    results = load_results()

    # Report-only mode
    if args.report:
        if not results:
            print("No results found. Run the benchmark first.")
            return
        print_console_summary(results)
        report_md = generate_report(results)
        REPORT_FILE.write_text(report_md)
        print(f"Full report saved to: {REPORT_FILE}")
        return

    # Load models
    print("\nLoading models…")
    seg_model = YOLO("yolov8x-seg.pt")
    world_model = YOLOWorld("yolov8x-worldv2.pt")
    world_model.set_classes(CUSTOM_CLASSES)
    print("Models loaded.\n")

    # Camera
    backend = cv2.CAP_AVFOUNDATION if platform.system() == "Darwin" else cv2.CAP_DSHOW
    cap = cv2.VideoCapture(CAMERA_INDEX, backend)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    if not cap.isOpened():
        raise RuntimeError("Cannot open camera.")

    cv2.namedWindow("Benchmark", cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)

    # FP-only mode
    if args.fp:
        out = run_fp_test(cap, seg_model, world_model)
        if out and out[1] > 0:
            fp_counts, total_frames, duration = out
            record_fp_result(results, fp_counts, total_frames, duration)
            print("\nFP test saved.")
        cap.release()
        cv2.destroyAllWindows()
        return

    # Count remaining tests
    all_tests = [(cls, dist)
                 for cls in ALL_DANGER_CLASSES
                 for dist in DISTANCES]
    remaining = [(cls, dist) for cls, dist in all_tests
                 if not is_done(results, cls, dist)]

    done_count   = len(all_tests) - len(remaining)
    total_count  = len(all_tests)
    print(f"Tests: {done_count}/{total_count} already complete.")
    print(f"Remaining: {len(remaining)} tests × {RECORD_SECONDS}s ≈ "
          f"{len(remaining)*RECORD_SECONDS//60} min\n")

    if not remaining:
        print("All tests complete! Running report…")
    else:
        print("Instructions:")
        print("  1. When prompted, place the specified object at the specified distance")
        print("  2. Press SPACE to begin recording (30 seconds)")
        print("  3. Keep the object visible and move it slightly")
        print("  4. Press S to skip a test, Q to quit and save progress\n")
        input("Press ENTER to begin…")

    # Run remaining detection-rate tests
    test_num = done_count + 1
    quit_requested = False

    for cls, dist in remaining:
        result = run_single_test(
            cap, seg_model, world_model,
            cls, dist, test_num, total_count
        )

        if result is None:          # quit
            quit_requested = True
            break
        elif result == (-1, -1):    # skip
            test_num += 1
            continue
        else:
            total_f, detected_f = result
            record_result(results, cls, dist, total_f, detected_f)
            dr = detected_f / total_f if total_f else 0
            status = "PASS ✓" if dr >= 0.70 else "FAIL ✗"
            print(f"  → {cls} | {dist}: {dr*100:.1f}% [{status}]")
            test_num += 1

    if not quit_requested:
        # Offer FP test if not already done
        if "_false_positive_test" not in results:
            print("\n" + "─" * 55)
            print("All detection tests done. Run false-positive test? (Y/N)")
            key_ch = input("> ").strip().lower()
            if key_ch == "y":
                out = run_fp_test(cap, seg_model, world_model)
                if out and out[1] > 0:
                    fp_counts, total_frames, duration = out
                    record_fp_result(results, fp_counts, total_frames, duration)

    cap.release()
    cv2.destroyAllWindows()
    cv2.waitKey(1)

    # Final report
    print_console_summary(results)
    report_md = generate_report(results)
    REPORT_FILE.write_text(report_md)
    print(f"Full report saved to: {REPORT_FILE}\n")


if __name__ == "__main__":
    main()
