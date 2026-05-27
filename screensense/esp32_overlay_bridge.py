"""
ESP32 + camera tracker + smartboard overlay - single entry point.

This wires three things together:

  1. The smartboard overlay (smartboard_camera_merge_candidate.py) shown
     on the main Qt thread, spanning every connected monitor.
  2. The ESP32 button receiver (src/input_receiver.py) - a TCP server on
     port 65432 listening for {"erase": bool, "write": bool} JSON lines.
     The receiver runs in its own daemon thread inside that module.
  3. The headless screen-mode pen tracker - a daemon thread that opens
     both cameras, runs orange-tip detection, fuses + Kalman-smooths the
     two screen-pixel estimates, and moves the OS cursor via Win32
     SetCursorPos. Mirrors src/tracker.run_screen() minus the cv2.imshow
     debug windows (OpenCV HighGUI conflicts with Qt's event loop).

Tool-selection mapping (ESP32 -> overlay):

    write=True         -> pen tool (drawing)
    erase=True         -> eraser tool
    both False         -> scroll/pointer mode (the overlay's scroll-mode
                          mask makes everything but the toolbar click-
                          through, so the laptop behaves like normal)
    both True          -> pen wins (only one tool can be active)

The first non-pointer state also shows the overlay if it is hidden, so
the ESP32 can bring up the UI without the user also pressing W.

Mouse-button mapping (ESP32 -> synthetic mouse, in the tracker thread):

    write transitions  -> left mouse down / up  (drives the pen stroke)
    erase transitions  -> right mouse down / up

Run from the project root:
    python esp32_overlay_bridge.py
"""

import os
import sys
import threading
import time

import cv2
import numpy as np
import yaml
from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import QApplication

# Make `src.*` importable regardless of cwd.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import the overlay module FIRST so its top-level Windows DPI-awareness
# call runs before we create QApplication; without that, the overlay
# geometry is wrong on the secondary monitor.
import smartboard_camera_merge_candidate as overlay_module

from src import input_receiver
from src.detect import OrangeTipDetector
from src.fuse import SensorFuser
from src.kalman import PenKalmanFilter
from src.cursor import CursorController
from src.utils import load_calibration


POLL_INTERVAL_MS = 33   # ~30 Hz: fast enough to feel instant on a button press
CONFIG_PATH      = "config.yaml"


# ── ESP32 -> overlay tool selection ──────────────────────────────────────────

def _tool_for(erase, write):
    """Translate the (erase, write) pair into an overlay tool name. write
    takes priority over erase if both happen to be on simultaneously so a
    user can't accidentally erase the moment they start drawing."""
    if write:
        return "pen"
    if erase:
        return "eraser"
    return "scroll"


def _make_state_handler(overlay):
    """Closure that only acts on transitions of (erase, write). Acting only
    on transitions means a manual tool change from the on-screen toolbar
    sticks until the ESP32 state actually changes again, instead of being
    overwritten every poll tick."""
    last_erase = None
    last_write = None

    def tick():
        nonlocal last_erase, last_write
        erase = input_receiver.is_erase_pressed()
        write = input_receiver.is_write_pressed()
        if erase == last_erase and write == last_write:
            return
        last_erase, last_write = erase, write

        tool = _tool_for(erase, write)
        # Single line per state change. If you press a physical button
        # and don't see one of these lines, the ESP32 isn't connected or
        # input_receiver isn't reading it - check the [InputReceiver]
        # startup line and confirm the ESP32 has actually established a
        # TCP connection back to this machine.
        print(f"[esp32] write={write} erase={erase} -> tool={tool} "
              f"overlay_on={overlay.overlay_on}", flush=True)

        # Auto-show the overlay the first time the user picks a drawing
        # tool, so they don't also have to press W. Don't auto-show for
        # pure pointer mode - if both buttons are off and the overlay was
        # hidden, leave it hidden.
        if tool != "scroll" and not overlay.overlay_on:
            overlay.show_overlay()

        if overlay.overlay_on:
            overlay.select_tool(tool)

    return tick


# ── Headless camera tracker (background thread) ──────────────────────────────

def _open_camera(index, cam_cfg):
    """DSHOW -> MSMF -> default cascade. Mirrors src/tracker._open_camera so
    the bridge stays standalone (importing tracker.py would also drag in
    its cv2.imshow-bound main loop). On Windows, cameras that don't speak
    MSMF properly need DSHOW, hence the order."""
    width  = cam_cfg["width"]
    height = cam_cfg["height"]
    fps    = cam_cfg["fps"]

    for backend, _ in [(cv2.CAP_DSHOW, "DSHOW"),
                       (cv2.CAP_MSMF,  "MSMF"),
                       (None,          "DEFAULT")]:
        cap = (cv2.VideoCapture(index, backend) if backend is not None
               else cv2.VideoCapture(index))
        if not cap.isOpened():
            cap.release()
            continue
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_FPS,          fps)
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
        for _ in range(30):
            ret, frame = cap.read()
            if ret and frame is not None:
                return cap
        cap.release()
    return None


def _run_camera_tracker(cfg, stop_event):
    """Headless port of tracker.run_screen(): two cameras -> detection ->
    confidence fusion -> single Kalman -> Win32 SetCursorPos. ESP32 write/
    erase transitions become synthetic mouse buttons so the overlay sees
    real mouseDown / mouseUp events at the tracked cursor position.

    A cv2 debug window mirrors tracker.run_screen()'s preview (both camera
    frames side-by-side with detection overlays + a status bar). cv2 GUI
    calls from a worker thread work on Windows as long as imshow AND
    waitKey both run on this thread - we never touch cv2 windows from
    the Qt main thread, so there's no event-loop conflict."""
    cam_cfg   = cfg["cameras"]
    out_cfg   = cfg.get("output", {}) or {}
    show_debug = bool(out_cfg.get("show_debug", True))
    min_conf  = cfg["fusion"]["min_confidence"]
    color_min = float(cfg["detection"].get("color_confidence_min", 0.0))

    try:
        cal1 = load_calibration("calibration/cam1_screen_calibration.npz")
        cal2 = load_calibration("calibration/cam2_screen_calibration.npz")
    except FileNotFoundError as exc:
        print(f"[tracker] screen calibration missing: {exc}. "
              f"Run `python calibration/screen_calibrate.py` first. "
              f"Pen tracking disabled; overlay/ESP32 will still work.",
              flush=True)
        return

    rect = cal1.get("rect")
    if rect is None or cal2.get("rect") is None:
        print("[tracker] calibration .npz missing 'rect' - re-run "
              "calibration/screen_calibrate.py.", flush=True)
        return

    detector1 = OrangeTipDetector.from_config(cfg, cam_key="cam1")
    detector2 = OrangeTipDetector.from_config(cfg, cam_key="cam2")
    fuser     = SensorFuser.from_config(cfg)
    kalman    = PenKalmanFilter.from_config(cfg, screen_mode=True)

    cursor_enabled = bool((cfg.get("cursor") or {}).get("enabled", False))
    cursor = CursorController.from_config(cfg) if cursor_enabled else None

    cap1 = _open_camera(cam_cfg["cam1_index"], cam_cfg)
    cap2 = _open_camera(cam_cfg["cam2_index"], cam_cfg)
    if cap1 is None or cap2 is None:
        print(f"[tracker] could not open one or both cameras "
              f"(cam1={cam_cfg['cam1_index']}, cam2={cam_cfg['cam2_index']}). "
              f"Close any app holding the camera (Windows Camera, Teams, "
              f"Zoom, browser tabs) and try again. Pen tracking disabled; "
              f"overlay/ESP32 will still work.", flush=True)
        if cap1 is not None: cap1.release()
        if cap2 is not None: cap2.release()
        return

    rx0, ry0, rx1, ry1 = int(rect[0]), int(rect[1]), int(rect[2]), int(rect[3])
    prev_write = False
    prev_erase = False

    def _to_screen(pixel_point, H):
        pt  = np.array([[[pixel_point[0], pixel_point[1]]]], dtype=np.float32)
        out = cv2.perspectiveTransform(pt, H)
        return float(out[0][0][0]), float(out[0][0][1])

    window_name = "Pen Tracker - Screen Mode"
    if show_debug:
        cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    prev_time = time.time()

    try:
        while not stop_event.is_set():
            ret1, frame1 = cap1.read()
            ret2, frame2 = cap2.read()
            if not ret1 or not ret2:
                continue

            frame1 = cv2.undistort(frame1, cal1["mtx"], cal1["dist"])
            frame2 = cv2.undistort(frame2, cal2["mtx"], cal2["dist"])
            det1 = detector1.detect(frame1)
            det2 = detector2.detect(frame2)

            ok1 = (det1.pixel_point is not None
                   and det1.confidence       >= min_conf
                   and det1.color_confidence >= color_min)
            ok2 = (det2.pixel_point is not None
                   and det2.confidence       >= min_conf
                   and det2.color_confidence >= color_min)
            pt1   = _to_screen(det1.pixel_point, cal1["H"]) if ok1 else None
            pt2   = _to_screen(det2.pixel_point, cal2["H"]) if ok2 else None
            conf1 = det1.confidence if ok1 else 0.0
            conf2 = det2.confidence if ok2 else 0.0

            fused  = fuser.fuse(pt1, conf1, pt2, conf2)
            sx, sy = kalman.update(
                measurement    = fused.plane_point,
                confidence     = fused.confidence,
                min_confidence = min_conf,
            )

            if cursor is not None and fused.source != "lost":
                cx = max(rx0, min(rx1 - 1, sx))
                cy = max(ry0, min(ry1 - 1, sy))
                cursor.move_screen(cx, cy)

            # ESP32 write/erase transitions -> synthetic mouse buttons. The
            # overlay reads these as mouseDown / mouseUp events and starts /
            # ends the pen stroke. Tool selection (pen vs eraser) is handled
            # separately by the QTimer-driven tick in _make_state_handler.
            write_now = input_receiver.is_write_pressed()
            erase_now = input_receiver.is_erase_pressed()
            if cursor is not None:
                if write_now and not prev_write:
                    cursor.left_down()
                elif prev_write and not write_now:
                    cursor.left_up()
                if erase_now and not prev_erase:
                    cursor.right_down()
                elif prev_erase and not erase_now:
                    cursor.right_up()
            prev_write, prev_erase = write_now, erase_now

            # ── Debug preview (side-by-side cameras + status bar) ──────────
            now = time.time()
            fps = 1.0 / max(now - prev_time, 1e-6)
            prev_time = now

            if show_debug:
                h, w   = frame1.shape[:2]
                half_w = w // 2
                left   = cv2.resize(frame1, (half_w, h))
                right  = cv2.resize(frame2, (half_w, h))
                for img, det, label, accepted in (
                    (left,  det1, "CAM1", ok1),
                    (right, det2, "CAM2", ok2),
                ):
                    # Green when the detection passed BOTH area- and color-
                    # confidence gates (i.e. actually contributed to fusion);
                    # red otherwise - that's the colour the user needs to
                    # see to know why the cursor isn't moving.
                    color = (0, 200, 0) if accepted else (0, 0, 200)
                    if det.pixel_point is not None:
                        # Only x scales: the resize halves width but keeps
                        # height, so scaling y here would float the circle
                        # off the actual tip.
                        px = int(det.pixel_point[0] * (half_w / w))
                        py = int(det.pixel_point[1])
                        cv2.circle(img, (px, py), 8, color, 2)
                    cv2.putText(
                        img,
                        f"{label} conf={det.confidence:.2f} "
                        f"col={det.color_confidence:.2f}",
                        (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2,
                    )
                combined = np.hstack([left, right])
                bar = np.zeros((40, combined.shape[1], 3), dtype=np.uint8)
                cv2.putText(
                    bar,
                    f"src: {fused.source:<10}  fused_conf: {fused.confidence:.2f}  "
                    f"smoothed: ({sx:.0f},{sy:.0f})  FPS: {fps:.1f}",
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1,
                )
                cv2.imshow(window_name, np.vstack([combined, bar]))

            # waitKey pumps the cv2 window's message queue and must run on
            # the same thread that created the window. 1 ms is enough; the
            # cameras gate the loop rate anyway.
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                # Closing the debug window quits the tracker but leaves the
                # overlay/ESP32 alive - the user can still draw with W.
                break
    except Exception as exc:
        print(f"[tracker] crashed: {exc}", flush=True)
    finally:
        # Release any synthetic buttons that might still be held down so
        # quitting mid-press doesn't leave the OS with a stuck mouse.
        if cursor is not None:
            if prev_write: cursor.left_up()
            if prev_erase: cursor.right_up()
        cap1.release()
        cap2.release()
        if show_debug:
            try:
                cv2.destroyWindow(window_name)
            except cv2.error:
                pass
        print("[tracker] stopped.", flush=True)


# ── Entry point ──────────────────────────────────────────────────────────────

def main():
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)

    app = QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(False)
    overlay_module.configure_macos_app_activation()
    overlay = overlay_module.SmartboardOverlay()

    input_receiver.start_background_listener()

    # ESP32 -> overlay tool, polled on the Qt main thread.
    tool_timer = QTimer()
    tool_timer.setInterval(POLL_INTERVAL_MS)
    tool_timer.timeout.connect(_make_state_handler(overlay))
    tool_timer.start()

    # Camera tracker -> cursor + mouse buttons, daemon thread so it dies
    # with the process if the user closes the Qt window. stop_event lets
    # the loop break out cleanly on aboutToQuit (camera release runs).
    stop_event = threading.Event()
    tracker_thread = threading.Thread(
        target=_run_camera_tracker, args=(cfg, stop_event), daemon=True,
    )
    tracker_thread.start()
    app.aboutToQuit.connect(stop_event.set)

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
