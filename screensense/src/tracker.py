import cv2
import numpy as np
import yaml
import json
import time
from pathlib import Path

from detect import OrangeTipDetector, Detection
from fuse   import SensorFuser
from kalman import PenKalmanFilter
from cursor import CursorController
from input_receiver import (
    start_background_listener,
    is_erase_pressed,
    is_write_pressed,
)
from utils  import (
    load_calibration,
    map_to_plane,
    draw_debug_overlay,
    StrokeRecorder,
)


# Number of consecutive frames both cameras must detect the pen for the
# initialization phase to consider the system ready.
INIT_STABLE_FRAMES = 10


def _open_camera(index, cam_cfg):
    """Try DSHOW, MSMF, then default backend until one yields real frames.
    Returns the opened cv2.VideoCapture or None if every attempt failed."""
    width  = cam_cfg["width"]
    height = cam_cfg["height"]
    fps    = cam_cfg["fps"]

    attempts = [
        (cv2.CAP_DSHOW, "DSHOW"),
        (cv2.CAP_MSMF,  "MSMF"),
        (None,          "DEFAULT"),
    ]
    for backend, name in attempts:
        cap = (cv2.VideoCapture(index, backend) if backend is not None
               else cv2.VideoCapture(index))
        if not cap.isOpened():
            cap.release()
            continue

        # MJPG is widely supported and uses less USB bandwidth than raw YUYV,
        # which is what lets two cameras share a controller without one
        # silently failing to start streaming.
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_FPS,          fps)
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)

        # Confirm the backend actually delivers a frame.
        for _ in range(30):
            ret, frame = cap.read()
            if ret and frame is not None:
                actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                print(f"cam{index} opened via {name} at {actual_w}x{actual_h}")
                return cap
        cap.release()
    return None


def initialize_tracking(cap1, cap2, cal1, cal2, detector1, detector2, cfg):
    """
    Pre-flight check before the main loop:
      - Make sure both cameras are streaming
      - Make sure the pen is detected by BOTH cameras at >= min_confidence
        for a sustained burst (INIT_STABLE_FRAMES) so the Kalman filter is
        seeded with a real measurement and the cursor doesn't jump on start
      - Wait for the user to press SPACE to begin tracking

    Returns True if initialization completed, False if the user aborted (Q).
    """
    min_conf  = cfg["fusion"]["min_confidence"]
    window    = "Tracker Initialization"
    cv2.namedWindow(window)

    print("\n=== Initialization ===")
    print("Hold the orange pen tip so BOTH cameras can see it.")
    print("When both indicators turn GREEN, press SPACE to begin tracking.")
    print("Press Q to abort.")

    stable_count = 0
    ready        = False

    while True:
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()
        if not ret1 or not ret2:
            continue

        frame1 = cv2.undistort(frame1, cal1["mtx"], cal1["dist"])
        frame2 = cv2.undistort(frame2, cal2["mtx"], cal2["dist"])

        det1 = detector1.detect(frame1)
        det2 = detector2.detect(frame2)

        cam1_ok = det1.confidence >= min_conf
        cam2_ok = det2.confidence >= min_conf

        if cam1_ok and cam2_ok:
            stable_count += 1
        else:
            stable_count = 0
        if stable_count >= INIT_STABLE_FRAMES:
            ready = True

        # Side-by-side preview with status badges
        h, w   = frame1.shape[:2]
        half_w = w // 2
        left   = cv2.resize(frame1, (half_w, h))
        right  = cv2.resize(frame2, (half_w, h))

        def badge(img, label, ok, det):
            color = (0, 200, 0) if ok else (0, 0, 200)
            cv2.rectangle(img, (10, 10), (half_w - 10, 60), (0, 0, 0), -1)
            cv2.rectangle(img, (10, 10), (half_w - 10, 60), color, 2)
            text = f"{label}: {'OK' if ok else 'NO DETECT'}  conf={det.confidence:.2f}"
            cv2.putText(img, text, (20, 42),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
            if det.pixel_point is not None:
                # The resize maps width w -> half_w but leaves height untouched,
                # so only x gets scaled. Scaling y here is what made the circle
                # float about a frame-height above the actual tip.
                scale_x = half_w / w
                px = int(det.pixel_point[0] * scale_x)
                py = int(det.pixel_point[1])
                cv2.circle(img, (px, py), 8, color, 2)

        badge(left,  "CAM1", cam1_ok, det1)
        badge(right, "CAM2", cam2_ok, det2)
        combined = np.hstack([left, right])

        # Footer message
        footer = np.zeros((60, combined.shape[1], 3), dtype=np.uint8)
        if ready:
            msg, col = "READY  -  press SPACE to start tracking", (0, 220, 0)
        else:
            remaining = max(INIT_STABLE_FRAMES - stable_count, 0)
            msg = (f"Hold pen steady in view of BOTH cameras "
                   f"(stable frames: {stable_count}/{INIT_STABLE_FRAMES})")
            col = (200, 200, 200)
        cv2.putText(footer, msg, (15, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2)
        display = np.vstack([combined, footer])

        cv2.imshow(window, display)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            cv2.destroyWindow(window)
            return False
        if key == ord(' ') and ready:
            cv2.destroyWindow(window)
            print("Initialization complete - starting tracker.")
            return True


def run_screen(cfg):
    """
    Screen-mode runtime: each camera has its own homography mapping cam pixels
    directly to screen pixels inside its rectangle (see calibration/screen_calibrate.py).
    cam2 controls the LEFT rect, cam1 the RIGHT rect. No SensorFuser - the pen
    is normally in only one camera's zone at a time, so we pick whichever camera
    has the higher-confidence detection this frame.

    The plane-mm pipeline and the two-camera initialization handshake do not
    apply here, so this path is deliberately separate from run().
    """
    cam_cfg    = cfg["cameras"]
    out_cfg    = cfg["output"]
    min_conf   = cfg["fusion"]["min_confidence"]
    # Color reliability: freeze the cursor when the detected blob's mean HSV
    # sits too close to the edge of the configured range (i.e. it's marginal,
    # the kind of detection that usually turns out to be a reflection rather
    # than the actual pen). 0.0 disables the freeze.
    color_min  = float(cfg["detection"].get("color_confidence_min", 0.0))

    # cal["H"] now maps cam-pixels -> screen-pixels; cal["rect"] is that
    # camera's screen rectangle.
    cal1 = load_calibration("calibration/cam1_screen_calibration.npz")
    cal2 = load_calibration("calibration/cam2_screen_calibration.npz")
    rect1 = cal1.get("rect")
    rect2 = cal2.get("rect")
    if rect1 is None or rect2 is None:
        raise RuntimeError(
            "Screen calibration .npz missing 'rect' field - re-run "
            "calibration/screen_calibrate.py."
        )

    detector1 = OrangeTipDetector.from_config(cfg, cam_key="cam1")
    detector2 = OrangeTipDetector.from_config(cfg, cam_key="cam2")
    # One Kalman per camera. State is in screen pixels here, but the filter
    # is unit-agnostic - same constant-velocity model still applies.
    kalman1   = PenKalmanFilter.from_config(cfg)
    kalman2   = PenKalmanFilter.from_config(cfg)

    cursor_cfg     = cfg.get("cursor", {}) or {}
    cursor_enabled = bool(cursor_cfg.get("enabled", False))
    use_smoothed   = bool(cursor_cfg.get("use_smoothed", False))
    cursor         = CursorController.from_config(cfg) if cursor_enabled else None

    start_background_listener()
    prev_write = False
    prev_erase = False

    cap1 = _open_camera(cam_cfg["cam1_index"], cam_cfg)
    cap2 = _open_camera(cam_cfg["cam2_index"], cam_cfg)
    if cap1 is None or cap2 is None:
        print("ERROR: could not open one or both cameras "
              f"(cam1={cam_cfg['cam1_index']}, cam2={cam_cfg['cam2_index']}).")
        if cap1 is not None: cap1.release()
        if cap2 is not None: cap2.release()
        return

    print("Screen-mode tracker running. Press Q to quit.")
    prev_time = time.time()

    def _to_screen(pixel_point, H):
        pt  = np.array([[[pixel_point[0], pixel_point[1]]]], dtype=np.float32)
        out = cv2.perspectiveTransform(pt, H)
        return float(out[0][0][0]), float(out[0][0][1])

    def _clip_to_rect(sx, sy, rect):
        x0, y0, x1, y1 = int(rect[0]), int(rect[1]), int(rect[2]), int(rect[3])
        return max(x0, min(x1 - 1, sx)), max(y0, min(y1 - 1, sy))

    while True:
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()
        if not ret1 or not ret2:
            print("WARNING: Frame capture failed - skipping frame.")
            continue

        frame1 = cv2.undistort(frame1, cal1["mtx"], cal1["dist"])
        frame2 = cv2.undistort(frame2, cal2["mtx"], cal2["dist"])

        det1 = detector1.detect(frame1)
        det2 = detector2.detect(frame2)

        # A detection contributes only if its area-confidence AND its color
        # reliability clear their respective thresholds. Marginal-color
        # detections are filtered out here so the cursor "freezes" rather
        # than jumping to whatever reflection slipped through the HSV range.
        ok1 = (det1.pixel_point is not None
               and det1.confidence       >= min_conf
               and det1.color_confidence >= color_min)
        ok2 = (det2.pixel_point is not None
               and det2.confidence       >= min_conf
               and det2.color_confidence >= color_min)
        raw1 = _to_screen(det1.pixel_point, cal1["H"]) if ok1 else None
        raw2 = _to_screen(det2.pixel_point, cal2["H"]) if ok2 else None

        # Per-camera Kalman smoothing (in screen pixels).
        sm1 = kalman1.update(measurement=raw1, confidence=det1.confidence, min_confidence=min_conf)
        sm2 = kalman2.update(measurement=raw2, confidence=det2.confidence, min_confidence=min_conf)

        # Pick which camera drives the cursor this frame. The pen is normally
        # only physically in one zone, so the camera that didn't see it has
        # near-zero confidence and naturally loses the tiebreak.
        chosen = None
        if raw1 is not None and (raw2 is None or det1.confidence >= det2.confidence):
            chosen = ("cam1", raw1, sm1, rect1)
        elif raw2 is not None:
            chosen = ("cam2", raw2, sm2, rect2)

        if cursor is not None and chosen is not None:
            _, raw, sm, rect = chosen
            sx, sy = sm if use_smoothed else raw
            sx, sy = _clip_to_rect(sx, sy, rect)
            cursor.move_screen(sx, sy)

        # ── ESP write/erase buttons -> synthetic mouse buttons ──────────────
        write_now = is_write_pressed()
        erase_now = is_erase_pressed()
        if cursor is not None:
            if write_now and not prev_write:
                cursor.left_down()
            elif prev_write and not write_now:
                cursor.left_up()
            if erase_now and not prev_erase:
                cursor.right_down()
            elif prev_erase and not erase_now:
                cursor.right_up()
        prev_write = write_now
        prev_erase = erase_now

        now       = time.time()
        fps       = 1.0 / max(now - prev_time, 1e-6)
        prev_time = now

        if out_cfg["show_debug"]:
            h, w   = frame1.shape[:2]
            half_w = w // 2
            left   = cv2.resize(frame1, (half_w, h))
            right  = cv2.resize(frame2, (half_w, h))
            for img, det, label in ((left, det1, "CAM1->RIGHT"),
                                    (right, det2, "CAM2->LEFT")):
                ok = det.pixel_point is not None and det.confidence >= min_conf
                color = (0, 200, 0) if ok else (0, 0, 200)
                if det.pixel_point is not None:
                    px = int(det.pixel_point[0] * (half_w / w))
                    py = int(det.pixel_point[1])
                    cv2.circle(img, (px, py), 8, color, 2)
                cv2.putText(img, f"{label} conf={det.confidence:.2f}",
                            (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
            combined = np.hstack([left, right])
            bar = np.zeros((40, combined.shape[1], 3), dtype=np.uint8)
            src = chosen[0] if chosen else "lost"
            cv2.putText(bar, f"src: {src}   FPS: {fps:.1f}", (10, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            cv2.imshow("Pen Tracker - Screen Mode", np.vstack([combined, bar]))

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    if cursor is not None:
        if prev_write: cursor.left_up()
        if prev_erase: cursor.right_up()

    cap1.release()
    cap2.release()
    cv2.destroyAllWindows()


def run(config_path: str = "config.yaml"):
    # ── Load config ───────────────────────────────────────────────────────────
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    # Screen mode: cam-pixel -> screen-pixel per camera, no fusion. Different
    # enough from the plane-mm path that it gets its own runtime function.
    if bool((cfg.get("screen", {}) or {}).get("enabled", False)):
        return run_screen(cfg)

    cam_cfg    = cfg["cameras"]
    out_cfg    = cfg["output"]

    # ── Load calibration data ─────────────────────────────────────────────────
    # Each .npz contains: mtx (camera matrix), dist (distortion), H (homography)
    cal1 = load_calibration("calibration/cam1_calibration.npz")
    cal2 = load_calibration("calibration/cam2_calibration.npz")

    # ── Build pipeline components ─────────────────────────────────────────────
    detector1 = OrangeTipDetector.from_config(cfg, cam_key="cam1")
    detector2 = OrangeTipDetector.from_config(cfg, cam_key="cam2")
    fuser     = SensorFuser.from_config(cfg)
    kalman    = PenKalmanFilter.from_config(cfg)
    recorder  = StrokeRecorder(
        min_move_mm=out_cfg["min_move_mm"],
        enabled=out_cfg["save_strokes"],
    )

    # Cursor control (Win32 SetCursorPos via ctypes — single syscall per move).
    cursor_cfg     = cfg.get("cursor", {}) or {}
    cursor_enabled = bool(cursor_cfg.get("enabled", False))
    use_smoothed   = bool(cursor_cfg.get("use_smoothed", False))
    cursor         = CursorController.from_config(cfg) if cursor_enabled else None

    # ESP32-C3 button receiver — daemon thread updates module-level state.
    start_background_listener()
    prev_write = False
    prev_erase = False

    # ── Open cameras ─────────────────────────────────────────────────────────
    # MSMF (Windows default) frequently fails to start streaming when two USB
    # cameras share a controller. DSHOW handles multi-cam much better, so try
    # it first and fall back to MSMF / default. Mirrors tools/hsv_tuner.py.
    cap1 = _open_camera(cam_cfg["cam1_index"], cam_cfg)
    cap2 = _open_camera(cam_cfg["cam2_index"], cam_cfg)
    if cap1 is None or cap2 is None:
        print("ERROR: could not open one or both cameras "
              f"(cam1={cam_cfg['cam1_index']}, cam2={cam_cfg['cam2_index']}).")
        if cap1 is not None: cap1.release()
        if cap2 is not None: cap2.release()
        return

    # ── Initialization phase ──────────────────────────────────────────────
    # Ensures both cameras are working and seeds the Kalman with a real
    # measurement before the cursor starts moving.
    if not initialize_tracking(cap1, cap2, cal1, cal2, detector1, detector2, cfg):
        print("Initialization aborted by user.")
        cap1.release()
        cap2.release()
        cv2.destroyAllWindows()
        return

    print("Tracker running. Press Q to quit, S to save current stroke.")

    prev_time = time.time()

    while True:
        # ── Capture ───────────────────────────────────────────────────────────
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()

        if not ret1 or not ret2:
            print("WARNING: Frame capture failed — skipping frame.")
            continue

        # ── Undistort ─────────────────────────────────────────────────────────
        # Removes lens barrel/pincushion distortion using the intrinsic
        # calibration data. Must be done before detection so pixel coordinates
        # correspond to the same geometric space as the homography.
        frame1 = cv2.undistort(frame1, cal1["mtx"], cal1["dist"])
        frame2 = cv2.undistort(frame2, cal2["mtx"], cal2["dist"])

        # ── Detect ────────────────────────────────────────────────────────────
        det1: Detection = detector1.detect(frame1)
        det2: Detection = detector2.detect(frame2)

        # ── Map pixel → plane coords ──────────────────────────────────────────
        # Applies the homography to convert each detected pixel centroid into
        # real-world plane coordinates (mm from top-left corner of plane).
        pt1 = map_to_plane(det1.pixel_point, cal1["H"]) if det1.pixel_point else None
        pt2 = map_to_plane(det2.pixel_point, cal2["H"]) if det2.pixel_point else None

        # ── Fuse ──────────────────────────────────────────────────────────────
        fused = fuser.fuse(pt1, det1.confidence, pt2, det2.confidence)

        # ── Move cursor as soon as we have a fused point ─────────────────────
        # Done BEFORE Kalman so the cursor sees the latest measurement with no
        # filter lag. If you prefer smoother (slower) motion, set
        # cursor.use_smoothed: true in config.yaml.
        if cursor is not None and not use_smoothed and fused.plane_point is not None:
            cursor.move(*fused.plane_point)

        # ── ESP button edges -> mouse buttons + stroke segmentation ──────────
        # write = left mouse (draw),  erase = right mouse (alt action).
        # Edge detection so the OS sees a single down / single up per press.
        write_now = is_write_pressed()
        erase_now = is_erase_pressed()
        if cursor is not None:
            if write_now and not prev_write:
                cursor.left_down()
            elif prev_write and not write_now:
                cursor.left_up()
                recorder.save_stroke()   # close the current stroke on release
            if erase_now and not prev_erase:
                cursor.right_down()
            elif prev_erase and not erase_now:
                cursor.right_up()
        prev_write = write_now
        prev_erase = erase_now

        # ── Kalman smooth ─────────────────────────────────────────────────────
        x, y = kalman.update(
            measurement    = fused.plane_point,
            confidence     = fused.confidence,
            min_confidence = cfg["fusion"]["min_confidence"],
        )

        # Smoothed cursor path (opt-in: trades latency for less jitter).
        if cursor is not None and use_smoothed and fused.source != "lost":
            cursor.move(x, y)

        # ── Record stroke ─────────────────────────────────────────────────────
        if fused.source != "lost":
            recorder.add_point(x, y)

        # ── FPS counter ───────────────────────────────────────────────────────
        now      = time.time()
        fps      = 1.0 / max(now - prev_time, 1e-6)
        prev_time = now

        # ── Debug display ─────────────────────────────────────────────────────
        if out_cfg["show_debug"]:
            debug = draw_debug_overlay(
                frame1=frame1, frame2=frame2,
                det1=det1,     det2=det2,
                pt1=pt1,       pt2=pt2,
                fused=fused,   smoothed=(x, y),
                fps=fps,
            )
            cv2.imshow("Pen Tracker — Debug", debug)

        # ── Key handling ──────────────────────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            recorder.save_stroke()
            print(f"Stroke saved ({len(recorder.current_stroke)} points)")

    # ── Cleanup ───────────────────────────────────────────────────────────────
    # Release any synthetic buttons that might still be held down so the OS
    # isn't left in a stuck-mouse state if you quit mid-press.
    if cursor is not None:
        if prev_write: cursor.left_up()
        if prev_erase: cursor.right_up()

    cap1.release()
    cap2.release()
    cv2.destroyAllWindows()

    if out_cfg["save_strokes"]:
        recorder.flush("output/strokes.json")
        print("All strokes written to output/strokes.json")


if __name__ == "__main__":
    run()
