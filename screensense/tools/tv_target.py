"""
Two-camera detection of an orange object inside a defined target polygon
on a TV (or any planar surface). Both cameras face the TV; the object
moves in front of the screen. Each camera has its own polygon outlining
the target as seen from that camera's angle. "CONFIRMED" requires both
cameras to see the object inside their respective polygons, which
implicitly acts as a depth check: an object NOT at the TV plane only
lands inside both polygons at very specific (unlikely) 3D positions.

First run prompts for polygon vertices in each camera view and saves to
calibration/tv_target.npz. Re-run with --recalibrate to redo it.

Usage:
    python tools/tv_target.py
    python tools/tv_target.py --recalibrate

Calibration keys (per camera):
    LMB    add a vertex
    ENTER  finish polygon (need >=3 points)
    U      undo last vertex
    ESC    abort

Live keys:
    Q  quit
    R  recalibrate
"""

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import yaml


CAL_PATH = "calibration/tv_target.npz"


# ── Camera open ──────────────────────────────────────────────────────────────

def open_camera(index, width, height, fps):
    """Try DSHOW/MSMF/DEFAULT with and without requested res; warm up."""
    attempts = [
        (cv2.CAP_DSHOW, "DSHOW",   True),
        (cv2.CAP_DSHOW, "DSHOW",   False),
        (cv2.CAP_MSMF,  "MSMF",    True),
        (cv2.CAP_MSMF,  "MSMF",    False),
        (None,          "DEFAULT", True),
        (None,          "DEFAULT", False),
    ]
    for backend, name, set_res in attempts:
        cap = (cv2.VideoCapture(index, backend) if backend is not None
               else cv2.VideoCapture(index))
        if not cap.isOpened():
            cap.release()
            continue
        if set_res:
            cap.set(cv2.CAP_PROP_FOURCC,       cv2.VideoWriter_fourcc(*'MJPG'))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH,  width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            cap.set(cv2.CAP_PROP_FPS,          fps)
        for _ in range(30):
            ret, frame = cap.read()
            if ret and frame is not None:
                aw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                ah = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                print(f"  cam{index} via {name} at {aw}x{ah}")
                return cap
            time.sleep(0.05)
        cap.release()
    return None


# ── Calibration ──────────────────────────────────────────────────────────────

def collect_polygon_clicks(cap, label):
    """Show live feed, capture arbitrary polygon vertices. ENTER to finish."""
    clicks = []

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            clicks.append((x, y))
            print(f"  {label} vertex {len(clicks)} -> ({x}, {y})")

    win = f"Calibrate {label}: click polygon, ENTER to finish"
    cv2.namedWindow(win)
    cv2.setMouseCallback(win, on_mouse)

    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        disp = frame.copy()

        # Draw committed polygon edges (closed once >=3 points)
        if len(clicks) >= 2:
            pts_arr = np.array(clicks, dtype=np.int32)
            cv2.polylines(disp, [pts_arr],
                          isClosed=(len(clicks) >= 3),
                          color=(0, 255, 255), thickness=2)
        for i, (x, y) in enumerate(clicks):
            cv2.circle(disp, (x, y), 6, (0, 255, 255), -1)
            cv2.putText(disp, str(i + 1), (x + 8, y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        cv2.putText(disp,
                    f"{label}: {len(clicks)} pts | ENTER finish (>=3) | U undo | ESC abort",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 255, 255), 2)
        cv2.imshow(win, disp)

        key = cv2.waitKey(1) & 0xFF
        if key == 13 or key == 10:        # ENTER
            if len(clicks) >= 3:
                break
            else:
                print(f"  Need at least 3 vertices (have {len(clicks)})")
        elif key == 27:                    # ESC aborts
            cv2.destroyWindow(win)
            return None
        elif key == ord('u') or key == 8:  # U or Backspace: undo
            if clicks:
                removed = clicks.pop()
                print(f"  undo {removed}")

    cv2.destroyWindow(win)
    return np.array(clicks, dtype=np.float32)


def run_calibration(cap1, cap2):
    print("\n=== Calibration ===")
    print("Click any number of polygon vertices (>=3) outlining the target")
    print("area on the TV as seen from each camera. ENTER to finish, U to undo,")
    print("ESC to abort.\n")

    # External cam first.
    poly2 = collect_polygon_clicks(cap2, "CAM2 (external)")
    if poly2 is None:
        print("Calibration aborted.")
        return None
    poly1 = collect_polygon_clicks(cap1, "CAM1 (built-in)")
    if poly1 is None:
        print("Calibration aborted.")
        return None

    Path(CAL_PATH).parent.mkdir(parents=True, exist_ok=True)
    np.savez(CAL_PATH, poly1=poly1, poly2=poly2)
    print(f"Saved {len(poly1)}-vertex + {len(poly2)}-vertex polygons to {CAL_PATH}")
    return poly1, poly2


def load_calibration():
    if not Path(CAL_PATH).exists():
        return None
    data = np.load(CAL_PATH)
    # Tolerate old 4-corner files saved under "pts1"/"pts2"
    if "poly1" in data.files and "poly2" in data.files:
        return data["poly1"], data["poly2"]
    if "pts1" in data.files and "pts2" in data.files:
        return data["pts1"], data["pts2"]
    return None


def point_in_polygon(point, polygon):
    """True if point is inside or on the polygon edge."""
    return cv2.pointPolygonTest(polygon.astype(np.float32),
                                 (float(point[0]), float(point[1])),
                                 False) >= 0


# ── Detection ────────────────────────────────────────────────────────────────

def detect_tip(frame, hsv_lower, hsv_upper, min_area, kernel):
    hsv  = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, hsv_lower, hsv_upper)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    best = max(contours, key=cv2.contourArea)
    if cv2.contourArea(best) < min_area:
        return None
    M = cv2.moments(best)
    if M["m00"] == 0:
        return None
    return (M["m10"] / M["m00"], M["m01"] / M["m00"])


# ── Display ──────────────────────────────────────────────────────────────────

def draw_pane(frame, polygon, tip_pixel, in_box, label):
    disp = frame.copy()
    cv2.polylines(disp, [polygon.astype(int)], True, (255, 255, 0), 2)
    if tip_pixel is not None:
        color = (0, 255, 0) if in_box else (0, 165, 255)
        cv2.circle(disp, (int(tip_pixel[0]), int(tip_pixel[1])), 8, color, 2)
    txt_color = (0, 255, 0) if in_box else ((0, 165, 255) if tip_pixel else (0, 0, 255))
    state = "IN" if in_box else ("OUT" if tip_pixel else "LOST")
    cv2.putText(disp, f"{label}: {state}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, txt_color, 2)
    return disp


# ── Main loop ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--recalibrate", action="store_true",
                        help="Re-click the corners even if a saved file exists")
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    idx1 = cfg["cameras"]["cam1_index"]
    idx2 = cfg["cameras"]["cam2_index"]
    w    = cfg["cameras"]["width"]
    h    = cfg["cameras"]["height"]
    fps  = cfg["cameras"]["fps"]

    # External cam first so it gets DSHOW cleanly.
    print(f"Opening cam{idx2} (external)...")
    cap2 = open_camera(idx2, w, h, fps)
    print("  pausing 2s before opening second camera...")
    time.sleep(2.0)
    print(f"Opening cam{idx1} (built-in)...")
    cap1 = open_camera(idx1, w, h, fps)

    if cap1 is None or cap2 is None:
        print(f"FAIL: cam{idx1}={'OK' if cap1 else 'FAIL'}, "
              f"cam{idx2}={'OK' if cap2 else 'FAIL'}")
        for c in (cap1, cap2):
            if c is not None:
                c.release()
        return

    cal = None if args.recalibrate else load_calibration()
    if cal is None:
        cal = run_calibration(cap1, cap2)
        if cal is None:
            cap1.release()
            cap2.release()
            return
    poly1, poly2 = cal

    d         = cfg["detection"]
    hsv_lower = np.array(d["hsv_lower"], dtype=np.uint8)
    hsv_upper = np.array(d["hsv_upper"], dtype=np.uint8)
    min_area  = d["min_area"]
    ksize     = d["morph_kernel_size"]
    kernel    = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))

    print("\nTracking: Q quit  |  R recalibrate")

    while True:
        # grab both, then retrieve — keeps cameras roughly synced
        cap1.grab()
        cap2.grab()
        ret1, frame1 = cap1.retrieve()
        ret2, frame2 = cap2.retrieve()
        if not ret1 or not ret2:
            continue

        tip1 = detect_tip(frame1, hsv_lower, hsv_upper, min_area, kernel)
        tip2 = detect_tip(frame2, hsv_lower, hsv_upper, min_area, kernel)

        in1 = tip1 is not None and point_in_polygon(tip1, poly1)
        in2 = tip2 is not None and point_in_polygon(tip2, poly2)

        if in1 and in2:
            status = "CONFIRMED (both cams)"
            color  = (0, 255, 0)
        elif in1 or in2:
            status = f"TENTATIVE ({'CAM1 only' if in1 else 'CAM2 only'})"
            color  = (0, 165, 255)
        elif tip1 is not None or tip2 is not None:
            status = "OUTSIDE target"
            color  = (0, 0, 255)
        else:
            status = "LOST"
            color  = (0, 0, 255)

        disp1 = draw_pane(frame1, poly1, tip1, in1, "CAM1")
        disp2 = draw_pane(frame2, poly2, tip2, in2, "CAM2")

        # Match heights, side-by-side
        H_disp = max(disp1.shape[0], disp2.shape[0])
        W_each = min(disp1.shape[1], disp2.shape[1])
        disp1 = cv2.resize(disp1, (W_each, H_disp))
        disp2 = cv2.resize(disp2, (W_each, H_disp))
        combined = np.hstack([disp1, disp2])

        bar = np.zeros((60, combined.shape[1], 3), dtype=np.uint8)
        cv2.putText(bar, status, (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        cv2.imshow("TV Target", np.vstack([combined, bar]))

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            print("Recalibrating...")
            new_cal = run_calibration(cap1, cap2)
            if new_cal is not None:
                poly1, poly2 = new_cal

    cap1.release()
    cap2.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
