"""
Single-camera stroke drawing test.

Opens one camera, detects the orange pen tip via HSV thresholding, and
accumulates the tip's path on an overlay over the live feed. No calibration
required — uses pixel coordinates directly.

Usage:
    python tools/draw_test.py            # uses cam2_index from config (external)
    python tools/draw_test.py --cam 0    # override (e.g. built-in laptop cam)

Keys (in the live window):
    Q  quit
    C  clear the canvas
    S  save the current canvas to output/draw_test_<timestamp>.png

If the trail doesn't follow your pen, the HSV range needs tuning:
    python tools/hsv_tuner.py --cam <N>
"""

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import yaml


def open_camera(index, width, height, fps):
    """Open across DSHOW/MSMF/DEFAULT, with and without requested resolution."""
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
                print(f"cam{index} opened via {name} at {aw}x{ah}")
                return cap
            time.sleep(0.05)
        cap.release()
    return None


def detect_tip(frame, hsv_lower, hsv_upper, min_area, kernel):
    hsv  = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, hsv_lower, hsv_upper)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, mask

    best = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(best)
    if area < min_area:
        return None, mask

    M = cv2.moments(best)
    if M["m00"] == 0:
        return None, mask

    return (M["m10"] / M["m00"], M["m01"] / M["m00"]), mask


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cam",    type=int, default=None,
                        help="Camera index (defaults to cam2_index from config)")
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    cam_index = args.cam if args.cam is not None else cfg["cameras"]["cam2_index"]
    w   = cfg["cameras"]["width"]
    h   = cfg["cameras"]["height"]
    fps = cfg["cameras"]["fps"]

    d         = cfg["detection"]
    hsv_lower = np.array(d["hsv_lower"], dtype=np.uint8)
    hsv_upper = np.array(d["hsv_upper"], dtype=np.uint8)
    min_area  = d["min_area"]
    ksize     = d["morph_kernel_size"]
    kernel    = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))

    cap = open_camera(cam_index, w, h, fps)
    if cap is None:
        print(f"FAIL: could not open camera {cam_index}")
        return

    aw  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    ah  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    canvas  = np.zeros((ah, aw, 3), dtype=np.uint8)
    last_pt = None

    print("Q quit  |  C clear canvas  |  S save canvas to output/")

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        tip, mask = detect_tip(frame, hsv_lower, hsv_upper, min_area, kernel)

        if tip is not None:
            pt = (int(tip[0]), int(tip[1]))
            if last_pt is not None:
                cv2.line(canvas, last_pt, pt, (0, 255, 0), 2)
            last_pt = pt
        else:
            # Tip lost → break stroke so next detection starts a fresh line.
            last_pt = None

        display = cv2.addWeighted(frame, 1.0, canvas, 1.0, 0)
        if tip is not None:
            cv2.circle(display, (int(tip[0]), int(tip[1])), 6, (0, 0, 255), 2)
        status = "TRACKING" if tip is not None else "NO TIP"
        cv2.putText(display, status, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0, 255, 0) if tip is not None else (0, 0, 255), 2)

        cv2.imshow("Draw test — live", display)
        cv2.imshow("HSV mask", mask)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            canvas[:] = 0
            last_pt   = None
            print("canvas cleared")
        elif key == ord('s'):
            Path("output").mkdir(exist_ok=True)
            out_path = f"output/draw_test_{int(time.time())}.png"
            cv2.imwrite(out_path, canvas)
            print(f"saved {out_path}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
