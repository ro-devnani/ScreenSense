"""
Interactive HSV range tuner.

Six sliders + live mask. Adjust until only the pen tip / target object is
white in the mask, then press S to save the values back to config.yaml.

Usage:
    python tools/hsv_tuner.py            # uses cam2_index from config (external)
    python tools/hsv_tuner.py --cam 0    # override (e.g. built-in laptop cam)

Keys:
    S  save current HSV values to config.yaml
    Q  quit
"""

import argparse
import time

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


def nothing(x):
    pass


def run(cam_index, config_path="config.yaml"):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    w   = cfg["cameras"]["width"]
    h   = cfg["cameras"]["height"]
    fps = cfg["cameras"]["fps"]

    if cam_index is None:
        cam_index = cfg["cameras"]["cam2_index"]

    cap = open_camera(cam_index, w, h, fps)
    if cap is None:
        print(f"FAIL: could not open camera {cam_index}")
        return

    # Seed sliders from current config values
    lo = cfg["detection"]["hsv_lower"]
    up = cfg["detection"]["hsv_upper"]

    cv2.namedWindow("HSV Tuner")
    cv2.createTrackbar("H min", "HSV Tuner", lo[0], 179, nothing)
    cv2.createTrackbar("H max", "HSV Tuner", up[0], 179, nothing)
    cv2.createTrackbar("S min", "HSV Tuner", lo[1], 255, nothing)
    cv2.createTrackbar("S max", "HSV Tuner", up[1], 255, nothing)
    cv2.createTrackbar("V min", "HSV Tuner", lo[2], 255, nothing)
    cv2.createTrackbar("V max", "HSV Tuner", up[2], 255, nothing)

    print("Adjust sliders until only the tip is white in the mask.")
    print("S = save to config.yaml,  Q = quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        hsv   = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower = np.array([
            cv2.getTrackbarPos("H min", "HSV Tuner"),
            cv2.getTrackbarPos("S min", "HSV Tuner"),
            cv2.getTrackbarPos("V min", "HSV Tuner"),
        ])
        upper = np.array([
            cv2.getTrackbarPos("H max", "HSV Tuner"),
            cv2.getTrackbarPos("S max", "HSV Tuner"),
            cv2.getTrackbarPos("V max", "HSV Tuner"),
        ])
        mask = cv2.inRange(hsv, lower, upper)

        mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        combined = np.hstack([
            cv2.resize(frame,    (w // 2, h // 2)),
            cv2.resize(mask_rgb, (w // 2, h // 2)),
        ])
        cv2.imshow("HSV Tuner", combined)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            cfg["detection"]["hsv_lower"] = lower.tolist()
            cfg["detection"]["hsv_upper"] = upper.tolist()
            with open(config_path, "w") as f:
                yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
            print(f"saved: lower={lower.tolist()}  upper={upper.tolist()}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cam", type=int, default=None,
                        help="Camera index (defaults to cam2_index from config)")
    args = parser.parse_args()
    run(args.cam)
