"""
Quick sanity check that both cameras open and stream.

Usage:
    python tools/camera_test.py

Press Q to quit. Each pane shows the camera index, actual frame size,
and live FPS. A red "NO SIGNAL" pane means that camera index failed
to open or is returning empty frames.
"""

import cv2
import numpy as np
import time
import yaml


def open_camera(index: int, width: int, height: int, fps: int):
    # DSHOW works best for most USB cams on Windows, but built-in laptop
    # webcams often only deliver frames via MSMF or the default backend.
    # Try each backend, both with the requested resolution and at native.
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
            # Force MJPG before setting resolution — raw YUY2 at 720p needs
            # ~110 MB/s per cam, which starves USB 2.0 when two cams share a
            # controller (symptom: ~3 fps). MJPG compresses on-camera and
            # drops bandwidth ~10x.
            cap.set(cv2.CAP_PROP_FOURCC,       cv2.VideoWriter_fourcc(*'MJPG'))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH,  width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            cap.set(cv2.CAP_PROP_FPS,          fps)
        for _ in range(30):
            ret, frame = cap.read()
            if ret and frame is not None:
                actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                actual_fps = cap.get(cv2.CAP_PROP_FPS)
                fourcc_int = int(cap.get(cv2.CAP_PROP_FOURCC))
                fourcc_str = "".join([chr((fourcc_int >> 8 * i) & 0xFF)
                                      for i in range(4)])
                res_note = f"{width}x{height}" if set_res else "native"
                print(f"    cam{index} warm-up OK via {name} ({res_note}), "
                      f"actual: {actual_w}x{actual_h} @ {actual_fps:.1f}fps "
                      f"fourcc={fourcc_str!r}")
                return cap
            time.sleep(0.05)
        print(f"    cam{index} no frames via {name} "
              f"({'requested' if set_res else 'native'} res)")
        cap.release()
    return None


def no_signal_pane(width: int, height: int, label: str) -> np.ndarray:
    pane = np.zeros((height, width, 3), dtype=np.uint8)
    cv2.putText(pane, f"{label}: NO SIGNAL", (20, height // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
    return pane


def annotate(frame: np.ndarray, label: str, fps: float) -> np.ndarray:
    h, w = frame.shape[:2]
    text = f"{label}  {w}x{h}  {fps:.1f} FPS"
    cv2.rectangle(frame, (0, 0), (w, 32), (0, 0, 0), -1)
    cv2.putText(frame, text, (10, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
    return frame


def run(config_path: str = "config.yaml"):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    cam_cfg = cfg["cameras"]
    idx1 = cam_cfg["cam1_index"]
    idx2 = cam_cfg["cam2_index"]
    w    = cam_cfg["width"]
    h    = cam_cfg["height"]
    fps  = cam_cfg["fps"]

    # Open the external camera first so it gets a clean DSHOW grab.
    # Opening the built-in cam first can leave it on MSMF/DEFAULT, which on
    # Windows often blocks a subsequent DSHOW open on a different device.
    print(f"Opening camera {idx2} at {w}x{h}@{fps}fps...")
    cap2 = open_camera(idx2, w, h, fps)
    print(f"  cam{idx2}: {'OK' if cap2 else 'FAILED to open'}")

    # Pause so the first camera fully initializes before opening the second.
    # Without this, Windows often invalidates the first camera's handle.
    print("  pausing 2s before opening second camera...")
    time.sleep(2.0)

    # Sanity-read cam 2 right before opening cam 1:
    if cap2 is not None:
        ret, _ = cap2.read()
        print(f"  cam{idx2} read just before opening cam{idx1}: ret={ret}")

    print(f"Opening camera {idx1} at {w}x{h}@{fps}fps...")
    cap1 = open_camera(idx1, w, h, fps)
    print(f"  cam{idx1}: {'OK' if cap1 else 'FAILED to open'}")

    # Sanity-read cam 2 right after opening cam 1:
    if cap2 is not None:
        ret, _ = cap2.read()
        print(f"  cam{idx2} read just after opening cam{idx1}: ret={ret}")

    print("Press Q to quit.")

    half_w, half_h = w // 2, h // 2
    prev_t1 = prev_t2 = time.time()
    fps1 = fps2 = 0.0

    while True:
        # Camera 1
        if cap1 is not None:
            ret1, frame1 = cap1.read()
            now = time.time()
            fps1 = 0.9 * fps1 + 0.1 * (1.0 / max(now - prev_t1, 1e-6))
            prev_t1 = now
            if ret1 and frame1 is not None:
                pane1 = annotate(cv2.resize(frame1, (half_w, half_h)),
                                 "CAM1", fps1)
            else:
                pane1 = no_signal_pane(half_w, half_h, "CAM1")
        else:
            pane1 = no_signal_pane(half_w, half_h, "CAM1")

        # Camera 2
        if cap2 is not None:
            ret2, frame2 = cap2.read()
            now = time.time()
            fps2 = 0.9 * fps2 + 0.1 * (1.0 / max(now - prev_t2, 1e-6))
            prev_t2 = now
            if ret2 and frame2 is not None:
                pane2 = annotate(cv2.resize(frame2, (half_w, half_h)),
                                 "CAM2", fps2)
            else:
                pane2 = no_signal_pane(half_w, half_h, "CAM2")
        else:
            pane2 = no_signal_pane(half_w, half_h, "CAM2")

        cv2.imshow("Camera Test (Q to quit)", np.hstack([pane1, pane2]))
        if (cv2.waitKey(1) & 0xFF) == ord('q'):
            break

    if cap1 is not None: cap1.release()
    if cap2 is not None: cap2.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run()
