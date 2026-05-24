"""
Probe camera indices 0..N to find every device OpenCV can open.

Usage:
    python tools/probe_cameras.py

Tries both default and DSHOW backends — on Windows, the right backend
varies by device.  A camera that responds at a given index here is one
you can put in config.yaml.
"""

import cv2

MAX_INDEX = 8

def try_backend(index, backend, name):
    cap = cv2.VideoCapture(index, backend) if backend else cv2.VideoCapture(index)
    if not cap.isOpened():
        return None
    ret, frame = cap.read()
    cap.release()
    if ret and frame is not None:
        h, w = frame.shape[:2]
        return f"{name} OK ({w}x{h})"
    return f"{name} opened but returned no frame"

def main():
    print(f"Probing camera indices 0..{MAX_INDEX - 1}")
    print("-" * 60)
    for i in range(MAX_INDEX):
        results = []
        for backend, name in [(None, "DEFAULT"), (cv2.CAP_DSHOW, "DSHOW"),
                              (cv2.CAP_MSMF, "MSMF")]:
            r = try_backend(i, backend, name)
            if r:
                results.append(r)
        if results:
            print(f"index {i}:")
            for r in results:
                print(f"  - {r}")
        else:
            print(f"index {i}: (none)")
    print("-" * 60)
    print("Put the index of your EXTERNAL camera into config.yaml as cam2_index.")

if __name__ == "__main__":
    main()
