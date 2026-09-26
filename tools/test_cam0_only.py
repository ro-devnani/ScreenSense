"""
Minimal: open camera 0 only, with a warm-up loop.

Some Windows laptop cameras need 10–30 frames to start delivering
real data. This script keeps reading until it gets a valid frame
or hits the timeout, then displays the live feed.
"""

import cv2
import time

INDEX   = 0
TIMEOUT = 5.0     # seconds to wait for the first valid frame

cap = cv2.VideoCapture(INDEX)
if not cap.isOpened():
    print(f"FAIL: VideoCapture({INDEX}).isOpened() is False")
    raise SystemExit(1)

print("Waiting for first valid frame...")
t0 = time.time()
got = False
while time.time() - t0 < TIMEOUT:
    ret, frame = cap.read()
    if ret and frame is not None:
        got = True
        h, w = frame.shape[:2]
        print(f"  got first frame after {time.time() - t0:.2f}s  ({w}x{h})")
        break
    time.sleep(0.05)

if not got:
    print(f"FAIL: no valid frame from index {INDEX} within {TIMEOUT}s")
    cap.release()
    raise SystemExit(1)

print("Streaming. Press Q to quit.")
while True:
    ret, frame = cap.read()
    if ret and frame is not None:
        cv2.imshow(f"Camera {INDEX}", frame)
    if (cv2.waitKey(1) & 0xFF) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
