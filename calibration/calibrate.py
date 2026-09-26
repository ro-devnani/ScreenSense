"""
Calibration script for both cameras.

Usage:
    python calibration/calibrate.py --cam 0 --out calibration/cam1_calibration.npz
    python calibration/calibrate.py --cam 1 --out calibration/cam2_calibration.npz

The two cameras provide the position estimate together: each one maps its
detected pen-tip pixel to plane mm via its own homography H, and src/fuse.py
combines the two plane estimates (weighted by confidence). So intrinsic
checkerboard calibration is not required to triangulate position — this script
captures only the homography per camera and stores a no-op distortion model
(identity-ish mtx, zero dist) so the rest of the pipeline can still call
cv2.undistort harmlessly.

Steps performed:
    1. Open the camera.
    2. Track the pen tip and press ENTER at each of the four plane corners,
       then SPACE to finish — this gives the homography H.
    3. Save mtx (identity-ish), dist (zeros), and H to the output .npz file.
"""

import os
import sys
import cv2
import numpy as np
import argparse
import yaml

# Allow `from src.detect import ...` when running this file from the project root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.detect import OrangeTipDetector


def default_intrinsics(width, height):
    """
    Build a no-op camera matrix + zero distortion. With dist all zero,
    cv2.undistort returns the input unchanged, so downstream code that always
    calls undistort works correctly without any real intrinsic calibration.
    The focal length is set to the image width as a reasonable placeholder
    (the value does not affect anything because dist is zero).
    """
    mtx = np.array([
        [float(width), 0.0,           width  / 2.0],
        [0.0,          float(width),  height / 2.0],
        [0.0,          0.0,           1.0          ],
    ], dtype=np.float64)
    dist = np.zeros(5, dtype=np.float64)
    return mtx, dist


# ── Homography calibration ────────────────────────────────────────────────────

CORNER_NAMES = ["top-left", "top-right", "bottom-right", "bottom-left"]


def _draw_corner_hint(display, next_idx):
    """Render a small rectangle in the top-right of `display` with the next
    corner highlighted in red. Pass next_idx=None when no corner is pending."""
    h, w = display.shape[:2]
    # Diagram box: anchored to top-right of the frame
    box_w, box_h = 110, 80
    pad          = 12
    x0 = w - box_w - pad
    y0 = pad
    x1 = x0 + box_w
    y1 = y0 + box_h

    # Translucent background so it doesn't obscure the live frame too much
    overlay = display.copy()
    cv2.rectangle(overlay, (x0 - 4, y0 - 4), (x1 + 4, y1 + 4), (40, 40, 40), -1)
    cv2.addWeighted(overlay, 0.55, display, 0.45, 0, display)

    # The plane outline
    cv2.rectangle(display, (x0, y0), (x1, y1), (200, 200, 200), 1)

    # Corner dots in plane order: TL, TR, BR, BL
    corner_pts = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
    for i, pt in enumerate(corner_pts):
        if i == next_idx:
            cv2.circle(display, pt, 7, (0, 0, 255), -1)   # next: red filled
        elif next_idx is not None and i < next_idx:
            cv2.circle(display, pt, 5, (0, 255, 255), -1)  # done: yellow
        else:
            cv2.circle(display, pt, 4, (180, 180, 180), 1)  # pending: outline

    cv2.putText(display, "next corner", (x0 - 4, y1 + 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (220, 220, 220), 1)


def compute_homography(cam_index, mtx, dist, width, height, plane_corners_mm, cfg):
    """
    Displays a live undistorted frame with the detected pen tip overlaid.
    The user moves the pen to each plane corner and presses ENTER to register
    the current pen-tip pixel position. Once all four corners are registered,
    pressing SPACE closes the polygon and finalises the homography. Order:
        top-left → top-right → bottom-right → bottom-left

    Keys:
        ENTER  register the current pen-tip position as the next corner
        SPACE  finish the shape (only after all 4 corners are set)
        U      undo the last registered point
        Q      abort

    Returns:
        H: 3×3 homography matrix mapping pixel coords → plane mm coords
    """
    detector = OrangeTipDetector.from_config(cfg)
    corner_points = []

    cap = cv2.VideoCapture(cam_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    window = f"Homography — Camera {cam_index}"
    cv2.namedWindow(window)

    print(f"\n[CAM {cam_index}] Place the pen tip at each corner of the plane.")
    print("  Order: top-left -> top-right -> bottom-right -> bottom-left")
    print("  ENTER = register current pen tip,  U = undo,")
    print("  SPACE = finish the shape (after all 4 corners),  Q = abort")

    finished = False
    while not finished:
        ret, frame = cap.read()
        if not ret:
            continue

        # Detector runs on the undistorted frame so the recorded corner
        # coordinates are in the same pixel space the rest of the pipeline uses.
        undistorted = cv2.undistort(frame, mtx, dist)
        detection   = detector.detect(undistorted)
        display     = undistorted.copy()

        # Draw the live pen-tip crosshair
        if detection.pixel_point is not None:
            px, py = int(detection.pixel_point[0]), int(detection.pixel_point[1])
            cv2.circle(display, (px, py), 10, (0, 255, 0), 2)
            cv2.line(display, (px - 18, py), (px + 18, py), (0, 255, 0), 1)
            cv2.line(display, (px, py - 18), (px, py + 18), (0, 255, 0), 1)
            cv2.putText(display, f"tip ({px},{py})  conf={detection.confidence:.2f}",
                        (px + 14, py - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 255, 0), 1)
        else:
            cv2.putText(display, "Pen tip not detected", (20, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Draw points already registered, plus polygon edges between them
        for i, pt in enumerate(corner_points):
            cv2.circle(display, pt, 6, (0, 255, 255), -1)
            cv2.putText(display, str(i + 1), (pt[0] + 8, pt[1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        if len(corner_points) >= 2:
            pts_arr = np.array(corner_points, dtype=np.int32)
            cv2.polylines(display, [pts_arr], isClosed=(len(corner_points) == 4),
                          color=(0, 255, 255), thickness=1)

        # Status / next-action prompt + corner-indicator diagram
        if len(corner_points) < 4:
            next_idx  = len(corner_points)
            next_name = CORNER_NAMES[next_idx]
            prompt = (f"Step {next_idx + 1}/4: place pen at {next_name.upper()} "
                      f"corner of the plane, then press ENTER")
        else:
            next_idx = None
            prompt = "All 4 corners set — press SPACE to finish, U to undo"
        cv2.putText(display, prompt, (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

        # Mini-diagram in the top-right showing which corner is next
        _draw_corner_hint(display, next_idx)

        # Side-by-side mask preview so detection issues are visible
        mask_bgr = cv2.cvtColor(detection.mask, cv2.COLOR_GRAY2BGR)
        cv2.putText(mask_bgr, f"mask  area={detection.area:.0f}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        combined = np.hstack([display, mask_bgr])

        cv2.imshow(window, combined)
        key = cv2.waitKey(1) & 0xFF

        # ENTER on Windows/Linux is 13, on some OpenCV builds 10.
        if key in (10, 13):
            if len(corner_points) >= 4:
                print("  4 corners already set — press SPACE to finish or U to undo.")
            elif detection.pixel_point is None:
                print("  No pen tip detected — cannot register a point.")
            else:
                px, py = detection.pixel_point
                pt = (int(round(px)), int(round(py)))
                corner_points.append(pt)
                print(f"  Registered {CORNER_NAMES[len(corner_points) - 1]}: {pt}")
        elif key == ord(' '):
            if len(corner_points) < 4:
                print(f"  Need 4 corners before finishing "
                      f"(have {len(corner_points)}).")
            else:
                finished = True
        elif key == ord('u') and corner_points:
            removed = corner_points.pop()
            print(f"  Undid point: {removed}")
        elif key == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            raise RuntimeError("Homography calibration aborted by user (Q pressed).")

    cap.release()
    cv2.destroyAllWindows()

    pixel_pts = np.array(corner_points,    dtype=np.float32)
    plane_pts = np.array(plane_corners_mm, dtype=np.float32)

    # findHomography: maps FROM pixel coords TO plane mm coords
    H, mask = cv2.findHomography(pixel_pts, plane_pts, cv2.RANSAC, 5.0)
    if H is None:
        # Happens when corners are collinear or duplicated (e.g. the pen
        # never moved between ENTER presses). Saving None would produce an
        # .npz the tracker can't load.
        raise RuntimeError("Homography solve failed - the four corners are "
                           "degenerate. Re-run and spread them around the plane.")
    print(f"  Homography computed. Inlier ratio: {int(mask.sum())}/{len(mask)}")
    return H

# ── Entry point ───────────────────────────────────────────────────────────────

def _run_one(cam_index, out_path, W, H_res, corners, cfg):
    """Capture one camera's homography and write its calibration .npz."""
    mtx, dist = default_intrinsics(W, H_res)
    print(f"\n=== Homography calibration (Camera {cam_index}) ===")
    H_mat = compute_homography(cam_index, mtx, dist, W, H_res, corners, cfg)
    np.savez(out_path, mtx=mtx, dist=dist, H=H_mat, rms=np.float64(0.0))
    print(f"Calibration saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cam",  type=int,
                        help="Camera index (omit if using --both)")
    parser.add_argument("--out",  type=str,
                        help="Output .npz path (omit if using --both)")
    parser.add_argument("--both", action="store_true",
                        help="Calibrate both cameras back-to-back, using "
                             "cam1_index/cam2_index from config and writing "
                             "to calibration/cam1_calibration.npz and "
                             "calibration/cam2_calibration.npz.")
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    W       = cfg["cameras"]["width"]
    H_res   = cfg["cameras"]["height"]
    corners = cfg["plane"]["corners_mm"]

    if args.both:
        cam1_idx = cfg["cameras"]["cam1_index"]
        cam2_idx = cfg["cameras"]["cam2_index"]
        # Run cam1 first, then cam2. compute_homography raises RuntimeError
        # if the user aborts with Q — catch it so the second cam doesn't
        # silently get skipped (and so it's clear which one was aborted).
        try:
            _run_one(cam1_idx, "calibration/cam1_calibration.npz",
                     W, H_res, corners, cfg)
        except RuntimeError as e:
            print(f"cam1 calibration aborted: {e}")
            raise SystemExit(1)
        try:
            _run_one(cam2_idx, "calibration/cam2_calibration.npz",
                     W, H_res, corners, cfg)
        except RuntimeError as e:
            print(f"cam2 calibration aborted: {e}")
            raise SystemExit(1)
        print("\nBoth cameras calibrated.")
    else:
        if args.cam is None or args.out is None:
            parser.error("must specify --cam and --out, or use --both")
        _run_one(args.cam, args.out, W, H_res, corners, cfg)
