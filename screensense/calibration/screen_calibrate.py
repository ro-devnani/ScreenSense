"""
Screen-rectangle calibration for the dual-camera pen tracker.

Both cameras look at the same physical drawing surface and are calibrated
against a SINGLE on-screen rectangle. Each camera gets its own homography
that maps its pixel coordinates into that rectangle in screen-pixel space;
at runtime the two estimates are confidence-fused and a single Kalman
filter smooths the result, so there are no per-camera state jumps when
one camera takes over from the other.

Workflow (3 phases, run by this script in order):

    1. Cursor phase  - the OS cursor is confined to the screen rectangle;
                       click N points anywhere inside it. These N screen-
                       pixel positions are the shared destination set.
    2. Pen phase 1   - for each of the N screen points (highlighted in turn
                       as a red target), place the orange pen on that
                       physical spot so cam1 detects the tip there.
                       Press ENTER to register the cam1 pixel.
    3. Pen phase 2   - same for cam2.

    H_cam1 maps cam1 pixels -> screen pixels inside the rectangle.
    H_cam2 maps cam2 pixels -> screen pixels inside the same rectangle.

Outputs (same shape as the plane-mm .npz so tracker.py can swap them in):
    calibration/cam1_screen_calibration.npz
    calibration/cam2_screen_calibration.npz

Each file contains: mtx, dist, H, rect ([x0,y0,x1,y1]), rms. `rect` is
identical between the two files.

Cursor confinement uses Win32 ClipCursor and is released the moment a
phase ends - it never persists beyond this script.
"""

import os
import sys
import cv2
import numpy as np
import argparse
import yaml
import ctypes

# Allow `from src.detect import ...` when run from the project root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.detect import OrangeTipDetector
from src.utils import set_config_list


WINDOW    = "Screen Calibration"
PT_COLOR  = (180, 120,   0)   # cursor-click point markers (BGR)
CAM_COLOR = ( 40, 160,  40)   # registered pen-pixel markers (BGR)


# ── Win32 helpers ─────────────────────────────────────────────────────────────

class _RECT(ctypes.Structure):
    _fields_ = [("left",   ctypes.c_long),
                ("top",    ctypes.c_long),
                ("right",  ctypes.c_long),
                ("bottom", ctypes.c_long)]


def _user32():
    return ctypes.windll.user32


def _clip_cursor(rect):
    x0, y0, x1, y1 = rect
    r = _RECT(int(x0), int(y0), int(x1), int(y1))
    _user32().ClipCursor(ctypes.byref(r))


def _unclip_cursor():
    _user32().ClipCursor(None)


def _set_cursor_pos(x, y):
    _user32().SetCursorPos(int(x), int(y))


def _get_screen_size():
    u = _user32()
    try:
        u.SetProcessDPIAware()
    except Exception:
        pass
    return u.GetSystemMetrics(0), u.GetSystemMetrics(1)


class _MONITORINFO(ctypes.Structure):
    _fields_ = [("cbSize",    ctypes.c_ulong),
                ("rcMonitor", _RECT),
                ("rcWork",    _RECT),
                ("dwFlags",   ctypes.c_ulong)]


def _enum_monitors():
    """Return [(x, y, w, h, is_primary), ...] for every connected display,
    sorted with the primary first and then by x-origin. Coordinates are in
    Win32 virtual-screen space (the primary monitor's top-left is (0,0);
    secondary monitors are at positive or negative offsets)."""
    _user32().SetProcessDPIAware()
    monitors = []

    MONITORENUMPROC = ctypes.WINFUNCTYPE(
        ctypes.c_int,
        ctypes.c_void_p,            # HMONITOR
        ctypes.c_void_p,            # HDC
        ctypes.POINTER(_RECT),      # LPRECT
        ctypes.c_void_p,            # LPARAM
    )

    def callback(hmon, hdc, lprect, data):
        info = _MONITORINFO()
        info.cbSize = ctypes.sizeof(_MONITORINFO)
        _user32().GetMonitorInfoW(hmon, ctypes.byref(info))
        r = info.rcMonitor
        monitors.append((int(r.left), int(r.top),
                         int(r.right - r.left), int(r.bottom - r.top),
                         bool(info.dwFlags & 1)))
        return 1

    _user32().EnumDisplayMonitors(0, 0, MONITORENUMPROC(callback), 0)
    # Primary first, then left-to-right by origin.
    monitors.sort(key=lambda m: (0 if m[4] else 1, m[0]))
    return monitors


def _resolve_monitor(monitor_index):
    """Return (x, y, w, h) for the requested monitor, falling back to the
    primary if the index is out of range."""
    mons = _enum_monitors()
    if not mons:
        w, h = _get_screen_size()
        return (0, 0, w, h)
    idx = max(0, min(len(mons) - 1, int(monitor_index)))
    if idx != int(monitor_index):
        print(f"WARNING: monitor_index={monitor_index} out of range; "
              f"using monitor {idx} of {len(mons)}.")
    x, y, w, h, _ = mons[idx]
    return (x, y, w, h)


# ── Layout ────────────────────────────────────────────────────────────────────

def _default_rect(origin_x, origin_y, w, h, pad=40):
    """A single rectangle filling the monitor inside `pad` pixels of margin,
    in GLOBAL virtual-screen coordinates."""
    return (origin_x + pad,
            origin_y + pad,
            origin_x + w - pad,
            origin_y + h - pad)


def _rect_from_cfg(rect_cfg, default):
    if rect_cfg is None:
        return default
    return tuple(int(v) for v in rect_cfg)


def _to_local_rect(rect, ox, oy):
    """Translate a GLOBAL rect into window-local coords for drawing."""
    return (rect[0] - ox, rect[1] - oy, rect[2] - ox, rect[3] - oy)


def _to_local_pt(pt, ox, oy):
    return (pt[0] - ox, pt[1] - oy)


# ── Drawing primitives ───────────────────────────────────────────────────────

def _blank(w, h):
    # White background - the physical drawing surface is matte white, so
    # matching the screen avoids reflections through the camera that a
    # black background would amplify.
    return np.full((h, w, 3), 255, dtype=np.uint8)


def _draw_rect(img, rect, color, thickness=2):
    cv2.rectangle(img, (rect[0], rect[1]), (rect[2], rect[3]), color, thickness)


def _draw_point(img, pt, idx, color):
    cv2.circle(img, (int(pt[0]), int(pt[1])),  8, color, -1)
    cv2.circle(img, (int(pt[0]), int(pt[1])), 14, color,  1)
    cv2.putText(img, str(idx + 1),
                (int(pt[0]) + 14, int(pt[1]) - 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)


def _draw_target(img, pt, idx):
    cv2.circle(img, pt, 26, (0,   0, 255), 2)
    cv2.circle(img, pt,  6, (0,   0, 255), -1)
    cv2.line(img, (pt[0] - 36, pt[1]), (pt[0] + 36, pt[1]), (0, 0, 255), 1)
    cv2.line(img, (pt[0], pt[1] - 36), (pt[0], pt[1] + 36), (0, 0, 255), 1)
    cv2.putText(img, f"#{idx + 1}", (pt[0] + 28, pt[1] - 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)


def _draw_prompt(img, lines, color=(0, 0, 0)):
    y = 36
    for line in lines:
        cv2.putText(img, line, (40, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        y += 32


def _composite_camera_inset(img, frame, det, inset_w=320, pad=24):
    """Paint a camera preview with detection overlay into the bottom-right."""
    if frame is None:
        return
    H, W = img.shape[:2]
    inset_h = int(frame.shape[0] * (inset_w / frame.shape[1]))
    preview = cv2.resize(frame, (inset_w, inset_h))
    if det.pixel_point is not None:
        sx = inset_w / frame.shape[1]
        sy = inset_h / frame.shape[0]
        px = int(det.pixel_point[0] * sx)
        py = int(det.pixel_point[1] * sy)
        cv2.circle(preview, (px, py), 8, (0, 255, 0), 2)
        cv2.line(preview, (px - 14, py), (px + 14, py), (0, 255, 0), 1)
        cv2.line(preview, (px, py - 14), (px, py + 14), (0, 255, 0), 1)
    else:
        cv2.putText(preview, "NO PEN DETECTED", (6, inset_h - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    x0 = W - inset_w - pad
    y0 = H - inset_h - pad
    img[y0:y0 + inset_h, x0:x0 + inset_w] = preview
    cv2.rectangle(img, (x0 - 2, y0 - 2),
                  (x0 + inset_w + 2, y0 + inset_h + 2),
                  (160, 160, 160), 1)


# ── Phase 1+2: cursor-click capture, confined to one rect ─────────────────────

def collect_cursor_points(monitor, rect, n_points, color):
    """Capture `n_points` mouse clicks anywhere inside `rect`. Cursor is
    clipped to `rect` for the duration. `rect` and the returned points are
    all in GLOBAL virtual-screen coordinates; drawing happens in window-
    local coords (= global minus the monitor's origin)."""
    ox, oy, mw, mh = monitor
    points = []  # GLOBAL coords

    def on_mouse(event, x, y, flags, param):
        # OpenCV mouse callback gives window-local pixels; the window is
        # placed at (ox, oy), so add that offset to recover global coords.
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < n_points:
            points.append((x + ox, y + oy))

    cv2.setMouseCallback(WINDOW, on_mouse)
    _set_cursor_pos((rect[0] + rect[2]) // 2, (rect[1] + rect[3]) // 2)
    _clip_cursor(rect)

    try:
        while True:
            canvas = _blank(mw, mh)
            _draw_rect(canvas, _to_local_rect(rect, ox, oy), (0, 0, 0), 2)
            for i, pt in enumerate(points):
                _draw_point(canvas, _to_local_pt(pt, ox, oy), i, color)
            done = (len(points) >= n_points)
            prompt = [
                "CURSOR phase",
                f"Click {n_points} points anywhere inside the rectangle. "
                "Spread them out for the best calibration.",
                f"Points: {len(points)}/{n_points}   "
                f"(U = undo,  Q = abort"
                + (",  SPACE = next phase" if done else "") + ")",
            ]
            _draw_prompt(canvas, prompt)
            cv2.imshow(WINDOW, canvas)

            key = cv2.waitKey(15) & 0xFF
            if key == ord('q'):
                raise RuntimeError("Aborted in cursor phase.")
            if key == ord('u') and points:
                points.pop()
            if key == ord(' ') and done:
                break
    finally:
        _unclip_cursor()
        # Remove the mouse callback so it cannot fire during later phases.
        cv2.setMouseCallback(WINDOW, lambda *a, **k: None)

    return points


# ── Phase 3+4: pen-detection capture for each previously-clicked screen point.

def collect_pen_points(cap, detector, mtx, dist,
                       monitor, rect, screen_points, label, color):
    """For each screen_point, wait for the user to put the pen on it (camera
    detection visible in the inset) and press ENTER to register the camera-
    pixel position, or S to skip it for this camera. Returns
    (pixel_points, screen_indices): pixel_points[i] in cam-pixel coords
    corresponds to screen_points[screen_indices[i]]. `rect` and
    `screen_points` are in GLOBAL coords."""
    ox, oy, mw, mh = monitor
    pixel_points   = []   # cam-pixel coords for registered points only
    screen_indices = []   # parallel: index into screen_points
    idx = 0

    while idx < len(screen_points):
        ret, frame = cap.read()
        if not ret or frame is None:
            continue
        undistorted = cv2.undistort(frame, mtx, dist)
        det = detector.detect(undistorted)

        canvas = _blank(mw, mh)
        _draw_rect(canvas, _to_local_rect(rect, ox, oy), (0, 0, 0), 2)

        # Mark already-handled points (registered or skipped) in a dimmed colour.
        dim = tuple(int(c * 0.35) for c in color)
        for i, pt in enumerate(screen_points):
            if i < idx:
                _draw_point(canvas, _to_local_pt(pt, ox, oy), i, dim)

        l_target = _to_local_pt(screen_points[idx], ox, oy)
        _draw_target(canvas, l_target, idx)
        _composite_camera_inset(canvas, undistorted, det)

        prompt = [
            f"PEN phase: {label}",
            f"Place the pen tip on the RED target ({idx + 1}/{len(screen_points)}).",
            "ENTER = register   S = skip (not visible)   U = undo   Q = abort",
        ]
        if det.pixel_point is None:
            prompt.append("(pen not detected - move it into the camera's view)")
        _draw_prompt(canvas, prompt)

        cv2.imshow(WINDOW, canvas)
        key = cv2.waitKey(15) & 0xFF

        if key == ord('q'):
            raise RuntimeError(f"Aborted in pen phase ({label}).")
        if key == ord('u') and idx > 0:
            idx -= 1
            # If the prior step was a register (not a skip), pop that record.
            if screen_indices and screen_indices[-1] == idx:
                pixel_points.pop()
                screen_indices.pop()
            print(f"  undid point {idx + 1}")
            continue
        if key == ord('s'):
            print(f"  skipped screen point {idx + 1} (not visible to {label})")
            idx += 1
            continue
        # ENTER is 10 or 13 depending on the OpenCV build.
        if key in (10, 13):
            if det.pixel_point is None:
                print("  no pen detected - cannot register this point.")
                continue
            pixel_points.append((float(det.pixel_point[0]),
                                 float(det.pixel_point[1])))
            screen_indices.append(idx)
            print(f"  cam pixel {pixel_points[-1]} -> screen {screen_points[idx]}")
            idx += 1

    return pixel_points, screen_indices


# ── Camera open (mirrors tracker.py's _open_camera) ──────────────────────────

def _open_camera(index, width, height, fps):
    attempts = [(cv2.CAP_DSHOW, "DSHOW"),
                (cv2.CAP_MSMF,  "MSMF"),
                (None,          "DEFAULT")]
    for backend, name in attempts:
        cap = (cv2.VideoCapture(index, backend) if backend is not None
               else cv2.VideoCapture(index))
        if not cap.isOpened():
            cap.release()
            continue
        cap.set(cv2.CAP_PROP_FOURCC,  cv2.VideoWriter_fourcc(*'MJPG'))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_FPS,          fps)
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
        for _ in range(30):
            ret, frame = cap.read()
            if ret and frame is not None:
                print(f"cam{index} opened via {name}")
                return cap
        cap.release()
    return None


def _default_intrinsics(w, h):
    """No-op intrinsics: identity-ish mtx, zero dist - matches calibrate.py."""
    mtx = np.array([[float(w), 0.0,      w / 2.0],
                    [0.0,      float(w), h / 2.0],
                    [0.0,      0.0,      1.0    ]], dtype=np.float64)
    dist = np.zeros(5, dtype=np.float64)
    return mtx, dist


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    cam_cfg  = cfg["cameras"]
    cam1_idx = cam_cfg["cam1_index"]
    cam2_idx = cam_cfg["cam2_index"]
    cam_w    = cam_cfg["width"]
    cam_h    = cam_cfg["height"]
    fps      = cam_cfg["fps"]

    screen_cfg    = cfg.get("screen", {}) or {}
    monitor_index = int(screen_cfg.get("monitor_index", 0))
    ox, oy, mw, mh = _resolve_monitor(monitor_index)
    monitor       = (ox, oy, mw, mh)
    print(f"Target monitor #{monitor_index}: origin=({ox},{oy})  size={mw}x{mh}")

    default_rect = _default_rect(ox, oy, mw, mh)
    rect         = _rect_from_cfg(screen_cfg.get("rect"), default_rect)
    n_points     = int(screen_cfg.get("calib_points", 8))
    if n_points < 4:
        raise RuntimeError(f"screen.calib_points must be >= 4 (got {n_points}).")
    print(f"Screen rect: {rect}")
    print(f"Calibration points per camera: {n_points}")

    # Place the window on the target monitor, then ask for fullscreen. On
    # Windows this fullscreens on whichever monitor the window currently
    # occupies, which is how we steer the UI to a non-primary display.
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.moveWindow(WINDOW, ox, oy)
    cv2.resizeWindow(WINDOW, mw, mh)
    cv2.setWindowProperty(WINDOW, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow(WINDOW, _blank(mw, mh))
    cv2.waitKey(50)

    try:
        print("\n=== Phase 1/3: cursor on rect ===")
        screen_pts = collect_cursor_points(monitor, rect, n_points, PT_COLOR)

        # Phase 2: pen via cam1
        print("\n=== Phase 2/3: pen on rect using cam1 ===")
        det1 = OrangeTipDetector.from_config(cfg, cam_key="cam1")
        cap1 = _open_camera(cam1_idx, cam_w, cam_h, fps)
        if cap1 is None:
            raise RuntimeError(f"could not open cam1 (index {cam1_idx})")
        mtx1, dist1 = _default_intrinsics(cam_w, cam_h)
        try:
            cam1_pixel_pts, cam1_screen_idx = collect_pen_points(
                cap1, det1, mtx1, dist1,
                monitor, rect, screen_pts, "cam1", CAM_COLOR)
        finally:
            cap1.release()

        # Phase 3: pen via cam2
        print("\n=== Phase 3/3: pen on rect using cam2 ===")
        det2 = OrangeTipDetector.from_config(cfg, cam_key="cam2")
        cap2 = _open_camera(cam2_idx, cam_w, cam_h, fps)
        if cap2 is None:
            raise RuntimeError(f"could not open cam2 (index {cam2_idx})")
        mtx2, dist2 = _default_intrinsics(cam_w, cam_h)
        try:
            cam2_pixel_pts, cam2_screen_idx = collect_pen_points(
                cap2, det2, mtx2, dist2,
                monitor, rect, screen_pts, "cam2", CAM_COLOR)
        finally:
            cap2.release()

    finally:
        cv2.destroyAllWindows()

    # ── Solve homographies (cam pixel -> screen pixel) ──────────────────────
    # Each camera may have skipped some screen points, so pair each cam's
    # registered pixels with only the screen points it actually saw.
    screen_pts_arr = np.array(screen_pts, dtype=np.float32)
    if len(cam1_pixel_pts) < 4:
        raise RuntimeError(f"cam1 registered only {len(cam1_pixel_pts)} points; "
                           f"need >= 4 to solve a homography.")
    if len(cam2_pixel_pts) < 4:
        raise RuntimeError(f"cam2 registered only {len(cam2_pixel_pts)} points; "
                           f"need >= 4 to solve a homography.")
    src1 = np.array(cam1_pixel_pts, dtype=np.float32)
    dst1 = screen_pts_arr[cam1_screen_idx]
    src2 = np.array(cam2_pixel_pts, dtype=np.float32)
    dst2 = screen_pts_arr[cam2_screen_idx]
    H1, mask1 = cv2.findHomography(src1, dst1, cv2.RANSAC, 5.0)
    if H1 is None:
        raise RuntimeError("cam1 homography solve failed - points may be degenerate.")
    print(f"cam1 homography inliers: {int(mask1.sum())}/{len(mask1)} "
          f"(of {len(screen_pts)} screen points)")
    H2, mask2 = cv2.findHomography(src2, dst2, cv2.RANSAC, 5.0)
    if H2 is None:
        raise RuntimeError("cam2 homography solve failed - points may be degenerate.")
    print(f"cam2 homography inliers: {int(mask2.sum())}/{len(mask2)} "
          f"(of {len(screen_pts)} screen points)")

    out1 = "calibration/cam1_screen_calibration.npz"
    out2 = "calibration/cam2_screen_calibration.npz"
    rect_arr = np.array(rect, dtype=np.int32)
    np.savez(out1, mtx=mtx1, dist=dist1, H=H1, rect=rect_arr, rms=np.float64(0.0))
    np.savez(out2, mtx=mtx2, dist=dist2, H=H2, rect=rect_arr, rms=np.float64(0.0))
    print(f"\nSaved:\n  {out1}\n  {out2}")

    # Persist the resolved rectangle back into config.yaml so subsequent
    # runs reuse the same bounds (and so the tracker has them on hand).
    if set_config_list(args.config, "rect", rect):
        print(f"Wrote rect={list(rect)} into {args.config}")
    else:
        print(f"NOTE: no `rect:` key under `screen:` in {args.config}; add "
              f"`rect: {list(rect)}` there to reuse these bounds.")

    print("\nSet screen.enabled: true in config.yaml to use these at runtime.")


if __name__ == "__main__":
    main()
