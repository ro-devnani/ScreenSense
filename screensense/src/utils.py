import cv2
import numpy as np
import json
import re
from pathlib import Path
from typing import Sequence, Tuple


# ── Calibration I/O ──────────────────────────────────────────────────────────

def load_calibration(path: str) -> dict:
    """
    Load a calibration .npz file saved by calibrate.py or screen_calibrate.py.

    Returns a dict with keys: mtx, dist, H, rms, and (for screen-mode files)
    rect — a 4-int [x0, y0, x1, y1] screen rectangle for that camera. The
    rect entry is None for plane-mm calibrations that don't store it.
    """
    data = np.load(path)
    return {
        "mtx"  : data["mtx"],
        "dist" : data["dist"],
        "H"    : data["H"],
        "rms"  : float(data["rms"]),
        "rect" : data["rect"] if "rect" in data.files else None,
    }


# ── Config I/O ────────────────────────────────────────────────────────────────

def set_config_list(config_path: str, key: str, values: Sequence[int]) -> bool:
    """
    Rewrite the value of `key:` in config.yaml as a flow list ([a, b, c]),
    leaving every other line (including comments) untouched.

    Handles both flow style (`key: [1, 2]`) and block style (`key:` followed
    by `- 1` lines), so tools that edit the same file can't leave orphaned
    list items behind. Returns False if the key isn't present.
    """
    fmt = "[" + ", ".join(str(int(v)) for v in values) + "]"
    with open(config_path, "r", encoding="utf-8") as f:
        text = f.read()
    pattern = rf"(?m)^([ \t]*{re.escape(key)}:)[^\n]*(?:\n[ \t]*-[^\n]*)*"
    text, count = re.subn(pattern, lambda m: f"{m.group(1)} {fmt}", text, count=1)
    if count:
        with open(config_path, "w", encoding="utf-8") as f:
            f.write(text)
    return bool(count)


# ── Coordinate mapping ────────────────────────────────────────────────────────

def map_to_plane(
    pixel_point: Tuple[float, float],
    H: np.ndarray,
) -> Tuple[float, float]:
    """
    Apply a homography matrix to map a pixel coordinate to plane mm coordinates.

    Parameters
    ----------
    pixel_point : (u, v) in undistorted pixel space
    H           : 3×3 homography matrix (pixel → plane mm)

    Returns
    -------
    (x, y) in plane mm coords
    """
    pt  = np.array([[[pixel_point[0], pixel_point[1]]]], dtype=np.float32)
    out = cv2.perspectiveTransform(pt, H)
    return (float(out[0][0][0]), float(out[0][0][1]))


# ── Debug overlay ─────────────────────────────────────────────────────────────

def draw_debug_overlay(
    frame1, frame2,
    det1, det2,
    pt1, pt2,
    fused, smoothed,
    fps: float,
) -> np.ndarray:
    """
    Compose a debug view:
    - Left half:  Camera 1 frame with mask overlay and detected centroid
    - Right half: Camera 2 frame with mask overlay and detected centroid
    - Bottom bar: fusion source, confidence, smoothed plane coord, FPS

    Returns a single BGR image.
    """
    h, w = frame1.shape[:2]

    # Convert masks to 3-channel for display
    mask1_rgb = cv2.cvtColor(det1.mask, cv2.COLOR_GRAY2BGR)
    mask2_rgb = cv2.cvtColor(det2.mask, cv2.COLOR_GRAY2BGR)

    # Blend mask over frame at 40% opacity
    overlay1  = cv2.addWeighted(frame1, 0.6, mask1_rgb, 0.4, 0)
    overlay2  = cv2.addWeighted(frame2, 0.6, mask2_rgb, 0.4, 0)

    # Draw detected centroid circles
    for overlay, det, label in [
        (overlay1, det1, "CAM1"),
        (overlay2, det2, "CAM2"),
    ]:
        color = (0, 255, 0) if det.pixel_point else (0, 0, 255)
        if det.pixel_point:
            cv2.circle(overlay, (int(det.pixel_point[0]), int(det.pixel_point[1])),
                       8, color, 2)
        # Hershey fonts are ASCII-only; a "²" here renders as "??".
        conf_text = f"{label}  conf={det.confidence:.2f}  area={det.area:.0f}px^2"
        cv2.putText(overlay, conf_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    # Resize to half width for side-by-side display
    half_w    = w // 2
    left_half  = cv2.resize(overlay1, (half_w, h))
    right_half = cv2.resize(overlay2, (half_w, h))
    combined   = np.hstack([left_half, right_half])

    # Status bar
    bar     = np.zeros((50, combined.shape[1], 3), dtype=np.uint8)
    src_col = {"both": (0, 200, 0), "cam1_only": (0, 165, 255),
               "cam2_only": (0, 165, 255), "lost": (0, 0, 200)}
    color   = src_col.get(fused.source, (255, 255, 255))
    status  = (
        f"Source: {fused.source:<10}  "
        f"Conf: {fused.confidence:.2f}  "
        f"Plane: ({smoothed[0]:.1f}, {smoothed[1]:.1f}) mm  "
        f"FPS: {fps:.1f}"
    )
    cv2.putText(bar, status, (10, 33),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)

    return np.vstack([combined, bar])


# ── Stroke recording ──────────────────────────────────────────────────────────

class StrokeRecorder:
    """
    Records pen strokes as lists of (x, y, timestamp) tuples.

    Filters out points that haven't moved more than min_move_mm to reduce
    duplicate entries from a stationary or barely-moving pen.
    """

    def __init__(self, min_move_mm: float = 1.0, enabled: bool = True):
        self.min_move_sq     = min_move_mm ** 2   # compare squared distance (faster)
        self.enabled         = enabled
        self.current_stroke  : list = []
        self.all_strokes     : list = []
        self._last_point     = None

    def add_point(self, x: float, y: float):
        if not self.enabled:
            return

        import time
        pt = (x, y, time.time())

        if self._last_point is not None:
            dx = x - self._last_point[0]
            dy = y - self._last_point[1]
            if dx * dx + dy * dy < self.min_move_sq:
                return     # point too close to last — skip

        self.current_stroke.append(pt)
        self._last_point = (x, y)

    def save_stroke(self):
        """Finish the current stroke and start a new one."""
        if self.current_stroke:
            self.all_strokes.append(self.current_stroke.copy())
            self.current_stroke = []
            self._last_point    = None

    def flush(self, path: str):
        """Write all strokes to a JSON file."""
        self.save_stroke()
        out = {
            "strokes": [
                [{"x": p[0], "y": p[1], "t": p[2]} for p in stroke]
                for stroke in self.all_strokes
            ]
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(out, f, indent=2)
