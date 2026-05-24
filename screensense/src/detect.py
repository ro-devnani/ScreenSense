import cv2
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class Detection:
    """
    Result of a single-camera detection attempt.

    Attributes:
        pixel_point      : (u, v) in undistorted pixel space, or None if not found
        confidence       : float in [0.0, 1.0] — 0.0 = not found, 1.0 = ideal blob size
        mask             : the binary HSV mask (for debugging / display)
        area             : contour area of the detected blob in pixels²
        color_confidence : float in [0.0, 1.0] — how centered the blob's mean
                           HSV sits inside the configured range. 1.0 = exactly
                           at range center on every channel; 0.0 = touching an
                           edge. Detections living at the range edge are the
                           ones most likely to be false positives (reflections,
                           ambient orange-ish pixels), so the tracker can
                           freeze the cursor when this drops too low.
        mean_hsv         : (H, S, V) means inside the blob — handy for debugging.
    """
    pixel_point      : Optional[Tuple[float, float]]
    confidence       : float
    mask             : np.ndarray
    area             : float
    color_confidence : float = 0.0
    mean_hsv         : Optional[Tuple[float, float, float]] = None


class OrangeTipDetector:
    """
    Detects an orange pen tip against a black background using HSV thresholding.

    The black background is key: V (value/brightness) near 0 means background
    pixels will never satisfy the V_min constraint in the orange range, giving
    a clean binary separation with no extra background removal needed.

    Parameters
    ----------
    hsv_lower    : (H_min, S_min, V_min) — lower bound of the orange HSV range
    hsv_upper    : (H_max, S_max, V_max) — upper bound of the orange HSV range
    min_area     : minimum blob area in px² to accept as a real detection
    ideal_area   : blob area in px² considered "fully visible" (used for confidence)
    kernel_size  : size of the morphological structuring element (must be odd)
    roi          : optional (x0, y0, x1, y1) crop rectangle to speed up processing
    """

    def __init__(
        self,
        hsv_lower    : Tuple[int, int, int] = (5,  160, 100),
        hsv_upper    : Tuple[int, int, int] = (22, 255, 255),
        min_area     : float = 20.0,
        ideal_area   : float = 300.0,
        kernel_size  : int   = 5,
        roi          : Optional[Tuple[int, int, int, int]] = None,
    ):
        self.lower       = np.array(hsv_lower, dtype=np.uint8)
        self.upper       = np.array(hsv_upper, dtype=np.uint8)
        self.min_area    = min_area
        self.ideal_area  = ideal_area
        self.roi         = roi

        # Elliptical kernel — better than rectangular for roughly circular blobs
        self.kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
        )

    def detect(self, frame: np.ndarray) -> Detection:
        """
        Run detection on a single undistorted BGR frame.

        Returns a Detection object. If no valid blob is found, pixel_point is
        None and confidence is 0.0, but mask is still returned for debugging.
        """
        # ── Step 1: Optionally crop to ROI ───────────────────────────────────
        if self.roi is not None:
            x0, y0, x1, y1 = self.roi
            region = frame[y0:y1, x0:x1]
            roi_offset = (x0, y0)
        else:
            region     = frame
            roi_offset = (0, 0)

        # ── Step 2: BGR → HSV ─────────────────────────────────────────────────
        # HSV separates hue (color identity) from saturation and brightness.
        # This makes the orange range consistent across different lighting levels
        # as long as the hue doesn't shift — which is why you tune S and V
        # minimums rather than relying on exact brightness values.
        hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)

        # ── Step 3: Threshold ─────────────────────────────────────────────────
        # inRange produces a binary image:
        #   white (255) = pixel falls within [lower, upper] in all three channels
        #   black (0)   = pixel is outside the range
        # Against a black background (V ≈ 0), background pixels always fail the
        # V_min constraint, so they are always black in the mask.
        mask = cv2.inRange(hsv, self.lower, self.upper)

        # ── Step 4: Morphological cleanup ─────────────────────────────────────
        # OPEN  = erosion then dilation: removes isolated white speckles
        #         that are smaller than the kernel. Eliminates salt noise.
        # CLOSE = dilation then erosion: fills small black holes inside white blobs.
        #         Handles cases where part of the tip is slightly under-exposed.
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  self.kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel)

        # ── Step 5: Find blobs ────────────────────────────────────────────────
        # RETR_EXTERNAL: only outer contours (no nested contours needed here)
        # CHAIN_APPROX_SIMPLE: compress horizontal/vertical/diagonal runs to
        #                      endpoints only — saves memory on large contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return Detection(pixel_point=None, confidence=0.0, mask=mask, area=0.0)

        # Pick the largest blob — the pen tip should always be the dominant
        # orange object in the frame when using a black background
        best   = max(contours, key=cv2.contourArea)
        area   = cv2.contourArea(best)

        if area < self.min_area:
            # Blob exists but is too small — likely a reflection or noise artifact
            return Detection(pixel_point=None, confidence=0.0, mask=mask, area=area)

        # ── Step 6: Locate the tip (not the centroid) ─────────────────────────
        # The centroid of an orange pen body is in the middle of the shaft,
        # not at the writing end. Use PCA on the contour points to find the
        # long axis, then pick the endpoint of that axis lowest in the frame
        # (the cameras look down at the drawing surface, so the tip is the
        # most-bottom extremity). For nearly-circular blobs PCA is unstable,
        # so we fall back to the centroid in that case.
        pts = best.reshape(-1, 2).astype(np.float32)
        mean, eigvecs, eigvals = cv2.PCACompute2(pts, mean=None)

        # Aspect ratio of variances along the two principal axes.
        # >> 1.0 means an elongated blob.
        long_var, short_var = float(eigvals[0]), float(eigvals[1])
        elongation = (long_var / short_var) if short_var > 1e-6 else float('inf')

        if elongation > 2.0:
            # Project every contour point onto the long axis and take the two
            # extremes — those are the ends of the elongated blob.
            axis     = eigvecs[0]                       # 2-vector, unit length
            centred  = pts - mean                       # (N, 2)
            proj     = centred @ axis                   # scalar per point
            end_lo   = pts[int(np.argmin(proj))]
            end_hi   = pts[int(np.argmax(proj))]
            # Pick the endpoint with the larger y (lower in the image).
            tip      = end_hi if end_hi[1] > end_lo[1] else end_lo
            cx, cy   = float(tip[0]), float(tip[1])
        else:
            # Round-ish blob — image moments are well-defined here.
            M  = cv2.moments(best)
            cx = M["m10"] / M["m00"]
            cy = M["m01"] / M["m00"]

        # ── Step 7: Offset back to full-frame pixel space ─────────────────────
        full_cx = cx + roi_offset[0]
        full_cy = cy + roi_offset[1]

        # ── Step 8: Confidence score ──────────────────────────────────────────
        # Ratio of detected area to ideal area, clamped to [0, 1].
        # A partially occluded tip has a smaller blob → lower confidence.
        # The Kalman filter and fusion step will weight this estimate accordingly.
        confidence = float(min(area / self.ideal_area, 1.0))

        # ── Step 9: Color reliability ─────────────────────────────────────────
        # Mean HSV of the blob's pixels; the closer that mean sits to the
        # center of [lower, upper] per channel, the more confident we are
        # that the blob is genuinely the pen and not a marginal range-edge
        # match (which is what reflections and stray bright spots tend to
        # be). Returns 1.0 at range center on every channel, 0.0 at any edge.
        mean_h, mean_s, mean_v = cv2.mean(hsv, mask=mask)[:3]
        color_confidence = self._color_confidence(mean_h, mean_s, mean_v)

        return Detection(
            pixel_point=(full_cx, full_cy),
            confidence=confidence,
            mask=mask,
            area=area,
            color_confidence=color_confidence,
            mean_hsv=(float(mean_h), float(mean_s), float(mean_v)),
        )

    def _color_confidence(self, mean_h, mean_s, mean_v) -> float:
        """How centered the blob's mean HSV is within the configured range.
        1.0 = exact center on every channel, 0.0 = at any edge."""
        def score(m, lo, hi):
            lo, hi = float(lo), float(hi)
            if hi <= lo:
                return 0.0
            center = (lo + hi) * 0.5
            half   = (hi - lo) * 0.5
            return max(0.0, 1.0 - abs(float(m) - center) / half)

        s_h = score(mean_h, self.lower[0], self.upper[0])
        s_s = score(mean_s, self.lower[1], self.upper[1])
        s_v = score(mean_v, self.lower[2], self.upper[2])
        # Hue gets double weight — for a coloured pen, hue mismatch is the
        # strongest signal that we're looking at the wrong object.
        return (2.0 * s_h + s_s + s_v) / 4.0

    @classmethod
    def from_config(cls, cfg: dict, cam_key: str = None) -> "OrangeTipDetector":
        """Construct a detector from a parsed config.yaml dict."""
        d   = cfg["detection"]
        roi = d.get(f"roi_{cam_key}") if cam_key else None
        return cls(
            hsv_lower   = tuple(d["hsv_lower"]),
            hsv_upper   = tuple(d["hsv_upper"]),
            min_area    = d["min_area"],
            ideal_area  = d["ideal_area"],
            kernel_size = d["morph_kernel_size"],
            roi         = tuple(roi) if roi else None,
        )
