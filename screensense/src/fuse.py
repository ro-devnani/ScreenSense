from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np


@dataclass
class FusedEstimate:
    """
    Result of fusing both camera detections into one plane coordinate.

    Attributes:
        plane_point : (x, y) in plane mm coords, or None if both cameras lost the tip
        confidence  : combined confidence — lower when only one camera sees the tip,
                      lowest when both cameras lost it
        source      : "both" | "cam1_only" | "cam2_only" | "lost"
    """
    plane_point : Optional[Tuple[float, float]]
    confidence  : float
    source      : str


class SensorFuser:
    """
    Confidence-weighted fusion of two independent camera estimates.

    When both cameras see the tip, their estimates are averaged weighted by
    confidence. When only one camera sees it, that estimate is used with a
    penalty applied (because a single viewpoint is less reliable).

    Parameters
    ----------
    single_camera_penalty : multiplier on confidence when only one camera detects
    min_confidence        : detections below this are treated as "not found"
    """

    def __init__(self, single_camera_penalty: float = 0.6, min_confidence: float = 0.2):
        self.penalty        = single_camera_penalty
        self.min_confidence = min_confidence

    def fuse(
        self,
        pt1  : Optional[Tuple[float, float]], conf1: float,
        pt2  : Optional[Tuple[float, float]], conf2: float,
    ) -> FusedEstimate:
        """
        Fuse two camera estimates.

        Parameters
        ----------
        pt1, conf1 : plane point and confidence from camera 1 (None if not detected)
        pt2, conf2 : plane point and confidence from camera 2 (None if not detected)

        Returns
        -------
        FusedEstimate with merged plane_point, combined confidence, and source tag.
        """
        # Apply minimum confidence threshold — treat weak detections as "not found"
        if conf1 < self.min_confidence:
            pt1, conf1 = None, 0.0
        if conf2 < self.min_confidence:
            pt2, conf2 = None, 0.0

        # ── Both lost ─────────────────────────────────────────────────────────
        if pt1 is None and pt2 is None:
            return FusedEstimate(plane_point=None, confidence=0.0, source="lost")

        # ── Only camera 2 sees the tip ────────────────────────────────────────
        if pt1 is None:
            return FusedEstimate(
                plane_point=pt2,
                confidence=conf2 * self.penalty,
                source="cam2_only",
            )

        # ── Only camera 1 sees the tip ────────────────────────────────────────
        if pt2 is None:
            return FusedEstimate(
                plane_point=pt1,
                confidence=conf1 * self.penalty,
                source="cam1_only",
            )

        # ── Both cameras see the tip — weighted average ───────────────────────
        # The point closer to "ideal" blob size gets more weight.
        # Example: conf1=0.9, conf2=0.4 → pt1 gets ~69% of the weight.
        total = conf1 + conf2
        x     = (pt1[0] * conf1 + pt2[0] * conf2) / total
        y     = (pt1[1] * conf1 + pt2[1] * conf2) / total

        # Combined confidence: mean of both, never exceeds 1.0
        combined_conf = min((conf1 + conf2) / 2.0, 1.0)

        return FusedEstimate(
            plane_point=(x, y),
            confidence=combined_conf,
            source="both",
        )

    @classmethod
    def from_config(cls, cfg: dict) -> "SensorFuser":
        f = cfg["fusion"]
        return cls(
            single_camera_penalty=f["single_camera_penalty"],
            min_confidence=f["min_confidence"],
        )
