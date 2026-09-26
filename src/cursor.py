"""
Direct OS cursor control from plane (mm) coordinates.

Uses Win32 SetCursorPos via ctypes — no third-party libraries, no PyAutoGUI
failsafe delays, just one syscall per move for the lowest latency available
from Python on Windows.
"""

import ctypes
import sys


class CursorController:
    """
    Maps plane mm coordinates to screen pixels and moves the OS cursor.

    Parameters
    ----------
    plane_width_mm, plane_height_mm : real-world plane dimensions
    screen_width, screen_height     : optional pixel overrides; auto-detected
                                       (DPI-aware) from the primary monitor
    invert_y                        : flip y axis if the plane's +y direction
                                       is opposite the screen's +y direction
    """

    def __init__(
        self,
        plane_width_mm : float,
        plane_height_mm: float,
        screen_width   : int  = None,
        screen_height  : int  = None,
        invert_y       : bool = False,
    ):
        if sys.platform != "win32":
            raise RuntimeError("CursorController currently supports Windows only.")

        self.plane_w  = float(plane_width_mm)
        self.plane_h  = float(plane_height_mm)
        self.invert_y = invert_y

        user32 = ctypes.windll.user32
        # Without this, GetSystemMetrics returns scaled (virtual) pixels on
        # high-DPI displays and the cursor lands in the wrong spot.
        try:
            user32.SetProcessDPIAware()
        except Exception:
            pass

        self.screen_w = int(screen_width  or user32.GetSystemMetrics(0))
        self.screen_h = int(screen_height or user32.GetSystemMetrics(1))

        # Virtual-screen bounds (the union of every connected monitor in
        # global Win32 coords). move_screen() clamps to this rect instead of
        # the primary monitor so screen-mode homographies that land on a
        # secondary display aren't clipped back onto the primary one.
        # SM_{X,Y,CX,CY}VIRTUALSCREEN = 76, 77, 78, 79.
        self.virt_x  = int(user32.GetSystemMetrics(76))
        self.virt_y  = int(user32.GetSystemMetrics(77))
        self.virt_w  = int(user32.GetSystemMetrics(78))
        self.virt_h  = int(user32.GetSystemMetrics(79))

        # Cache the function pointer — avoids the ctypes attribute lookup
        # on every move() call.
        self._set_cursor_pos = user32.SetCursorPos
        self._mouse_event    = user32.mouse_event

        # MOUSEEVENTF_* flags for synthetic mouse button events.
        self._FLAG_LEFT_DOWN  = 0x0002
        self._FLAG_LEFT_UP    = 0x0004
        self._FLAG_RIGHT_DOWN = 0x0008
        self._FLAG_RIGHT_UP   = 0x0010

    def move(self, plane_x_mm: float, plane_y_mm: float) -> None:
        """Move the OS cursor to the screen pixel that corresponds to
        (plane_x_mm, plane_y_mm). One syscall, no waiting."""
        sx = plane_x_mm / self.plane_w * self.screen_w
        sy = plane_y_mm / self.plane_h * self.screen_h
        if self.invert_y:
            sy = self.screen_h - sy

        # Clamp to the screen so off-plane jitter cannot send the cursor to
        # negative coordinates (SetCursorPos clips silently but int() on
        # negative floats rounds toward zero, which we'd rather not rely on).
        sxi = int(max(0, min(self.screen_w - 1, sx)))
        syi = int(max(0, min(self.screen_h - 1, sy)))

        self._set_cursor_pos(sxi, syi)

    def move_screen(self, sx: float, sy: float) -> None:
        """Move the OS cursor to a screen pixel directly, bypassing the
        plane-mm scaling. Used by screen-mode tracking where the per-camera
        homography already produces screen pixels in GLOBAL virtual-screen
        space (so values can sit on any monitor, not just the primary)."""
        sxi = int(max(self.virt_x, min(self.virt_x + self.virt_w - 1, sx)))
        syi = int(max(self.virt_y, min(self.virt_y + self.virt_h - 1, sy)))
        self._set_cursor_pos(sxi, syi)

    # ── Mouse button events ───────────────────────────────────────────────
    # Each call issues a single synthetic mouse_event at the cursor's
    # current position. Pair down() with up() — leaving a button stuck
    # down because of a crash will affect the whole OS, not just this app.

    def left_down(self)  -> None: self._mouse_event(self._FLAG_LEFT_DOWN,  0, 0, 0, 0)
    def left_up(self)    -> None: self._mouse_event(self._FLAG_LEFT_UP,    0, 0, 0, 0)
    def right_down(self) -> None: self._mouse_event(self._FLAG_RIGHT_DOWN, 0, 0, 0, 0)
    def right_up(self)   -> None: self._mouse_event(self._FLAG_RIGHT_UP,   0, 0, 0, 0)

    @classmethod
    def from_config(cls, cfg: dict) -> "CursorController":
        c = cfg.get("cursor", {}) or {}
        return cls(
            plane_width_mm  = cfg["plane"]["width_mm"],
            plane_height_mm = cfg["plane"]["height_mm"],
            screen_width    = c.get("screen_width"),
            screen_height   = c.get("screen_height"),
            invert_y        = bool(c.get("invert_y", False)),
        )
