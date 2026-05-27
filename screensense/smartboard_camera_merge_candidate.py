import base64
import ctypes
import html
import math
import os
import re
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from io import BytesIO
from pathlib import Path

# Tell Windows we'll handle our own DPI scaling. Without this, Qt receives
# coordinates from a virtualised "compatibility" coordinate space and the
# overlay's geometry math is off on the secondary monitor (the overlay
# ends up sized for the primary monitor's logical resolution and never
# crosses the boundary). Must run BEFORE QApplication is created, hence
# module-level rather than tucked inside __main__.
if sys.platform == "win32":
    try:
        # PROCESS_PER_MONITOR_DPI_AWARE = 2, available since Windows 8.1.
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except (AttributeError, OSError):
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except (AttributeError, OSError):
            pass

try:
    from anthropic import Anthropic
except ImportError:
    Anthropic = None

# Windows screenshot path: Pillow's ImageGrab supports `all_screens=True`
# for virtual-desktop capture across every monitor. If Pillow isn't
# installed, the overlay still runs but the AI tools can't snapshot
# the screen and we surface a clear hint.
try:
    from PIL import ImageGrab
except ImportError:
    ImageGrab = None

from pynput import keyboard
from PyQt6.QtCore import QByteArray, QBuffer, QIODevice, QObject, QPoint, QRect, QSize, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QFont, QImage, QKeySequence, QPainter, QPainterPath, QPen, QPixmap, QRegion
from PyQt6.QtWidgets import (
    QApplication,
    QFrame,
    QLabel,
    QMessageBox,
    QPushButton,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)


ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
CLAUDE_MODEL = "claude-sonnet-4-20250514"
CLAUDE_IMAGE_LIMIT_BYTES = 3_600_000
BOOKMARKS_DIR = Path(__file__).resolve().parent / "smartboard_bookmarks"
DEBUG_CAPTURE_DIR = Path(__file__).resolve().parent / "smartboard_debug_captures"

claude_client = Anthropic(api_key=ANTHROPIC_API_KEY) if Anthropic and ANTHROPIC_API_KEY else None


@dataclass
class FrontWindowCapture:
    png_bytes: bytes
    global_rect: QRect
    window_id: int | None
    owner: str


@dataclass
class Stroke:
    points: list[QPoint]
    color: QColor
    width: int
    highlighter: bool = False


@dataclass
class AiBox:
    tool: str
    rect: QRect


class HotkeyBridge(QObject):
    toggle_requested = pyqtSignal()


class AiBridge(QObject):
    result_ready = pyqtSignal(str, str)


class SmartboardOverlay(QWidget):
    AI_TOOLS = {"solve", "similar", "translate"}

    def __init__(self):
        super().__init__()
        self.overlay_on = False
        self.sidebar_open = True
        self.tool = "pen"
        self.drawing = False
        self.ai_busy = False

        self.start = QPoint()
        self.preview = QPoint()
        self.current_points: list[QPoint] = []

        self.strokes: list[Stroke] = []
        self.ai_boxes: list[AiBox] = []
        self.pending_ai_box: AiBox | None = None

        self.context: FrontWindowCapture | None = None
        self.context_pixmap = QPixmap()
        self.context_attempted = False
        self.translation_target = "English"
        self.ai_panel_tool = None

        self._w_is_down = False
        self._last_toggle_time = 0.0

        self.hotkey_bridge = HotkeyBridge()
        self.hotkey_bridge.toggle_requested.connect(self.toggle_overlay)
        self.ai_bridge = AiBridge()
        self.ai_bridge.result_ready.connect(self.show_ai_popup)

        self.configure_window()
        self.build_sidebar()
        self.build_ai_panel()
        self.hide_overlay()
        self.start_hotkey_listener()

    def configure_window(self):
        self.setWindowTitle("ScreenSense Smartboard Overlay")
        # On Windows, FramelessWindowHint + WindowStaysOnTopHint + Tool gives
        # us a click-through-capable overlay that floats above normal windows
        # and stays out of the taskbar. WA_TranslucentBackground gives us the
        # transparent canvas. NoDropShadowWindowHint avoids the system shadow
        # bleeding around the frameless rect.
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
            | Qt.WindowType.WindowDoesNotAcceptFocus
            | Qt.WindowType.NoDropShadowWindowHint
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground, True)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating, True)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self.setAttribute(Qt.WidgetAttribute.WA_NativeWindow, True)
        self.setAutoFillBackground(False)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.setStyleSheet("background: transparent;")
        self.winId()
        self.move_to_all_screens()

    def _virtual_screen_rect(self):
        """Bounding rect of every connected monitor in global pixel coords.
        Uses raw Win32 GetSystemMetrics on Windows - Qt's screen geometry
        can return scaled values when monitors have different DPI factors
        even with PROCESS_PER_MONITOR_DPI_AWARE, which makes the overlay
        come up sized for the primary monitor only."""
        if sys.platform == "win32":
            try:
                user32 = ctypes.windll.user32
                # SM_{X,Y,CX,CY}VIRTUALSCREEN = 76, 77, 78, 79.
                return QRect(
                    int(user32.GetSystemMetrics(76)),
                    int(user32.GetSystemMetrics(77)),
                    int(user32.GetSystemMetrics(78)),
                    int(user32.GetSystemMetrics(79)),
                )
            except Exception as exc:
                print(f"GetSystemMetrics fallback: {exc}", flush=True)

        screens = QApplication.screens()
        if not screens:
            return None
        return screens[0].virtualGeometry()

    def move_to_all_screens(self):
        rect = self._virtual_screen_rect()
        if rect is None:
            return
        self.setGeometry(rect)
        self.move(rect.topLeft())
        self.resize(rect.size())
        # On Windows, frameless+topmost tool windows are occasionally
        # re-snapped to the primary monitor by the DWM after Qt's
        # setGeometry. Bypass Qt with the raw SetWindowPos call to lock
        # the HWND across the full virtual desktop. SWP_NOACTIVATE keeps
        # focus on whatever the user was using; SWP_FRAMECHANGED forces
        # the new rect to take effect immediately.
        if sys.platform == "win32":
            try:
                hwnd = int(self.winId())
                HWND_TOPMOST     = -1
                SWP_NOACTIVATE   = 0x0010
                SWP_FRAMECHANGED = 0x0020
                ctypes.windll.user32.SetWindowPos(
                    hwnd, HWND_TOPMOST,
                    rect.x(), rect.y(), rect.width(), rect.height(),
                    SWP_NOACTIVATE | SWP_FRAMECHANGED,
                )
            except Exception as exc:
                print(f"SetWindowPos warning: {exc}", flush=True)

        # Diagnostic: lets us tell whether the overlay's "stuck on primary"
        # is caused by a wrong target rect (virt mismatch) or by Windows
        # ignoring our setGeometry/SetWindowPos. Compare `target` to
        # `actual` to find out.
        actual = self.geometry()
        screens = [f"({s.geometry().x()},{s.geometry().y()} "
                   f"{s.geometry().width()}x{s.geometry().height()})"
                   for s in QApplication.screens()]
        print(
            f"[overlay] target={rect.x()},{rect.y()} {rect.width()}x{rect.height()} | "
            f"actual={actual.x()},{actual.y()} {actual.width()}x{actual.height()} | "
            f"Qt screens={','.join(screens)}",
            flush=True,
        )

    def build_sidebar(self):
        self.sidebar = QFrame(self)
        self.sidebar.setObjectName("sidebar")
        self.sidebar.setGeometry(self.expanded_sidebar_rect())
        self.sidebar.setStyleSheet(
            """
            QFrame#sidebar {
                background-color: rgba(14, 18, 26, 226);
                border: 1px solid rgba(255, 255, 255, 42);
                border-radius: 24px;
            }
            """
        )

        layout = QVBoxLayout()
        layout.setContentsMargins(10, 12, 10, 12)
        layout.setSpacing(9)
        self.sidebar.setLayout(layout)

        # Icons use Unicode glyphs covered by Segoe UI Emoji (Windows 10+).
        # Each tuple: (internal name, glyph, hover tooltip). The internal
        # names are stable - everything else (select_tool, the ESP32
        # bridge, paint code) keys off them, so don't rename.
        buttons = [
            ("menu",      "☰",   "Collapse menu"),                                # ☰
            ("scroll",    "\U0001F5B1", "Cursor mode - use laptop normally"),          # 🖱
            ("pen",       "✏",   "Draw pen"),                                      # ✏
            ("eraser",    "⌫",   "Erase strokes"),                                 # ⌫
            ("highlight", "\U0001F58D", "Highlight"),                                   # 🖍
            ("similar",   "?",        "Box a problem and generate similar questions"),
            ("solve",     "⚡",   "Box a problem and solve step by step"),          # ⚡
            ("translate", "\U0001F310", "Box text and translate"),                      # 🌐
            ("clear",     "✕",   "Clear overlay drawings"),                        # ✕
        ]

        self.buttons: dict[str, QPushButton] = {}
        for name, icon, tooltip in buttons:
            button = QPushButton(icon)
            button.setToolTip(tooltip)
            button.setCursor(Qt.CursorShape.PointingHandCursor)
            button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            button.setFixedSize(54, 46)
            # Segoe UI Emoji covers both the BMP symbols and the SMP emoji
            # used above. Falls back gracefully if it isn't installed.
            button.setFont(QFont("Segoe UI Emoji", 17, QFont.Weight.Bold))
            button.clicked.connect(lambda checked=False, selected=name: self.select_tool(selected))
            layout.addWidget(button)
            self.buttons[name] = button

        self.refresh_buttons()

    def build_ai_panel(self):
        self.ai_panel = QFrame(self)
        self.ai_panel.setObjectName("ai_panel")
        self.ai_panel.setStyleSheet(
            """
            QFrame#ai_panel {
                background-color: rgba(9, 17, 28, 238);
                border: 1px solid rgba(255, 255, 255, 42);
                border-radius: 18px;
            }
            QLabel {
                color: white;
            }
            QTextBrowser {
                background-color: rgba(20, 34, 51, 220);
                color: #eef6ff;
                border: none;
                border-radius: 12px;
                padding: 10px;
                font-size: 15px;
            }
            QPushButton {
                background-color: rgba(255, 255, 255, 230);
                color: #111827;
                border: none;
                border-radius: 9px;
                padding: 8px 16px;
            }
            """
        )

        layout = QVBoxLayout()
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(10)
        self.ai_panel.setLayout(layout)

        self.ai_title = QLabel("ScreenSense AI")
        self.ai_title.setFont(QFont("Segoe UI", 20, QFont.Weight.Bold))
        layout.addWidget(self.ai_title)

        self.ai_body = QTextBrowser()
        self.ai_body.setOpenExternalLinks(False)
        self.ai_body.setMinimumSize(480, 230)
        layout.addWidget(self.ai_body)

        close_button = QPushButton("Close")
        close_button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        close_button.clicked.connect(self.ai_panel.hide)
        layout.addWidget(close_button, alignment=Qt.AlignmentFlag.AlignRight)
        self.ai_panel.hide()

    def expanded_sidebar_rect(self):
        return QRect(24, 44, 76, 510)

    def button_style(self, active=False):
        if active:
            return """
                QPushButton {
                    background-color: rgba(255, 255, 255, 240);
                    color: #111827;
                    border: 1px solid rgba(255, 255, 255, 140);
                    border-radius: 16px;
                }
            """
        return """
            QPushButton {
                background-color: rgba(255, 255, 255, 34);
                color: white;
                border: 1px solid rgba(255, 255, 255, 20);
                border-radius: 16px;
            }
            QPushButton:hover {
                background-color: rgba(255, 255, 255, 82);
            }
            QPushButton:pressed {
                background-color: rgba(125, 211, 252, 210);
                color: #111827;
            }
        """

    def refresh_buttons(self):
        for name, button in self.buttons.items():
            button.setStyleSheet(self.button_style(name == self.tool))

    def select_tool(self, name):
        if name == "menu":
            self.toggle_sidebar()
            return
        if name == "clear":
            self.exit_scroll_mode()
            self.strokes.clear()
            self.ai_boxes.clear()
            self.pending_ai_box = None
            self.ai_busy = False
            self.ai_panel.hide()
            self.tool = "pen"
            self.refresh_buttons()
            self.update()
            return
        if name == "scroll":
            self.enter_scroll_mode()
            return
        if name == "translate":
            self.translation_target = "English"
        self.exit_scroll_mode()
        self.tool = name
        self.refresh_buttons()

    def enter_scroll_mode(self):
        self.tool = "scroll"
        self.drawing = False
        self.current_points.clear()
        self.refresh_buttons()
        self.apply_scroll_mode_mask()
        self.update()

    def exit_scroll_mode(self):
        self.clearMask()

    def apply_scroll_mode_mask(self):
        self.clearMask()
        sidebar_rect = QRect(self.sidebar.geometry()).adjusted(-2, -2, 2, 2)
        self.setMask(QRegion(sidebar_rect))

    def toggle_sidebar(self):
        self.sidebar_open = not self.sidebar_open
        if self.sidebar_open:
            self.sidebar.setGeometry(self.expanded_sidebar_rect())
            self.buttons["menu"].setText("☰")
            for button in self.buttons.values():
                button.show()
            if self.tool == "scroll":
                self.apply_scroll_mode_mask()
            return

        self.sidebar.setGeometry(QRect(24, 44, 76, 70))
        self.buttons["menu"].setText("▶")
        for name, button in self.buttons.items():
            if name != "menu":
                button.hide()
        if self.tool == "scroll":
            self.apply_scroll_mode_mask()

    def start_hotkey_listener(self):
        def on_press(key):
            try:
                if key.char and key.char.lower() == "w" and not self._w_is_down:
                    self._w_is_down = True
                    self.hotkey_bridge.toggle_requested.emit()
            except AttributeError:
                pass

        def on_release(key):
            try:
                if key.char and key.char.lower() == "w":
                    self._w_is_down = False
            except AttributeError:
                pass

        listener = keyboard.Listener(on_press=on_press, on_release=on_release)
        listener.daemon = True
        listener.start()

    def toggle_overlay(self):
        now = time.monotonic()
        if now - self._last_toggle_time < 0.25:
            return
        self._last_toggle_time = now
        if self.overlay_on:
            self.hide_overlay()
        else:
            self.show_overlay()

    def show_overlay(self):
        self.clearMask()
        # First call: get the geometry right BEFORE the screen capture
        # so capture_context_before_overlay() snapshots the full virtual
        # desktop rect that we're about to draw on.
        self.move_to_all_screens()
        self.context = self.capture_context_before_overlay()
        self.context_attempted = True
        self.overlay_on = True
        self.drawing = False
        self.current_points.clear()
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, False)
        self.sidebar.show()
        self.show()
        self.raise_()
        QApplication.processEvents()
        # Second call AFTER the HWND is realised and mapped. show() can
        # cause Windows to re-snap the window to the primary monitor;
        # this re-applies SetWindowPos with the virtual-desktop rect so
        # the overlay actually spans both screens.
        self.move_to_all_screens()
        self.update()
        if not self.context:
            self.show_notice(
                "Screen Capture Unavailable",
                "The overlay could not freeze the visible screen. Install Pillow "
                "(`pip install pillow`) and restart the app so the AI tools have "
                "something to send to Claude.",
            )

    def hide_overlay(self):
        self.overlay_on = False
        self.drawing = False
        self.current_points.clear()
        self.clearMask()
        self.sidebar.hide()
        self.ai_panel.hide()
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self.hide()
        self.update()

    def capture_context_before_overlay(self):
        if ImageGrab is None:
            print(
                "Pillow is not installed; cannot freeze the visible screen. "
                "Run `pip install pillow` to enable the AI tools.",
                flush=True,
            )
            self.context_pixmap = QPixmap()
            return None

        rect = QRect(self.geometry())
        data = self.screencapture_rect(rect)
        if data:
            pixmap = QPixmap()
            if pixmap.loadFromData(data):
                self.context_pixmap = pixmap
                return FrontWindowCapture(data, rect, None, "frozen visible screen")

        self.context_pixmap = QPixmap()
        print("Could not freeze the visible screen. AI tools will be disabled until the next toggle.", flush=True)
        return None

    def screencapture_rect(self, rect):
        """Grab the requested virtual-screen rectangle on Windows via Pillow
        and return PNG-encoded bytes. `all_screens=True` lets us reach
        monitors that aren't the primary one (without it, Pillow crops to
        the primary screen and ignores anything else)."""
        if ImageGrab is None:
            return None

        bbox = (rect.x(), rect.y(), rect.x() + rect.width(), rect.y() + rect.height())
        try:
            image = ImageGrab.grab(bbox=bbox, all_screens=True)
        except Exception as exc:
            print(f"ImageGrab failed for bbox={bbox}: {exc}", flush=True)
            return None

        buffer = BytesIO()
        try:
            image.save(buffer, format="PNG")
        except Exception as exc:
            print(f"PNG encode failed: {exc}", flush=True)
            return None
        return buffer.getvalue()

    def mousePressEvent(self, event):
        if not self.overlay_on or event.button() != Qt.MouseButton.LeftButton:
            return
        if self.tool == "scroll":
            event.ignore()
            return
        if self.ai_busy:
            self.show_notice("ScreenSense AI", "Wait for the current answer to finish first.")
            return

        self.drawing = True
        self.start = event.position().toPoint()
        self.preview = QPoint(self.start)
        self.current_points = [QPoint(self.start)]
        if self.tool == "eraser":
            self.erase_at(self.start)
        self.update()
        event.accept()

    def mouseMoveEvent(self, event):
        if not self.overlay_on or not self.drawing:
            return
        if self.tool == "scroll":
            event.ignore()
            return
        self.preview = event.position().toPoint()
        if self.tool in {"pen", "highlight"}:
            self.current_points.append(QPoint(self.preview))
        elif self.tool == "eraser":
            self.erase_at(self.preview)
        self.update()
        event.accept()

    def mouseReleaseEvent(self, event):
        if not self.overlay_on or not self.drawing:
            return
        if self.tool == "scroll":
            event.ignore()
            return

        end = event.position().toPoint()
        self.preview = QPoint(end)
        if self.tool == "pen" and len(self.current_points) > 1:
            self.strokes.append(Stroke(list(self.current_points), QColor(255, 70, 92, 245), 5))
        elif self.tool == "highlight" and len(self.current_points) > 1:
            self.strokes.append(Stroke(list(self.current_points), QColor(255, 230, 75, 115), 20, True))
        elif self.tool in self.AI_TOOLS:
            self.create_ai_box(self.tool, self.start, end)

        self.drawing = False
        self.current_points.clear()
        self.update()
        event.accept()

    def keyPressEvent(self, event):
        if self.overlay_on and event.matches(QKeySequence.StandardKey.Cancel):
            self.hide_overlay()
            event.accept()
            return
        super().keyPressEvent(event)

    def erase_at(self, point):
        radius = 24
        kept = []
        for stroke in self.strokes:
            if any(self.distance(point, item) <= radius for item in stroke.points):
                continue
            kept.append(stroke)
        self.strokes = kept
        self.ai_boxes = [box for box in self.ai_boxes if not box.rect.adjusted(-radius, -radius, radius, radius).contains(point)]

    def distance(self, a, b):
        return math.hypot(a.x() - b.x(), a.y() - b.y())

    def create_ai_box(self, tool, start, end):
        rect = QRect(start, end).normalized()
        if rect.width() < 45 or rect.height() < 45:
            rect = QRect(end.x() - 160, end.y() - 100, 320, 200)
        box = AiBox(tool, rect)
        self.ai_boxes.append(box)
        self.pending_ai_box = box
        self.ai_busy = True
        self.show_ai_loading(tool)
        self.update()
        QApplication.processEvents()

        png_bytes = self.ai_capture_for_box(box)
        if not png_bytes:
            self.ai_bridge.result_ready.emit(
                tool,
                "I could not freeze the visible screen, so I did not send Claude anything. "
                "Install Pillow (`pip install pillow`), restart the app, and press W while "
                "the worksheet/browser is visible.",
            )
            return

        prompt = self.ai_prompt(tool)
        thread = threading.Thread(target=self.run_ai, args=(tool, prompt, png_bytes), daemon=True)
        thread.start()

    def ai_capture_for_box(self, box):
        if not self.context:
            print("No frozen visible-screen capture; refusing to capture live wallpaper.", flush=True)
            return None

        background = QImage()
        if not background.loadFromData(self.context.png_bytes):
            print("Stored front-window capture could not be decoded.", flush=True)
            return None

        full_image = QImage(background.size(), QImage.Format.Format_ARGB32)
        full_image.fill(Qt.GlobalColor.transparent)
        painter = QPainter(full_image)
        painter.drawImage(0, 0, background)

        selection_in_image = self.overlay_rect_to_context_pixels(box.rect, background.size())
        if selection_in_image.isEmpty():
            painter.end()
            print("The AI box did not overlap the captured front window.", flush=True)
            return None

        self.paint_ai_box_on_context(painter, box.tool, selection_in_image)
        painter.end()

        crop = selection_in_image.adjusted(-80, -80, 80, 80)
        crop = crop.intersected(QRect(0, 0, full_image.width(), full_image.height()))
        if crop.isEmpty():
            return None

        cropped = full_image.copy(crop)
        data = QByteArray()
        buffer = QBuffer(data)
        buffer.open(QIODevice.OpenModeFlag.WriteOnly)
        cropped.save(buffer, "PNG")
        buffer.close()
        return bytes(data)

    def save_debug_capture(self, image_bytes, label):
        # Kept for parity with the original Mac build but no longer called
        # in the normal flow - the "Saved debug capture" stream was noise.
        # Re-add the call inside capture_context_before_overlay or
        # ai_capture_for_box if you need to inspect what's being sent.
        try:
            DEBUG_CAPTURE_DIR.mkdir(exist_ok=True)
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            path = DEBUG_CAPTURE_DIR / f"{label}_{stamp}.png"
            path.write_bytes(image_bytes)
        except Exception as exc:
            print(f"Could not save debug capture: {exc}", flush=True)

    def overlay_rect_to_context_pixels(self, overlay_rect, image_size):
        overlay_global = QRect(self.mapToGlobal(overlay_rect.topLeft()), overlay_rect.size())
        overlap = overlay_global.intersected(self.context.global_rect)
        if overlap.isEmpty():
            return QRect()

        sx = image_size.width() / max(1, self.context.global_rect.width())
        sy = image_size.height() / max(1, self.context.global_rect.height())
        x = int((overlap.x() - self.context.global_rect.x()) * sx)
        y = int((overlap.y() - self.context.global_rect.y()) * sy)
        w = max(1, int(overlap.width() * sx))
        h = max(1, int(overlap.height() * sy))
        return QRect(x, y, w, h)

    def paint_ai_box_on_context(self, painter, tool, rect):
        colors = {
            "solve": QColor(65, 165, 255, 235),
            "similar": QColor(95, 200, 255, 235),
            "translate": QColor(80, 225, 150, 235),
        }
        color = colors.get(tool, QColor(65, 165, 255, 235))
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setPen(QPen(color, 6, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
        painter.setBrush(QColor(color.red(), color.green(), color.blue(), 26))
        painter.drawRoundedRect(rect.adjusted(3, 3, -3, -3), 12, 12)

        if tool == "similar":
            self.draw_context_badge(painter, rect.topRight() + QPoint(22, 0), "?", color)
        elif tool == "translate":
            self.draw_context_globe(painter, rect.topRight() + QPoint(22, 0), color)

    def draw_context_badge(self, painter, center, text, color):
        rect = QRect(center.x() - 22, center.y() - 22, 44, 44)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(color)
        painter.drawEllipse(rect)
        painter.setPen(QColor(8, 12, 18))
        painter.setFont(QFont("Segoe UI", 23, QFont.Weight.Bold))
        painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, text)

    def draw_context_globe(self, painter, center, color):
        rect = QRect(center.x() - 22, center.y() - 22, 44, 44)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(color)
        painter.drawEllipse(rect)
        painter.setPen(QPen(QColor(8, 12, 18), 2))
        painter.drawEllipse(rect.adjusted(10, 6, -10, -6))
        painter.drawLine(QPoint(rect.left() + 7, center.y()), QPoint(rect.right() - 7, center.y()))
        painter.drawLine(QPoint(center.x(), rect.top() + 7), QPoint(center.x(), rect.bottom() - 7))

    def ai_prompt(self, tool):
        base = (
            "You are receiving a cropped screenshot taken from the frozen visible screen captured when the overlay opened. "
            "Focus only on the content inside the colored box. Ignore anything outside the box. "
            "Do not mention wallpaper or background unless it is literally inside the selected box. "
            "Be concise, accurate, and student-friendly. Do not use Markdown syntax."
        )
        if tool == "solve":
            return (
                f"{base} Solve or explain the boxed problem step by step. "
                "If it is a language question, give the correct answer and explain why. "
                "If it is math, show the needed steps and the final answer. "
                "If the image is unreadable, say exactly what is unreadable."
            )
        if tool == "similar":
            return (
                f"{base} Generate 3 similar practice questions that test the same skill as the boxed question. "
                "Include a short answer key after the questions."
            )
        if tool == "translate":
            return (
                f"{base} Translate the boxed content to {self.translation_target}. "
                "If it is already in that language, explain what it means and identify the source language if possible."
            )
        return base

    def run_ai(self, tool, prompt, png_bytes):
        if Anthropic is None:
            self.ai_bridge.result_ready.emit(tool, "Install the Anthropic package first: pip install anthropic")
            return
        if claude_client is None:
            self.ai_bridge.result_ready.emit(tool, "Claude API key is not configured.")
            return

        try:
            media_type, image_bytes = self.prepare_claude_image(png_bytes)
            if not image_bytes:
                self.ai_bridge.result_ready.emit(tool, "Could not compress the selected image under Claude's image limit.")
                return

            response = claude_client.messages.create(
                model=CLAUDE_MODEL,
                max_tokens=900,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": media_type,
                                    "data": base64.b64encode(image_bytes).decode("ascii"),
                                },
                            },
                            {"type": "text", "text": prompt},
                        ],
                    }
                ],
            )
            self.ai_bridge.result_ready.emit(tool, self.extract_claude_text(response) or "No answer returned.")
        except Exception as exc:
            self.ai_bridge.result_ready.emit(tool, f"Claude error: {exc}")

    def prepare_claude_image(self, image_bytes):
        if len(image_bytes) <= CLAUDE_IMAGE_LIMIT_BYTES:
            return "image/png", image_bytes

        image = QImage()
        if not image.loadFromData(image_bytes):
            return None, None

        max_sides = [1700, 1400, 1100, 900, 760, 620]
        qualities = [82, 74, 66, 58, 48, 38]
        for max_side in max_sides:
            scaled = image
            if image.width() > max_side or image.height() > max_side:
                scaled = image.scaled(
                    QSize(max_side, max_side),
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
            for quality in qualities:
                compressed = self.encode_image(scaled, "JPG", quality)
                if compressed and len(compressed) <= CLAUDE_IMAGE_LIMIT_BYTES:
                    print(f"Compressed Claude image: {len(image_bytes)} -> {len(compressed)} bytes", flush=True)
                    return "image/jpeg", compressed
        return None, None

    def encode_image(self, image, fmt, quality):
        data = QByteArray()
        buffer = QBuffer(data)
        buffer.open(QIODevice.OpenModeFlag.WriteOnly)
        ok = image.save(buffer, fmt, quality)
        buffer.close()
        return bytes(data) if ok else None

    def extract_claude_text(self, response):
        parts = []
        for block in getattr(response, "content", []):
            text = getattr(block, "text", None)
            if text:
                parts.append(text)
        return "\n".join(parts)

    def show_ai_popup(self, tool, answer):
        self.finish_ai_action()
        titles = {
            "solve": "Step-by-Step",
            "similar": "Similar Questions",
            "translate": "Translation",
        }
        colors = {
            "solve": "#93c5fd",
            "similar": "#7dd3fc",
            "translate": "#86efac",
        }
        title = titles.get(tool, "ScreenSense AI")
        accent = colors.get(tool, "#93c5fd")

        self.ai_title.setText(title)
        self.ai_title.setStyleSheet(f"color: {accent};")
        self.ai_body.setHtml(self.answer_html(title, answer, accent))
        self.layout_ai_panel()
        self.ai_panel.show()
        self.ai_panel.raise_()

    def show_ai_loading(self, tool):
        titles = {
            "solve": "Working on Step-by-Step",
            "similar": "Generating Similar Questions",
            "translate": "Translating to English",
        }
        accents = {
            "solve": "#93c5fd",
            "similar": "#7dd3fc",
            "translate": "#86efac",
        }
        title = titles.get(tool, "ScreenSense AI Working")
        accent = accents.get(tool, "#93c5fd")
        self.ai_panel_tool = tool
        self.ai_title.setText(title)
        self.ai_title.setStyleSheet(f"color: {accent};")
        self.ai_body.setHtml(
            f"""
            <div style="font-family: Segoe UI, Arial, sans-serif; color: #eef6ff;">
                <p style="font-size: 16px; line-height: 1.45; margin: 0;">
                    Analyzing...
                </p>
            </div>
            """
        )
        self.layout_ai_panel()
        self.ai_panel.show()
        self.ai_panel.raise_()

    def layout_ai_panel(self):
        width = min(620, max(420, self.width() - 160))
        height = min(390, max(300, self.height() - 120))
        x = max(116, self.width() - width - 34)
        y = 42
        self.ai_panel.setGeometry(QRect(x, y, width, height))

    def answer_html(self, title, answer, accent):
        body = self.markdownish_to_html(answer)
        return f"""
        <div style="font-family: Segoe UI, Arial, sans-serif; color: #eef6ff;">
            <div style="font-size: 22px; font-weight: 800; color: {accent}; margin-bottom: 10px;">
                {html.escape(title)}
            </div>
            <div style="background-color: #142233; border-left: 5px solid {accent}; padding: 14px 16px; border-radius: 10px;">
                {body}
            </div>
        </div>
        """

    def markdownish_to_html(self, text):
        lines = text.strip().splitlines() or ["No answer returned."]
        parts = []
        list_mode = None

        def close_list():
            nonlocal list_mode
            if list_mode:
                parts.append(f"</{list_mode}>")
                list_mode = None

        for raw in lines:
            line = raw.strip()
            if not line or line.startswith("```"):
                close_list()
                continue

            heading = re.match(r"^#{1,6}\s+(.+)$", line)
            numbered = re.match(r"^\d+[.)]\s+(.+)$", line)
            bullet = re.match(r"^[-*â¢]\s+(.+)$", line)

            if heading:
                close_list()
                parts.append(
                    "<div style='color:#bfdbfe; font-size:18px; font-weight:800; margin:10px 0 6px;'>"
                    + self.inline_html(heading.group(1))
                    + "</div>"
                )
            elif numbered:
                if list_mode != "ol":
                    close_list()
                    parts.append("<ol style='margin:8px 0 8px 22px; padding:0;'>")
                    list_mode = "ol"
                parts.append("<li style='margin:7px 0;'>" + self.inline_html(numbered.group(1)) + "</li>")
            elif bullet:
                if list_mode != "ul":
                    close_list()
                    parts.append("<ul style='margin:8px 0 8px 22px; padding:0;'>")
                    list_mode = "ul"
                parts.append("<li style='margin:7px 0;'>" + self.inline_html(bullet.group(1)) + "</li>")
            else:
                close_list()
                parts.append("<p style='line-height:1.45; margin:8px 0;'>" + self.inline_html(line) + "</p>")

        close_list()
        return "\n".join(parts)

    def inline_html(self, text):
        escaped = html.escape(text)
        escaped = re.sub(r"`([^`]+)`", r"<span style='color:#fde68a; font-family: Consolas, monospace;'>\1</span>", escaped)
        escaped = re.sub(r"\*\*(.+?)\*\*", r"<b style='color:#fef3c7;'>\1</b>", escaped)
        escaped = re.sub(r"__(.+?)__", r"<b style='color:#fef3c7;'>\1</b>", escaped)
        escaped = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"<i>\1</i>", escaped)
        return escaped

    def finish_ai_action(self):
        if self.pending_ai_box in self.ai_boxes:
            self.ai_boxes.remove(self.pending_ai_box)
        self.pending_ai_box = None
        self.ai_busy = False
        self.ai_panel_tool = None
        self.tool = "pen"
        self.refresh_buttons()
        self.update()

    def show_notice(self, title, message):
        box = QMessageBox(self)
        box.setWindowTitle(title)
        box.setText(message)
        box.setStandardButtons(QMessageBox.StandardButton.Ok)
        box.show()
        box.raise_()

    def paintEvent(self, event):
        if not self.overlay_on:
            return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceOver)
        self.paint_frozen_context(painter)
        self.paint_strokes(painter)
        self.paint_ai_boxes(painter)
        self.paint_preview(painter)

    def paint_frozen_context(self, painter):
        if self.context_pixmap.isNull() or not self.context:
            painter.fillRect(self.rect(), QColor(0, 0, 0, 24))
            return

        local_top_left = self.mapFromGlobal(self.context.global_rect.topLeft())
        target = QRect(local_top_left, self.context.global_rect.size())
        painter.drawPixmap(target, self.context_pixmap)

    def paint_strokes(self, painter):
        for stroke in self.strokes:
            painter.setPen(self.stroke_pen(stroke))
            self.draw_points(painter, stroke.points)

        if self.drawing and self.tool in {"pen", "highlight"}:
            color = QColor(255, 70, 92, 245) if self.tool == "pen" else QColor(255, 230, 75, 115)
            width = 5 if self.tool == "pen" else 20
            painter.setPen(QPen(color, width, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
            self.draw_points(painter, self.current_points)

    def stroke_pen(self, stroke):
        return QPen(
            stroke.color,
            stroke.width,
            Qt.PenStyle.SolidLine,
            Qt.PenCapStyle.RoundCap,
            Qt.PenJoinStyle.RoundJoin,
        )

    def draw_points(self, painter, points):
        if len(points) < 2:
            return
        path = QPainterPath()
        path.moveTo(float(points[0].x()), float(points[0].y()))
        for point in points[1:]:
            path.lineTo(float(point.x()), float(point.y()))
        painter.drawPath(path)

    def paint_ai_boxes(self, painter):
        for box in self.ai_boxes:
            self.draw_ai_box(painter, box.tool, box.rect, preview=False)

    def paint_preview(self, painter):
        if not self.drawing or self.tool not in self.AI_TOOLS:
            return
        self.draw_ai_box(painter, self.tool, QRect(self.start, self.preview).normalized(), preview=True)

    def draw_ai_box(self, painter, tool, rect, preview=False):
        colors = {
            "solve": QColor(65, 165, 255, 175 if preview else 235),
            "similar": QColor(95, 200, 255, 175 if preview else 235),
            "translate": QColor(80, 225, 150, 175 if preview else 235),
        }
        color = colors.get(tool, QColor(65, 165, 255, 235))
        painter.setPen(QPen(color, 5, Qt.PenStyle.DashLine if preview else Qt.PenStyle.SolidLine))
        painter.setBrush(QColor(color.red(), color.green(), color.blue(), 24))
        painter.drawRoundedRect(rect, 10, 10)
        if tool == "similar":
            self.draw_badge(painter, rect.topRight() + QPoint(17, 0), "?", color)
        elif tool == "translate":
            self.draw_globe(painter, rect.topRight() + QPoint(18, 0), color)

    def draw_badge(self, painter, center, text, color):
        rect = QRect(center.x() - 20, center.y() - 20, 40, 40)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(color)
        painter.drawEllipse(rect)
        painter.setPen(QColor(9, 13, 20))
        painter.setFont(QFont("Segoe UI", 20, QFont.Weight.Bold))
        painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, text)

    def draw_globe(self, painter, center, color):
        rect = QRect(center.x() - 20, center.y() - 20, 40, 40)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(color)
        painter.drawEllipse(rect)
        painter.setPen(QPen(QColor(9, 13, 20), 2))
        painter.drawEllipse(rect.adjusted(9, 5, -9, -5))
        painter.drawLine(QPoint(rect.left() + 6, center.y()), QPoint(rect.right() - 6, center.y()))
        painter.drawLine(QPoint(center.x(), rect.top() + 6), QPoint(center.x(), rect.bottom() - 6))


def configure_macos_app_activation():
    """Kept for API compatibility with esp32_overlay_bridge.py and any other
    caller from the original macOS build. No-op on Windows - Qt's
    Tool + WindowDoesNotAcceptFocus flags already give us the same
    'floating accessory window' behaviour without needing AppKit."""
    return


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(False)
    overlay = SmartboardOverlay()
    sys.exit(app.exec())
