import base64
import html
import math
import os
import re
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

os.environ["QT_MAC_WANTS_LAYER"] = "1"

try:
    from anthropic import Anthropic
except ImportError:
    Anthropic = None

try:
    import AppKit
except ImportError:
    AppKit = None

try:
    import Quartz
except ImportError:
    Quartz = None

from pynput import keyboard
from PyQt6.QtCore import QByteArray, QBuffer, QIODevice, QObject, QPoint, QRect, QSize, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QFont, QImage, QKeySequence, QPainter, QPainterPath, QPen, QPixmap
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
CLAUDE_MODEL = "claude-opus-5"
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
        self.apply_macos_overlay_behavior()

    def move_to_all_screens(self):
        screens = QApplication.screens()
        if not screens:
            return

        geometry = QRect(screens[0].geometry())
        for screen in screens[1:]:
            geometry = geometry.united(screen.geometry())
        self.setGeometry(geometry)

    def macos_window(self):
        if sys.platform != "darwin" or AppKit is None:
            return None
        try:
            for window in AppKit.NSApp.windows():
                if str(window.title()) == self.windowTitle():
                    return window
        except Exception:
            return None
        return None

    def apply_macos_overlay_behavior(self):
        window = self.macos_window()
        if window is None:
            return
        try:
            behavior = 0
            for name in (
                "NSWindowCollectionBehaviorCanJoinAllSpaces",
                "NSWindowCollectionBehaviorFullScreenAuxiliary",
                "NSWindowCollectionBehaviorStationary",
                "NSWindowCollectionBehaviorIgnoresCycle",
            ):
                behavior |= getattr(AppKit, name, 0)
            level = getattr(
                AppKit,
                "NSScreenSaverWindowLevel",
                getattr(AppKit, "NSStatusWindowLevel", getattr(AppKit, "NSFloatingWindowLevel", 3)),
            )
            window.setOpaque_(False)
            window.setHasShadow_(False)
            window.setIgnoresMouseEvents_(False)
            window.setCollectionBehavior_(behavior)
            window.setLevel_(level)
            window.setReleasedWhenClosed_(False)
            if hasattr(window, "setHidesOnDeactivate_"):
                window.setHidesOnDeactivate_(False)
            window.orderFrontRegardless()
        except Exception as exc:
            print(f"macOS overlay warning: {exc}", flush=True)

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

        buttons = [
            ("menu", "☰", "Collapse menu"),
            ("pen", "✎", "Draw pen"),
            ("eraser", "⌫", "Erase"),
            ("highlight", "▰", "Highlight"),
            ("similar", "?", "Box a problem and generate similar questions"),
            ("solve", "□", "Box a problem and solve step by step"),
            ("translate", "🌐", "Box text and translate"),
            ("clear", "×", "Clear overlay drawings"),
        ]

        self.buttons: dict[str, QPushButton] = {}
        for name, icon, tooltip in buttons:
            button = QPushButton(icon)
            button.setToolTip(tooltip)
            button.setCursor(Qt.CursorShape.PointingHandCursor)
            button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            button.setFixedSize(54, 46)
            button.setFont(QFont("Apple Color Emoji" if name == "translate" else "Arial", 17, QFont.Weight.Bold))
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
        self.ai_title.setFont(QFont("Arial", 20, QFont.Weight.Bold))
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
            self.strokes.clear()
            self.ai_boxes.clear()
            self.pending_ai_box = None
            self.ai_busy = False
            self.ai_panel.hide()
            self.update()
            return
        if name == "translate":
            self.translation_target = "English"
        self.tool = name
        self.refresh_buttons()

    def toggle_sidebar(self):
        self.sidebar_open = not self.sidebar_open
        if self.sidebar_open:
            self.sidebar.setGeometry(self.expanded_sidebar_rect())
            self.buttons["menu"].setText("☰")
            for button in self.buttons.values():
                button.show()
            return

        self.sidebar.setGeometry(QRect(24, 44, 76, 70))
        self.buttons["menu"].setText("›")
        for name, button in self.buttons.items():
            if name != "menu":
                button.hide()

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
        self.move_to_all_screens()
        self.context = self.capture_context_before_overlay()
        self.context_attempted = True
        self.overlay_on = True
        self.drawing = False
        self.current_points.clear()
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, False)
        self.apply_macos_overlay_behavior()
        self.sidebar.show()
        self.show()
        self.raise_()
        QApplication.processEvents()
        self.apply_macos_overlay_behavior()
        self.update()
        if not self.context:
            self.show_notice(
                "Screen Capture Blocked",
                "The overlay could not freeze the visible screen. Enable Screen Recording for Terminal, VS Code, or Python, restart smartboard.py, then press W again.",
            )

    def hide_overlay(self):
        self.overlay_on = False
        self.drawing = False
        self.current_points.clear()
        self.sidebar.hide()
        self.ai_panel.hide()
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self.hide()
        self.update()

    def capture_context_before_overlay(self):
        if not self.has_screen_capture_permission():
            print(
                "macOS Screen Recording permission is not available. Cannot freeze the visible screen.",
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
                self.save_debug_capture(data, "frozen_visible_screen")
                print(
                    f"Frozen visible screen for overlay/AI: "
                    f"rect={rect.x()},{rect.y()},{rect.width()},{rect.height()}, bytes={len(data)}",
                    flush=True,
                )
                return FrontWindowCapture(data, rect, None, "frozen visible screen")

        self.context_pixmap = QPixmap()
        print("Could not freeze the visible screen. AI tools will not capture the wallpaper as a fallback.", flush=True)
        return None

    def has_screen_capture_permission(self):
        if sys.platform != "darwin":
            return True
        if Quartz is None:
            print("Quartz is not installed, so macOS Screen Recording permission cannot be checked.", flush=True)
            return False
        try:
            preflight = getattr(Quartz, "CGPreflightScreenCaptureAccess", None)
            if preflight is None:
                return True
            if bool(preflight()):
                return True

            request = getattr(Quartz, "CGRequestScreenCaptureAccess", None)
            if request is not None:
                request()
            return bool(preflight())
        except Exception as exc:
            print(f"Could not check Screen Recording permission: {exc}", flush=True)
            return False

    def front_window_bounds_via_applescript(self):
        if sys.platform != "darwin" or not os.path.exists("/usr/bin/osascript"):
            return None

        script = """
        tell application "System Events"
            set frontApp to first application process whose frontmost is true
            set appName to name of frontApp
            if (count of windows of frontApp) is 0 then return ""
            set frontWindow to front window of frontApp
            set windowPosition to position of frontWindow
            set windowSize to size of frontWindow
            return appName & "|" & (item 1 of windowPosition) & "," & (item 2 of windowPosition) & "," & (item 1 of windowSize) & "," & (item 2 of windowSize)
        end tell
        """

        try:
            result = subprocess.run(
                ["/usr/bin/osascript", "-e", script],
                capture_output=True,
                text=True,
                timeout=4,
                check=False,
            )
            output = result.stdout.strip()
            if result.returncode != 0 or not output or "|" not in output:
                return None
            owner, raw_rect = output.split("|", 1)
            x, y, width, height = [int(float(part.strip())) for part in raw_rect.split(",")]
            if width < 120 or height < 80:
                return None
            return {"owner": owner.strip() or "front window", "rect": QRect(x, y, width, height)}
        except Exception as exc:
            print(f"AppleScript front-window bounds failed: {exc}", flush=True)
            return None

    def frontmost_real_window(self):
        if sys.platform != "darwin" or Quartz is None:
            return None

        own_pid = os.getpid()
        front_pid = None
        if AppKit is not None:
            try:
                front_app = AppKit.NSWorkspace.sharedWorkspace().frontmostApplication()
                front_pid = int(front_app.processIdentifier()) if front_app else None
            except Exception:
                front_pid = None

        try:
            options = Quartz.kCGWindowListOptionOnScreenOnly | Quartz.kCGWindowListExcludeDesktopElements
            windows = Quartz.CGWindowListCopyWindowInfo(options, Quartz.kCGNullWindowID)
        except Exception as exc:
            print(f"Could not list macOS windows: {exc}", flush=True)
            return None

        candidates = []
        for window in windows:
            owner = str(window.get("kCGWindowOwnerName", ""))
            pid = int(window.get("kCGWindowOwnerPID", -1))
            layer = int(window.get("kCGWindowLayer", 999))
            bounds = window.get("kCGWindowBounds", {})
            alpha = float(window.get("kCGWindowAlpha", 1.0))
            window_id = int(window.get("kCGWindowNumber", 0))
            width = int(bounds.get("Width", 0))
            height = int(bounds.get("Height", 0))

            if pid == own_pid or layer != 0 or alpha <= 0 or width < 120 or height < 80:
                continue
            if owner in {"Dock", "Window Server", "SystemUIServer", "Control Center", "Notification Center"}:
                continue

            area = width * height
            candidates.append(
                {
                    "window_id": window_id,
                    "owner": owner,
                    "pid": pid,
                    "bounds": bounds,
                    "area": area,
                    "front_pid_match": pid == front_pid,
                }
            )

        if not candidates:
            return None

        if front_pid is not None:
            front_matches = [candidate for candidate in candidates if candidate["front_pid_match"]]
            if front_matches:
                return max(front_matches, key=lambda item: item["area"])

        return candidates[0]

    def rect_from_cg_bounds(self, bounds):
        return QRect(
            int(bounds.get("X", 0)),
            int(bounds.get("Y", 0)),
            int(bounds.get("Width", 0)),
            int(bounds.get("Height", 0)),
        )

    def screencapture_window(self, window_id):
        return self.run_screencapture(["/usr/sbin/screencapture", "-x", "-l", str(window_id)])

    def screencapture_rect(self, rect):
        region = f"{rect.x()},{rect.y()},{rect.width()},{rect.height()}"
        return self.run_screencapture(["/usr/sbin/screencapture", "-x", "-R", region])

    def run_screencapture(self, base_command):
        if sys.platform != "darwin" or not os.path.exists("/usr/sbin/screencapture"):
            return None

        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                tmp_path = tmp.name

            result = subprocess.run(
                [*base_command, tmp_path],
                capture_output=True,
                timeout=8,
                check=False,
            )
            if result.returncode != 0 or not os.path.exists(tmp_path) or os.path.getsize(tmp_path) == 0:
                stderr = result.stderr.decode("utf-8", errors="ignore").strip()
                if stderr:
                    print(f"screencapture failed: {stderr}", flush=True)
                return None
            with open(tmp_path, "rb") as image_file:
                return image_file.read()
        except Exception as exc:
            print(f"screencapture exception: {exc}", flush=True)
            return None
        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass

    def mousePressEvent(self, event):
        if not self.overlay_on or event.button() != Qt.MouseButton.LeftButton:
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
                "I could not freeze the visible screen, so I did not send Claude anything. Enable macOS Screen Recording for Terminal, VS Code, or Python, restart the app, and press W while the worksheet/browser is visible.",
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
        self.save_debug_capture(bytes(data), "ai_crop")
        print(
            f"AI crop from stored {self.context.owner}: "
            f"{crop.x()},{crop.y()},{crop.width()},{crop.height()}, bytes={len(data)}",
            flush=True,
        )
        return bytes(data)

    def save_debug_capture(self, image_bytes, label):
        try:
            DEBUG_CAPTURE_DIR.mkdir(exist_ok=True)
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            path = DEBUG_CAPTURE_DIR / f"{label}_{stamp}.png"
            path.write_bytes(image_bytes)
            print(f"Saved debug capture: {path}", flush=True)
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
        painter.setFont(QFont("Arial", 23, QFont.Weight.Bold))
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
            self.ai_bridge.result_ready.emit(
                tool, "Claude API key is not configured. Set the ANTHROPIC_API_KEY environment variable and restart."
            )
            return

        try:
            media_type, image_bytes = self.prepare_claude_image(png_bytes)
            if not image_bytes:
                self.ai_bridge.result_ready.emit(tool, "Could not compress the selected image under Claude's image limit.")
                return

            # fallbacks="default": if a safety classifier declines the request,
            # the API retries it on Anthropic's recommended fallback model
            # instead of returning a refusal.
            response = claude_client.beta.messages.create(
                model=CLAUDE_MODEL,
                max_tokens=16000,
                betas=["server-side-fallback-2026-07-01"],
                fallbacks="default",
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
            if response.stop_reason == "refusal":
                self.ai_bridge.result_ready.emit(tool, "Claude declined to answer this request.")
                return
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
            """
            <div style="font-family: Arial, sans-serif; color: #eef6ff;">
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
        <div style="font-family: Arial, sans-serif; color: #eef6ff;">
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
            bullet = re.match(r"^[-*•]\s+(.+)$", line)

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
        escaped = re.sub(r"`([^`]+)`", r"<span style='color:#fde68a; font-family: Menlo, monospace;'>\1</span>", escaped)
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
        painter.setFont(QFont("Arial", 20, QFont.Weight.Bold))
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
    if sys.platform != "darwin" or AppKit is None:
        return
    try:
        policy = getattr(AppKit, "NSApplicationActivationPolicyAccessory", 1)
        AppKit.NSApp.setActivationPolicy_(policy)
    except Exception as exc:
        print(f"macOS activation policy warning: {exc}", flush=True)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(False)
    configure_macos_app_activation()
    overlay = SmartboardOverlay()
    sys.exit(app.exec())
