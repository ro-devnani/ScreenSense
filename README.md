# ScreenSense

Turn a monitor, TV or whiteboard into a smartboard. Two webcams track a
colored pen tip and move the mouse cursor. A transparent overlay lets you
draw on top of anything on screen. Draw a box around a problem and Claude
will solve it, write similar practice questions, or translate it.

- **Dual-camera pen tracking**: HSV color detection, confidence-weighted
  fusion of both cameras, and Kalman smoothing so the cursor doesn't jump
  when one camera loses sight of the pen.
- **Smartboard overlay**: pen, highlighter and eraser drawn over a frozen
  copy of the screen, across every monitor.
- **AI tools**: box part of the screen and send it to Claude.
- **ESP32 buttons (optional)**: a wireless *write* and *erase* button on
  the pen, sent to the PC over Wi-Fi.

## How it works

```
Camera 1 ─► detect tip ─► homography ─► point + confidence ─┐
                                                            ├─► fuse ─► Kalman ─► cursor
Camera 2 ─► detect tip ─► homography ─► point + confidence ─┘

ESP32 buttons ─► TCP :65432 ─► pen / eraser tool + mouse button ─► overlay
```

Each camera finds the pen tip by color, then maps it through its own
calibration homography. There are two modes:

| Mode | Maps pen to | Calibrate with | Use it for |
|---|---|---|---|
| **Screen** (`screen.enabled: true`) | screen pixels | `calibration/screen_calibrate.py` | driving the cursor / overlay |
| **Plane** (`screen.enabled: false`) | millimetres on a flat surface | `calibration/calibrate.py` | recording strokes to JSON |

## Requirements

- Python 3.10+
- Two USB webcams (MJPG capable if they share a USB controller)
- A pen with a brightly colored tip. The shipped `config.yaml` is tuned for
  a green tip; retune it for yours (see [HSV tuning](#2-tune-the-pen-color)).
- Windows for cursor control, screen calibration and `overlay.py`.
  `overlay_macos.py` is the macOS version of the overlay. On macOS/Linux the
  tracker still runs with `cursor.enabled: false`.
- Optional: an ESP32 with two buttons, and an
  [Anthropic API key](https://console.anthropic.com/) for the AI tools.

## Installation

```bash
git clone https://github.com/ro-devnani/ScreenSense.git
cd ScreenSense
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux
pip install -r requirements.txt
```

For the AI tools, set your API key before launching the overlay:

```bash
setx ANTHROPIC_API_KEY "sk-ant-..."     # Windows (open a new terminal afterwards)
export ANTHROPIC_API_KEY="sk-ant-..."   # macOS / Linux
```

Run every command below from the repository root. The scripts read
`config.yaml` and `calibration/` relative to the current directory.

## Setup

### 1. Find your cameras

```bash
python tools/probe_cameras.py   # lists every camera index OpenCV can open
python tools/camera_test.py     # shows both configured cameras side by side
```

Put the two indices into `cameras.cam1_index` / `cam2_index` in `config.yaml`.

### 2. Tune the pen color

```bash
python tools/hsv_tuner.py --cam 1
```

Drag the sliders until only the pen tip is white in the mask, then press
**S** to save `hsv_lower` / `hsv_upper` to `config.yaml` (**Q** quits).
`tools/draw_test.py` draws a trail behind the detected tip, which is a quick
way to check the result with one camera.

| Problem | Fix |
|---|---|
| Background shows up white | Raise V min |
| Tip disappears in shadow | Lower V min slightly |
| Skin bleeds into the mask | Raise S min |
| Tip only partly white | Lower S min or widen the H range by about 3 |
| Tip splits into two blobs | Lower `morph_kernel_size` or widen the V range |

### 3. Calibrate

**Screen mode** (Windows): calibrates both cameras against one on-screen
rectangle on `screen.monitor_index`.

```bash
python calibration/screen_calibrate.py
```

1. **Cursor phase**: the cursor is confined to the rectangle. Click
   `screen.calib_points` points spread across it, then press **SPACE**.
2. **Pen phase, cam1 then cam2**: put the pen tip on each red target and
   press **ENTER**. Press **S** to skip a target the camera can't see.

**U** undoes and **Q** aborts in every phase. The results are saved to
`calibration/cam{1,2}_screen_calibration.npz`, and the rectangle is written
back to `screen.rect`.

**Plane mode**: place the pen on each corner of the surface (top-left,
top-right, bottom-right, bottom-left) and press **ENTER** at each one, then
**SPACE** to finish.

```bash
python calibration/calibrate.py --both
# or a single camera:
python calibration/calibrate.py --cam 1 --out calibration/cam1_calibration.npz
```

Recalibrate whenever a camera or the surface moves.

## Running

**Everything together** (overlay + camera tracking + ESP32 buttons, Windows):

```bash
python esp32_overlay_bridge.py
```

| Input | Action |
|---|---|
| **W** key | show / hide the overlay |
| ESP32 *write* held | pen tool, left mouse button down (draw) |
| ESP32 *erase* held | eraser tool, left mouse button down (erase) |
| both released | cursor mode: clicks pass through to the apps underneath |
| **Q** in the debug window | stop camera tracking (overlay keeps running) |

The overlay's sidebar also has the highlighter, **?** (similar questions),
**⚡** (solve step by step), **🌐** (translate to English) and **✕** (clear).
For the AI tools, drag a box around the content and the answer appears in a
panel.

**Just the tracker** (OpenCV debug window, moves the cursor if enabled):

```bash
python src/tracker.py
```

In screen mode, press **Q** to quit. In plane mode, a pre-flight window
waits until both cameras see the pen; press **SPACE** to start. **S** ends
the current stroke, and **Q** quits and writes `output/strokes.json`. When
the tracker runs on its own, ESP32 *write* maps to the left mouse button and
*erase* to the right.

**Just the overlay** (no cameras): `python overlay.py`, or
`python overlay_macos.py` on macOS.

### ESP32 protocol

The PC listens on TCP port **65432**. The ESP32 connects and sends one
JSON object per line whenever a button changes:

```json
{"write": true, "erase": false}
```

If the connection drops, both buttons are treated as released.

### Stroke output (plane mode)

```json
{
  "strokes": [
    [ {"x": 142.3, "y": 87.1, "t": 1718123456.123},
      {"x": 143.1, "y": 88.0, "t": 1718123456.140} ]
  ]
}
```

`x` and `y` are millimetres from the plane's top-left corner, and `t` is a
Unix timestamp. Each inner list is one stroke.

## Project structure

```
├── esp32_overlay_bridge.py   # overlay + tracker + ESP32 in one process
├── overlay.py                # smartboard overlay (Windows)
├── overlay_macos.py          # smartboard overlay (macOS)
├── config.yaml               # all tunable settings
├── calibration/
│   ├── screen_calibrate.py   # screen-mode calibration (Windows)
│   └── calibrate.py          # plane-mode calibration
├── src/
│   ├── tracker.py            # tracker entry point (screen and plane mode)
│   ├── detect.py             # HSV pen-tip detection
│   ├── fuse.py               # confidence-weighted fusion of both cameras
│   ├── kalman.py             # constant-velocity Kalman filter
│   ├── cursor.py             # Win32 cursor and mouse-button control
│   ├── input_receiver.py     # ESP32 TCP listener
│   └── utils.py              # calibration I/O, config editing, debug overlay, strokes
└── tools/
    ├── probe_cameras.py      # list camera indices
    ├── camera_test.py        # live view of both cameras
    ├── test_cam0_only.py     # minimal single-camera check
    ├── hsv_tuner.py          # interactive HSV range tuning
    ├── draw_test.py          # single-camera drawing test
    └── tv_target.py          # "is the pen inside this region" demo
```

## Troubleshooting

**Pen never detected.** Re-run `tools/hsv_tuner.py` under your actual
lighting, and check the camera indices with `tools/probe_cameras.py`.

**A camera won't open, or both run at ~3 fps.** Close other apps using the
camera (Camera, Teams, Zoom, browser tabs). Two cameras on one USB
controller need MJPG, which the scripts request. Try another port if one
camera still starves.

**Cursor jumps when the pen reappears.** Increase
`kalman.screen_measurement_noise` (or `measurement_noise` in plane mode), or
raise `fusion.min_confidence`.

**Cursor freezes even though the pen is visible.** The detection is failing
the color check. Lower `detection.color_confidence_min`, or retune the HSV
range so the pen sits in the middle of it.

**Cursor lands in the wrong place.** A camera or the screen moved: recalibrate.

**Slow frame rate.** Set `roi_cam1` / `roi_cam2` to crop to the area you
draw in, keep 640x480, or set `output.show_debug: false`.

**`Could not listen on port 65432`.** Another copy of the tracker or bridge
is already running.

## License

[MIT](LICENSE)
