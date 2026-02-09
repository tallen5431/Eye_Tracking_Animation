#!/usr/bin/env python3
"""
Eye Tracking Animation - Animated eyes controlled by USB camera eye tracking
Displays on external display (HDMI-2) and tracks user's eye movement via USB camera.
"""

import os
import json
import sys
import time
import math
import subprocess
import re
import threading
import ctypes
from ctypes import wintypes
from pathlib import Path

# Add parent directory for tracker imports
THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
sys.path.insert(0, str(REPO_ROOT / "trackers"))

import tkinter as tk

# X11 convenience (SSH-friendly)
os.environ.setdefault("DISPLAY", ":0")
os.environ.setdefault("XAUTHORITY", os.path.expanduser("~/.Xauthority"))

# Check for OpenCV availability early
CV2_AVAILABLE = False
try:
    import cv2
    import numpy as np
    CV2_AVAILABLE = True
except ImportError:
    pass

# ----------------------------
# Configuration via environment
# ----------------------------
TARGET_OUTPUT = os.environ.get("TARGET_OUTPUT", "HDMI-2")
EYE = os.environ.get("EYE", "center").lower()  # left | right | center
CAMERA_ID = int(os.environ.get("CAMERA_ID", "0"))
TRACKING_ENABLED = os.environ.get("TRACKING_ENABLED", "1") == "1"

# Tracking sensitivity (higher = more responsive)
TRACK_SENSITIVITY_X = float(os.environ.get("TRACK_SENSITIVITY_X", "2.5"))
TRACK_SENSITIVITY_Y = float(os.environ.get("TRACK_SENSITIVITY_Y", "2.0"))

# Smoothing factor (0-1, higher = smoother but slower response)
SMOOTHING = float(os.environ.get("SMOOTHING", "0.15"))


def get_monitor_rect_windows(index: int = 0):
    """Return (W, H, XOFF, YOFF) for monitor at given index on Windows."""
    user32 = ctypes.windll.user32

    class RECT(ctypes.Structure):
        _fields_ = [("left", ctypes.c_long),
                    ("top", ctypes.c_long),
                    ("right", ctypes.c_long),
                    ("bottom", ctypes.c_long)]

    class MONITORINFO(ctypes.Structure):
        _fields_ = [("cbSize", ctypes.c_ulong),
                    ("rcMonitor", RECT),
                    ("rcWork", RECT),
                    ("dwFlags", ctypes.c_ulong)]

    MonitorEnumProc = ctypes.WINFUNCTYPE(
        ctypes.c_int,
        wintypes.HMONITOR,
        wintypes.HDC,
        ctypes.POINTER(RECT),
        ctypes.c_double
    )

    monitors = []

    def _enum_proc(hMonitor, hdcMonitor, lprcMonitor, dwData):
        mi = MONITORINFO()
        mi.cbSize = ctypes.sizeof(MONITORINFO)
        user32.GetMonitorInfoW(hMonitor, ctypes.byref(mi))
        r = mi.rcMonitor
        monitors.append((r.right - r.left, r.bottom - r.top, r.left, r.top, mi.dwFlags))
        return 1

    user32.EnumDisplayMonitors(0, 0, MonitorEnumProc(_enum_proc), 0)

    if not monitors:
        w = user32.GetSystemMetrics(0)
        h = user32.GetSystemMetrics(1)
        return w, h, 0, 0

    if index < 0:
        index = 0
    if index >= len(monitors):
        index = len(monitors) - 1

    W, H, XOFF, YOFF, _flags = monitors[index]
    return W, H, XOFF, YOFF


def get_display_geometry(target_output: str = "HDMI-2"):
    """
    Cross-platform display geometry.

    Linux: uses xrandr output name like HDMI-2 (existing behavior).
    Windows: uses TARGET_MONITOR=0,1,2... (or TARGET_OUTPUT if numeric).
    Returns (W, H, XOFF, YOFF).
    """
    if os.name == "nt":
        raw = os.environ.get("TARGET_MONITOR", os.environ.get("TARGET_OUTPUT", "0")).strip()
        try:
            idx = int(raw)
        except Exception:
            idx = 0
        W, H, XOFF, YOFF = get_monitor_rect_windows(idx)
        return W, H, XOFF, YOFF

    # non-Windows: keep existing xrandr behavior
    return get_output_geometry(target_output)

def get_output_geometry(output_name: str):
    """
    Return (W, H, XOFF, YOFF) for a connected output like:
      HDMI-2 connected 800x400+2880+0 ...
    Falls back to finding any connected display with active mode.
    """
    try:
        env = os.environ.copy()
        out = subprocess.check_output(
            ["xrandr", "--query"],
            text=True,
            stderr=subprocess.STDOUT,
            env=env,
        )
    except Exception as e:
        raise RuntimeError(f"Failed to run xrandr: {e}")

    print(f"[DEBUG] Looking for display: {output_name}")

    # Pattern to match display with active geometry
    geo_pattern = re.compile(
        r"^(\S+)\s+connected(?:\s+primary)?\s+(\d+)x(\d+)\+(\d+)\+(\d+)",
        re.MULTILINE
    )

    # First try to find the requested output
    m = re.search(
        rf"^{re.escape(output_name)}\s+connected(?:\s+primary)?\s+(\d+)x(\d+)\+(\d+)\+(\d+)",
        out,
        flags=re.MULTILINE,
    )
    if m:
        W, H, XOFF, YOFF = map(int, m.groups())
        print(f"[INFO] Found {output_name}: {W}x{H}+{XOFF}+{YOFF}")
        return W, H, XOFF, YOFF

    # If requested output not found with geometry, try to find any other display
    print(f"[WARN] {output_name} not found with active mode, searching alternatives...")

    for match in geo_pattern.finditer(out):
        name = match.group(1)
        W, H, XOFF, YOFF = map(int, match.groups()[1:])
        # Skip the primary VR display (usually HDMI-1 at 0,0)
        if XOFF == 0 and YOFF == 0 and W > 1000:
            print(f"[DEBUG] Skipping primary display {name}: {W}x{H}+{XOFF}+{YOFF}")
            continue
        print(f"[INFO] Using alternative display {name}: {W}x{H}+{XOFF}+{YOFF}")
        return W, H, XOFF, YOFF

    # Last resort: use any connected display with geometry
    for match in geo_pattern.finditer(out):
        name = match.group(1)
        W, H, XOFF, YOFF = map(int, match.groups()[1:])
        print(f"[INFO] Using fallback display {name}: {W}x{H}+{XOFF}+{YOFF}")
        return W, H, XOFF, YOFF

    # No display with geometry found
    con = "\n".join([ln for ln in out.splitlines() if " connected" in ln])
    raise RuntimeError(
        f"No display found with active mode.\n"
        f"Requested: {output_name}\n"
        f"Connected outputs:\n{con}\n\n"
        f"Tip: Make sure your display has a mode set (run 'xrandr' to check)"
    )



# -------------------- Tracking sources --------------------
class SharedFileTracker:
    """Reads pupil center from a JSON file written by tuning_tool.py.

    Supports BOTH formats:
      1) Old: {"ts":..., "cx":..., "cy":..., "frame_w":..., "frame_h":..., "confidence":...}
      2) New: {"ts":..., "frame_w":..., "frame_h":..., "confidence":..., "ellipse": {"cx":..., "cy":..., ...}}

    Includes temporal smoothing and outlier rejection for stable animation.
    """
    def __init__(self, path: str):
        self.path = path
        self._last_ts = 0.0
        self._last = None
        self._center = None  # (cx, cy)
        self._frame = None   # (w, h)
        # Temporal smoothing state (exponential moving average on raw pixel coords)
        self._smooth_cx = None
        self._smooth_cy = None
        self._ema_alpha = 0.35  # blend factor: lower = smoother, higher = more responsive
        # Outlier rejection: max jump as fraction of frame dimension per read
        self._max_jump_frac = 0.15
        # File stat cache to skip re-reading unchanged files
        self._last_mtime = 0.0
        self._last_size = 0

    def recalibrate(self):
        self._center = None
        self._smooth_cx = None
        self._smooth_cy = None

    def get_pupil_position(self):
        # Returns dict with keys: normalized_offset=(dx,dy), confidence=float
        try:
            # Skip file read if it hasn't changed (stat is much cheaper than open+parse)
            try:
                st = os.stat(self.path)
                if st.st_mtime_ns == self._last_mtime and st.st_size == self._last_size:
                    return self._last
                self._last_mtime = st.st_mtime_ns
                self._last_size = st.st_size
            except OSError:
                return self._last

            with open(self.path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            ts = float(data.get('ts', 0.0))
            if ts <= self._last_ts:
                return self._last
            self._last_ts = ts

            fw = int(data.get('frame_w', 0) or 0)
            fh = int(data.get('frame_h', 0) or 0)
            conf = float(data.get('confidence', 0.0))

            # Prefer ellipse dict if present (modular tuning_tool format)
            cx = cy = None
            ell = data.get('ellipse', None)
            if isinstance(ell, dict):
                cx = ell.get('cx', None)
                cy = ell.get('cy', None)
            else:
                cx = data.get('cx', None)
                cy = data.get('cy', None)

            if cx is None or cy is None or fw <= 0 or fh <= 0:
                return self._last

            cx = float(cx)
            cy = float(cy)

            self._frame = (fw, fh)
            if self._center is None:
                self._center = (cx, cy)

            # Outlier rejection: reject sudden large jumps (likely detection errors)
            if self._smooth_cx is not None:
                jump_x = abs(cx - self._smooth_cx) / max(1.0, fw)
                jump_y = abs(cy - self._smooth_cy) / max(1.0, fh)
                if max(jump_x, jump_y) > self._max_jump_frac and conf < 0.7:
                    # Likely a mis-detection; reduce its influence
                    conf *= 0.3

            # Temporal smoothing on raw pixel coordinates
            if self._smooth_cx is None:
                self._smooth_cx = cx
                self._smooth_cy = cy
            else:
                # Adaptive alpha: use more smoothing when confidence is low
                alpha = self._ema_alpha * min(1.0, conf + 0.3)
                self._smooth_cx += alpha * (cx - self._smooth_cx)
                self._smooth_cy += alpha * (cy - self._smooth_cy)

            ccx, ccy = self._center

            # Normalize smoothed coords to -1..1 using half-frame
            dx = (self._smooth_cx - ccx) / max(1.0, fw * 0.5)
            dy = (self._smooth_cy - ccy) / max(1.0, fh * 0.5)
            dx = max(-1.0, min(1.0, dx))
            dy = max(-1.0, min(1.0, dy))

            self._last = {
                "normalized_offset": (dx, dy),
                "confidence": conf,
                "raw": {"cx": cx, "cy": cy, "frame_w": fw, "frame_h": fh},
            }
            return self._last

        except Exception:
            return self._last



class BasicCv2Tracker:
    """Simple OpenCV pupil tracker (borrowed from tuning_tool defaults).

    Provides dict interface compatible with optimized_tracker.
    """
    def __init__(self, camera_id: int):
        import cv2
        self.cv2 = cv2
        self.cap = cv2.VideoCapture(camera_id, cv2.CAP_DSHOW)

        # Reasonable defaults; can be overridden by env
        self.blur = int(os.environ.get('TUNER_BLUR', '5'))
        if self.blur % 2 == 0:
            self.blur += 1
        self.thresh = int(os.environ.get('TUNER_THRESH', '50'))
        self.min_area = int(os.environ.get('TUNER_MIN_AREA', '400'))
        self.max_area = int(os.environ.get('TUNER_MAX_AREA', '15000'))
        self.use_canny = int(os.environ.get('TUNER_USE_CANNY', '0'))
        self.canny1 = int(os.environ.get('TUNER_CANNY1', '40'))
        self.canny2 = int(os.environ.get('TUNER_CANNY2', '120'))

        self._center = None  # (cx, cy)

    def recalibrate(self):
        self._center = None

    def release(self):
        try:
            self.cap.release()
        except Exception:
            pass

    def get_pupil_position(self):
        cv2 = self.cv2
        ok, frame = self.cap.read()
        if not ok or frame is None:
            return None

        H, W = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self.blur >= 3:
            gray = cv2.GaussianBlur(gray, (self.blur, self.blur), 0)

        if self.use_canny:
            edges = cv2.Canny(gray, self.canny1, self.canny2)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        else:
            _, bw = cv2.threshold(gray, self.thresh, 255, cv2.THRESH_BINARY_INV)
            contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        best = None
        best_area = 0.0
        for c in contours:
            area = cv2.contourArea(c)
            if area < self.min_area or area > self.max_area:
                continue
            if area > best_area and len(c) >= 5:
                best = c
                best_area = area

        if best is None:
            return None

        try:
            (cx, cy), (w, h), angle = cv2.fitEllipse(best)
            cx = float(cx); cy = float(cy)
            if self._center is None:
                self._center = (cx, cy)
            ccx, ccy = self._center

            dx = (cx - ccx) / max(1.0, W * 0.5)
            dy = (cy - ccy) / max(1.0, H * 0.5)
            dx = max(-1.0, min(1.0, dx))
            dy = max(-1.0, min(1.0, dy))

            conf = min(1.0, best_area / float(max(1, self.max_area)))
            return {
                "normalized_offset": (dx, dy),
                "confidence": float(conf),
                "raw": {"cx": cx, "cy": cy, "frame_w": int(W), "frame_h": int(H)},
            }
        except Exception:
            return None


class EyeTracker:
    """Wrapper for the optimized pupil tracker with thread-safe position access."""

    def __init__(self, camera_id=0):
        self.camera_id = camera_id
        self.tracker = None
        self.running = False
        self.thread = None
        self.lock = threading.Lock()

        # Current tracked position (normalized -1 to 1)
        self.offset_x = 0.0
        self.offset_y = 0.0
        self.confidence = 0.0

        # Calibration offset
        self.cal_x = 0.0
        self.cal_y = 0.0

        # Stay centered until first calibration completes
        self._calibrated = False

    def start(self) -> bool:
        """Start tracker thread (Windows/Linux).

        Priority:
          1) If TRACK_SHARE_FILE is set -> read it (NEVER opens the camera).
             Optionally waits SHARE_WAIT_SECONDS for the file to appear.
          2) If SHARE_ONLY=1 and no share file -> fail (prevents camera grab).
          3) Otherwise fall back to camera-based trackers.
        """
        if self.running:
            return True

        share_path = os.environ.get("TRACK_SHARE_FILE", "").strip()
        share_only = os.environ.get("SHARE_ONLY", "0").strip().lower() in ("1", "true", "yes", "y")

        if share_path:
            if not os.path.isabs(share_path):
                share_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), share_path)

            wait_s = float(os.environ.get("SHARE_WAIT_SECONDS", "3.0"))
            t_end = time.time() + max(0.0, wait_s)
            while not os.path.exists(share_path) and time.time() < t_end:
                time.sleep(0.05)

            self.tracker = SharedFileTracker(share_path)
            self.running = True
            self.thread = threading.Thread(target=self._track_loop, daemon=True)
            self.thread.start()
            return True

        if share_only:
            return False

        # Optional fallback trackers (only if allowed)
        try:
            import optimized_tracker  # type: ignore
            self.tracker = optimized_tracker.EyeTracker(camera_id=self.camera_id)
        except Exception:
            self.tracker = None

        if self.tracker is None:
            try:
                self.tracker = BasicCv2Tracker(self.camera_id)
            except Exception:
                return False

        self.running = True
        self.thread = threading.Thread(target=self._track_loop, daemon=True)
        self.thread.start()
        return True


    def calibrate(self, duration=2.0, callback=None):
        """Request tracker recalibration (center gaze)."""
        self.recalibrate_requested = True
        self._calibrated = True
        # If tracker supports explicit recalibration, call it too.
        try:
            if self.tracker is not None and hasattr(self.tracker, 'recalibrate'):
                self.tracker.recalibrate()
        except Exception:
            pass
        if callback:
            try:
                callback()
            except Exception:
                pass

    def _track_loop(self):
        """Background tracking loop."""
        while self.running:
            try:
                pos = self.tracker.get_pupil_position()
                if pos:
                    with self.lock:
                        # Use normalized offset (-1 to 1)
                        self.offset_x = pos['normalized_offset'][0]
                        self.offset_y = pos['normalized_offset'][1]
                        self.confidence = pos['confidence']
            except Exception as e:
                print(f"[ERROR] Tracking error: {e}")

            time.sleep(0.016)  # ~60Hz

    def get_position(self):
        """Get current calibrated position. Returns zeros until first calibration."""
        if not self._calibrated:
            return (0.0, 0.0, 0.0)
        with self.lock:
            return (
                self.offset_x - self.cal_x,
                self.offset_y - self.cal_y,
                self.confidence
            )

    def stop(self):
        self.running = False
        try:
            if self.thread:
                self.thread.join(timeout=1.0)
        except Exception:
            pass
        try:
            if self.tracker is not None:
                if hasattr(self.tracker, 'release'):
                    self.tracker.release()
                elif hasattr(self.tracker, 'cap') and self.tracker.cap is not None:
                    self.tracker.cap.release()
        except Exception:
            pass

class EyeAnimation:
    """Animated eye display with tracking input."""

    def __init__(self, target_output="HDMI-2", eye_side="center"):
        # Get display geometry
        # Get display geometry (monitor rect) and apply optional window overrides
        try:
            mon_w, mon_h, mon_x, mon_y = get_display_geometry(target_output)
        except Exception as e:
            print(f"[WARN] get_display_geometry failed ({e}); falling back to 800x600 on primary monitor")
            mon_w, mon_h, mon_x, mon_y = 800, 600, 0, 0

        # Windowing controls (env overrides)
        fullscreen = os.environ.get("FULLSCREEN", "0").strip() in ("1","true","True","yes","YES")
        borderless = os.environ.get("BORDERLESS", "0").strip() in ("1","true","True","yes","YES")
        win_w = int(os.environ.get("WINDOW_W", str(mon_w)))
        win_h = int(os.environ.get("WINDOW_H", str(mon_h)))
        win_x = int(os.environ.get("WINDOW_X", "0"))
        win_y = int(os.environ.get("WINDOW_Y", "0"))

        # Clamp window size to monitor
        win_w = max(100, min(win_w, mon_w))
        win_h = max(100, min(win_h, mon_h))

        # Final geometry used by the Tk window
        self.W = mon_w if fullscreen else win_w
        self.H = mon_h if fullscreen else win_h
        self.XOFF = mon_x if fullscreen else (mon_x + win_x)
        self.YOFF = mon_y if fullscreen else (mon_y + win_y)
        self._FULLSCREEN = fullscreen
        self._BORDERLESS = borderless

        print(f"[INFO] Display target: {target_output} {mon_w}x{mon_h}+{mon_x}+{mon_y}")
        if fullscreen:
            print("[INFO] Mode: fullscreen")
        else:
            print(f"[INFO] Mode: windowed {self.W}x{self.H}+{self.XOFF}+{self.YOFF} (relative {win_x}+{win_y})")

        # Visual settings
        self.BG = "black"
        self.SCLERA_COLOR = "#050505"
        self.DOT_COLOR = "#00f6ff"
        self.DOT_R = 18

        # Detect two-eyes side-by-side output
        self.TWO_EYES = (self.W >= int(self.H * 1.8))
        self.EYE_W = (self.W // 2) if self.TWO_EYES else self.W
        self.EYE_H = self.H

        self.SCLERA_R = min(260, (min(self.EYE_W, self.EYE_H) // 2) - 6)
        self.MATRIX_R = min(235, self.SCLERA_R - 18)

        # Pixel matrix settings
        self.CELL = 8
        self.GAP = 2
        self.BASE_PIXEL = "#0a0a0a"
        self.MID_PIXEL = "#163033"
        self.HOT_PIXEL = self.DOT_COLOR

        # Motion settings
        self.FPS_MS = 16
        self.MAX_X = int(self.EYE_W * 0.22)
        self.MAX_Y = int(self.EYE_H * 0.18)
        self.CLAMP_PAD = 14

        # Current eye position
        self.target_dx = 0.0
        self.target_dy = 0.0
        self.current_dx = 0.0
        self.current_dy = 0.0

        # Eye tracker
        self.tracker = None

        # Demo mode animation (when no tracker)
        self.demo_mode = False
        self.demo_start_time = None
        self.SPARK_R = 5  # Sparkle radius constant

        # Setup window
        self.root = tk.Tk()
        self.root.configure(bg=self.BG)
        # Borderless (overrideredirect) is optional; fullscreen is controlled via env FULLSCREEN=1
        self.root.overrideredirect(bool(self._BORDERLESS))
        self.root.attributes("-topmost", True)
        self.root.geometry(f"{self.W}x{self.H}+{self.XOFF}+{self.YOFF}")
        if self._FULLSCREEN:
            try:
                self.root.attributes("-fullscreen", True)
            except Exception as e:
                print(f"[WARN] Could not enable fullscreen: {e}")
        self.canvas = tk.Canvas(
            self.root,
            width=self.W,
            height=self.H,
            bg=self.BG,
            highlightthickness=0
        )
        self.canvas.pack(fill="both", expand=True)

        # Calculate eye centers
        self._setup_eyes(eye_side)

        # Keyboard bindings
        self.root.bind("<Escape>", lambda e: self.quit())
        self.root.bind("<q>", lambda e: self.quit())
        self.root.bind("<c>", lambda e: self.calibrate())

    def _setup_eyes(self, eye_side):
        """Setup eye positions and draw static elements."""
        if self.TWO_EYES:
            self.left_cx = self.W // 4
            self.right_cx = (3 * self.W) // 4
        else:
            self.left_cx = self.W // 2
            self.right_cx = self.W // 2

        self.cy = self.H // 2

        # Create pixel matrices for both eyes
        # Each pixel entry: (canvas_item_id, center_x, center_y, current_color_zone)
        # color_zone: 0=base, 1=mid, 2=hot - avoids redundant itemconfig calls
        self.left_pixels = []
        self.right_pixels = []

        for cx, pixels in [(self.left_cx, self.left_pixels),
                           (self.right_cx, self.right_pixels)]:
            if not self.TWO_EYES and cx == self.right_cx and self.left_cx == self.right_cx:
                if pixels is self.right_pixels:
                    continue

            # Dark outer circle (sclera)
            self.canvas.create_oval(
                cx - self.SCLERA_R, self.cy - self.SCLERA_R,
                cx + self.SCLERA_R, self.cy + self.SCLERA_R,
                fill=self.SCLERA_COLOR, outline=""
            )

            # Pixel matrix
            step = self.CELL + self.GAP
            x0 = cx - self.MATRIX_R
            y0 = self.cy - self.MATRIX_R

            for gy in range(int((2 * self.MATRIX_R) // step) + 3):
                for gx in range(int((2 * self.MATRIX_R) // step) + 3):
                    px = x0 + gx * step
                    py = y0 + gy * step
                    pcx = px + self.CELL / 2
                    pcy = py + self.CELL / 2

                    if (pcx - cx) ** 2 + (pcy - self.cy) ** 2 <= (self.MATRIX_R - 2) ** 2:
                        item = self.canvas.create_rectangle(
                            px, py, px + self.CELL, py + self.CELL,
                            outline="", fill=self.BASE_PIXEL
                        )
                        pixels.append([item, pcx, pcy, 0])  # 0 = base zone

        # Create pupil dots and sparkles
        self.left_dot = self.canvas.create_oval(
            self.left_cx - self.DOT_R, self.cy - self.DOT_R,
            self.left_cx + self.DOT_R, self.cy + self.DOT_R,
            fill=self.DOT_COLOR, outline=""
        )

        self.left_spark = self.canvas.create_oval(
            self.left_cx - self.SPARK_R, self.cy - self.SPARK_R,
            self.left_cx + self.SPARK_R, self.cy + self.SPARK_R,
            fill="white", outline=""
        )

        if self.TWO_EYES:
            self.right_dot = self.canvas.create_oval(
                self.right_cx - self.DOT_R, self.cy - self.DOT_R,
                self.right_cx + self.DOT_R, self.cy + self.DOT_R,
                fill=self.DOT_COLOR, outline=""
            )
            self.right_spark = self.canvas.create_oval(
                self.right_cx - self.SPARK_R, self.cy - self.SPARK_R,
                self.right_cx + self.SPARK_R, self.cy + self.SPARK_R,
                fill="white", outline=""
            )
        else:
            self.right_dot = None
            self.right_spark = None

    def set_tracker(self, tracker):
        """Set the eye tracker instance."""
        self.tracker = tracker
        self.demo_mode = False

    def enable_demo_mode(self):
        """Enable demo mode (automatic animation without tracking)."""
        self.demo_mode = True
        self.demo_start_time = time.time()
        print("[INFO] Demo mode enabled - eyes will animate automatically")

    def calibrate(self):
        """Trigger calibration."""
        if self.tracker:
            self.tracker.calibrate(duration=2.0)

    def _clamp(self, dx, dy):
        """Clamp movement to stay within sclera."""
        max_move = self.SCLERA_R - self.DOT_R - self.CLAMP_PAD
        mag = math.hypot(dx, dy)
        if mag > max_move and mag > 1e-6:
            s = max_move / mag
            return dx * s, dy * s
        return dx, dy

    # Pre-computed color zone constants
    _ZONE_BASE = 0
    _ZONE_MID = 1
    _ZONE_HOT = 2
    _HOT_DIST = 55
    _MID_DIST = 110
    _HOT_DIST_SQ = _HOT_DIST * _HOT_DIST
    _MID_DIST_SQ = _MID_DIST * _MID_DIST

    def _update_eye(self, cx, dot, spark, pixels, dx, dy):
        """Update a single eye's position. Only recolors pixels whose zone changed."""
        px = cx + dx
        py = self.cy + dy

        self.canvas.coords(
            dot,
            px - self.DOT_R, py - self.DOT_R,
            px + self.DOT_R, py + self.DOT_R
        )

        self.canvas.coords(
            spark,
            px - self.SPARK_R - 7, py - self.SPARK_R - 7,
            px + self.SPARK_R - 7, py + self.SPARK_R - 7
        )

        # Update pixel matrix - only call itemconfig when color zone changes
        hot_sq = self._HOT_DIST_SQ
        mid_sq = self._MID_DIST_SQ
        hot_color = self.HOT_PIXEL
        mid_color = self.MID_PIXEL
        base_color = self.BASE_PIXEL
        itemconfig = self.canvas.itemconfig

        for pix in pixels:
            ddx = pix[1] - px
            ddy = pix[2] - py
            d_sq = ddx * ddx + ddy * ddy

            if d_sq <= hot_sq:
                new_zone = 2
            elif d_sq <= mid_sq:
                new_zone = 1
            else:
                new_zone = 0

            if new_zone != pix[3]:
                pix[3] = new_zone
                if new_zone == 2:
                    itemconfig(pix[0], fill=hot_color)
                elif new_zone == 1:
                    itemconfig(pix[0], fill=mid_color)
                else:
                    itemconfig(pix[0], fill=base_color)

    def _get_demo_position(self):
        """Get demo animation position (smooth figure-8 pattern)."""
        if self.demo_start_time is None:
            self.demo_start_time = time.time()

        t = time.time() - self.demo_start_time
        # Figure-8 pattern with varying speeds
        x = math.sin(t * 0.8) * self.MAX_X * 0.7
        y = math.sin(t * 1.6) * self.MAX_Y * 0.5
        return x, y

    def tick(self):
        """Animation tick - called every frame."""
        # Get tracking input or demo position
        confidence = 0.0
        if self.tracker:
            track_x, track_y, confidence = self.tracker.get_position()
            if confidence > 0.3:
                self.target_dx = track_x * self.MAX_X * TRACK_SENSITIVITY_X
                self.target_dy = track_y * self.MAX_Y * TRACK_SENSITIVITY_Y
        elif self.demo_mode:
            self.target_dx, self.target_dy = self._get_demo_position()
            confidence = 1.0

        # Adaptive smoothing: more smoothing (lower alpha) when confidence is low
        # This reduces jitter from noisy/uncertain detections
        alpha = SMOOTHING * (0.5 + 0.5 * min(1.0, confidence + 0.3))
        self.current_dx += (self.target_dx - self.current_dx) * alpha
        self.current_dy += (self.target_dy - self.current_dy) * alpha

        # Clamp to valid range
        dx, dy = self._clamp(self.current_dx, self.current_dy)

        # Update left eye
        self._update_eye(self.left_cx, self.left_dot, self.left_spark, self.left_pixels, dx, dy)

        # Update right eye (if exists)
        if self.TWO_EYES and self.right_dot:
            self._update_eye(self.right_cx, self.right_dot, self.right_spark, self.right_pixels, dx, dy)

        # Schedule next tick
        self.root.after(self.FPS_MS, self.tick)

    def quit(self):
        """Clean shutdown."""
        if self.tracker:
            self.tracker.stop()
        self.root.destroy()

    def run(self):
        """Start the animation loop."""
        self.tick()
        self.root.mainloop()


def main():
    print("=" * 50)
    print("Eye Tracking Animation")
    print("=" * 50)
    print(f"Display: {TARGET_OUTPUT}")
    print(f"Camera: {CAMERA_ID}")
    print(f"Tracking: {'enabled' if TRACKING_ENABLED else 'disabled'}")
    print(f"Sensitivity: X={TRACK_SENSITIVITY_X}, Y={TRACK_SENSITIVITY_Y}")
    print(f"DISPLAY env: {os.environ.get('DISPLAY', 'not set')}")
    print()
    print("Controls:")
    print("  ESC/q - Quit")
    print("  c     - Recalibrate")
    print("  d     - Toggle demo mode")
    print("=" * 50)
    sys.stdout.flush()

    try:
        print("[INFO] Creating animation window...")
        sys.stdout.flush()
        animation = EyeAnimation(target_output=TARGET_OUTPUT, eye_side=EYE)
        print(f"[INFO] Window created: {animation.W}x{animation.H} at +{animation.XOFF}+{animation.YOFF}")
        print(f"[INFO] Two eyes mode: {animation.TWO_EYES}")
        sys.stdout.flush()

        # Add demo mode toggle keybinding
        animation.root.bind("<d>", lambda e: animation.enable_demo_mode())

        tracking_started = False
        if TRACKING_ENABLED:
            print("[INFO] Initializing eye tracker...")
            sys.stdout.flush()

            # If a share file is configured, we can run WITHOUT OpenCV/camera.
            tracker = EyeTracker(camera_id=CAMERA_ID)
            if tracker.start():
                animation.set_tracker(tracker)
                animation.root.after(500, lambda: tracker.calibrate(duration=2.0))
                tracking_started = True

        if not tracking_started:
            print("[WARN] Tracking did not start - running demo mode")
            animation.enable_demo_mode()


        print("[INFO] Starting animation loop...")
        sys.stdout.flush()
        animation.run()

    except RuntimeError as e:
        # Handle display not found gracefully
        print(f"[ERROR] {e}")
        print("\nTip: On Linux, set TARGET_OUTPUT to your display name (e.g., HDMI-2).\n      On Windows, set TARGET_MONITOR=1 to use your second display.")
        print("      Linux: run 'xrandr --query' to see outputs.")
        sys.stdout.flush()
        sys.exit(1)
    except Exception as e:
        print(f"[ERROR] {e}")
        import traceback
        traceback.print_exc()
        sys.stdout.flush()
        sys.exit(1)


if __name__ == "__main__":
    main()