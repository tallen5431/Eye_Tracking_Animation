#!/usr/bin/env python3
"""
Eye Tracking Animation (Windows) - Single camera pupil detection & highlighting.

Opens a USB camera, processes each frame to find the pupil, and displays
the result with the pupil highlighted.  Uses the same pipeline as the
pupil_tuner so tuning parameters carry over.

Usage:
    python eye_tracking_animation.py              # camera 0, defaults
    python eye_tracking_animation.py --camera 1   # pick a specific camera
    python eye_tracking_animation.py --rotate 270  # rotate camera feed

Environment overrides (all optional):
    CAMERA_ID=0          Camera index
    CAMERA_ROTATION=270  Rotation degrees (0/90/180/270)
    ROI_SIZE=0.85        Center-crop fraction of the frame
"""

import os
import sys
import time
import argparse
import signal

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Resolve imports - pupil_tuner is a sibling package
# ---------------------------------------------------------------------------
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)

from pupil_tuner.config import TuningParams, TuningToggles, RuntimeConfig
from pupil_tuner.camera import open_camera
from pupil_tuner.pipeline import (
    rotate_frame,
    to_gray,
    preprocess,
    blur,
    detect_pupil_blob,
    ellipse_roi_mask_u8,
)
from pupil_tuner.overlay import eye_overlay


# ---------------------------------------------------------------------------
# Core tracker
# ---------------------------------------------------------------------------

class PupilHighlighter:
    """
    Captures frames from a single USB camera and runs the pupil detection
    pipeline on each frame.  The result is drawn directly on the camera
    image so you can see exactly what the detector finds.

    Processing steps shown in the grid window:
        0. Eye overlay     - final result with ellipse drawn on the frame
        1. Original ROI    - raw camera crop
        2. Grayscale       - luminance channel
        3. Preprocessed    - contrast / CLAHE / glare removal
        4. Threshold mask  - dark-blob percentile threshold
        5. Cleaned blob    - morphology-cleaned pupil blob
    """

    def __init__(self, camera_id: int = 0, rotation: int = 270, cap=None):
        self.camera_id = camera_id
        self.rotation = rotation
        self.cap = cap
        self._owns_camera = cap is None

        # Reuse tuner's dataclasses so all parameter names match
        self.params = TuningParams()
        self.toggles = TuningToggles()

        # Smoothing state
        self._iris_ellipse_smoothed = None
        self._pupil_ellipse_smoothed = None
        self._pupil_smooth_alpha = 0.55
        self._prev_pupil_center = None

        # Detection hold (keeps last good ellipse for a few dropped frames)
        self._held_ellipse = None
        self._held_confidence = 0.0
        self._hold_frames = 0
        self._HOLD_MAX = 8

        # FPS counter
        self._fps = 0.0
        self._frame_count = 0
        self._fps_time = time.time()

        # Grid layout for the debug composite window
        self.CELL_W = 400
        self.CELL_H = 300
        self.COLS = 3
        self.PAD = 8
        self._canvas = None

    # ---- camera --------------------------------------------------------

    def open(self) -> bool:
        """Open the camera (with warm-up and fallback scan)."""
        if self.cap is not None and self.cap.isOpened():
            print(f"[OK] Using shared camera {self.camera_id}")
            return True
        try:
            self.cap, self.camera_id = open_camera(self.camera_id)
            self._owns_camera = True
            print(f"[OK] Camera {self.camera_id} ready")
            return True
        except RuntimeError as exc:
            print(f"[ERROR] {exc}")
            return False

    def release(self):
        if self.cap is not None and self._owns_camera:
            self.cap.release()
        self.cap = None

    # ---- ROI -----------------------------------------------------------

    def _get_roi(self, frame: np.ndarray):
        """Center-crop by roi_size fraction."""
        h, w = frame.shape[:2]
        rw = int(w * self.params.roi_size)
        rh = int(h * self.params.roi_size)
        x1 = (w - rw) // 2
        y1 = (h - rh) // 2
        return frame[y1:y1 + rh, x1:x1 + rw]

    # ---- ellipse smoothing ---------------------------------------------

    def _smooth_ellipse(self, prev, new, alpha=0.55):
        if new is None:
            return prev
        if prev is None:
            return new
        (pcx, pcy), (pw, ph), pa = prev
        (ncx, ncy), (nw, nh), na = new
        # Unwrap angle
        da = na - pa
        while da > 90:
            da -= 180
        while da < -90:
            da += 180
        na = pa + da
        cx = alpha * pcx + (1 - alpha) * ncx
        cy = alpha * pcy + (1 - alpha) * ncy
        w_ = alpha * pw + (1 - alpha) * nw
        h_ = alpha * ph + (1 - alpha) * nh
        a_ = alpha * pa + (1 - alpha) * na
        return ((cx, cy), (w_, h_), a_)

    # ---- per-frame pipeline --------------------------------------------

    def process_frame(self, frame_bgr: np.ndarray):
        """
        Run the full detection pipeline on one frame.

        Returns:
            dict with keys:
                overlay   - BGR image with pupil + iris drawn
                original  - original ROI (BGR)
                gray      - grayscale
                preproc   - after contrast/CLAHE
                threshold - dark-blob mask
                blob      - cleaned blob mask
                ellipse   - cv2-style ellipse tuple or None
                confidence - float 0-1
        """
        # 1. Rotate
        if self.rotation != 0:
            frame_bgr = rotate_frame(frame_bgr, self.rotation)

        # 2. Center crop
        roi = self._get_roi(frame_bgr)

        # 3. Grayscale + preprocess
        gray = to_gray(roi)
        preprocessed, _steps = preprocess(gray, self.params, self.toggles)

        # 4. Iris ROI constraint (from previous frame's iris ellipse)
        iris_roi_mask = None
        if self.params.blob_use_iris_roi and self._iris_ellipse_smoothed is not None:
            iris_roi_mask = ellipse_roi_mask_u8(
                preprocessed.shape[:2],
                self._iris_ellipse_smoothed,
                scale=self.params.blob_cyan_roi_scale,
                dilate_k=self.params.blob_iris_roi_dilate_k,
                erode_k=self.params.blob_iris_roi_erode_k,
            )

        # 5. Blob-based pupil detection
        pupil_ellipse, pupil_conf, pupil_blob, raw_mask, _info = detect_pupil_blob(
            roi, self.params,
            iris_roi_mask_u8=iris_roi_mask,
            prev_center=self._prev_pupil_center,
        )

        # 6. Smooth the pupil ellipse
        if pupil_ellipse is not None:
            self._pupil_ellipse_smoothed = self._smooth_ellipse(
                self._pupil_ellipse_smoothed, pupil_ellipse, self._pupil_smooth_alpha,
            )
            pupil_ellipse = self._pupil_ellipse_smoothed
            self._prev_pupil_center = (pupil_ellipse[0][0], pupil_ellipse[0][1])

        # 7. Detection hold
        conf = float(pupil_conf or 0.0)
        if pupil_ellipse is not None:
            self._held_ellipse = pupil_ellipse
            self._held_confidence = conf
            self._hold_frames = 0
        else:
            self._hold_frames += 1
            if self._hold_frames <= self._HOLD_MAX and self._held_ellipse is not None:
                pupil_ellipse = self._held_ellipse
                conf = self._held_confidence * max(0.0, 1.0 - self._hold_frames * 0.12)

        # 8. Update iris estimate (simple: expand pupil ellipse)
        if pupil_ellipse is not None:
            (cx, cy), (w, h), ang = pupil_ellipse
            iris_est = ((cx, cy),
                        (w * self.params.iris_expand_ratio,
                         h * self.params.iris_expand_ratio),
                        ang)
            self._iris_ellipse_smoothed = self._smooth_ellipse(
                self._iris_ellipse_smoothed, iris_est, alpha=self.params.iris_smooth_alpha,
            )

        # 9. Draw overlay
        overlay, draw_conf = eye_overlay(
            roi, pupil_ellipse, conf,
            iris_ellipse=self._iris_ellipse_smoothed,
            pupil_blob=pupil_blob,
        )

        return {
            "overlay": overlay,
            "original": roi,
            "gray": gray,
            "preproc": preprocessed,
            "threshold": raw_mask,
            "blob": pupil_blob,
            "ellipse": pupil_ellipse,
            "confidence": draw_conf,
        }

    # ---- FPS -----------------------------------------------------------

    def _tick_fps(self):
        self._frame_count += 1
        elapsed = time.time() - self._fps_time
        if elapsed >= 1.0:
            self._fps = self._frame_count / elapsed
            self._frame_count = 0
            self._fps_time = time.time()

    # ---- grid display --------------------------------------------------

    def _show_grid(self, items):
        """
        Composite all pipeline images into a single window.
        items: list of (title, image_or_None)
        """
        n = len(items)
        cols = min(self.COLS, n)
        rows = -(-n // cols)  # ceil div
        cw, ch, pad = self.CELL_W, self.CELL_H, self.PAD

        canvas_w = pad + cols * (cw + pad)
        canvas_h = pad + rows * (ch + pad)

        # Reuse canvas buffer
        if self._canvas is None or self._canvas.shape[:2] != (canvas_h, canvas_w):
            self._canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
        else:
            self._canvas[:] = 0

        for idx, (title, img) in enumerate(items):
            r, c = divmod(idx, cols)
            x0 = pad + c * (cw + pad)
            y0 = pad + r * (ch + pad)

            cv2.rectangle(self._canvas, (x0, y0), (x0 + cw, y0 + ch), (60, 60, 60), 1)
            cv2.putText(self._canvas, title, (x0 + 6, y0 + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

            if img is None:
                cv2.putText(self._canvas, "---", (x0 + cw // 2 - 15, y0 + ch // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 50, 50), 1, cv2.LINE_AA)
                continue

            tile = img
            if tile.ndim == 2:
                tile = cv2.cvtColor(tile, cv2.COLOR_GRAY2BGR)
            if tile.dtype != np.uint8:
                tile = np.clip(tile, 0, 255).astype(np.uint8)

            h, w = tile.shape[:2]
            if h == 0 or w == 0:
                continue
            scale = min(cw / w, ch / h)
            nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
            tile_rs = cv2.resize(tile, (nw, nh), interpolation=cv2.INTER_AREA)

            x1 = x0 + (cw - nw) // 2
            y1 = y0 + (ch - nh) // 2
            self._canvas[y1:y1 + nh, x1:x1 + nw] = tile_rs

        win = "Eye Tracker - Pipeline"
        cv2.imshow(win, self._canvas)

    # ---- per-frame display ---------------------------------------------

    def process_and_display(self, frame):
        """Process one frame and update the display grid (no camera read)."""
        result = self.process_frame(frame)
        self._last_result = result
        self._tick_fps()

        overlay = result["overlay"]
        conf = result["confidence"]
        status = f"FPS: {self._fps:.0f}  Conf: {conf:.2f}"
        cv2.putText(overlay, status, (10, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1, cv2.LINE_AA)

        original_vis = result["original"].copy()
        ell = result["ellipse"]
        if ell is not None:
            cv2.ellipse(original_vis, ell, (0, 255, 255), 2)
            (cx, cy), _, _ = ell
            cv2.circle(original_vis, (int(cx), int(cy)), 3, (0, 255, 255), -1)

        items = [
            ("0. PUPIL OVERLAY", overlay),
            ("1. Original + Ellipse", original_vis),
            ("2. Grayscale", result["gray"]),
            ("3. Preprocessed", result["preproc"]),
            ("4. Threshold Mask", result["threshold"]),
            ("5. Cleaned Blob", result["blob"]),
        ]
        self._show_grid(items)

    def handle_key(self, key: int) -> bool:
        """Handle keyboard input. Returns True if user wants to quit."""
        if key == 255:
            return False
        if key == ord('q') or key == 27:
            return True
        elif key == ord('+') or key == ord('='):
            self.params.threshold_value = min(255, self.params.threshold_value + 1)
            print(f"  threshold = {self.params.threshold_value}")
        elif key == ord('-') or key == ord('_'):
            self.params.threshold_value = max(0, self.params.threshold_value - 1)
            print(f"  threshold = {self.params.threshold_value}")
        elif key == ord('9'):
            self.params.contrast_alpha = max(0.5, round(self.params.contrast_alpha - 0.1, 1))
            print(f"  contrast = {self.params.contrast_alpha:.1f}")
        elif key == ord('0'):
            self.params.contrast_alpha = min(3.0, round(self.params.contrast_alpha + 0.1, 1))
            print(f"  contrast = {self.params.contrast_alpha:.1f}")
        elif key == ord(','):
            self.params.brightness_beta = max(-100, self.params.brightness_beta - 10)
            print(f"  brightness = {self.params.brightness_beta:+d}")
        elif key == ord('.'):
            self.params.brightness_beta = min(100, self.params.brightness_beta + 10)
            print(f"  brightness = {self.params.brightness_beta:+d}")
        elif key == ord('['):
            self.params.iris_expand_ratio = max(1.5, round(self.params.iris_expand_ratio - 0.1, 1))
            print(f"  iris_expand = {self.params.iris_expand_ratio:.1f}")
        elif key == ord(']'):
            self.params.iris_expand_ratio = min(4.0, round(self.params.iris_expand_ratio + 0.1, 1))
            print(f"  iris_expand = {self.params.iris_expand_ratio:.1f}")
        elif key == ord('h'):
            self.toggles.use_histogram_eq = not self.toggles.use_histogram_eq
            print(f"  histogram_eq = {self.toggles.use_histogram_eq}")
        elif key == ord('g'):
            self.toggles.use_glint_removal = not self.toggles.use_glint_removal
            print(f"  glint_removal = {self.toggles.use_glint_removal}")
        elif key == ord('w'):
            self.toggles.use_glasses_mode = not self.toggles.use_glasses_mode
            print(f"  glasses_mode = {self.toggles.use_glasses_mode}")
        elif key == ord('r'):
            self.params = TuningParams()
            self.toggles = TuningToggles()
            self._iris_ellipse_smoothed = None
            self._pupil_ellipse_smoothed = None
            self._prev_pupil_center = None
            print("  [RESET] All parameters back to defaults")
        elif key == ord('s'):
            result = getattr(self, '_last_result', None)
            if result is not None:
                self._save_snapshot(result)
        return False

    # ---- main loop -----------------------------------------------------

    def run(self):
        """Main capture-process-display loop. Press q or ESC to exit."""
        if not self.open():
            return

        print()
        print("=" * 52)
        print("  Eye Tracking - Pupil Detection (Windows)")
        print("=" * 52)
        print(f"  Camera : {self.camera_id}")
        print(f"  Rotate : {self.rotation} deg")
        print()
        print("  Keys:")
        print("    q / ESC   Quit")
        print("    +/-       Threshold  ({})".format(self.params.threshold_value))
        print("    9/0       Contrast   ({:.1f})".format(self.params.contrast_alpha))
        print("    ,/.       Brightness ({:+d})".format(self.params.brightness_beta))
        print("    [/]       Iris ratio  ({:.1f})".format(self.params.iris_expand_ratio))
        print("    r         Reset defaults")
        print("    s         Save calibration snapshot")
        print("=" * 52)
        sys.stdout.flush()

        cv2.namedWindow("Eye Tracker - Pipeline", cv2.WINDOW_NORMAL)

        try:
            while True:
                ok, frame = self.cap.read()
                if not ok or frame is None:
                    print("[WARN] Frame grab failed, retrying...")
                    time.sleep(0.05)
                    continue

                self.process_and_display(frame)

                key = cv2.waitKey(1) & 0xFF
                if self.handle_key(key):
                    break

        except KeyboardInterrupt:
            print("\n[INFO] Interrupted")
        finally:
            print("[INFO] Cleaning up...")
            self.release()
            cv2.destroyAllWindows()
            cv2.waitKey(1)
            print("[INFO] Done")

    # ---- snapshot save -------------------------------------------------

    def _save_snapshot(self, result):
        """Save current calibration + detection state to JSON."""
        import json
        out = {
            "timestamp": time.time(),
            "camera_id": self.camera_id,
            "rotation": self.rotation,
            "confidence": float(result["confidence"]),
            "tuning_params": {
                "threshold_value": self.params.threshold_value,
                "contrast_alpha": self.params.contrast_alpha,
                "brightness_beta": self.params.brightness_beta,
                "blob_dark_percentile": self.params.blob_dark_percentile,
                "blob_min_area": self.params.blob_min_area,
                "iris_expand_ratio": self.params.iris_expand_ratio,
            },
        }
        ell = result["ellipse"]
        if ell is not None:
            (cx, cy), (w, h), ang = ell
            out["ellipse"] = {
                "cx": float(cx), "cy": float(cy),
                "width": float(w), "height": float(h),
                "angle": float(ang),
            }

        path = os.path.join(THIS_DIR, ".eye_calibration.json")
        try:
            with open(path, 'w') as f:
                json.dump(out, f, indent=2)
            print(f"  [SAVED] {path}")
        except Exception as exc:
            print(f"  [ERROR] save failed: {exc}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Eye tracking pupil highlighter (Windows)")
    parser.add_argument("--camera", type=int,
                        default=int(os.environ.get("CAMERA_ID", "0")),
                        help="Camera index (default: 0)")
    parser.add_argument("--rotate", type=int,
                        default=int(os.environ.get("CAMERA_ROTATION", "270")),
                        choices=[0, 90, 180, 270],
                        help="Rotate camera feed (default: 270)")
    args = parser.parse_args()

    tracker = PupilHighlighter(camera_id=args.camera, rotation=args.rotate)
    tracker.run()


if __name__ == "__main__":
    main()
