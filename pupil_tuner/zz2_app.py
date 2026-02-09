from __future__ import annotations

import time
import cv2
import numpy as np

from .config import TuningParams, TuningToggles, ViewFlags, RuntimeConfig
from .camera import open_camera
from .pipeline import (
    to_gray, preprocess, blur, threshold, morphology,
    contrast_normalize_u8,  # NEW helper
)
from .scoring import find_contours, score_contours, fit_ellipse_if_possible
from .overlay import contours_view, eye_overlay, info_panel
from .sharing import ShareWriter


class PupilTrackerTuner:
    """
    Modularized version of your original tuning_tool.py.
    Supports TWO ellipse targets:
      - PUPIL: threshold-based (dark blob)
      - IRIS : edge-based (limbus boundary)
    """

    def __init__(self, camera_id: int = 0):
        self.params = TuningParams()
        self.toggles = TuningToggles()
        self.views = ViewFlags()
        self.runtime = RuntimeConfig(camera_id=camera_id)

        self.cap = None
        self.camera_id = camera_id

        # Debug images
        self.img_eye_overlay = None
        self.img_gray = None
        self.img_preprocessed = None
        self.img_threshold = None
        self.img_morphology = None
        self.img_contours = None

        # Extra debug for IRIS mode
        self.img_edges = None

        # Output
        self.current_ellipse = None
        self.current_confidence = 0.0
        self.detected_contours = []

        # Mode: "pupil" or "iris"
        self.detect_mode = "pupil"

        self._frame_count = 0
        self._start_time = time.time()
        self.fps = 0.0

        self.share_writer = ShareWriter(self.runtime.share_file, hz=self.runtime.share_hz)

    # ---------------------- window layout helpers ----------------------

    def _grid_positions(self, start_x=30, start_y=30, cell_w=360, cell_h=260, cols=4, pad=10):
        def pos(i: int):
            r = i // cols
            c = i % cols
            x = start_x + c * (cell_w + pad)
            y = start_y + r * (cell_h + pad)
            return x, y
        return pos

    def _show_in_grid(self, items, start_x=30, start_y=30, cell_w=360, cell_h=260, cols=4, pad=10):
        """
        items: list of (title, img_or_None)
        Shows all provided windows in a neat grid. Skips None images.
        """
        pos = self._grid_positions(start_x, start_y, cell_w, cell_h, cols, pad)
        shown = 0
        for title, img in items:
            if img is None:
                continue

            cv2.imshow(title, img)
            try:
                cv2.resizeWindow(title, cell_w, cell_h)
            except Exception:
                pass

            x, y = pos(shown)
            cv2.moveWindow(title, x, y)
            shown += 1

    # ------------------------------------------------------------------

    def initialize_camera(self) -> bool:
        try:
            self.cap, actual_id = open_camera(self.camera_id)
            self.camera_id = actual_id
            print(f"Camera opened successfully (ID: {self.camera_id})")
            return True
        except Exception as e:
            print(f"Failed to open camera: {e}")
            return False

    def get_roi(self, frame_bgr: np.ndarray):
        h, w = frame_bgr.shape[:2]
        roi_w = int(w * self.params.roi_size)
        roi_h = int(h * self.params.roi_size)
        x1 = (w - roi_w) // 2
        y1 = (h - roi_h) // 2
        roi = frame_bgr[y1:y1 + roi_h, x1:x1 + roi_w]
        return roi, (x1, y1)

    def update_fps(self):
        self._frame_count += 1
        elapsed = time.time() - self._start_time
        if elapsed >= 1.0:
            self.fps = self._frame_count / elapsed
            self._frame_count = 0
            self._start_time = time.time()

    def print_settings(self):
        p = self.params
        t = self.toggles
        print("\nCurrent Settings:")
        print(f"  Mode: {self.detect_mode.upper()}")
        print(f"  Threshold: {p.threshold_value}")
        print(f"  Min Area: {p.min_area}")
        print(f"  Max Area: {p.max_area}")
        print(f"  Min Circularity: {p.min_circularity:.2f}")
        print(f"  Blur Kernel: {p.blur_kernel_size}")
        print(f"  Morph Close Iter: {p.morph_close_iterations}")
        print(f"  Morph Open Iter: {p.morph_open_iterations}")
        print(f"  Morph Kernel: {p.morph_kernel_size}")
        print(f"  Hist EQ (CLAHE): {t.use_histogram_eq}")
        print(f"  Glint Removal: {t.use_glint_removal}")
        print(f"  Auto Threshold: {t.use_auto_threshold}")
        print(f"  Adaptive Threshold: {t.use_adaptive_threshold}")
        print(f"  Bilateral Filter: {t.use_bilateral_filter}")
        print(f"  Share file: {self.runtime.share_file}")

    def reset_defaults(self):
        self.params = TuningParams()
        self.toggles = TuningToggles()
        print("Reset parameters to defaults.")

    def _binary_for_pupil(self, preprocessed_gray: np.ndarray) -> np.ndarray:
        """
        PUPIL mode: blur -> threshold(inv) -> morphology
        """
        blurred = blur(preprocessed_gray, self.params.blur_kernel_size)
        self.img_threshold, _ = threshold(blurred, self.params, self.toggles)
        self.img_morphology, _ = morphology(self.img_threshold, self.params)
        self.img_edges = None
        return self.img_morphology

    def _binary_for_iris(self, preprocessed_gray: np.ndarray) -> np.ndarray:
        """
        IRIS mode: blur -> contrast normalize -> Canny -> morphology
        """
        blurred = blur(preprocessed_gray, self.params.blur_kernel_size)

        # NEW: stretch contrast for stronger edges before Canny
        blurred = contrast_normalize_u8(blurred)

        # Auto Canny thresholds based on median intensity
        v = float(np.median(blurred))
        lower = int(max(0, 0.66 * v))
        upper = int(min(255, 1.33 * v))
        edges = cv2.Canny(blurred, lower, upper)

        self.img_edges = edges
        self.img_threshold = None

        self.img_morphology, _ = morphology(edges, self.params)
        return self.img_morphology

    def process_pipeline(self, roi_bgr: np.ndarray):
        """
        Build a binary image (pupil or iris) then find contours and fit ellipse.
        """
        self.img_gray = to_gray(roi_bgr)
        self.img_preprocessed = preprocess(self.img_gray, self.toggles)

        if self.detect_mode == "iris":
            bin_img = self._binary_for_iris(self.img_preprocessed)
            min_area = int(self.params.min_area * 2.0)
            max_area = int(self.params.max_area * 6.0)
        else:
            bin_img = self._binary_for_pupil(self.img_preprocessed)
            min_area = int(self.params.min_area)
            max_area = int(self.params.max_area)

        contours = find_contours(bin_img)
        best_contour, best_score, metas = score_contours(
            contours,
            frame_shape_hw=roi_bgr.shape[:2],
            min_area=min_area,
            max_area=max_area,
            min_circularity=self.params.min_circularity,
        )
        ellipse = fit_ellipse_if_possible(best_contour)

        self.current_ellipse = ellipse
        self.current_confidence = float(best_score)
        self.detected_contours = metas

        self.img_contours = contours_view(bin_img, metas, best_contour, ellipse)
        self.img_eye_overlay, conf = eye_overlay(roi_bgr, ellipse, best_score)
        self.current_confidence = conf

    def run(self):
        if not self.initialize_camera():
            print("Failed to initialize camera")
            return

        print("\n" + "=" * 70)
        print("PUPIL / IRIS TRACKER TUNING TOOL - (MODULAR)")
        print("=" * 70)
        print("\nKEYBOARD CONTROLS:")
        print("  q/ESC       - Quit")
        print("  p           - Print current settings")
        print("  r           - Reset to defaults")
        print("  m           - Toggle mode (PUPIL / IRIS)")
        print("\nPREPROCESSING:")
        print("  h           - Toggle contrast boost (CLAHE)")
        print("  g           - Toggle glint removal")
        print("  f           - Toggle bilateral filter")
        print("\nTHRESHOLD (PUPIL mode only):")
        print("  +/-         - Increase/Decrease threshold")
        print("  a           - Toggle auto-threshold (Otsu)")
        print("  d           - Toggle adaptive threshold")
        print("\nAREA:")
        print("  z/x         - Decrease/Increase min area")
        print("  c/v         - Decrease/Increase max area")
        print("\nFILTERS:")
        print("  b/n         - Decrease/Increase min circularity")
        print("  1/2         - Decrease/Increase blur kernel")
        print("  3/4         - Decrease/Increase morph close iterations")
        print("  5/6         - Decrease/Increase morph open iterations")
        print("  7/8         - Decrease/Increase morph kernel size")
        print("\nVIEWS:")
        print("  Space       - Toggle all views on/off")
        print("=" * 70)
        print("\nStarting...")

        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    break

                roi, _ = self.get_roi(frame)

                self.process_pipeline(roi)
                self.share_writer.maybe_write(self.current_ellipse, self.current_confidence, roi.shape[:2])

                self.update_fps()

                panel = info_panel(
                    fps=self.fps,
                    params=self.params,
                    toggles=self.toggles,
                    has_detection=self.current_ellipse is not None,
                    confidence=self.current_confidence,
                    valid_contours=len(self.detected_contours),
                )

                title0 = f"0. EYE TRACKING OVERLAY  [{self.detect_mode.upper()}]"
                stage4_title = "4. Edges (Canny)" if self.detect_mode == "iris" else "4. Threshold"
                stage4_img = self.img_edges if self.detect_mode == "iris" else self.img_threshold

                # Respect view toggles by passing None for hidden windows
                items = [
                    (title0, self.img_eye_overlay),
                    ("1. Original ROI", roi if self.views.show_original else None),
                    ("2. Grayscale", self.img_gray if self.views.show_grayscale else None),
                    ("3. Preprocessed", self.img_preprocessed if self.views.show_preprocessed else None),
                    (stage4_title, stage4_img if self.views.show_threshold else None),
                    ("5. Morphology", self.img_morphology if self.views.show_morphology else None),
                    ("6. Contours + Scores", self.img_contours if self.views.show_contours else None),
                    ("7. Parameters & Info", panel),
                ]

                # Clean 4x2 grid by default (tweak cell_w/cell_h for your monitor)
                self._show_in_grid(items, start_x=30, start_y=30, cell_w=360, cell_h=260, cols=4, pad=10)

                key = cv2.waitKey(1) & 0xFF

                if key == ord('q') or key == 27:
                    break
                elif key == ord('p'):
                    self.print_settings()
                elif key == ord('r'):
                    self.reset_defaults()
                elif key == ord('m'):
                    self.detect_mode = "iris" if self.detect_mode == "pupil" else "pupil"
                    print(f"[MODE] Now tracking: {self.detect_mode.upper()}")
                elif key == ord('+') or key == ord('='):
                    self.params.threshold_value = min(255, self.params.threshold_value + 1)
                elif key == ord('-') or key == ord('_'):
                    self.params.threshold_value = max(0, self.params.threshold_value - 1)
                elif key == ord('a'):
                    self.toggles.use_auto_threshold = not self.toggles.use_auto_threshold
                    if self.toggles.use_auto_threshold:
                        self.toggles.use_adaptive_threshold = False
                    print(f"Auto threshold: {'ON' if self.toggles.use_auto_threshold else 'OFF'}")
                elif key == ord('d'):
                    self.toggles.use_adaptive_threshold = not self.toggles.use_adaptive_threshold
                    if self.toggles.use_adaptive_threshold:
                        self.toggles.use_auto_threshold = False
                    print(f"Adaptive threshold: {'ON' if self.toggles.use_adaptive_threshold else 'OFF'}")
                elif key == ord('z'):
                    self.params.min_area = max(0, self.params.min_area - 50)
                elif key == ord('x'):
                    self.params.min_area = min(50000, self.params.min_area + 50)
                elif key == ord('c'):
                    self.params.max_area = max(0, self.params.max_area - 50)
                elif key == ord('v'):
                    self.params.max_area = min(500000, self.params.max_area + 50)
                elif key == ord('b'):
                    self.params.min_circularity = max(0.0, self.params.min_circularity - 0.05)
                elif key == ord('n'):
                    self.params.min_circularity = min(1.0, self.params.min_circularity + 0.05)
                elif key == ord('1'):
                    self.params.blur_kernel_size = max(1, self.params.blur_kernel_size - 2)
                elif key == ord('2'):
                    self.params.blur_kernel_size = min(15, self.params.blur_kernel_size + 2)
                elif key == ord('3'):
                    self.params.morph_close_iterations = max(0, self.params.morph_close_iterations - 1)
                elif key == ord('4'):
                    self.params.morph_close_iterations = min(5, self.params.morph_close_iterations + 1)
                elif key == ord('5'):
                    self.params.morph_open_iterations = max(0, self.params.morph_open_iterations - 1)
                elif key == ord('6'):
                    self.params.morph_open_iterations = min(5, self.params.morph_open_iterations + 1)
                elif key == ord('7'):
                    self.params.morph_kernel_size = max(1, self.params.morph_kernel_size - 2)
                elif key == ord('8'):
                    self.params.morph_kernel_size = min(21, self.params.morph_kernel_size + 2)
                elif key == ord('h'):
                    self.toggles.use_histogram_eq = not self.toggles.use_histogram_eq
                    print(f"Contrast boost (CLAHE): {'ON' if self.toggles.use_histogram_eq else 'OFF'}")
                elif key == ord('g'):
                    self.toggles.use_glint_removal = not self.toggles.use_glint_removal
                    print(f"Glint removal: {'ON' if self.toggles.use_glint_removal else 'OFF'}")
                elif key == ord('f'):
                    self.toggles.use_bilateral_filter = not self.toggles.use_bilateral_filter
                    print(f"Bilateral filter: {'ON' if self.toggles.use_bilateral_filter else 'OFF'}")
                elif key == ord(' '):
                    all_on = self.views.show_original and self.views.show_grayscale
                    self.views.show_original = not all_on
                    self.views.show_grayscale = not all_on
                    self.views.show_preprocessed = not all_on
                    self.views.show_threshold = not all_on
                    self.views.show_morphology = not all_on
                    self.views.show_contours = not all_on

        finally:
            try:
                if self.cap is not None:
                    self.cap.release()
            finally:
                cv2.destroyAllWindows()
