from __future__ import annotations

import time
import cv2
import numpy as np

from .config import TuningParams, TuningToggles, ViewFlags, RuntimeConfig
from .camera import open_camera
from .pipeline import (
    to_gray, preprocess, blur, threshold, morphology,
    contrast_normalize_u8,
    fill_holes_u8, find_external_contours,
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

        # Grid layout constants (tweak to taste)
        self.GRID_START_X = 30
        self.GRID_START_Y = 30
        self.GRID_CELL_W = 360
        self.GRID_CELL_H = 260
        self.GRID_COLS = 4
        self.GRID_PAD = 10

        # Info panel window sizing (keep it large & readable)
        self.PANEL_W = 700
        self.PANEL_H = 280

        # NEW: whether to draw ellipse on preview windows
        self.preview_show_ellipse = True

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

    def _show_info_panel(self, title: str, panel_img: np.ndarray):
        """
        Show the info panel at a guaranteed size so it doesn't get cut off by grid sizing.
        """
        cv2.imshow(title, panel_img)
        try:
            cv2.resizeWindow(title, self.PANEL_W, self.PANEL_H)
        except Exception:
            pass

        # Place directly under the 2-row grid
        rows = 2
        y = self.GRID_START_Y + rows * (self.GRID_CELL_H + self.GRID_PAD) + 10
        cv2.moveWindow(title, self.GRID_START_X, y)

    # ---------------------- preview overlay helper ----------------------

    def _preview_with_ellipse(self, img, ellipse, color=(0, 255, 255), thickness=2):
        """
        Return a copy of img with ellipse drawn on top (for preview windows).
        If img is grayscale, converts to BGR so the overlay is visible.
        """
        if img is None:
            return None

        # Convert gray -> BGR so overlay color shows up
        if img.ndim == 2:
            out = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        else:
            out = img.copy()

        if self.preview_show_ellipse and ellipse is not None:
            try:
                cv2.ellipse(out, ellipse, color, thickness)
                (cx, cy), _, _ = ellipse
                cv2.circle(out, (int(cx), int(cy)), 3, color, -1)
            except Exception:
                pass

        return out

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

    def reset_defaults(self):
        self.params = TuningParams()
        self.toggles = TuningToggles()
        print("Reset parameters to defaults.")

    # ---------------------- detection helpers ----------------------

    def _pick_contour_near_center(self, contours, shape_hw, min_area, max_area):
        """
        Pick the contour whose centroid is closest to ROI center, constrained by area.
        Helps lock onto the pupil/iris region instead of stray blobs.
        """
        if not contours:
            return None

        h, w = shape_hw
        cx0, cy0 = w * 0.5, h * 0.5

        best = None
        best_d = 1e18

        for c in contours:
            area = cv2.contourArea(c)
            if area < min_area or area > max_area:
                continue

            M = cv2.moments(c)
            if M["m00"] <= 1e-6:
                continue

            cx = M["m10"] / M["m00"]
            cy = M["m01"] / M["m00"]
            d = (cx - cx0) ** 2 + (cy - cy0) ** 2

            if d < best_d:
                best_d = d
                best = c

        return best

    # ------------------------------------------------------------------

    def _binary_for_pupil(self, preprocessed_gray: np.ndarray) -> np.ndarray:
        blurred = blur(preprocessed_gray, self.params.blur_kernel_size)
        self.img_threshold, _ = threshold(blurred, self.params, self.toggles)
        self.img_morphology, _ = morphology(self.img_threshold, self.params)
        self.img_edges = None
        return self.img_morphology

    def _binary_for_iris(self, preprocessed_gray: np.ndarray) -> np.ndarray:
        blurred = blur(preprocessed_gray, self.params.blur_kernel_size)
        blurred = contrast_normalize_u8(blurred)

        v = float(np.median(blurred))
        lower = int(max(0, 0.66 * v))
        upper = int(min(255, 1.33 * v))
        edges = cv2.Canny(blurred, lower, upper)

        self.img_edges = edges
        self.img_threshold = None

        morphed, _ = morphology(edges, self.params)

        # fill holes so the ring becomes one connected region and inner holes don't dominate
        self.img_morphology = fill_holes_u8(morphed)
        return self.img_morphology

    def process_pipeline(self, roi_bgr: np.ndarray):
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

        # Prefer external contours to avoid selecting hole contours
        ext_contours = find_external_contours(bin_img)

        # First attempt: pick contour closest to ROI center (stable lock)
        best_contour = self._pick_contour_near_center(
            ext_contours,
            shape_hw=roi_bgr.shape[:2],
            min_area=min_area,
            max_area=max_area,
        )

        # Fallback: original scorer (more robust if center-pick fails)
        if best_contour is None:
            contours_all = find_contours(bin_img)
            best_contour, best_score, metas = score_contours(
                contours_all,
                frame_shape_hw=roi_bgr.shape[:2],
                min_area=min_area,
                max_area=max_area,
                min_circularity=self.params.min_circularity,
            )
        else:
            best_score = 0.75
            metas = []

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

        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    break

                roi, _ = self.get_roi(frame)

                self.process_pipeline(roi)
                self.share_writer.maybe_write(self.current_ellipse, self.current_confidence, roi.shape[:2])

                self.update_fps()

                # Build panel image (larger + includes mode)
                panel = info_panel(
                    fps=self.fps,
                    params=self.params,
                    toggles=self.toggles,
                    has_detection=self.current_ellipse is not None,
                    confidence=self.current_confidence,
                    valid_contours=len(self.detected_contours),
                    detect_mode=self.detect_mode,
                    width=self.PANEL_W,
                    height=self.PANEL_H,
                )

                title0 = f"0. EYE TRACKING OVERLAY  [{self.detect_mode.upper()}]"
                stage4_title = "4. Edges (Canny)" if self.detect_mode == "iris" else "4. Threshold"
                stage4_img = self.img_edges if self.detect_mode == "iris" else self.img_threshold

                # NEW: overlay ellipse on preview images
                e = self.current_ellipse
                roi_vis = self._preview_with_ellipse(roi, e)
                gray_vis = self._preview_with_ellipse(self.img_gray, e)
                prep_vis = self._preview_with_ellipse(self.img_preprocessed, e)
                stage4_vis = self._preview_with_ellipse(stage4_img, e)
                morph_vis = self._preview_with_ellipse(self.img_morphology, e)

                # Grid windows only (no info panel here)
                items = [
                    (title0, self.img_eye_overlay),
                    ("1. Original ROI", roi_vis if self.views.show_original else None),
                    ("2. Grayscale", gray_vis if self.views.show_grayscale else None),
                    ("3. Preprocessed", prep_vis if self.views.show_preprocessed else None),
                    (stage4_title, stage4_vis if self.views.show_threshold else None),
                    ("5. Morphology", morph_vis if self.views.show_morphology else None),
                    ("6. Contours + Scores", self.img_contours if self.views.show_contours else None),
                ]

                self._show_in_grid(
                    items,
                    start_x=self.GRID_START_X,
                    start_y=self.GRID_START_Y,
                    cell_w=self.GRID_CELL_W,
                    cell_h=self.GRID_CELL_H,
                    cols=self.GRID_COLS,
                    pad=self.GRID_PAD,
                )

                # Dedicated info panel window (prevents cut-off)
                self._show_info_panel("7. Parameters & Info", panel)

                key = cv2.waitKey(1) & 0xFF

                if key == ord('q') or key == 27:
                    break
                elif key == ord('m'):
                    self.detect_mode = "iris" if self.detect_mode == "pupil" else "pupil"
                    print(f"[MODE] Now tracking: {self.detect_mode.upper()}")
                elif key == ord('r'):
                    self.reset_defaults()
                elif key == ord('+') or key == ord('='):
                    self.params.threshold_value = min(255, self.params.threshold_value + 1)
                elif key == ord('-') or key == ord('_'):
                    self.params.threshold_value = max(0, self.params.threshold_value - 1)
                elif key == ord('a'):
                    self.toggles.use_auto_threshold = not self.toggles.use_auto_threshold
                    if self.toggles.use_auto_threshold:
                        self.toggles.use_adaptive_threshold = False
                elif key == ord('d'):
                    self.toggles.use_adaptive_threshold = not self.toggles.use_adaptive_threshold
                    if self.toggles.use_adaptive_threshold:
                        self.toggles.use_auto_threshold = False
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
                    self.params.morph_kernel_size = min(31, self.params.morph_kernel_size + 2)
                elif key == ord('h'):
                    self.toggles.use_histogram_eq = not self.toggles.use_histogram_eq
                elif key == ord('g'):
                    self.toggles.use_glint_removal = not self.toggles.use_glint_removal
                elif key == ord('f'):
                    self.toggles.use_bilateral_filter = not self.toggles.use_bilateral_filter
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
