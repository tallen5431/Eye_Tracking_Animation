from __future__ import annotations

import time
import cv2
import numpy as np

from .config import TuningParams, TuningToggles, ViewFlags, RuntimeConfig
from .camera import open_camera
from .pipeline import to_gray, preprocess, blur, threshold, morphology
from .scoring import find_contours, score_contours, fit_ellipse_if_possible
from .overlay import contours_view, eye_overlay, info_panel
from .sharing import ShareWriter

class PupilTrackerTuner:
    """
    Modularized version of your original tuning_tool.py.
    The goal is to keep the same behavior, but make the pipeline easy to edit.
    """

    def __init__(self, camera_id: int = 0):
        self.params = TuningParams()
        self.toggles = TuningToggles()
        self.views = ViewFlags()
        self.runtime = RuntimeConfig(camera_id=camera_id)

        self.cap = None
        self.camera_id = camera_id

        self.img_eye_overlay = None
        self.img_gray = None
        self.img_preprocessed = None
        self.img_threshold = None
        self.img_morphology = None
        self.img_contours = None

        self.current_ellipse = None
        self.current_confidence = 0.0
        self.detected_contours = []

        self._frame_count = 0
        self._start_time = time.time()
        self.fps = 0.0

        self.share_writer = ShareWriter(self.runtime.share_file, hz=self.runtime.share_hz)

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
        roi = frame_bgr[y1:y1+roi_h, x1:x1+roi_w]
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
        print(f"  Threshold: {p.threshold_value}")
        print(f"  Min Area: {p.min_area}")
        print(f"  Max Area: {p.max_area}")
        print(f"  Min Circularity: {p.min_circularity:.2f}")
        print(f"  Blur Kernel: {p.blur_kernel_size}")
        print(f"  Morph Close Iter: {p.morph_close_iterations}")
        print(f"  Morph Open Iter: {p.morph_open_iterations}")
        print(f"  Morph Kernel: {p.morph_kernel_size}")
        print(f"  Hist EQ: {t.use_histogram_eq}")
        print(f"  Glint Removal: {t.use_glint_removal}")
        print(f"  Auto Threshold: {t.use_auto_threshold}")
        print(f"  Adaptive Threshold: {t.use_adaptive_threshold}")
        print(f"  Bilateral Filter: {t.use_bilateral_filter}")
        print(f"  Share file: {self.runtime.share_file}")

    def reset_defaults(self):
        self.params = TuningParams()
        self.toggles = TuningToggles()
        print("Reset parameters to defaults.")

    def process_pipeline(self, roi_bgr: np.ndarray):
        # This is the function you will likely edit the most.
        self.img_gray = to_gray(roi_bgr)
        self.img_preprocessed = preprocess(self.img_gray, self.toggles)
        blurred = blur(self.img_preprocessed, self.params.blur_kernel_size)
        self.img_threshold, _ = threshold(blurred, self.params, self.toggles)
        self.img_morphology, _ = morphology(self.img_threshold, self.params)

        contours = find_contours(self.img_morphology)
        best_contour, best_score, metas = score_contours(
            contours,
            frame_shape_hw=roi_bgr.shape[:2],
            min_area=self.params.min_area,
            max_area=self.params.max_area,
            min_circularity=self.params.min_circularity,
        )
        ellipse = fit_ellipse_if_possible(best_contour)

        self.current_ellipse = ellipse
        self.current_confidence = float(best_score)
        self.detected_contours = metas

        self.img_contours = contours_view(self.img_morphology, metas, best_contour, ellipse)
        self.img_eye_overlay, conf = eye_overlay(roi_bgr, ellipse, best_score)
        self.current_confidence = conf

    def run(self):
        if not self.initialize_camera():
            print("Failed to initialize camera")
            return

        print("\n" + "="*70)
        print("PUPIL TRACKER TUNING TOOL - ENHANCED (MODULAR)")
        print("="*70)
        print("\nWINDOWS:")
        print("  0. EYE TRACKING OVERLAY - Main view with fitted ellipse on your eye")
        print("  1-6. Processing stages   - Detection pipeline visualization")
        print("  7. Parameters & Info     - Current settings and FPS")
        print("\nKEYBOARD CONTROLS:")
        print("  q/ESC       - Quit")
        print("  p           - Print current settings")
        print("  r           - Reset to defaults")
        print("\nTHRESHOLD:")
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
        print("\nPREPROCESSING:")
        print("  h           - Toggle histogram equalization")
        print("  g           - Toggle glint removal")
        print("  f           - Toggle bilateral filter")
        print("\nVIEWS:")
        print("  Space       - Toggle all views on/off")
        print("="*70)
        print("\n⭐ WATCH WINDOW 0 (EYE OVERLAY) - Shows ellipse fitted to your pupil!")
        print("   The cyan ellipse should align with your pupil boundary.")
        print("   Adjust parameters until confidence is > 0.7 (GREEN)")
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

                cv2.imshow("0. EYE TRACKING OVERLAY", self.img_eye_overlay)
                cv2.moveWindow("0. EYE TRACKING OVERLAY", 50, 100)

                if self.views.show_original:
                    cv2.imshow("1. Original ROI", roi)
                cv2.moveWindow("1. Original ROI", 400, 100)

                if self.views.show_grayscale and self.img_gray is not None:
                    cv2.imshow("2. Grayscale", self.img_gray)
                cv2.moveWindow("2. Grayscale", 750, 100)

                if self.views.show_preprocessed and self.img_preprocessed is not None:
                    cv2.imshow("3. Preprocessed", self.img_preprocessed)
                cv2.moveWindow("3. Preprocessed", 1100, 100)

                if self.views.show_threshold and self.img_threshold is not None:
                    cv2.imshow("4. Threshold", self.img_threshold)
                cv2.moveWindow("4. Threshold", 400, 450)

                if self.views.show_morphology and self.img_morphology is not None:
                    cv2.imshow("5. Morphology", self.img_morphology)
                cv2.moveWindow("5. Morphology", 750, 450)

                if self.views.show_contours and self.img_contours is not None:
                    cv2.imshow("6. Contours + Scores", self.img_contours)
                cv2.moveWindow("6. Contours + Scores", 1100, 450)

                cv2.imshow("7. Parameters & Info", panel)
                cv2.moveWindow("7. Parameters & Info", 50, 650)

                key = cv2.waitKey(1) & 0xFF

                if key == ord('q') or key == 27:
                    break
                elif key == ord('p'):
                    self.print_settings()
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
                    print(f"Histogram equalization: {'ON' if self.toggles.use_histogram_eq else 'OFF'}")
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
