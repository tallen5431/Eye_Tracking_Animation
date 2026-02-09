from __future__ import annotations

import time
import cv2
import numpy as np

from .config import TuningParams, TuningToggles, ViewFlags, RuntimeConfig
from .camera import open_camera
from .pipeline import (
    to_gray,
    preprocess,
    blur,
    threshold,
    morphology,
    detect_pupil_blob,
    contrast_normalize_u8,
    fill_holes_u8,
    find_external_contours,
    auto_canny_u8,
    create_iris_mask_from_sclera,
    ellipse_roi_mask_u8,
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

        # Debug images - main pipeline
        self.img_eye_overlay = None
        self.img_gray = None
        self.img_preprocessed = None
        self.img_threshold = None
        self.img_morphology = None
        self.img_contours = None
        self.img_edges = None
        
        # Debug images - preprocessing steps (NEW)
        self.preprocessing_steps = {}  # Dict of step images
        
        # Iris detection
        self.iris_ellipse = None
        self.img_iris_mask = None

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

        # Grid layout constants - LARGER to show full rotated images
        self.GRID_START_X = 30
        self.GRID_START_Y = 30
        self.GRID_CELL_W = 480  # Increased from 360
        self.GRID_CELL_H = 360  # Increased from 260
        self.GRID_COLS = 4
        self.GRID_PAD = 10

        # Info panel window sizing (keep it large & readable)
        self.PANEL_W = 900  # Increased from 700
        self.PANEL_H = 320  # Increased from 280

        # Whether to draw ellipse on preview windows
        self.preview_show_ellipse = True

        # Iris ellipse smoothing (reduces jitter)
        self._iris_ellipse_smoothed = None

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
            print(f"Camera rotation: {self.runtime.camera_rotation}° CCW")
            return True
        except Exception as e:
            print(f"Failed to open camera: {e}")
            return False

    def get_roi(self, frame_bgr: np.ndarray):
        """Extract ROI, handling rotated frames properly."""
        if frame_bgr is None or frame_bgr.size == 0:
            # Return dummy ROI
            return np.zeros((100, 100, 3), dtype=np.uint8), (0, 0)
        
        h, w = frame_bgr.shape[:2]
        roi_w = int(w * self.params.roi_size)
        roi_h = int(h * self.params.roi_size)
        
        # Ensure ROI doesn't exceed frame bounds
        roi_w = min(roi_w, w)
        roi_h = min(roi_h, h)
        
        x1 = (w - roi_w) // 2
        y1 = (h - roi_h) // 2
        
        # Clamp to valid range
        x1 = max(0, min(x1, w - roi_w))
        y1 = max(0, min(y1, h - roi_h))
        
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
        ENHANCED: Also checks circularity to avoid picking glare reflections.
        """
        if not contours:
            return None

        h, w = shape_hw
        cx0, cy0 = w * 0.5, h * 0.5

        best = None
        best_score = 0.0

        for c in contours:
            area = cv2.contourArea(c)
            if area < min_area or area > max_area:
                continue

            # Check circularity (reject glare reflections which are often irregular)
            perimeter = cv2.arcLength(c, True)
            if perimeter <= 0:
                continue
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            if circularity < 0.3:  # Reject very non-circular shapes
                continue

            M = cv2.moments(c)
            if M["m00"] <= 1e-6:
                continue

            cx = M["m10"] / M["m00"]
            cy = M["m01"] / M["m00"]
            d = np.sqrt((cx - cx0) ** 2 + (cy - cy0) ** 2)
            
            # Score: closer to center AND more circular = better
            center_score = 1.0 - (d / (min(w, h) * 0.5))
            score = center_score * 0.6 + circularity * 0.4

            if score > best_score:
                best_score = score
                best = c

        return best





    def _contour_touches_border(self, contour, shape_hw, margin: int = 2) -> bool:
        """Reject contours that touch ROI borders (often eyelids/skin edges)."""
        h, w = shape_hw
        x, y, cw, ch = cv2.boundingRect(contour)
        if x <= margin or y <= margin:
            return True
        if (x + cw) >= (w - margin) or (y + ch) >= (h - margin):
            return True
        return False

    def _choose_iris_contour(self, contours, shape_hw, min_area: float, max_area: float,
                             border_margin: int = 2, top_k_by_center: int = 6):
        """
        Choose a contour likely to be the iris boundary:
          - area in [min_area, max_area]
          - does NOT touch border (within margin)
          - among closest-to-center contours, prefer the LARGEST (outer boundary)
        """
        if not contours:
            return None

        h, w = shape_hw
        cx0, cy0 = w * 0.5, h * 0.5

        candidates = []
        for c in contours:
            area = float(cv2.contourArea(c))
            if area < min_area or area > max_area:
                continue
            if self._contour_touches_border(c, (h, w), margin=border_margin):
                continue
            M = cv2.moments(c)
            if M["m00"] <= 1e-6:
                continue
            cx = M["m10"] / M["m00"]
            cy = M["m01"] / M["m00"]
            d2 = (cx - cx0) ** 2 + (cy - cy0) ** 2
            candidates.append((d2, -area, c))  # closer center first; if tie, bigger area wins

        if not candidates:
            return None

        candidates.sort(key=lambda t: (t[0], t[1]))
        short = candidates[:max(1, int(top_k_by_center))]

        # From the closest few, pick the largest area (outer boundary preference)
        best = min(short, key=lambda t: t[1])  # most negative area == largest
        return best[2]

    def _ellipse_point_error(self, ellipse, pts_xy: np.ndarray) -> np.ndarray:
        """
        Approx normalized radial error of points w.r.t. ellipse:
        error ~ |sqrt((x'/a)^2 + (y'/b)^2) - 1|
        """
        (cx, cy), (w, h), angle_deg = ellipse
        a = max(1e-6, float(w) * 0.5)
        b = max(1e-6, float(h) * 0.5)
        ang = np.deg2rad(float(angle_deg))
        ca, sa = np.cos(ang), np.sin(ang)

        x = pts_xy[:, 0] - float(cx)
        y = pts_xy[:, 1] - float(cy)

        # rotate by -angle
        xr =  ca * x + sa * y
        yr = -sa * x + ca * y

        r = np.sqrt((xr / a) ** 2 + (yr / b) ** 2)
        return np.abs(r - 1.0)

    def _trimmed_fit_ellipse(self, contour, iters: int = 3, keep_frac: float = 0.7):
        """
        Robust ellipse fit via iterative trimming (outlier rejection).
        Helps ignore eyelashes/glints/eyelid outliers.
        """
        if contour is None or len(contour) < 5:
            return None

        pts = contour.reshape(-1, 2).astype(np.float32)
        if pts.shape[0] < 5:
            return None

        cur = pts
        for _ in range(max(1, int(iters))):
            if cur.shape[0] < 5:
                break
            try:
                e = cv2.fitEllipse(cur.reshape(-1, 1, 2))
            except Exception:
                break

            err = self._ellipse_point_error(e, cur)
            if err.size < 5:
                break

            k = max(5, int(round(float(keep_frac) * err.size)))
            idx = np.argsort(err)[:k]
            cur = cur[idx]

        if cur.shape[0] < 5:
            return None
        try:
            return cv2.fitEllipse(cur.reshape(-1, 1, 2))
        except Exception:
            return None

    def _smooth_ellipse(self, prev, new, alpha: float = 0.7):
        """Exponential smoothing of ellipse params."""
        if new is None:
            return prev
        if prev is None:
            return new

        (pcx, pcy), (pw, ph), pa = prev
        (ncx, ncy), (nw, nh), na = new

        # unwrap angle around prev
        da = (na - pa)
        while da > 90:
            da -= 180
        while da < -90:
            da += 180
        na_u = pa + da

        cx = alpha * float(pcx) + (1 - alpha) * float(ncx)
        cy = alpha * float(pcy) + (1 - alpha) * float(ncy)
        w = alpha * float(pw) + (1 - alpha) * float(nw)
        h = alpha * float(ph) + (1 - alpha) * float(nh)
        a = alpha * float(pa) + (1 - alpha) * float(na_u)

        return ((cx, cy), (w, h), a)


    def _fit_iris_ellipse_simple(self, preprocessed_gray: np.ndarray, pupil_ellipse):
        """
        SIMPLIFIED: Detect iris using sclera contrast + pupil anchor.
        ENHANCED: Sanity checks to prevent absurd results (like 528x2281px!).
        """
        if preprocessed_gray is None:
            self.img_iris_mask = None
            return self._iris_ellipse_smoothed
        
        from .pipeline import create_iris_mask_from_sclera, fit_iris_from_mask_and_pupil
        
        # Create mask by excluding bright sclera
        mask = create_iris_mask_from_sclera(
            preprocessed_gray, 
            self.params,
            pupil_center=pupil_ellipse[0] if pupil_ellipse else None,
            pupil_radius=((pupil_ellipse[1][0] + pupil_ellipse[1][1])/4.0) if pupil_ellipse else None
        )
        self.img_iris_mask = mask
        
        if mask is None or pupil_ellipse is None:
            return self._iris_ellipse_smoothed
        
        # Fit ellipse using mask + pupil constraints
        ellipse, conf = fit_iris_from_mask_and_pupil(mask, pupil_ellipse, self.params)
        
        if ellipse is None:
            return self._iris_ellipse_smoothed
        
        # SANITY CHECK: Reject absurd ellipse sizes
        (cx, cy), (w, h), angle = ellipse
        frame_h, frame_w = preprocessed_gray.shape[:2]
        
        # Iris should not be larger than 80% of frame or smaller than 10%
        max_dim = max(w, h)
        frame_diag = np.sqrt(frame_w**2 + frame_h**2)
        
        if max_dim > frame_diag * 0.8:  # Way too big
            print(f"[WARN] Rejecting absurd iris: {int(w)}x{int(h)}px (too large)")
            return self._iris_ellipse_smoothed
        
        if max_dim < frame_diag * 0.1:  # Way too small
            print(f"[WARN] Rejecting absurd iris: {int(w)}x{int(h)}px (too small)")
            return self._iris_ellipse_smoothed
        
        # Check aspect ratio (iris should be roughly circular)
        aspect = max(w, h) / max(1.0, min(w, h))
        if aspect > 2.5:  # Too elongated
            print(f"[WARN] Rejecting elongated iris: {int(w)}x{int(h)}px (aspect {aspect:.1f})")
            return self._iris_ellipse_smoothed
        
        # Temporal smoothing
        alpha = float(self.params.iris_smooth_alpha)
        self._iris_ellipse_smoothed = self._smooth_ellipse(
            self._iris_ellipse_smoothed, ellipse, alpha=alpha
        )
        
        return self._iris_ellipse_smoothed

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
        from .pipeline import rotate_frame
        
        # Apply camera rotation if needed
        if self.runtime.camera_rotation != 0:
            roi_bgr = rotate_frame(roi_bgr, self.runtime.camera_rotation)
        
        self.img_gray = to_gray(roi_bgr)
        
        # Enhanced preprocessing with step-by-step debug images
        self.img_preprocessed, self.preprocessing_steps = preprocess(
            self.img_gray, self.params, self.toggles
        )

        if self.detect_mode == "iris":
            bin_img = self._binary_for_iris(self.img_preprocessed)
            min_area = int(self.params.min_area * 2.0)
            max_area = int(self.params.max_area * 6.0)
            pupil_blob = None
            pupil_conf = None
        else:
            # NEW: blob-based pupil detection (dark circular region)
            # Optional: use iris ROI (from sclera segmentation) to constrain pupil blob detection.

            iris_roi_mask = None
            if bool(getattr(self.params, "blob_use_iris_roi", True)):
                # Use the cyan iris ellipse (smoothed) from the previous frame/step as an ROI.
                # This keeps the blob detector from bouncing to eyelashes/eyelids when contrast changes.
                prev_iris = getattr(self, "_iris_ellipse_smoothed", None) or getattr(self, "iris_ellipse", None)
                iris_roi_mask = ellipse_roi_mask_u8(
                    self.img_preprocessed.shape[:2],
                    prev_iris,
                    scale=float(getattr(self.params, "blob_cyan_roi_scale", 1.25)),
                    dilate_k=int(getattr(self.params, "blob_iris_roi_dilate_k", 0)),
                    erode_k=int(getattr(self.params, "blob_iris_roi_erode_k", 0)),
                )
            self.img_iris_roi_mask = iris_roi_mask

            pupil_ellipse, pupil_conf, pupil_blob, raw_mask, _info = detect_pupil_blob(roi_bgr, self.params, iris_roi_mask_u8=iris_roi_mask)

            # Debug views: keep the existing window meanings:
            #   Stage 4 ("Threshold")   -> raw_mask
            #   Stage 5 ("Morphology")  -> final filled blob
            self.img_threshold = raw_mask
            self.img_morphology = pupil_blob

            # Use the detected ellipse for downstream (sharing + iris anchor)
            bin_img = (pupil_blob if (pupil_blob is not None and np.count_nonzero(pupil_blob) > 0) else (raw_mask if raw_mask is not None else self._binary_for_pupil(self.img_preprocessed)))
            min_area = int(self.params.min_area)
            max_area = int(self.params.max_area)

        # Prefer external contours to avoid selecting hole contours
        ext_contours = find_external_contours(bin_img)

        if self.detect_mode != "iris":
            # Blob mode: use the filled blob (already cleaned in detect_pupil_blob) as the truth.
            # This avoids unstable scoring against eyelids / lashes / glasses rims.
            best_contour = max(ext_contours, key=cv2.contourArea) if ext_contours else None
            best_score = float(pupil_conf or 0.0)
            metas = []
            ellipse = pupil_ellipse if pupil_ellipse is not None else fit_ellipse_if_possible(best_contour)

        else:
            # Iris mode: contour -> center-pick -> fallback scorer
            best_contour = self._pick_contour_near_center(
                ext_contours,
                shape_hw=roi_bgr.shape[:2],
                min_area=min_area,
                max_area=max_area,
            )

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

        # Compute iris ellipse AFTER we have pupil ellipse
        self.iris_ellipse = self._fit_iris_ellipse_simple(self.img_preprocessed, ellipse)

        self.current_ellipse = ellipse
        self.current_confidence = float(best_score)
        self.detected_contours = metas

        self.img_contours = contours_view(bin_img, metas, best_contour, ellipse)
        self.img_eye_overlay, conf = eye_overlay(
            roi_bgr,
            ellipse,
            best_score,
            iris_ellipse=self.iris_ellipse,
            pupil_blob=(pupil_blob if self.detect_mode != "iris" else None),
        )
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
                ie = getattr(self, 'iris_ellipse', None)
                prep_vis = self._preview_with_ellipse(self.img_preprocessed, ie, color=(255, 255, 0), thickness=2)
                prep_vis = self._preview_with_ellipse(prep_vis, e, color=(0, 255, 255), thickness=1)
                stage4_vis = self._preview_with_ellipse(stage4_img, e)
                morph_vis = self._preview_with_ellipse(self.img_morphology, e)

                # Simplified grid windows
                iris_mask_vis = self._preview_with_ellipse(
                    self.img_iris_mask, ie, color=(0, 255, 255), thickness=2
                ) if self.img_iris_mask is not None else None
                
                items = [
                    (title0, self.img_eye_overlay),
                    ("1. Original ROI", roi_vis if self.views.show_original else None),
                    ("2. Grayscale", gray_vis if self.views.show_grayscale else None),
                    ("3. Preprocessed", prep_vis if self.views.show_preprocessed else None),
                    (stage4_title, stage4_vis if self.views.show_threshold else None),
                    ("5. Morphology", morph_vis if self.views.show_morphology else None),
                    ("6. Iris Mask", iris_mask_vis if self.views.show_iris_mask else None),
                    ("7. Contours", self.img_contours if self.views.show_contours else None),
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
                self._show_info_panel("8. Parameters & Info", panel)

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
                # Simplified iris controls
                elif key == ord('i'):
                    self.params.iris_sclera_threshold = max(100, self.params.iris_sclera_threshold - 10)
                    print(f"[IRIS] Sclera threshold: {self.params.iris_sclera_threshold}")
                elif key == ord('o'):
                    self.params.iris_sclera_threshold = min(220, self.params.iris_sclera_threshold + 10)
                    print(f"[IRIS] Sclera threshold: {self.params.iris_sclera_threshold}")
                elif key == ord('k'):
                    self.params.iris_blur = max(3, self.params.iris_blur - 2)
                    print(f"[IRIS] Blur: {self.params.iris_blur}")
                elif key == ord('l'):
                    self.params.iris_blur = min(15, self.params.iris_blur + 2)
                    print(f"[IRIS] Blur: {self.params.iris_blur}")
                elif key == ord('['):
                    self.params.iris_expand_ratio = max(1.5, self.params.iris_expand_ratio - 0.1)
                    print(f"[IRIS] Expand ratio: {self.params.iris_expand_ratio:.1f}")
                elif key == ord(']'):
                    self.params.iris_expand_ratio = min(4.0, self.params.iris_expand_ratio + 0.1)
                    print(f"[IRIS] Expand ratio: {self.params.iris_expand_ratio:.1f}")
                elif key == ord('h'):
                    self.toggles.use_histogram_eq = not self.toggles.use_histogram_eq
                elif key == ord('g'):
                    self.toggles.use_glint_removal = not self.toggles.use_glint_removal
                elif key == ord('f'):
                    self.toggles.use_bilateral_filter = not self.toggles.use_bilateral_filter
                elif key == ord('w'):
                    # Toggle glasses mode (enhanced glare removal)
                    self.toggles.use_glasses_mode = not self.toggles.use_glasses_mode
                    print(f"[GLASSES MODE] {'ON' if self.toggles.use_glasses_mode else 'OFF'}")
                elif key == ord('e'):
                    # Adjust glare threshold
                    self.params.glare_threshold = max(200, self.params.glare_threshold - 10)
                    print(f"[GLARE] Threshold: {self.params.glare_threshold}")
                elif key == ord('t'):
                    # Adjust glare threshold
                    self.params.glare_threshold = min(250, self.params.glare_threshold + 10)
                    print(f"[GLARE] Threshold: {self.params.glare_threshold}")
                elif key == ord('y'):
                    # Adjust inpaint radius
                    self.params.glare_inpaint_radius = max(3, self.params.glare_inpaint_radius - 1)
                    print(f"[GLARE] Inpaint radius: {self.params.glare_inpaint_radius}")
                elif key == ord('u'):
                    # Adjust inpaint radius
                    self.params.glare_inpaint_radius = min(10, self.params.glare_inpaint_radius + 1)
                    print(f"[GLARE] Inpaint radius: {self.params.glare_inpaint_radius}")
                # New granular image processing controls
                elif key == ord('9'):
                    self.params.contrast_alpha = max(0.5, self.params.contrast_alpha - 0.1)
                    print(f"[IMG] Contrast: {self.params.contrast_alpha:.1f}")
                elif key == ord('0'):
                    self.params.contrast_alpha = min(3.0, self.params.contrast_alpha + 0.1)
                    print(f"[IMG] Contrast: {self.params.contrast_alpha:.1f}")
                elif key == ord(','):
                    self.params.brightness_beta = max(-100, self.params.brightness_beta - 10)
                    print(f"[IMG] Brightness: {self.params.brightness_beta:+d}")
                elif key == ord('.'):
                    self.params.brightness_beta = min(100, self.params.brightness_beta + 10)
                    print(f"[IMG] Brightness: {self.params.brightness_beta:+d}")
                elif key == ord('/'):
                    self.params.gamma_value = max(0.5, self.params.gamma_value - 0.1)
                    print(f"[IMG] Gamma: {self.params.gamma_value:.1f}")
                elif key == ord(';'):
                    self.params.gamma_value = min(2.0, self.params.gamma_value + 0.1)
                    print(f"[IMG] Gamma: {self.params.gamma_value:.1f}")
                elif key == ord('['):
                    self.params.sharpen_amount = max(0.0, self.params.sharpen_amount - 0.2)
                    print(f"[IMG] Sharpen: {self.params.sharpen_amount:.1f}")
                elif key == ord(']'):
                    self.params.sharpen_amount = min(2.0, self.params.sharpen_amount + 0.2)
                    print(f"[IMG] Sharpen: {self.params.sharpen_amount:.1f}")
                elif key == ord('\\'):
                    self.params.clahe_clip_limit = max(1.0, self.params.clahe_clip_limit - 0.5)
                    print(f"[IMG] CLAHE Clip: {self.params.clahe_clip_limit:.1f}")
                elif key == ord("'"):
                    self.params.clahe_clip_limit = min(8.0, self.params.clahe_clip_limit + 0.5)
                    print(f"[IMG] CLAHE Clip: {self.params.clahe_clip_limit:.1f}")
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
