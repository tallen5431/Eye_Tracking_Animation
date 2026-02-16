from __future__ import annotations
import cv2
import numpy as np
from typing import List
from .scoring import ContourMeta
from .config import TuningParams, TuningToggles


def contours_view(morph_img: np.ndarray, metas: List[ContourMeta], best_contour, ellipse) -> np.ndarray:
    img = cv2.cvtColor(morph_img, cv2.COLOR_GRAY2BGR)

    for m in metas:
        cv2.drawContours(img, [m.contour], -1, (0, 255, 255), 2)
        cx, cy = int(m.centroid[0]), int(m.centroid[1])
        cv2.putText(img, m.label, (cx - 40, cy - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    if best_contour is not None:
        cv2.drawContours(img, [best_contour], -1, (0, 255, 0), 3)

    if ellipse is not None:
        cv2.ellipse(img, ellipse, (255, 0, 0), 2)

    return img


def eye_overlay(frame_bgr: np.ndarray, ellipse, best_score: float, iris_ellipse=None, pupil_blob=None):
    """
    ENHANCED: Iris ellipse is now PRIMARY tracking target (cyan).
    Pupil ellipse shown as secondary reference (dimmer orange).
    
    Args:
        frame_bgr: Original eye image
        ellipse: Ellipse fitted to pupil (secondary)
        best_score: Confidence of pupil detection
        iris_ellipse: Ellipse fitted to iris (PRIMARY, bright cyan)
    
    Returns:
        (overlay_image, confidence)
    """
    overlay = frame_bgr.copy()

    # Optional: visualize blob-mask pupil detection (filled cyan tint)
    if pupil_blob is not None:
        try:
            m = pupil_blob
            if m.dtype != np.uint8:
                m = m.astype(np.uint8)
            if m.ndim == 3:
                m = cv2.cvtColor(m, cv2.COLOR_BGR2GRAY)
            sel = (m == 255)
            if np.any(sel):
                # Integer blend on masked pixels only (avoids full-frame float32)
                # 0.45 * pixel + 0.55 * [255,255,0] using fixed-point: *115/256 + const
                px = overlay[sel].astype(np.uint16)
                # cyan tint: B=255, G=255, R=0 scaled by 0.55 * 256 = 140.8 ≈ 140
                overlay[sel] = np.clip(
                    (px * 115 + np.array([35700, 35700, 0], dtype=np.uint16)) >> 8,
                    0, 255
                ).astype(np.uint8)
        except Exception:
            pass

    # Determine which ellipse to use for tracking
    if iris_ellipse is not None:
        primary_ellipse = iris_ellipse
        primary_name = "IRIS"
        # Iris detection is more stable, boost confidence
        confidence = min(1.0, best_score * 1.2) if best_score > 0 else 0.85
    elif ellipse is not None:
        primary_ellipse = ellipse
        primary_name = "PUPIL"
        confidence = best_score
    else:
        primary_ellipse = None
        primary_name = "NONE"
        confidence = 0.0

    if primary_ellipse is None:
        cv2.putText(overlay, "NO DETECTION", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
        cv2.putText(overlay, "Enable CLAHE (h) for iris mode", (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
        return overlay, 0.0

    try:
        # Draw PRIMARY ellipse (iris in BRIGHT CYAN)
        (cx, cy), (ew, eh), angle = primary_ellipse
        cv2.ellipse(overlay, primary_ellipse, (0, 255, 255), 4)  # Thicker for prominence
        
        # Draw secondary pupil ellipse if both available (dimmer)
        if iris_ellipse is not None and ellipse is not None:
            try:
                cv2.ellipse(overlay, ellipse, (0, 200, 255), 2)  # Thinner, orange-ish
            except:
                pass

        cx_i, cy_i = int(cx), int(cy)
        crosshair_size = 12
        cv2.line(overlay, (cx_i - crosshair_size, cy_i), (cx_i + crosshair_size, cy_i), (0, 255, 255), 2)
        cv2.line(overlay, (cx_i, cy_i - crosshair_size), (cx_i, cy_i + crosshair_size), (0, 255, 255), 2)

        cv2.circle(overlay, (cx_i, cy_i), 5, (0, 255, 255), -1)
        cv2.circle(overlay, (cx_i, cy_i), 6, (255, 255, 255), 1)

        conf_text = f"TRACKING: {primary_name}  ({confidence:.2f})"
        text_size = cv2.getTextSize(conf_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
        text_x, text_y = 10, 35
        cv2.rectangle(overlay,
                      (text_x - 5, text_y - text_size[1] - 5),
                      (text_x + text_size[0] + 10, text_y + 10),
                      (0, 0, 0), -1)
        cv2.rectangle(overlay,
                      (text_x - 5, text_y - text_size[1] - 5),
                      (text_x + text_size[0] + 10, text_y + 10),
                      (0, 255, 255), 2)

        if best_score > 0.8:
            color = (0, 255, 0); status = "EXCELLENT"
        elif best_score > 0.6:
            color = (0, 255, 255); status = "GOOD"
        elif best_score > 0.4:
            color = (0, 165, 255); status = "FAIR"
        else:
            color = (0, 0, 255); status = "POOR"

        cv2.putText(overlay, conf_text, (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        dim_text = f"{primary_name}: {int(ew)}x{int(eh)}px | {status}"
        text_y2 = text_y + 40
        cv2.rectangle(overlay,
                      (text_x - 5, text_y2 - 25 - 5),
                      (text_x + 360, text_y2 + 10),
                      (0, 0, 0), -1)
        cv2.rectangle(overlay,
                      (text_x - 5, text_y2 - 25 - 5),
                      (text_x + 360, text_y2 + 10),
                      color, 2)
        cv2.putText(overlay, dim_text, (text_x, text_y2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

        quality_x = overlay.shape[1] - 120
        quality_y = 35
        quality_text = primary_name
        cv2.rectangle(overlay, (quality_x - 10, 5), (overlay.shape[1] - 5, 50), (0, 0, 0), -1)
        cv2.rectangle(overlay, (quality_x - 10, 5), (overlay.shape[1] - 5, 50), color, 3)
        cv2.putText(overlay, quality_text, (quality_x, quality_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # Add secondary info if both available
        if iris_ellipse is not None and ellipse is not None:
            try:
                (pcx, pcy), (pew, peh), _ = ellipse
                secondary_text = f"Pupil: {int(pew)}x{int(peh)}px"
                text_y3 = text_y2 + 35
                cv2.putText(overlay, secondary_text, (text_x, text_y3),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
            except:
                pass
        
        return overlay, float(confidence)
    except Exception:
        cv2.putText(overlay, "NO DETECTION", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
        cv2.putText(overlay, "Adjust threshold (+/-)", (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
        return overlay, 0.0


def info_panel(
    fps: float,
    params: TuningParams,
    toggles: TuningToggles,
    has_detection: bool,
    confidence: float,
    valid_contours: int,
    detect_mode: str = "pupil",
    width: int = 900,
    height: int = 460,
) -> np.ndarray:
    """
    Returns a HUD panel with organised sections and clear key bindings.
    Three columns: Detection | Image Processing | Toggles & Status.
    """
    panel = np.zeros((height, width, 3), dtype=np.uint8)

    # Layout
    col1_x = 12
    col2_x = width // 3 + 5
    col3_x = 2 * width // 3 + 5
    y_top = 22
    line_h = 19
    section_gap = 6
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_s = 0.43
    font_hdr = 0.52
    white = (210, 210, 210)
    cyan = (0, 255, 255)
    dim = (120, 120, 120)
    green = (0, 220, 0)
    red = (0, 0, 255)

    def _hdr(x, y, text):
        cv2.putText(panel, text, (x, y), font, font_hdr, cyan, 1, cv2.LINE_AA)
        return y + line_h + 2

    def _line(x, y, text, color=white):
        cv2.putText(panel, text, (x, y), font, font_s, color, 1, cv2.LINE_AA)
        return y + line_h

    # ------- Header -------
    header = f"MODE: {detect_mode.upper()}  (m)  |  SAVE: s  LOAD: l  |  PROFILES: Shift+1/2/3/4  |  SPACE: views  |  r: reset  |  q: quit"
    cv2.putText(panel, header, (col1_x, y_top), font, 0.45, cyan, 1, cv2.LINE_AA)
    y_start = y_top + line_h + section_gap

    # ============ COLUMN 1: Detection ============
    y = _hdr(col1_x, y_start, "DETECTION")
    y = _line(col1_x, y, f"Threshold: {params.threshold_value}        +/-")
    y = _line(col1_x, y, f"Min Area: {params.min_area}           z/x")
    y = _line(col1_x, y, f"Max Area: {params.max_area}          c/v")
    y = _line(col1_x, y, f"Min Circ: {params.min_circularity:.2f}          b/n")
    y = _line(col1_x, y, f"Blur Kern: {params.blur_kernel_size}             1/2")
    y += section_gap
    y = _hdr(col1_x, y, "MORPHOLOGY")
    y = _line(col1_x, y, f"Close Iter: {params.morph_close_iterations}            3/4")
    y = _line(col1_x, y, f"Open Iter: {params.morph_open_iterations}             5/6")
    y = _line(col1_x, y, f"Kernel: {params.morph_kernel_size}                7/8")
    y += section_gap
    y = _hdr(col1_x, y, "IRIS")
    y = _line(col1_x, y, f"Sclera Thr: {params.iris_sclera_threshold}         i/o")
    y = _line(col1_x, y, f"Blur: {params.iris_blur}                  k/l")
    y = _line(col1_x, y, f"Expand: {params.iris_expand_ratio:.1f}x            [/]")

    # ============ COLUMN 2: Image Processing ============
    y = _hdr(col2_x, y_start, "IMAGE PROCESSING")
    y = _line(col2_x, y, f"Contrast: {params.contrast_alpha:.1f}           9/0")
    y = _line(col2_x, y, f"Brightness: {params.brightness_beta:+d}          ,/.")
    y = _line(col2_x, y, f"Gamma: {params.gamma_value:.1f}              ;/ /")
    y = _line(col2_x, y, f"CLAHE Clip: {params.clahe_clip_limit:.1f}          \\/'" )
    y += section_gap
    y = _hdr(col2_x, y, "GLARE REMOVAL")
    y = _line(col2_x, y, f"Glare Thr: {params.glare_threshold}           e/t")
    y = _line(col2_x, y, f"Inpaint Rad: {params.glare_inpaint_radius}           y/u")
    y += section_gap
    y = _hdr(col2_x, y, "BLOB PARAMS")
    y = _line(col2_x, y, f"Dark %ile: {params.blob_dark_percentile:.0f}")
    y = _line(col2_x, y, f"Close K: {params.blob_close_ksize}")
    y = _line(col2_x, y, f"Min Circ: {params.blob_min_circularity:.2f}")
    y = _line(col2_x, y, f"Sat Max: {params.blob_sat_max}")
    y = _line(col2_x, y, f"ROI Dilate: {params.blob_iris_roi_dilate_k}")

    # ============ COLUMN 3: Toggles & Status ============
    y = _hdr(col3_x, y_start, "TOGGLES")
    toggles_list = [
        ("CLAHE (Hist EQ)", toggles.use_histogram_eq, "h"),
        ("Glint Removal", toggles.use_glint_removal, "g"),
        ("Glasses Mode", toggles.use_glasses_mode, "w"),
        ("Auto Thresh", toggles.use_auto_threshold, "a"),
        ("Adaptive Thresh", toggles.use_adaptive_threshold, "d"),
        ("Bilateral Filter", toggles.use_bilateral_filter, "f"),
    ]
    for name, state, key in toggles_list:
        color = green if state else dim
        label = "ON" if state else "OFF"
        y = _line(col3_x, y, f"{label:>3}  {name}  ({key})", color)

    y += section_gap + 4
    y = _hdr(col3_x, y, "STATUS")

    fps_color = green if fps >= 20 else (0, 165, 255) if fps >= 10 else red
    y = _line(col3_x, y, f"FPS: {fps:.1f}", fps_color)

    if has_detection:
        conf_pct = confidence * 100
        if conf_pct >= 70:
            conf_color = green
        elif conf_pct >= 40:
            conf_color = (0, 165, 255)
        else:
            conf_color = red
        y = _line(col3_x, y, f"Confidence: {conf_pct:.0f}%", conf_color)
        y = _line(col3_x, y, f"Contours: {valid_contours}", white)
    else:
        y = _line(col3_x, y, "NO DETECTION", red)

    return panel
