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


def eye_overlay(frame_bgr: np.ndarray, ellipse, best_score: float):
    overlay = frame_bgr.copy()

    if ellipse is None:
        cv2.putText(overlay, "NO DETECTION", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
        cv2.putText(overlay, "Adjust threshold (+/-)", (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
        return overlay, 0.0

    try:
        (cx, cy), (ew, eh), angle = ellipse

        cv2.ellipse(overlay, ellipse, (0, 255, 255), 3)

        cx_i, cy_i = int(cx), int(cy)
        crosshair_size = 12
        cv2.line(overlay, (cx_i - crosshair_size, cy_i), (cx_i + crosshair_size, cy_i), (0, 255, 255), 2)
        cv2.line(overlay, (cx_i, cy_i - crosshair_size), (cx_i, cy_i + crosshair_size), (0, 255, 255), 2)

        cv2.circle(overlay, (cx_i, cy_i), 5, (0, 255, 255), -1)
        cv2.circle(overlay, (cx_i, cy_i), 6, (255, 255, 255), 1)

        conf_text = f"CONFIDENCE: {best_score:.2f}"
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

        dim_text = f"Ellipse: {int(ew)}x{int(eh)}px | {status}"
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
        quality_text = "TRACKING"
        cv2.rectangle(overlay, (quality_x - 10, 5), (overlay.shape[1] - 5, 50), (0, 0, 0), -1)
        cv2.rectangle(overlay, (quality_x - 10, 5), (overlay.shape[1] - 5, 50), color, 3)
        cv2.putText(overlay, quality_text, (quality_x, quality_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        return overlay, float(best_score)
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
    width: int = 700,
    height: int = 280,
) -> np.ndarray:
    """
    Returns a HUD image. Updated to reflect controls currently in app.py:
      - Mode toggle: m
      - Morph kernel size: 7/8
      - CLAHE contrast toggle uses 'h' (shown as Hist EQ (CLAHE))
      - Threshold controls are PUPIL-only
    """
    panel = np.zeros((height, width, 3), dtype=np.uint8)

    # Layout constants
    left_x = 12
    right_x = width // 2 + 10
    y = 22
    line_h = 20

    # Header
    header = f"TUNING PARAMETERS   |   MODE: {detect_mode.upper()}  (m)"
    cv2.putText(panel, header, (left_x, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    y += line_h + 10

    # Left column: numeric params + keys
    params_lines = [
        f"Threshold: {params.threshold_value}  ( +/- )   [PUPIL only]",
        f"Min Area: {params.min_area}  ( z / x )",
        f"Max Area: {params.max_area}  ( c / v )",
        f"Min Circularity: {params.min_circularity:.2f}  ( b / n )",
        f"Blur Kernel: {params.blur_kernel_size}  ( 1 / 2 )",
        f"Morph Close: {params.morph_close_iterations}  ( 3 / 4 )",
        f"Morph Open: {params.morph_open_iterations}  ( 5 / 6 )",
        f"Morph Kernel: {params.morph_kernel_size}  ( 7 / 8 )",
    ]
    yy = y
    for s in params_lines:
        cv2.putText(panel, s, (left_x, yy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (210, 210, 210), 1)
        yy += line_h

    # Right column: toggles
    yy = y
    options = [
        ("Hist EQ (CLAHE)", toggles.use_histogram_eq, "h"),
        ("Glint Removal", toggles.use_glint_removal, "g"),
        ("Auto Thresh (Otsu)", toggles.use_auto_threshold, "a"),
        ("Adaptive Thresh", toggles.use_adaptive_threshold, "d"),
        ("Bilateral Filter", toggles.use_bilateral_filter, "f"),
    ]
    for name, state, key in options:
        color = (0, 255, 0) if state else (100, 100, 100)
        cv2.putText(panel, f"{name}: {'ON' if state else 'OFF'}  ({key})",
                    (right_x, yy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        yy += line_h

    yy += 8
    # Status box
    cv2.putText(panel, f"FPS: {fps:.1f}", (right_x, yy),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
    yy += line_h

    if has_detection:
        cv2.putText(panel, f"Confidence: {confidence*100:.0f}%",
                    (right_x, yy), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
        yy += line_h
        cv2.putText(panel, f"Valid Contours: {valid_contours}",
                    (right_x, yy), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
    else:
        cv2.putText(panel, "NO DETECTION",
                    (right_x, yy), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)

    return panel
