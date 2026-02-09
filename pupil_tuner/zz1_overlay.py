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

        dim_text = f"Pupil: {int(ew)}x{int(eh)}px | {status}"
        text_y2 = text_y + 40
        cv2.rectangle(overlay,
                      (text_x - 5, text_y2 - 25 - 5),
                      (text_x + 350, text_y2 + 10),
                      (0, 0, 0), -1)
        cv2.rectangle(overlay,
                      (text_x - 5, text_y2 - 25 - 5),
                      (text_x + 350, text_y2 + 10),
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
    height: int = 200,
) -> np.ndarray:
    panel = np.zeros((height, 640, 3), dtype=np.uint8)

    y = 20
    line_h = 20

    cv2.putText(panel, "TUNING PARAMETERS", (10, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    y += line_h + 10

    params_lines = [
        f"Threshold: {params.threshold_value} (Manual: +/-)",
        f"Min Area: {params.min_area} (Keys: z/x)",
        f"Max Area: {params.max_area} (Keys: c/v)",
        f"Min Circularity: {params.min_circularity:.2f} (Keys: b/n)",
        f"Blur Kernel: {params.blur_kernel_size} (Keys: 1/2)",
        f"Morph Close: {params.morph_close_iterations} (Keys: 3/4)",
        f"Morph Open: {params.morph_open_iterations} (Keys: 5/6)",
    ]
    for s in params_lines:
        cv2.putText(panel, s, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
        y += line_h

    y = 20
    x = 350
    options = [
        f"Hist EQ: {'ON' if toggles.use_histogram_eq else 'OFF'} (h)",
        f"Glint Removal: {'ON' if toggles.use_glint_removal else 'OFF'} (g)",
        f"Auto Thresh: {'ON' if toggles.use_auto_threshold else 'OFF'} (a)",
        f"Adaptive: {'ON' if toggles.use_adaptive_threshold else 'OFF'} (d)",
        f"Bilateral: {'ON' if toggles.use_bilateral_filter else 'OFF'} (f)",
    ]
    for opt in options:
        color = (0, 255, 0) if "ON" in opt else (100, 100, 100)
        cv2.putText(panel, opt, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
        y += line_h

    y += 10
    cv2.putText(panel, f"FPS: {fps:.1f}", (x, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    y += line_h
    if has_detection:
        cv2.putText(panel, f"Confidence: {confidence*100:.0f}%", (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        y += line_h
        cv2.putText(panel, f"Valid Contours: {valid_contours}", (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    else:
        cv2.putText(panel, "NO DETECTION", (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    return panel
