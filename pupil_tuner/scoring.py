from __future__ import annotations
import math
import cv2
import numpy as np
from dataclasses import dataclass
from typing import Optional, List

_4PI = 4.0 * math.pi

@dataclass
class ContourMeta:
    contour: np.ndarray
    area: float
    perimeter: float
    circularity: float
    centroid: tuple[float, float]
    score: float
    label: str

def find_contours(morph_img: np.ndarray):
    contours, _ = cv2.findContours(morph_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def score_contours(
    contours,
    frame_shape_hw: tuple[int, int],
    min_area: int,
    max_area: int,
    min_circularity: float,
):
    """
    Filter + score contours:
      - area range
      - circularity threshold
      - score = 0.5*circularity + 0.5*center_score
    """
    h, w = frame_shape_hw
    cx0 = w * 0.5
    cy0 = h * 0.5
    half_min = min(w, h) * 0.5

    best_contour = None
    best_score = 0.0
    valid: List[ContourMeta] = []

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area or area > max_area:
            continue

        perimeter = cv2.arcLength(contour, True)
        if perimeter <= 0:
            continue

        circularity = _4PI * area / (perimeter * perimeter)
        if circularity < min_circularity:
            continue

        M = cv2.moments(contour)
        m00 = M["m00"]
        if m00 <= 1e-6:
            continue

        cx = M["m10"] / m00
        cy = M["m01"] / m00

        ddx = cx - cx0
        ddy = cy - cy0
        dist = math.sqrt(ddx * ddx + ddy * ddy)
        center_score = 1.0 - (dist / half_min) if half_min > 0 else 0.0

        score = 0.5 * circularity + 0.5 * center_score

        label = f"S:{score:.2f} C:{circularity:.2f} A:{int(area)}"
        valid.append(ContourMeta(
            contour=contour,
            area=area,
            perimeter=perimeter,
            circularity=circularity,
            centroid=(cx, cy),
            score=score,
            label=label,
        ))

        if score > best_score:
            best_score = score
            best_contour = contour

    valid.sort(key=lambda m: m.score, reverse=True)
    return best_contour, best_score, valid

def fit_ellipse_if_possible(contour: Optional[np.ndarray]):
    if contour is None:
        return None
    if len(contour) < 5:
        return None
    try:
        return cv2.fitEllipse(contour)
    except Exception:
        return None
