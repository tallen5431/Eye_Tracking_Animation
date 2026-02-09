from __future__ import annotations

import cv2
import numpy as np
from .config import TuningParams, TuningToggles


def to_gray(frame_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)


def _odd(k: int, minimum: int = 1) -> int:
    k = max(minimum, int(k))
    return k if (k % 2 == 1) else k + 1


def contrast_normalize_u8(img: np.ndarray) -> np.ndarray:
    """
    Boost global contrast by stretching min..max to 0..255.
    Great right before Canny when the frame is "flat".
    """
    out = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
    return out.astype(np.uint8, copy=False)


def apply_clahe(img_u8: np.ndarray, clip_limit: float = 2.0, tile_grid_size: tuple[int, int] = (8, 8)) -> np.ndarray:
    """
    Local-contrast enhancer (usually better than equalizeHist for eye imagery).
    """
    clahe = cv2.createCLAHE(clipLimit=float(clip_limit), tileGridSize=tile_grid_size)
    return clahe.apply(img_u8)


def preprocess(gray: np.ndarray, toggles: TuningToggles) -> np.ndarray:
    """mo
    Optional preprocessing:
      - bilateral filter (edge-preserving)
      - contrast boost (CLAHE, toggled by use_histogram_eq)
      - glint removal via threshold + inpaint
    """
    img = gray.copy()

    if toggles.use_bilateral_filter:
        img = cv2.bilateralFilter(img, 5, 50, 50)

    # NOTE: we reuse the existing toggle name "use_histogram_eq"
    # but apply CLAHE instead of global equalizeHist for better eye contrast.
    if toggles.use_histogram_eq:
        img = apply_clahe(img, clip_limit=2.0, tile_grid_size=(8, 8))

    if toggles.use_glint_removal:
        _, glint_mask = cv2.threshold(img, 245, 255, cv2.THRESH_BINARY)
        if cv2.countNonZero(glint_mask) > 10:
            glint_mask = cv2.dilate(glint_mask, np.ones((3, 3), np.uint8), iterations=1)
            img = cv2.inpaint(img, glint_mask, 2, cv2.INPAINT_NS)

    return img


def blur(img: np.ndarray, blur_kernel_size: int) -> np.ndarray:
    k = _odd(blur_kernel_size, minimum=1)
    return cv2.GaussianBlur(img, (k, k), 0)


def threshold(img_blurred: np.ndarray, params: TuningParams, toggles: TuningToggles):
    if toggles.use_adaptive_threshold:
        th = cv2.adaptiveThreshold(
            img_blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 11, 2
        )
        return th, None
    elif toggles.use_auto_threshold:
        thresh_val, th = cv2.threshold(
            img_blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )
        return th, float(thresh_val)
    else:
        _, th = cv2.threshold(
            img_blurred, int(params.threshold_value), 255, cv2.THRESH_BINARY_INV
        )
        return th, None


def morphology(bin_img: np.ndarray, params: TuningParams) -> tuple[np.ndarray, np.ndarray]:
    k = _odd(params.morph_kernel_size, minimum=1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))

    out = bin_img.copy()
    if params.morph_close_iterations > 0:
        out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, kernel, iterations=int(params.morph_close_iterations))
    if params.morph_open_iterations > 0:
        out = cv2.morphologyEx(out, cv2.MORPH_OPEN, kernel, iterations=int(params.morph_open_iterations))

    return out, kernel

def fill_holes_u8(mask_u8: np.ndarray) -> np.ndarray:
    """
    Fill internal holes in a binary mask (0/255) using flood fill.
    Returns a new mask with holes filled.
    """
    if mask_u8 is None:
        return mask_u8

    # Ensure single-channel uint8
    m = mask_u8
    if m.dtype != np.uint8:
        m = m.astype(np.uint8)
    if m.ndim == 3:
        m = cv2.cvtColor(m, cv2.COLOR_BGR2GRAY)

    # Need a copy because floodFill modifies in-place
    inv = cv2.bitwise_not(m)
    h, w = inv.shape[:2]
    ff = inv.copy()
    flood_mask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(ff, flood_mask, (0, 0), 0)  # remove background-connected "holes"
    holes = cv2.bitwise_not(ff)               # now only holes remain
    filled = cv2.bitwise_or(m, holes)
    return filled


def find_external_contours(mask_u8: np.ndarray):
    """
    Find only EXTERNAL contours. Helps avoid selecting inner hole contours.
    Returns list of contours.
    """
    m = mask_u8
    if m is None:
        return []
    if m.dtype != np.uint8:
        m = m.astype(np.uint8)
    if m.ndim == 3:
        m = cv2.cvtColor(m, cv2.COLOR_BGR2GRAY)

    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

