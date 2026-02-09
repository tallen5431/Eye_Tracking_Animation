from __future__ import annotations

import cv2
import numpy as np
from .config import TuningParams, TuningToggles


def rotate_frame(frame: np.ndarray, rotation: int) -> np.ndarray:
    """
    Rotate frame by specified angle (0, 90, 180, 270 degrees).
    
    Args:
        frame: Input image
        rotation: Rotation angle in degrees (0, 90, 180, 270)
    
    Returns:
        Rotated image
    """
    if rotation == 0:
        return frame
    elif rotation == 90:
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    elif rotation == 180:
        return cv2.rotate(frame, cv2.ROTATE_180)
    elif rotation == 270:
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    else:
        return frame


def to_gray(frame_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)


def _odd(k: int, minimum: int = 0) -> int:
    k = max(minimum, int(k))
    return k if (k % 2 == 1) else k + 1


_se_cache: dict[tuple[int, int], np.ndarray] = {}

def _get_ellipse_se(k: int) -> np.ndarray:
    """Get (and cache) an elliptical structuring element of size k x k."""
    key = (k, k)
    se = _se_cache.get(key)
    if se is None:
        se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, key)
        _se_cache[key] = se
    return se


def ellipse_roi_mask_u8(shape_hw, ellipse, scale=1.25, dilate_k=0, erode_k=0):
    """Create a filled elliptical ROI mask (0/255) from a cv2.fitEllipse-style tuple.

    Used to constrain blob detection to the (stable) iris region to reduce bouncing.
    """
    if ellipse is None:
        return None
    (cx, cy), (MA, ma), ang = ellipse
    if MA <= 1 or ma <= 1:
        return None

    h, w = shape_hw
    mask = np.zeros((h, w), dtype=np.uint8)

    e_scaled = ((float(cx), float(cy)), (float(MA) * float(scale), float(ma) * float(scale)), float(ang))
    try:
        cv2.ellipse(mask, e_scaled, 255, -1)
    except cv2.error:
        return None

    dk = int(dilate_k or 0)
    ek = int(erode_k or 0)

    if dk >= 3:
        kk = _odd(dk, minimum=3)
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kk, kk))
        mask = cv2.dilate(mask, k, iterations=1)
    if ek >= 3:
        kk = _odd(ek, minimum=3)
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kk, kk))
        mask = cv2.erode(mask, k, iterations=1)

    return mask


def contrast_normalize_u8(img: np.ndarray) -> np.ndarray:
    """
    Boost global contrast by stretching min..max to 0..255.
    Great right before Canny when the frame is "flat".
    """
    out = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
    return out.astype(np.uint8, copy=False)


def adjust_contrast_brightness(img: np.ndarray, alpha: float = 1.0, beta: int = 0) -> np.ndarray:
    """
    Adjust contrast and brightness.
    
    Args:
        img: Input image (grayscale)
        alpha: Contrast multiplier (1.0 = no change, >1.0 = more contrast)
        beta: Brightness offset (0 = no change, +50 = brighter)
    
    Returns:
        Adjusted image
        
    Formula: output = alpha * input + beta
    """
    if alpha == 1.0 and beta == 0:
        return img
    
    adjusted = cv2.convertScaleAbs(img, alpha=float(alpha), beta=int(beta))
    return adjusted


_gamma_lut_cache: dict[float, np.ndarray] = {}

def adjust_gamma(img: np.ndarray, gamma: float = 1.0) -> np.ndarray:
    """
    Apply gamma correction with cached LUT for repeated gamma values.
    """
    if gamma == 1.0:
        return img

    # Cache the LUT so we don't rebuild it every frame
    table = _gamma_lut_cache.get(gamma)
    if table is None:
        inv_gamma = 1.0 / float(gamma)
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)], dtype=np.uint8)
        _gamma_lut_cache[gamma] = table
    return cv2.LUT(img, table)


def sharpen_image(img: np.ndarray, amount: float = 0.0) -> np.ndarray:
    """
    Sharpen image using unsharp mask.
    
    Args:
        img: Input image (grayscale)
        amount: Sharpening strength (0.0 = no sharpening, 1.0 = standard, 2.0 = strong)
    
    Returns:
        Sharpened image
    """
    if amount <= 0.0:
        return img
    
    # Gaussian blur for unsharp mask
    blurred = cv2.GaussianBlur(img, (0, 0), 3)
    sharpened = cv2.addWeighted(img, 1.0 + float(amount), blurred, -float(amount), 0)
    return sharpened


_clahe_cache: dict[tuple[float, int], cv2.CLAHE] = {}

def apply_clahe(img_u8: np.ndarray, clip_limit: float = 2.0, tile_grid_size: int = 8) -> np.ndarray:
    """
    Local-contrast enhancer (usually better than equalizeHist for eye imagery).
    Caches the CLAHE object to avoid recreation every frame.
    """
    grid = max(4, min(16, int(tile_grid_size)))
    key = (float(clip_limit), grid)
    clahe = _clahe_cache.get(key)
    if clahe is None:
        clahe = cv2.createCLAHE(clipLimit=float(clip_limit), tileGridSize=(grid, grid))
        _clahe_cache[key] = clahe
    return clahe.apply(img_u8)


def auto_canny_u8(img_u8: np.ndarray, sigma: float = 0.33) -> np.ndarray:
    """
    Auto-tuned Canny edge detection using median intensity.
    Useful for iris boundary extraction on preprocessed frames.
    """
    if img_u8 is None:
        return img_u8
    v = float(np.median(img_u8))
    lower = int(max(0, (1.0 - float(sigma)) * v))
    upper = int(min(255, (1.0 + float(sigma)) * v))
    return cv2.Canny(img_u8, lower, upper)


def preprocess(gray: np.ndarray, params: TuningParams, toggles: TuningToggles) -> tuple[np.ndarray, dict]:
    """
    Step-by-step preprocessing with adjustable parameters at each stage.

    Pipeline:
      1. Contrast/brightness adjustment
      2. Gamma correction
      3. Sharpening (optional)
      4. Bilateral filter (optional)
      5. CLAHE (optional)
      6. Glint/glare removal (optional, enhanced for glasses)

    Returns:
        (processed_image, debug_images_dict)
    """
    debug = {}
    # Each processing step returns a new array so we don't need explicit copies
    # except for the original which we snapshot once.
    img = gray.copy()
    debug['0_original'] = gray  # read-only reference; gray is not mutated

    # Step 1: Contrast & Brightness
    if params.contrast_alpha != 1.0 or params.brightness_beta != 0:
        img = adjust_contrast_brightness(img, params.contrast_alpha, params.brightness_beta)
    debug['1_contrast_brightness'] = img

    # Step 2: Gamma Correction
    if params.gamma_value != 1.0:
        img = adjust_gamma(img, params.gamma_value)
    debug['2_gamma'] = img

    # Step 3: Sharpening
    if params.sharpen_amount > 0.0:
        img = sharpen_image(img, params.sharpen_amount)
    debug['3_sharpened'] = img

    # Step 4: Bilateral Filter (noise reduction, edge-preserving)
    if toggles.use_bilateral_filter:
        img = cv2.bilateralFilter(img, 5, 50, 50)
    debug['4_bilateral'] = img

    # Step 5: CLAHE (local contrast enhancement)
    if toggles.use_histogram_eq:
        img = apply_clahe(img, params.clahe_clip_limit, params.clahe_grid_size)
    debug['5_clahe'] = img

    # Step 6: Enhanced Glint/Glare Removal
    if toggles.use_glint_removal:
        thresh = max(200, min(250, int(params.glare_threshold)))
        radius = max(3, min(10, int(params.glare_inpaint_radius)))

        _, glare_mask = cv2.threshold(img, thresh, 255, cv2.THRESH_BINARY)

        if toggles.use_glasses_mode:
            kernel = np.ones((7, 7), np.uint8)
            glare_mask = cv2.dilate(glare_mask, kernel, iterations=2)
        else:
            kernel = np.ones((3, 3), np.uint8)
            glare_mask = cv2.dilate(glare_mask, kernel, iterations=1)

        if cv2.countNonZero(glare_mask) > 10:
            img = cv2.inpaint(img, glare_mask, radius, cv2.INPAINT_TELEA)

    debug['6_glint_removed'] = img

    return img, debug


def create_iris_mask_from_sclera(gray: np.ndarray, params: TuningParams, pupil_center=None, pupil_radius=None) -> np.ndarray:
    """
    SIMPLIFIED APPROACH: Detect iris by EXCLUDING the bright sclera (white of eye).
    
    Key insight: Sclera is very bright (180-255), iris is darker (60-160).
    Rather than detecting iris edges, we:
    1. Threshold out the bright sclera
    2. Get the remaining dark region (iris + pupil)
    3. Use the largest circular region as iris boundary
    
    This is much more robust than edge detection!
    
    Args:
        gray: Grayscale eye image
        params: Tuning parameters
        pupil_center: Optional pupil center for validation
        pupil_radius: Optional pupil radius for size estimation
    
    Returns:
        Binary mask where iris region is white (255)
    """
    if gray is None:
        return None
    
    # Blur to reduce noise (more blur = smoother boundaries)
    k = max(3, int(params.iris_blur) | 1)  # Force odd
    blurred = cv2.GaussianBlur(gray, (k, k), 0)
    
    # Threshold: anything brighter than this is sclera (exclude it)
    thresh_val = max(100, min(220, int(params.iris_sclera_threshold)))
    _, sclera_mask = cv2.threshold(blurred, thresh_val, 255, cv2.THRESH_BINARY)
    
    # Invert: dark regions (iris + pupil) become white
    iris_region = cv2.bitwise_not(sclera_mask)
    
    # Clean up with morphology
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    iris_region = cv2.morphologyEx(iris_region, cv2.MORPH_CLOSE, kernel, iterations=2)
    iris_region = cv2.morphologyEx(iris_region, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # Fill holes (important: iris might have reflections)
    iris_region = fill_holes_u8(iris_region)
    
    return iris_region


def fit_iris_from_mask_and_pupil(mask: np.ndarray, pupil_ellipse, params: TuningParams) -> tuple:
    """
    Fit iris ellipse using:
    1. Mask contour (from sclera exclusion)
    2. Pupil position as anchor point
    3. Expected size ratio (iris ≈ 2.5x pupil radius)
    
    This combines geometric constraints with image data for robustness.
    
    Returns:
        (ellipse, confidence) or (None, 0.0)
    """
    if mask is None or pupil_ellipse is None:
        return None, 0.0
    
    # Extract pupil info
    (px, py), (pw, ph), _ = pupil_ellipse
    pupil_r = (pw + ph) / 4.0  # Average radius
    
    # Expected iris radius
    expected_iris_r = pupil_r * float(params.iris_expand_ratio)
    
    # Find contours in mask
    contours = find_external_contours(mask)
    if not contours:
        return None, 0.0
    
    # Find largest contour near pupil center
    best = None
    best_score = 0.0
    
    for c in contours:
        if len(c) < 5:
            continue
        
        area = cv2.contourArea(c)
        if area < 100:
            continue
        
        # Check if contour contains pupil
        M = cv2.moments(c)
        if M["m00"] <= 1e-6:
            continue
        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]
        
        # Distance from pupil center
        dist = np.sqrt((cx - px)**2 + (cy - py)**2)
        
        # Penalize contours far from pupil
        dist_score = 1.0 / (1.0 + dist / 20.0)
        
        # Check size consistency
        expected_area = np.pi * expected_iris_r * expected_iris_r
        area_ratio = min(area, expected_area) / max(area, expected_area)
        
        score = dist_score * 0.4 + area_ratio * 0.6
        
        if score > best_score:
            best_score = score
            best = c
    
    if best is None or len(best) < 5:
        return None, 0.0
    
    # Fit ellipse
    try:
        ellipse = cv2.fitEllipse(best)
        confidence = float(best_score)
        return ellipse, confidence
    except:
        return None, 0.0


def blur(img: np.ndarray, blur_kernel_size: int) -> np.ndarray:
    k = _odd(blur_kernel_size, minimum=1)
    return cv2.GaussianBlur(img, (k, k), 0)



def safe_gauss(img: np.ndarray, k: int) -> np.ndarray:
    """Gaussian blur helper used by blob detection.
    If k < 3, returns the input unchanged. Ensures k is odd.
    """
    if not k or int(k) < 3:
        return img
    kk = _odd(int(k), minimum=3)
    return cv2.GaussianBlur(img, (kk, kk), 0)
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


def keep_largest_cc(mask_u8: np.ndarray, min_area_frac: float = 0.0) -> np.ndarray:
    """Keep only the largest connected component in a binary mask (0/255)."""
    if mask_u8 is None or np.count_nonzero(mask_u8) == 0:
        return mask_u8
    num, labels, stats, _ = cv2.connectedComponentsWithStats((mask_u8 > 0).astype(np.uint8), connectivity=8)
    if num <= 1:
        return mask_u8
    areas = stats[1:, cv2.CC_STAT_AREA]
    best_idx = 1 + int(np.argmax(areas))
    best_area = float(stats[best_idx, cv2.CC_STAT_AREA])

    out = np.zeros_like(mask_u8)
    out[labels == best_idx] = 255

    # Optional guard: if the "largest" is still too small, return original mask
    if min_area_frac and min_area_frac > 0.0:
        # (Since we're keeping only largest, this is mostly defensive)
        if best_area < (best_area * float(min_area_frac)):
            return mask_u8
    return out


def clean_blob_mask(blob_u8: np.ndarray, params) -> np.ndarray:
    """Post-process the filled blob so it better matches the iris region.

    This is intentionally conservative: it mainly removes detached junk (glasses rims),
    fills specular holes, and smooths the boundary.
    """
    if blob_u8 is None:
        return None
    blob = blob_u8.copy()

    keep_largest = bool(getattr(params, "blob_clean_keep_largest", True))
    fill_holes = bool(getattr(params, "blob_clean_fill_holes", True))
    open_k = _odd(getattr(params, "blob_clean_open_k", 3))
    close_k = _odd(getattr(params, "blob_clean_close_k", 9))
    erode_k = _odd(getattr(params, "blob_clean_erode_k", 0))
    dilate_k = _odd(getattr(params, "blob_clean_dilate_k", 0))
    min_area_frac = float(getattr(params, "blob_clean_min_area_frac", 0.0))

    if keep_largest:
        blob = keep_largest_cc(blob, min_area_frac=min_area_frac)

    if fill_holes:
        blob = fill_holes_u8(blob)

    if open_k >= 3:
        blob = cv2.morphologyEx(blob, cv2.MORPH_OPEN, _get_ellipse_se(open_k), iterations=1)

    if close_k >= 3:
        blob = cv2.morphologyEx(blob, cv2.MORPH_CLOSE, _get_ellipse_se(close_k), iterations=1)

    if erode_k >= 3:
        blob = cv2.erode(blob, _get_ellipse_se(erode_k), iterations=1)

    if dilate_k >= 3:
        blob = cv2.dilate(blob, _get_ellipse_se(dilate_k), iterations=1)

    if keep_largest:
        blob = keep_largest_cc(blob, min_area_frac=0.0)

    return blob



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



# ---------------------------------------------------------------------------
# Blob-based pupil detection (dark circular region) — robust for glasses
# ---------------------------------------------------------------------------

def _ellipse_aspect(e) -> float:
    (_, _), (MA, ma), _ = e
    return float(max(MA, ma) / (min(MA, ma) + 1e-6))

def _contour_circularity(cnt) -> float:
    area = float(cv2.contourArea(cnt))
    if area <= 1.0:
        return 0.0
    peri = float(cv2.arcLength(cnt, True))
    return float((4.0 * np.pi * area) / (peri * peri + 1e-6))

def _contour_solidity(cnt) -> float:
    area = float(cv2.contourArea(cnt))
    if area <= 1.0:
        return 0.0
    hull = cv2.convexHull(cnt)
    hull_area = float(cv2.contourArea(hull))
    if hull_area <= 1.0:
        return 0.0
    return float(area / (hull_area + 1e-6))

def _contour_extent(cnt) -> float:
    x, y, w, h = cv2.boundingRect(cnt)
    area = float(cv2.contourArea(cnt))
    rect = float(w * h)
    if rect <= 1.0:
        return 0.0
    return float(area / (rect + 1e-6))

def _contour_thinness(cnt) -> float:
    area = float(cv2.contourArea(cnt))
    if area <= 1.0:
        return 1e9
    peri = float(cv2.arcLength(cnt, True))
    return float((peri * peri) / (area + 1e-6))

def _contour_centroid(cnt):
    M = cv2.moments(cnt)
    if M["m00"] <= 1e-6:
        return None
    return (float(M["m10"] / M["m00"]), float(M["m01"] / M["m00"]))

def _mean_intensity_in_contour(gray_f: np.ndarray, cnt) -> float:
    """Mean intensity inside contour. Uses bounding-rect crop for speed."""
    x, y, w, h = cv2.boundingRect(cnt)
    if w <= 0 or h <= 0:
        return 255.0
    # Work on a small ROI instead of full image
    roi = gray_f[y:y+h, x:x+w]
    mask_roi = np.zeros((h, w), dtype=np.uint8)
    shifted = cnt - np.array([x, y])
    cv2.drawContours(mask_roi, [shifted], -1, 255, -1)
    return float(cv2.mean(roi, mask=mask_roi)[0])

def detect_pupil_blob(frame_bgr: np.ndarray, params: TuningParams, iris_roi_mask_u8: np.ndarray | None = None):
    """
    Detect pupil as the darkest *circular* region.

    Returns:
        (ellipse, confidence, blob_mask_u8, raw_mask_u8, debug_info)

    Notes:
    - raw_mask_u8 is the cleaned threshold mask (good for a "threshold" debug view)
    - blob_mask_u8 is the final filled blob used to drive overlays/ellipse fit
    """
    if frame_bgr is None or frame_bgr.size == 0:
        return None, 0.0, None, None, None

    H, W = frame_bgr.shape[:2]

    # 1) grayscale + blur (use existing blur_kernel_size for speed)
    gray = to_gray(frame_bgr).astype(np.float32)
    gray_f = safe_gauss(gray, int(getattr(params, "blob_blur_kernel_size", 5)))

    # 2) dark percentile threshold (robust to exposure changes)
    pct = float(getattr(params, "blob_dark_percentile", 6.0))
    pct = float(np.clip(pct, 0.5, 50.0))
    thr = float(np.percentile(gray_f, pct))
    mask_dark = (gray_f <= thr).astype(np.uint8) * 255

    # 3) optional low-saturation filter (helps ignore dark skin shadows / glasses rims)
    if bool(getattr(params, "blob_use_sat_filter", True)):
        sat_max = int(getattr(params, "blob_sat_max", 140))
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        S = hsv[:, :, 1]
        mask_lowsat = (S <= sat_max).astype(np.uint8) * 255
        raw = cv2.bitwise_and(mask_dark, mask_lowsat)
    else:
        raw = mask_dark
    # Optional: restrict blob search to an iris ROI mask (from sclera segmentation).
    # This helps avoid selecting dark glasses rims / eyelashes outside the iris region.
    if iris_roi_mask_u8 is not None and bool(getattr(params, 'blob_use_iris_roi', True)):
        roi = (iris_roi_mask_u8 > 0).astype(np.uint8) * 255
        dk = int(getattr(params, 'blob_iris_roi_dilate_k', 21))
        ek = int(getattr(params, 'blob_iris_roi_erode_k', 0))
        if dk >= 3:
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (_odd(dk, minimum=3), _odd(dk, minimum=3)))
            roi = cv2.dilate(roi, k, iterations=1)
        if ek >= 3:
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (_odd(ek, minimum=3), _odd(ek, minimum=3)))
            roi = cv2.erode(roi, k, iterations=1)
        raw = cv2.bitwise_and(raw, roi)

    # Choose a center reference for scoring (prefer blobs near iris ROI center if provided).
    center_x, center_y = (W / 2.0), (H / 2.0)
    if iris_roi_mask_u8 is not None and bool(getattr(params, 'blob_use_iris_roi', True)):
        mm = cv2.moments((iris_roi_mask_u8 > 0).astype(np.uint8))
        if mm.get('m00', 0.0) > 1e-6:
            center_x = float(mm['m10'] / mm['m00'])
            center_y = float(mm['m01'] / mm['m00'])

    # 4) morphology cleanup (using cached structuring elements)
    open_k = int(getattr(params, "blob_open_ksize", 5))
    close_k = int(getattr(params, "blob_close_ksize", 21))
    if open_k >= 3:
        raw = cv2.morphologyEx(raw, cv2.MORPH_OPEN, _get_ellipse_se(_odd(open_k, 3)), iterations=1)
    if close_k >= 3:
        raw = cv2.morphologyEx(raw, cv2.MORPH_CLOSE, _get_ellipse_se(_odd(close_k, 3)), iterations=2)

    # 5) score candidate contours
    cnts = find_external_contours(raw)
    if not cnts:
        return None, 0.0, None, raw, {"thr": thr, "pct": pct, "cand": 0}

    min_area = int(getattr(params, "blob_min_area", 300))
    keep_k = int(getattr(params, "blob_keep_top_k", 6))
    min_circ = float(getattr(params, "blob_min_circularity", 0.40))
    max_aspect = float(getattr(params, "blob_max_aspect", 2.0))

    # weights
    w_center = float(getattr(params, "blob_center_weight", 1.2))
    w_circ = float(getattr(params, "blob_circularity_weight", 14.0))
    w_sol = float(getattr(params, "blob_solidity_weight", 6.0))
    w_ext = float(getattr(params, "blob_extent_weight", 2.5))
    w_thin = float(getattr(params, "blob_thinness_weight", 5.0))
    w_dark = float(getattr(params, "blob_darkness_weight", 1.8))
    w_area = float(getattr(params, "blob_area_weight", 0.0005))

    def score(cnt) -> tuple:
        area = float(cv2.contourArea(cnt))
        if area < min_area:
            return -1e18, None

        circ = _contour_circularity(cnt)
        if circ < min_circ:
            return -1e18, None

        c = _contour_centroid(cnt)
        if c is None:
            return -1e18, None
        cx, cy = c
        dist = float(np.hypot(cx - center_x, cy - center_y) / (max(W, H) + 1e-6))

        # ellipse stretch guard
        e = None
        if len(cnt) >= 5:
            try:
                e = cv2.fitEllipse(cnt)
            except Exception:
                e = None
        if e is not None and _ellipse_aspect(e) > max_aspect:
            return -1e18, None

        mean_int = _mean_intensity_in_contour(gray_f, cnt)
        dark_term = (255.0 - mean_int) / 255.0  # 0..1

        sol = _contour_solidity(cnt)
        ext = _contour_extent(cnt)
        thin = _contour_thinness(cnt)

        # strong nonlinear circle preference
        circ_boost = circ * circ
        thin_penalty = float(np.clip((thin - 120.0) / 200.0, 0.0, 4.0))

        s = (
            (w_circ * circ_boost)
            + (w_sol * sol)
            + (w_ext * ext)
            + (w_dark * dark_term)
            + (w_area * area)
            - (w_center * dist)
            - (w_thin * thin_penalty)
        )
        return float(s), {"circ": float(circ), "sol": float(sol), "ext": float(ext), "thin": float(thin), "mean_gray": float(mean_int), "dist": float(dist), "area": float(area), "dark": float(dark_term), "ellipse": e}

    scored = []
    for c in cnts:
        s, meta = score(c)
        scored.append((s, c, meta))
    scored.sort(key=lambda t: t[0], reverse=True)
    scored = scored[:max(1, keep_k)]
    scored = [t for t in scored if t[0] > -1e17]
    if not scored:
        return None, 0.0, None, raw, {"thr": thr, "pct": pct, "cand": len(cnts)}

    best_s, best_cnt, meta = scored[0]
    e = meta.get("ellipse") if meta else None

    # 6) build filled blob mask (contour + hull + ellipse fill)
    blob = np.zeros_like(raw)
    cv2.drawContours(blob, [best_cnt], -1, 255, -1)

    hull = cv2.convexHull(best_cnt)
    cv2.drawContours(blob, [hull], -1, 255, -1)

    if e is not None:
        scale = float(getattr(params, "blob_ellipse_scale", 1.05))
        (cx, cy), (MA, ma), ang = e
        e2 = ((cx, cy), (MA * scale, ma * scale), ang)
        cv2.ellipse(blob, e2, 255, -1)
        e = e2

    # Close gaps then fill any internal holes (specular reflections, etc.)
    blob = cv2.morphologyEx(blob, cv2.MORPH_CLOSE, _get_ellipse_se(15), iterations=2)
    blob = fill_holes_u8(blob)

    # Flood fill from blob centroid to capture the full connected region
    if e is not None:
        seed = (int(np.clip(e[0][0], 1, W - 2)), int(np.clip(e[0][1], 1, H - 2)))
        if blob[seed[1], seed[0]] == 255:
            ff_mask = np.zeros((H + 2, W + 2), np.uint8)
            cv2.floodFill(blob, ff_mask, seed, 255)

    # 7) final ellipse: prefer fitted ellipse; fallback to minEnclosingCircle
    ellipse = None
    if e is not None:
        ellipse = e
    else:
        (x, y), r = cv2.minEnclosingCircle(best_cnt)
        ellipse = ((float(x), float(y)), (2.0 * float(r), 2.0 * float(r)), 0.0)

    # Map score -> confidence (0..1) based on the most important terms
    circ = float(meta.get("circ", 0.0)) if meta else 0.0
    sol = float(meta.get("sol", 0.0)) if meta else 0.0
    dark = float(meta.get("dark", 0.0)) if meta else 0.0
    conf = float(np.clip(0.10 + 0.55 * circ + 0.20 * sol + 0.15 * dark, 0.0, 1.0))

    info = {
        "thr": float(thr),
        "pct": float(pct),
        "cand": int(len(cnts)),
        "score": float(best_s),
        "circ": float(circ),
        "sol": float(sol),
        "ext": float(meta.get("ext", 0.0)) if meta else 0.0,
        "mean_gray": float(meta.get("mean_gray", 255.0)) if meta else 255.0,
        "confidence": float(conf),
    }

    return ellipse, conf, blob, raw, info
