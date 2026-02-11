"""
calibration.py

Startup calibration to compute a stable ellipse ROI for eye tracking.

Goal:
- capture a short burst of frames at startup
- find sclera-ish pixels (sample-based HSV gates)
- build a center-weighted composite heatmap
- build a centered eye-region mask (prevents downward drift)
- expand mask (prevents narrow ellipse)
- fit + inflate ellipse to form ROI usable for blob detection

This produces `ellipse_roi` in OpenCV fitEllipse format:
    ((cx, cy), (MA, ma), angle_degrees)

And saves/loads to JSON for reuse across runs.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np


Ellipse = Tuple[Tuple[float, float], Tuple[float, float], float]


# ============================================================
# Config
# ============================================================

@dataclass
class CalibrationConfig:
    # --- capture ---
    camera_id: int = 0
    warmup_frames: int = 10
    capture_frames: int = 40
    capture_stride: int = 1          # sample every Nth frame
    max_capture_seconds: float = 3.0 # safety cap
    resize_w: Optional[int] = None   # set e.g. 640 for speed; None keeps size
    flip: Optional[int] = None       # OpenCV flipCode: 0=vertical, 1=horizontal, -1=both

    # --- sample from click tool (H,S,V) ---
    sclera_hsv_sample: Tuple[int, int, int] = (111, 33, 100)

    # --- Per-image white mask (HSV gates) ---
    s_margin: int = 25
    v_margin_low: int = 12
    v_margin_high: int = 55
    v_step: int = 5

    # --- Per-image cleanup ---
    close_k: int = 7
    close_iters: int = 2

    # --- Per-image composite mode (within V scan) ---
    per_image_mode: str = "vote"     # "union" or "vote"
    per_image_min_frac: float = 0.22

    # --- Across-image composite baseline ---
    final_min_image_frac: float = 0.25
    final_close_k: int = 11
    final_close_iters: int = 2

    # --- Center preference (weighted composite) ---
    use_center_weight: bool = True
    center_sigma_frac: float = 0.28
    center_power: float = 2.0
    weighted_min_frac: Optional[float] = 0.23  # None -> uses final_min_image_frac

    # ============================================================
    # Centered eye region (tight drift + bigger eyeball)
    # ============================================================
    eye_min_frac: float = 0.20
    row_half_width_frac: float = 0.34
    row_min_support_frac: float = 0.12

    y_top_frac: float = 0.10
    y_bot_frac: float = 0.72

    row_peak_min_frac: float = 0.28
    global_peak_x_tol_frac: float = 0.22

    eye_dilate_k: int = 13
    eye_close_k: int = 21
    eye_close_iters: int = 2

    keep_largest_centered: bool = True
    center_dist_frac: float = 0.28

    # --- ellipse fit & inflate ---
    ellipse_expand_k: int = 19
    ellipse_expand_iters: int = 1
    ellipse_min_area: int = 1200
    ellipse_size_mult: float = 1.22

    # --- output ---
    save_path: str = "logs/calibration_roi.json"


# ============================================================
# Utility
# ============================================================

def _odd(k: int) -> int:
    k = int(k)
    return max(1, k | 1)

def _clip01(x: np.ndarray) -> np.ndarray:
    return np.clip(x, 0.0, 1.0)

def ellipse_to_bbox(ell: Ellipse, pad_frac: float = 0.15) -> Tuple[int, int, int, int]:
    (cx, cy), (MA, ma), _ang = ell
    rx = 0.5 * float(MA)
    ry = 0.5 * float(ma)
    r = max(rx, ry)

    x0 = float(cx) - r
    x1 = float(cx) + r
    y0 = float(cy) - r
    y1 = float(cy) + r

    bw = (x1 - x0)
    bh = (y1 - y0)
    x0 -= pad_frac * bw
    x1 += pad_frac * bw
    y0 -= pad_frac * bh
    y1 += pad_frac * bh

    return int(np.floor(x0)), int(np.floor(y0)), int(np.ceil(x1)), int(np.ceil(y1))

def clamp_bbox(x0: int, y0: int, x1: int, y1: int, w: int, h: int) -> Tuple[int, int, int, int]:
    x0 = max(0, min(w - 1, x0))
    y0 = max(0, min(h - 1, y0))
    x1 = max(x0 + 1, min(w, x1))
    y1 = max(y0 + 1, min(h, y1))
    return x0, y0, x1, y1


# ============================================================
# Core masks
# ============================================================

def make_center_weight_map(h: int, w: int, sigma_frac: float, power: float) -> np.ndarray:
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cx = (w - 1) * 0.5
    cy = (h - 1) * 0.5
    sigma = max(1.0, float(sigma_frac) * float(min(h, w)))
    d2 = (xx - cx) ** 2 + (yy - cy) ** 2
    g = np.exp(-0.5 * d2 / (sigma ** 2))
    g = (g - g.min()) / (g.max() - g.min() + 1e-8)
    if power != 1.0:
        g = np.power(g, float(power))
    return g.astype(np.float32)

def per_image_white_mask(img_bgr: np.ndarray, cfg: CalibrationConfig) -> np.ndarray:
    """
    Returns per-image binary mask (0/255) selecting white-ish sclera:
      S <= S_sample + s_margin
      V >= scanned vmin range around V_sample
    Then collapses scan masks using union or vote.
    """
    _, ss, vs = map(int, cfg.sclera_hsv_sample)

    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    s = hsv[..., 1]
    v = hsv[..., 2]

    s_max = int(np.clip(ss + int(cfg.s_margin), 0, 255))
    v_start = int(np.clip(vs - int(cfg.v_margin_low), 0, 255))
    v_end = int(np.clip(vs + int(cfg.v_margin_high), 0, 255))

    v_step = max(1, int(cfg.v_step))
    v_mins = list(range(v_start, v_end + 1, v_step)) or [v_start]

    hits = np.zeros(v.shape, dtype=np.uint16)
    union = np.zeros(v.shape, dtype=np.uint8)

    for vmin in v_mins:
        m = ((s <= s_max) & (v >= int(vmin))).astype(np.uint8) * 255

        if cfg.close_iters > 0:
            se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (_odd(cfg.close_k), _odd(cfg.close_k)))
            m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, se, iterations=int(cfg.close_iters))

        union = cv2.bitwise_or(union, m)
        hits += (m == 255).astype(np.uint16)

    if str(cfg.per_image_mode).lower() == "union":
        return union

    # vote composite inside scan
    num_tests = len(v_mins)
    min_hits = max(1, int(np.ceil(num_tests * float(cfg.per_image_min_frac))))
    vote = (hits >= min_hits).astype(np.uint8) * 255
    return vote

def compute_sclera_composites(images_bgr: List[np.ndarray], cfg: CalibrationConfig) -> Dict[str, Any]:
    """
    Build both count-based and center-weighted sclera presence maps.
    """
    assert len(images_bgr) > 0, "No images passed."

    h, w = images_bgr[0].shape[:2]
    N = len(images_bgr)

    sum_u16 = np.zeros((h, w), dtype=np.uint16)
    sum_w = np.zeros((h, w), dtype=np.float32)

    mean_img = np.mean(np.stack(images_bgr, axis=0).astype(np.float32), axis=0).astype(np.uint8)

    if cfg.use_center_weight:
        wmap = make_center_weight_map(h, w, cfg.center_sigma_frac, cfg.center_power)
    else:
        wmap = np.ones((h, w), dtype=np.float32)

    per_image_masks = []
    for img in images_bgr:
        m = per_image_white_mask(img, cfg)
        per_image_masks.append(m)
        bin01 = (m == 255).astype(np.uint16)
        sum_u16 += bin01
        sum_w += bin01.astype(np.float32) * wmap

    sum_norm_u8 = (sum_u16.astype(np.float32) * (255.0 / float(N))).astype(np.uint8)

    denom = (float(N) * wmap)
    sum_w_frac = np.zeros_like(sum_w, dtype=np.float32)
    valid = denom > 1e-6
    sum_w_frac[valid] = sum_w[valid] / denom[valid]
    sum_w_norm_u8 = np.clip(sum_w_frac * 255.0, 0, 255).astype(np.uint8)

    # count-based blob
    min_count = int(np.ceil(N * float(cfg.final_min_image_frac)))
    final_blob = (sum_u16 >= min_count).astype(np.uint8) * 255

    # weighted blob
    weighted_min_frac = cfg.weighted_min_frac
    if weighted_min_frac is None:
        weighted_min_frac = float(cfg.final_min_image_frac)
    final_blob_w = (sum_w_frac >= float(weighted_min_frac)).astype(np.uint8) * 255

    # cleanup
    if cfg.final_close_iters > 0:
        se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (_odd(cfg.final_close_k), _odd(cfg.final_close_k)))
        final_blob = cv2.morphologyEx(final_blob, cv2.MORPH_CLOSE, se, iterations=int(cfg.final_close_iters))
        final_blob_w = cv2.morphologyEx(final_blob_w, cv2.MORPH_CLOSE, se, iterations=int(cfg.final_close_iters))

    return {
        "per_image_masks": per_image_masks,
        "sum_u16": sum_u16,
        "sum_norm_u8": sum_norm_u8,
        "final_blob": final_blob,
        "min_count": min_count,
        "wmap": wmap,
        "sum_w": sum_w,
        "sum_w_frac": sum_w_frac,
        "sum_w_norm_u8": sum_w_norm_u8,
        "final_blob_w": final_blob_w,
        "mean_img": mean_img,
        "N": N,
        "final_min_image_frac": float(cfg.final_min_image_frac),
        "weighted_min_frac": float(weighted_min_frac),
        "use_center_weight": bool(cfg.use_center_weight),
        "center_sigma_frac": float(cfg.center_sigma_frac),
        "center_power": float(cfg.center_power),
    }

def compute_eye_region_from_sclera_centered(frac_map: np.ndarray, cfg: CalibrationConfig) -> np.ndarray:
    """
    Centered eye-region builder that prevents downward drift and stays wide enough for eyeball.
    """
    h, w = frac_map.shape[:2]

    y0 = int(np.clip(cfg.y_top_frac * h, 0, h - 1))
    y1 = int(np.clip(cfg.y_bot_frac * h, y0 + 1, h))

    on = (frac_map >= float(cfg.eye_min_frac)).astype(np.uint8)
    band_vals = frac_map[y0:y1]
    if band_vals.size == 0:
        return np.zeros((h, w), dtype=np.uint8)

    # stable global x peak
    _gy, gx = np.unravel_index(np.argmax(band_vals), band_vals.shape)
    gx = int(gx)

    x_tol = int(max(6, w * float(cfg.global_peak_x_tol_frac)))
    half_w = int(max(10, w * float(cfg.row_half_width_frac)))
    min_support = int(max(18, w * float(cfg.row_min_support_frac)))

    eye = np.zeros((h, w), dtype=np.uint8)

    for y in range(y0, y1):
        xs = np.flatnonzero(on[y])
        if xs.size < min_support:
            continue

        row_vals = frac_map[y]
        x_peak = int(xs[np.argmax(row_vals[xs])])

        if float(row_vals[x_peak]) < float(cfg.row_peak_min_frac):
            continue
        if abs(x_peak - gx) > x_tol:
            continue

        xL = max(0, x_peak - half_w)
        xR = min(w, x_peak + half_w + 1)
        eye[y, xL:xR] = 1

    eye_u8 = (eye * 255).astype(np.uint8)

    if cfg.eye_dilate_k > 0:
        se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (_odd(cfg.eye_dilate_k), _odd(cfg.eye_dilate_k)))
        eye_u8 = cv2.dilate(eye_u8, se, iterations=1)

    if cfg.eye_close_k > 0 and cfg.eye_close_iters > 0:
        se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (_odd(cfg.eye_close_k), _odd(cfg.eye_close_k)))
        eye_u8 = cv2.morphologyEx(eye_u8, cv2.MORPH_CLOSE, se, iterations=int(cfg.eye_close_iters))

    if cfg.keep_largest_centered:
        num, labels, stats, centroids = cv2.connectedComponentsWithStats((eye_u8 > 0).astype(np.uint8), connectivity=8)
        if num > 1:
            cx0 = (w - 1) * 0.5
            cy0 = (h - 1) * 0.5
            max_dist = float(cfg.center_dist_frac) * float(min(h, w))

            best_k = 0
            best_score = -1e18
            for k in range(1, num):
                area = float(stats[k, cv2.CC_STAT_AREA])
                cx, cy = centroids[k]
                dist = float(np.hypot(cx - cx0, cy - cy0))
                if dist > max_dist:
                    continue
                score = area - 2.0 * dist
                if score > best_score:
                    best_score = score
                    best_k = k

            if best_k > 0:
                eye_u8 = (labels == best_k).astype(np.uint8) * 255

    return eye_u8

def expand_mask_u8(mask_u8: np.ndarray, k: int, iters: int) -> np.ndarray:
    if iters <= 0 or k <= 0:
        return mask_u8
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (_odd(k), _odd(k)))
    return cv2.dilate(mask_u8, se, iterations=int(iters))

def fit_ellipse_from_mask(mask_u8: np.ndarray, min_area: int) -> Optional[Ellipse]:
    if mask_u8 is None:
        return None
    m = mask_u8
    if m.ndim == 3:
        m = cv2.cvtColor(m, cv2.COLOR_BGR2GRAY)

    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    best = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(best)
    if area < float(min_area) or len(best) < 5:
        return None

    ell = cv2.fitEllipse(best)
    # Normalize to python floats
    (cx, cy), (MA, ma), ang = ell
    return ((float(cx), float(cy)), (float(MA), float(ma)), float(ang))

def inflate_ellipse(ellipse: Optional[Ellipse], mult: float) -> Optional[Ellipse]:
    if ellipse is None:
        return None
    (cx, cy), (MA, ma), ang = ellipse
    return ((float(cx), float(cy)), (float(MA) * float(mult), float(ma) * float(mult)), float(ang))


# ============================================================
# Public API
# ============================================================

@dataclass
class CalibrationResult:
    ellipse_roi: Optional[Ellipse]
    ellipse_fit: Optional[Ellipse]
    image_w: int
    image_h: int
    timestamp: float
    config: Dict[str, Any]

    # Helpful extras
    bbox_pad_frac: float = 0.18  # recommended crop padding


def calibrate_from_frames(frames_bgr: List[np.ndarray], cfg: CalibrationConfig) -> CalibrationResult:
    """
    Core offline calibration given pre-captured frames.
    """
    if len(frames_bgr) == 0:
        raise RuntimeError("No frames provided to calibration.")

    h, w = frames_bgr[0].shape[:2]
    comp = compute_sclera_composites(frames_bgr, cfg)

    # Prefer weighted fraction map when enabled
    frac_map = comp["sum_w_frac"] if ("sum_w_frac" in comp) else (comp["sum_u16"].astype(np.float32) / float(max(1, comp["N"])))
    frac_map = _clip01(frac_map)

    eye_region = compute_eye_region_from_sclera_centered(frac_map, cfg)

    # Expand before fit so ellipse is not narrow
    eye_for_fit = expand_mask_u8(eye_region, cfg.ellipse_expand_k, cfg.ellipse_expand_iters)

    ellipse_fit = fit_ellipse_from_mask(eye_for_fit, min_area=int(cfg.ellipse_min_area))
    ellipse_roi = inflate_ellipse(ellipse_fit, mult=float(cfg.ellipse_size_mult))

    return CalibrationResult(
        ellipse_roi=ellipse_roi,
        ellipse_fit=ellipse_fit,
        image_w=int(w),
        image_h=int(h),
        timestamp=float(time.time()),
        config=asdict(cfg),
        bbox_pad_frac=0.18,
    )

def calibrate_camera(cfg: CalibrationConfig) -> CalibrationResult:
    """
    Online calibration: opens camera, captures burst of frames, calibrates ROI.
    """
    cap = cv2.VideoCapture(int(cfg.camera_id))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera_id={cfg.camera_id}")

    frames: List[np.ndarray] = []
    t0 = time.time()
    grabbed = 0
    kept = 0

    # warmup
    for _ in range(max(0, int(cfg.warmup_frames))):
        cap.read()

    while kept < int(cfg.capture_frames):
        if (time.time() - t0) > float(cfg.max_capture_seconds):
            break

        ok, frame = cap.read()
        if not ok or frame is None:
            continue

        grabbed += 1
        if cfg.capture_stride > 1 and (grabbed % int(cfg.capture_stride)) != 0:
            continue

        if cfg.flip is not None:
            frame = cv2.flip(frame, int(cfg.flip))

        if cfg.resize_w is not None and int(cfg.resize_w) > 0:
            h, w = frame.shape[:2]
            new_w = int(cfg.resize_w)
            new_h = int(round(h * (new_w / float(w))))
            frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

        frames.append(frame)
        kept += 1

    cap.release()

    if len(frames) < max(8, int(cfg.capture_frames) // 3):
        raise RuntimeError(f"Too few calibration frames captured: {len(frames)}")

    return calibrate_from_frames(frames, cfg)

def save_calibration(result: CalibrationResult, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    payload: Dict[str, Any] = {
        "timestamp": result.timestamp,
        "image_w": result.image_w,
        "image_h": result.image_h,
        "bbox_pad_frac": float(result.bbox_pad_frac),
        "ellipse_fit": result.ellipse_fit,
        "ellipse_roi": result.ellipse_roi,
        "config": result.config,
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

def load_calibration(path: str) -> Optional[CalibrationResult]:
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return None

    def _as_ellipse(obj: Any) -> Optional[Ellipse]:
        if obj is None:
            return None
        # expected: [[cx,cy],[MA,ma],ang] or ((cx,cy),(MA,ma),ang)
        try:
            (cx, cy), (MA, ma), ang = obj
            return ((float(cx), float(cy)), (float(MA), float(ma)), float(ang))
        except Exception:
            return None

    return CalibrationResult(
        ellipse_roi=_as_ellipse(data.get("ellipse_roi")),
        ellipse_fit=_as_ellipse(data.get("ellipse_fit")),
        image_w=int(data.get("image_w", 0)),
        image_h=int(data.get("image_h", 0)),
        timestamp=float(data.get("timestamp", 0.0)),
        config=dict(data.get("config", {})),
        bbox_pad_frac=float(data.get("bbox_pad_frac", 0.18)),
    )

def ensure_calibration(cfg: CalibrationConfig, force: bool = False) -> CalibrationResult:
    """
    Load calibration if present, otherwise run calibration and save it.
    """
    if (not force):
        loaded = load_calibration(cfg.save_path)
        if loaded is not None and loaded.ellipse_roi is not None:
            return loaded

    res = calibrate_camera(cfg)
    save_calibration(res, cfg.save_path)
    return res


# ============================================================
# Optional: simple CLI test
# ============================================================

if __name__ == "__main__":
    cfg = CalibrationConfig()
    res = ensure_calibration(cfg, force=True)
    print("Calibration complete.")
    print("Saved:", cfg.save_path)
    print("ellipse_roi:", res.ellipse_roi)
