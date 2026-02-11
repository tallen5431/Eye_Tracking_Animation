from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
import os

@dataclass
class TuningParams:
    roi_size: float = 0.6

    # Pupil detection params
    threshold_value: int = 60
    min_area: int = 500
    max_area: int = 6000
    min_circularity: float = 0.40

    blur_kernel_size: int = 5

    morph_close_iterations: int = 1
    morph_open_iterations: int = 1
    morph_kernel_size: int = 5
    
    # Step-by-step image processing controls
    contrast_alpha: float = 2.5  # Contrast multiplier (0.5-3.0)
    brightness_beta: int = 70     # Brightness offset (-100 to +100)
    gamma_value: float = 1.0     # Gamma correction (0.5-2.0)
    sharpen_amount: float = 0.0  # Sharpening strength (0.0-2.0)
    
    # CLAHE parameters (now adjustable)
    clahe_clip_limit: float = 2.0    # CLAHE clip limit (1.0-8.0)
    clahe_grid_size: int = 8         # CLAHE tile grid size (4-16)
    
    # Glare/reflection handling (CRITICAL for glasses)
    glare_threshold: int = 220       # Threshold for glare detection (200-250)
    glare_inpaint_radius: int = 5    # Inpainting radius for glare removal (3-10)
    
    # Simplified iris detection params
    iris_blur: int = 7  # Gaussian blur before sclera detection
    iris_sclera_threshold: int = 180  # Brightness threshold for sclera (white)
    iris_expand_ratio: float = 2.5  # Iris radius = pupil_radius * this
    iris_smooth_alpha: float = 0.75  # Temporal smoothing (0-1)
    iris_glint_threshold: int = 240  # Specular highlights above this are suppressed before sclera threshold
    iris_mask_close_k: int = 17  # Close kernel for iris mask (bridges glare gaps; odd, 5-25)

    # Blob-based pupil detection (dark circular region) — robust for glasses
    blob_dark_percentile: float = 8.0     # 3–12 typical; higher = include more pixels
    blob_use_sat_filter: bool = True     # require low saturation to avoid skin shadows
    blob_sat_max: int = 140              # 80–180; lower = stricter gray/black
    blob_blur_kernel_size: int = 5      # gaussian blur before percentile threshold (odd 0/3/5/7)
    blob_use_iris_roi: bool = True      # constrain blob search to iris ROI (from sclera segmentation)
    blob_iris_roi_dilate_k: int = 21    # dilate iris ROI to include iris boundary (odd 0/11/21/31)
    blob_iris_roi_erode_k: int = 0      # optional shrink ROI if too wide (odd 0/3/5)
    blob_cyan_roi_scale: float = 1.25  # scale ellipse ROI (1.05–1.60) when guiding blob detection
    blob_open_ksize: int = 5             # remove specks (odd 3–9)
    blob_close_ksize: int = 25           # connect/fill (odd 11–35)
    blob_min_area: int = 300             # reject tiny blobs
    blob_keep_top_k: int = 6             # score top candidates
    blob_min_circularity: float = 0.35   # raise to avoid rims/edges
    blob_max_aspect: float = 2.0         # reject very stretched ellipses
    blob_ellipse_scale: float = 1.05

    # Blob cleanup (post-process the filled blob)
    blob_clean_keep_largest: bool = True
    blob_clean_fill_holes: bool = True
    blob_clean_open_k: int = 3
    blob_clean_close_k: int = 9
    blob_clean_erode_k: int = 0
    blob_clean_dilate_k: int = 0
    blob_clean_min_area_frac: float = 0.0     # expand fitted ellipse slightly
    blob_center_weight: float = 1.2
    blob_circularity_weight: float = 14.0
    blob_solidity_weight: float = 6.0
    blob_extent_weight: float = 2.5
    blob_thinness_weight: float = 5.0
    blob_darkness_weight: float = 1.8
    blob_area_weight: float = 0.0005

@dataclass
class TuningToggles:
    use_histogram_eq: bool = False
    use_glint_removal: bool = False
    use_auto_threshold: bool = False
    use_adaptive_threshold: bool = False
    use_bilateral_filter: bool = False
    use_glasses_mode: bool = False  # Enhanced glare removal for glasses

@dataclass
class ViewFlags:
    show_original: bool = True
    show_grayscale: bool = True
    show_preprocessed: bool = True
    show_threshold: bool = True
    show_morphology: bool = True
    show_contours: bool = True
    show_iris_mask: bool = True  # Simplified: just show the mask

@dataclass
class RuntimeConfig:
    camera_id: int = 0
    share_file: Path = Path(os.environ.get("TRACK_SHARE_FILE", "logs/pupil_share.json"))
    share_hz: float = 30.0  # max write rate to share file
    camera_rotation: int = 270  # Rotate camera feed (0, 90, 180, 270)

def to_dict(params: TuningParams, toggles: TuningToggles, runtime: RuntimeConfig) -> dict:
    d = asdict(params)
    d.update(asdict(toggles))
    d.update({
        "camera_id": runtime.camera_id,
        "share_file": str(runtime.share_file),
        "share_hz": runtime.share_hz,
    })
    return d
