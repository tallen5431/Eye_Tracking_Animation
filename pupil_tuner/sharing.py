# pupil_tuner/sharing.py
from __future__ import annotations
import json, os, time
from pathlib import Path

class ShareWriter:
    def __init__(self, share_file: Path, hz: float = 30.0):
        self.share_file = Path(share_file)
        self.share_file.parent.mkdir(parents=True, exist_ok=True)
        self.min_dt = 1.0 / float(hz) if hz and hz > 0 else 0.0
        self._last_write = 0.0

    def maybe_write(self, ellipse, confidence: float, frame_shape_hw: tuple[int,int]):
        now = time.time()
        if self.min_dt and (now - self._last_write) < self.min_dt:
            return
        self._last_write = now

        h, w = frame_shape_hw
        payload = {
            "ts": now,
            "frame_w": int(w),
            "frame_h": int(h),
            "confidence": float(confidence),
            "ellipse": None,
        }
        if ellipse is not None:
            (cx, cy), (ew, eh), angle = ellipse
            payload["ellipse"] = {
                "cx": float(cx), "cy": float(cy),
                "ew": float(ew), "eh": float(eh),
                "angle": float(angle),
            }

        # Write a temp file (unique per write reduces collisions)
        tmp = self.share_file.with_suffix(self.share_file.suffix + f".{os.getpid()}.tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f)

        # Try atomic replace with a short retry loop (handles Windows locks)
        for attempt in range(10):
            try:
                os.replace(tmp, self.share_file)
                return
            except PermissionError:
                time.sleep(0.01 * (attempt + 1))  # small backoff

        # Fallback: write directly (non-atomic) so we don't crash the tuner
        try:
            with open(self.share_file, "w", encoding="utf-8") as f:
                json.dump(payload, f)
        finally:
            try:
                if tmp.exists():
                    tmp.unlink()
            except Exception:
                pass

def save_configuration(path: Path, camera_id: int, params, toggles):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")

    config = {
        "camera_id": int(camera_id),
        "roi_size": float(params.roi_size),
        "threshold_value": int(params.threshold_value),
        "min_area": int(params.min_area),
        "max_area": int(params.max_area),
        "min_circularity": float(params.min_circularity),
        "blur_kernel_size": int(params.blur_kernel_size),
        "morph_close_iterations": int(params.morph_close_iterations),
        "morph_open_iterations": int(params.morph_open_iterations),
        "morph_kernel_size": int(params.morph_kernel_size),
        "use_histogram_eq": bool(toggles.use_histogram_eq),
        "use_glint_removal": bool(toggles.use_glint_removal),
        "use_auto_threshold": bool(toggles.use_auto_threshold),
        "use_adaptive_threshold": bool(toggles.use_adaptive_threshold),
        "use_bilateral_filter": bool(toggles.use_bilateral_filter),
    }

    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    os.replace(tmp, path)
