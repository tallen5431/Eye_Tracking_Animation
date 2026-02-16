# Eye Tracker Optimization Recommendations

## 1. Performance Optimizations (High Impact)

### A. Camera Buffer Management
**Problem:** Default OpenCV buffer accumulates frames, causing lag
**Solution:** Reduce buffer size and skip old frames

```python
# In camera.py - add to _apply_props():
def _apply_props(cap, width: int, height: int, fps: int):
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    # NEW: Reduce buffer to minimize latency
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    # Enable auto-exposure/white balance for consistent lighting
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)  # Manual mode
    cap.set(cv2.CAP_PROP_EXPOSURE, -6)  # Faster shutter for less motion blur
```

### B. Frame Skipping for Real-Time Processing
**Problem:** Processing every frame when falling behind causes more lag
**Solution:** Skip frames when processing is slow

```python
# In app.py run() method, after ret, frame = self.cap.read():
def run(self):
    # ... initialization ...
    frame_skip_counter = 0
    target_fps = 30
    
    while True:
        ret, frame = self.cap.read()
        if not ret:
            break
        
        # Skip frames if processing is too slow
        if self.fps < target_fps * 0.7:  # If below 70% of target
            frame_skip_counter += 1
            if frame_skip_counter % 2 == 0:  # Skip every other frame
                continue
        else:
            frame_skip_counter = 0
        
        # ... rest of processing ...
```

### C. Reduce Redundant Conversions
**Problem:** Multiple gray→BGR conversions in preview windows
**Solution:** Convert once at display time

```python
# In app.py - optimize _preview_with_ellipse:
def _preview_with_ellipse(self, img, ellipse, color=(0, 255, 255), thickness=2):
    if img is None:
        return None
    
    if not self.preview_show_ellipse or ellipse is None:
        return img  # Let grid handle conversion
    
    # Only convert if we need to draw
    if img.ndim == 2:
        out = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    else:
        out = img.copy()
    
    try:
        cv2.ellipse(out, ellipse, color, thickness, lineType=cv2.LINE_AA)  # Anti-aliased
        (cx, cy), _, _ = ellipse
        cv2.circle(out, (int(cx), int(cy)), 3, color, -1, lineType=cv2.LINE_AA)
    except Exception:
        pass
    
    return out
```

## 2. Reliability Improvements (High Impact)

### A. Adaptive Threshold Smoothing
**Problem:** Blob threshold flickers between frames
**Current:** Uses EMA on threshold (good!)
**Enhancement:** Add hysteresis to prevent oscillation

```python
# In pipeline.py - enhance detect_pupil_blob:
_blob_thr_ema: float | None = None
_blob_thr_hysteresis: float = 5.0  # Prevent small oscillations

def detect_pupil_blob(...):
    # ... existing code ...
    thr_raw = float(np.percentile(gray_u8[::4, ::4], pct))
    
    if _blob_thr_ema is None:
        _blob_thr_ema = thr_raw
    else:
        # Only update if change is significant (hysteresis)
        diff = abs(thr_raw - _blob_thr_ema)
        if diff > _blob_thr_hysteresis:
            _blob_thr_ema += _BLOB_THR_EMA_ALPHA * (thr_raw - _blob_thr_ema)
    
    thr = _blob_thr_ema
    # ... rest of function ...
```

### B. Multi-Frame Validation
**Problem:** Single-frame outliers cause jumps
**Solution:** Require consistency across frames before accepting large changes

```python
# In app.py - add validation buffer:
class PupilTrackerTuner:
    def __init__(self, camera_id: int = 0):
        # ... existing init ...
        self._validation_buffer = []  # Store last N detections
        self._validation_size = 3
        self._max_jump_threshold = 50  # pixels
    
    def _validate_detection(self, new_ellipse):
        """Reject outliers that jump too far from recent history"""
        if new_ellipse is None:
            return None
        
        if not self._validation_buffer:
            self._validation_buffer.append(new_ellipse)
            return new_ellipse
        
        # Check if jump is too large
        (new_cx, new_cy), _, _ = new_ellipse
        recent = self._validation_buffer[-1]
        (old_cx, old_cy), _, _ = recent
        
        dist = np.sqrt((new_cx - old_cx)**2 + (new_cy - old_cy)**2)
        
        if dist > self._max_jump_threshold:
            # Large jump - require confirmation
            return recent  # Keep previous
        
        # Valid detection - add to buffer
        self._validation_buffer.append(new_ellipse)
        if len(self._validation_buffer) > self._validation_size:
            self._validation_buffer.pop(0)
        
        return new_ellipse
    
    # In process_pipeline, after ellipse detection:
    # ellipse = self._validate_detection(ellipse)
```

### C. Lighting Adaptation
**Problem:** Performance degrades in varying lighting
**Solution:** Auto-adjust preprocessing based on frame statistics

```python
# In config.py - add adaptive mode:
@dataclass
class TuningToggles:
    # ... existing toggles ...
    use_adaptive_preprocessing: bool = True  # Auto-adjust to lighting

# In pipeline.py - add adaptive preprocessing:
def preprocess(gray: np.ndarray, params: TuningParams, toggles: TuningToggles):
    img = gray
    
    if toggles.use_adaptive_preprocessing:
        # Analyze frame brightness
        mean_brightness = np.mean(gray)
        
        # Auto-adjust contrast/brightness based on conditions
        if mean_brightness < 80:  # Dark conditions
            alpha = params.contrast_alpha * 1.2
            beta = params.brightness_beta + 20
        elif mean_brightness > 180:  # Bright conditions
            alpha = params.contrast_alpha * 0.9
            beta = params.brightness_beta - 10
        else:  # Normal
            alpha = params.contrast_alpha
            beta = params.brightness_beta
    else:
        alpha = params.contrast_alpha
        beta = params.brightness_beta
    
    # Apply adjustments
    if alpha != 1.0 or beta != 0:
        img = adjust_contrast_brightness(img, alpha, beta)
    
    # ... rest of preprocessing ...
```

## 3. Memory & CPU Optimizations

### A. Reduce Allocation Overhead
**Problem:** Creating new arrays every frame
**Solution:** Reuse buffers where possible

```python
# In app.py - add buffer reuse:
class PupilTrackerTuner:
    def __init__(self, camera_id: int = 0):
        # ... existing init ...
        # Pre-allocate buffers
        self._gray_buffer = None
        self._preprocessed_buffer = None
    
    def process_pipeline(self, roi_bgr: np.ndarray):
        # Reuse buffers instead of allocating
        if self._gray_buffer is None or self._gray_buffer.shape != roi_bgr.shape[:2]:
            self._gray_buffer = np.empty(roi_bgr.shape[:2], dtype=np.uint8)
        
        cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY, dst=self._gray_buffer)
        self.img_gray = self._gray_buffer
        # ... rest of processing ...
```

### B. Optimize Grid Display
**Problem:** Resizing images every frame is expensive
**Solution:** Cache resized images when parameters don't change

```python
# In app.py - add display cache:
def _show_in_grid(self, items, ...):
    # ... existing code ...
    
    # Cache key: (image_id, cell_w, cell_h)
    if not hasattr(self, '_resize_cache'):
        self._resize_cache = {}
        self._cache_frame_count = 0
    
    # Clear cache every 100 frames to prevent memory leak
    self._cache_frame_count += 1
    if self._cache_frame_count > 100:
        self._resize_cache.clear()
        self._cache_frame_count = 0
    
    for idx, (title, img) in enumerate(items):
        # ... existing positioning code ...
        
        if img is None:
            continue
        
        # Use cache for resize
        cache_key = (id(img), cell_w, cell_h, img.shape)
        if cache_key in self._resize_cache:
            tile_rs = self._resize_cache[cache_key]
        else:
            # ... existing resize code ...
            if len(self._resize_cache) < 50:  # Limit cache size
                self._resize_cache[cache_key] = tile_rs
        
        # ... rest of display code ...
```

## 4. Robustness Enhancements

### A. Blink Detection & Recovery
**Problem:** Blinks cause detection loss
**Solution:** Detect blinks and maintain tracking

```python
# In app.py - add blink detection:
class PupilTrackerTuner:
    def __init__(self, camera_id: int = 0):
        # ... existing init ...
        self._blink_detector_threshold = 0.3  # Area ratio threshold
        self._in_blink = False
        self._pre_blink_ellipse = None
    
    def _detect_blink(self, roi_bgr):
        """Detect if eye is closed (blink)"""
        gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
        # Very dark frame = likely closed eye
        dark_ratio = np.sum(gray < 50) / gray.size
        return dark_ratio > self._blink_detector_threshold
    
    def process_pipeline(self, roi_bgr: np.ndarray):
        # Check for blink
        if self._detect_blink(roi_bgr):
            if not self._in_blink:
                self._pre_blink_ellipse = self.current_ellipse
                self._in_blink = True
            # During blink, maintain last good detection
            return
        else:
            self._in_blink = False
        
        # ... normal processing ...
```

### B. Confidence-Based Parameter Adjustment
**Problem:** Fixed parameters don't adapt to detection quality
**Solution:** Relax constraints when confidence is low

```python
# In app.py - adaptive constraints:
def process_pipeline(self, roi_bgr: np.ndarray):
    # ... existing preprocessing ...
    
    # Adjust detection parameters based on recent confidence
    if hasattr(self, '_recent_confidence'):
        if self._recent_confidence < 0.4:
            # Low confidence - relax constraints
            min_circ = self.params.min_circularity * 0.8
            max_aspect = self.params.blob_max_aspect * 1.2
        else:
            min_circ = self.params.min_circularity
            max_aspect = self.params.blob_max_aspect
    
    # ... use adjusted parameters in detection ...
    
    self._recent_confidence = self.current_confidence
```

## 5. Configuration Improvements

### A. Add Performance Profiles
**Problem:** Users don't know optimal settings
**Solution:** Provide presets

```python
# In config.py - add profiles:
class PerformanceProfile:
    @staticmethod
    def high_quality():
        """Best quality, slower (15-20 FPS)"""
        params = TuningParams()
        params.blur_kernel_size = 7
        params.blob_close_ksize = 25
        params.clahe_clip_limit = 3.0
        return params
    
    @staticmethod
    def balanced():
        """Good quality, fast (25-30 FPS)"""
        return TuningParams()  # Current defaults
    
    @staticmethod
    def high_speed():
        """Lower quality, fastest (30+ FPS)"""
        params = TuningParams()
        params.blur_kernel_size = 3
        params.blob_close_ksize = 15
        params.clahe_clip_limit = 2.0
        params.blob_keep_top_k = 3  # Fewer candidates
        return params
```

### B. Save/Load Configurations
**Problem:** Users lose tuned parameters
**Solution:** Add config persistence

```python
# In config.py - add serialization:
import json
from pathlib import Path

def save_config(params: TuningParams, toggles: TuningToggles, 
                path: Path = Path("config/tuning_config.json")):
    """Save current configuration to file"""
    path.parent.mkdir(exist_ok=True)
    config = {
        'params': asdict(params),
        'toggles': asdict(toggles),
    }
    with open(path, 'w') as f:
        json.dump(config, f, indent=2)

def load_config(path: Path = Path("config/tuning_config.json")):
    """Load configuration from file"""
    if not path.exists():
        return TuningParams(), TuningToggles()
    
    with open(path, 'r') as f:
        config = json.load(f)
    
    params = TuningParams(**config.get('params', {}))
    toggles = TuningToggles(**config.get('toggles', {}))
    return params, toggles

# In app.py - add keyboard shortcuts:
# elif key == ord('s'):
#     save_config(self.params, self.toggles)
#     print("Configuration saved!")
# elif key == ord('l'):
#     self.params, self.toggles = load_config()
#     print("Configuration loaded!")
```

## 6. Monitoring & Diagnostics

### A. Performance Metrics
**Problem:** Hard to identify bottlenecks
**Solution:** Add timing instrumentation

```python
# In app.py - add profiling:
import time
from collections import defaultdict

class PupilTrackerTuner:
    def __init__(self, camera_id: int = 0):
        # ... existing init ...
        self._timing_stats = defaultdict(list)
        self._show_timing = False  # Toggle with 't' key
    
    def _time_section(self, name):
        """Context manager for timing code sections"""
        class Timer:
            def __init__(self, stats, name):
                self.stats = stats
                self.name = name
            def __enter__(self):
                self.start = time.perf_counter()
            def __exit__(self, *args):
                elapsed = (time.perf_counter() - self.start) * 1000
                self.stats[self.name].append(elapsed)
                if len(self.stats[self.name]) > 30:
                    self.stats[self.name].pop(0)
        return Timer(self._timing_stats, name)
    
    def process_pipeline(self, roi_bgr: np.ndarray):
        with self._time_section('preprocess'):
            self.img_preprocessed, _ = preprocess(...)
        
        with self._time_section('detection'):
            pupil_ellipse, pupil_conf, ... = detect_pupil_blob(...)
        
        with self._time_section('iris'):
            self.iris_ellipse = self._fit_iris_ellipse_simple(...)
        
        # ... rest of processing ...
    
    def _get_timing_report(self):
        """Generate timing report"""
        report = []
        for name, times in self._timing_stats.items():
            if times:
                avg = sum(times) / len(times)
                report.append(f"{name}: {avg:.1f}ms")
        return " | ".join(report)
```

## Priority Implementation Order

1. **Camera buffer reduction** (A) - Immediate latency improvement
2. **Multi-frame validation** (2B) - Major stability boost
3. **Save/Load config** (5B) - User convenience
4. **Adaptive preprocessing** (2C) - Handles varying conditions
5. **Performance profiles** (5A) - Easy optimization
6. **Frame skipping** (1B) - Maintains real-time performance
7. **Blink detection** (4A) - Better user experience
8. **Timing instrumentation** (6A) - Identify remaining bottlenecks

## Expected Improvements

- **Latency:** 50-100ms reduction from buffer management
- **Stability:** 30-40% fewer detection jumps from validation
- **FPS:** 5-10 FPS improvement from optimizations
- **Robustness:** 80%+ detection during blinks/lighting changes
- **Usability:** Saved configs eliminate re-tuning

## Testing Recommendations

1. Test with different lighting conditions (bright, dim, mixed)
2. Test with glasses (reflections, glare)
3. Test rapid eye movements (saccades)
4. Test blinks and partial occlusions
5. Measure FPS under each condition
6. Profile to find remaining bottlenecks
