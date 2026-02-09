# Preprocessing Step Viewer

## Overview
The tuning tool now shows **step-by-step preprocessing** stages.
Each image processing operation is applied sequentially and can be adjusted independently.

## Preprocessing Pipeline

```
Original Grayscale
       ↓
1. Contrast/Brightness Adjustment  (keys: 9/0, ,/.)
       ↓
2. Gamma Correction               (keys: / / ;)
       ↓
3. Sharpening (optional)          (keys: [ / ])
       ↓
4. Bilateral Filter (optional)    (key: f to toggle)
       ↓
5. CLAHE (optional)               (key: h to toggle, \\ / ' for strength)
       ↓
6. Glint Removal (optional)       (key: g to toggle)
       ↓
Final Preprocessed Image
```

## New Keyboard Controls

### Image Processing (NEW!)

| Keys | Parameter | Range | Default | Effect |
|------|-----------|-------|---------|--------|
| `9` / `0` | Contrast | 0.5 - 3.0 | 1.0 | Multiply pixel values |
| `,` / `.` | Brightness | -100 - +100 | 0 | Add to pixel values |
| `/` / `;` | Gamma | 0.5 - 2.0 | 1.0 | Non-linear brightness |
| `[` / `]` | Sharpen | 0.0 - 2.0 | 0.0 | Edge enhancement |
| `\\` / `'` | CLAHE Clip | 1.0 - 8.0 | 2.0 | Local contrast |

### Original Controls (unchanged)

| Key | Function |
|-----|----------|
| `+` / `-` | Threshold value |
| `z` / `x` | Min area |
| `c` / `v` | Max area |
| `b` / `n` | Min circularity |
| `1` / `2` | Blur kernel |
| `3` / `4` | Morph close iterations |
| `5` / `6` | Morph open iterations |
| `7` / `8` | Morph kernel size |
| `h` | Toggle CLAHE |
| `g` | Toggle glint removal |
| `f` | Toggle bilateral filter |
| `Space` | Toggle all views |
| `r` | Reset to defaults |
| `q` / `ESC` | Quit |

## How to Use

### Step 1: Basic Adjustments
```
1. Start the tuner normally
2. Look at Window 3: Preprocessed
3. Adjust basic parameters first:
   - Contrast (9/0): If image is flat/gray
   - Brightness (,/.): If image is too dark/bright
   - Gamma (/;): For non-linear brightness adjustment
```

### Step 2: Enhancement
```
4. Enable/adjust enhancement:
   - Press 'f' to enable bilateral filter (smooths noise)
   - Press 'h' to enable CLAHE (boosts local contrast)
   - Adjust CLAHE strength with \\ / '
   - Add sharpening with [ / ] if edges are soft
```

### Step 3: Fine-Tuning
```
5. Adjust detection parameters:
   - Threshold (+/-)
   - Area constraints (z/x, c/v)
   - Morphology (3/4, 5/6, 7/8)
```

## Parameter Explanations

### Contrast (Alpha)
```
alpha < 1.0  →  Lower contrast (flatter image)
alpha = 1.0  →  No change
alpha > 1.0  →  Higher contrast (more separation)

Example:
  Original: [50, 100, 150, 200]
  alpha=1.5: [75, 150, 225, 255]  (more spread)
  alpha=0.7: [35, 70, 105, 140]   (less spread)
```

### Brightness (Beta)
```
beta < 0  →  Darker image
beta = 0  →  No change
beta > 0  →  Brighter image

Example:
  Original: [50, 100, 150, 200]
  beta=+30: [80, 130, 180, 230]  (shifted up)
  beta=-30: [20, 70, 120, 170]   (shifted down)
```

### Gamma
```
gamma < 1.0  →  Brighten dark regions more (expand shadows)
gamma = 1.0  →  No change
gamma > 1.0  →  Darken bright regions more (compress highlights)

Useful for:
  - gamma < 1.0: Dark eyes (boost visibility)
  - gamma > 1.0: Overexposed eyes (reduce glare)
```

### Sharpening
```
amount = 0.0  →  No sharpening
amount = 1.0  →  Standard sharpening
amount = 2.0  →  Strong sharpening

Method: Unsharp mask
  1. Blur the image
  2. Subtract blur from original
  3. Add result back to original
  
Effect: Enhances edges, makes features more distinct
```

### CLAHE Clip Limit
```
clip = 1.0  →  Minimal contrast enhancement
clip = 2.0  →  Standard (default)
clip = 4.0  →  Aggressive enhancement
clip = 8.0  →  Very aggressive (may amplify noise)

CLAHE = Contrast Limited Adaptive Histogram Equalization
  - Works on local regions (tiles)
  - Prevents over-amplification
  - Grid size controls tile size (4-16)
```

## Camera Rotation

**Camera is now rotated 90° counter-clockwise** to show upright orientation.

To change rotation, edit `config.py`:
```python
camera_rotation: int = 270  # 0, 90, 180, or 270
```

Rotation values:
- `0` = No rotation
- `90` = 90° clockwise
- `180` = Upside down
- `270` = 90° counter-clockwise (current default)

## Typical Workflows

### Workflow 1: Dark/Underexposed Eye
```
1. Increase brightness: Press '.' several times (+30 to +50)
2. Boost contrast: Press '0' a few times (1.2 to 1.4)
3. Adjust gamma: Press ';' slightly (1.1 to 1.2)
4. Enable CLAHE: Press 'h'
5. Fine-tune detection threshold: Adjust with +/-
```

### Workflow 2: Overexposed/Bright Eye
```
1. Decrease brightness: Press ',' several times (-20 to -40)
2. Reduce gamma: Press '/' slightly (0.8 to 0.9)
3. Enable bilateral filter: Press 'f' (reduce glare noise)
4. Fine-tune detection
```

### Workflow 3: Low Contrast (Flat Image)
```
1. Increase contrast: Press '0' several times (1.5 to 2.0)
2. Enable CLAHE: Press 'h'
3. Increase CLAHE strength: Press ' several times (3.0 to 4.0)
4. Optional: Add slight sharpening (0.5 to 1.0)
```

### Workflow 4: Soft/Blurry Edges
```
1. Add sharpening: Press ']' several times (1.0 to 1.5)
2. Increase contrast: Press '0' (1.2 to 1.5)
3. Enable CLAHE: Press 'h'
```

## Debugging Tips

### Problem: Can't see pupil clearly
**Solution**: Adjust preprocessing first, then detection
```
1. Look at Window 3: Preprocessed
2. Pupil should be clearly darker than iris/sclera
3. Adjust contrast/brightness/gamma until clear
4. Then adjust threshold (+/-) for detection
```

### Problem: Too much noise in image
**Solution**: Apply smoothing
```
1. Enable bilateral filter: Press 'f'
2. Reduce sharpening: Press '[' to 0.0
3. May need to increase blur kernel (1/2)
```

### Problem: Image processing too aggressive
**Solution**: Reset and start over
```
1. Press 'r' to reset all parameters
2. Make small adjustments (0.1-0.2 at a time)
3. Check Window 3 after each change
```

## Advanced: Custom Presets

You can create presets by noting optimal values for your setup:

### Preset Example: Indoor Lighting
```python
contrast_alpha: 1.3
brightness_beta: +20
gamma_value: 1.1
sharpen_amount: 0.5
clahe_clip_limit: 3.0
use_bilateral_filter: True
use_histogram_eq: True
```

### Preset Example: Bright Lighting
```python
contrast_alpha: 0.9
brightness_beta: -10
gamma_value: 0.9
sharpen_amount: 0.0
clahe_clip_limit: 2.0
use_bilateral_filter: True
use_histogram_eq: False
```

## Summary

This update adds **6-stage adjustable preprocessing** with:
- ✅ Contrast adjustment (9/0)
- ✅ Brightness adjustment (,/.)
- ✅ Gamma correction (/;)
- ✅ Sharpening ([/])
- ✅ CLAHE strength control (\\/')
- ✅ Camera rotation (270° CCW)

You now have **complete control** over every image processing step!
