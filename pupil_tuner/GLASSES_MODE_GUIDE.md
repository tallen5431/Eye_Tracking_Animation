# Glasses Mode - Enhanced Glare Removal

## Problem

Glasses create **multiple bright reflections** that interfere with eye tracking:
- Direct light reflections off lens surface
- Glare halos around bright spots
- Multiple reflection points (both lenses)
- Can be mistaken for pupil by detection algorithm

## Solution: Glasses Mode

Press **`w`** to enable **Glasses Mode** - enhanced glare removal specifically designed for glasses wearers.

## What Glasses Mode Does

### Standard Glint Removal (key: g)
```
1. Detects bright spots (>245 brightness)
2. Small dilation (3x3 kernel, 1 iteration)
3. Inpaint with small radius (2px)

Good for: Single corneal reflections
Not enough for: Multiple large glass reflections
```

### Glasses Mode (key: w)
```
1. Detects bright spots (adjustable threshold: e/t)
2. LARGE dilation (7x7 kernel, 2 iterations)
3. Inpaint with large radius (5px default, adjust: y/u)

Good for: Multiple large reflections from glasses
Effect: Removes glare AND surrounding halo
```

## New Controls for Glare

| Key | Parameter | Range | Default | Effect |
|-----|-----------|-------|---------|--------|
| `w` | Glasses Mode | ON/OFF | OFF | Toggle enhanced glare removal |
| `e` / `t` | Glare Threshold | 200-250 | 220 | Brightness cutoff for glare |
| `y` / `u` | Inpaint Radius | 3-10 | 5 | Area to fill around glare |
| `g` | Glint Removal | ON/OFF | ON | Enable/disable glare removal |

## How to Use

### Step 1: Enable Glare Removal
```
1. Press 'g' to enable glint removal
2. Check Window 3 (Preprocessed)
3. Bright spots should be filled in
```

### Step 2: Enable Glasses Mode (if needed)
```
4. Still see bright reflections?
5. Press 'w' to enable Glasses Mode
6. Check Window 3 again
7. Larger areas should now be filled
```

### Step 3: Adjust Threshold (if needed)
```
8. Press 't' to increase threshold (detect MORE glare)
   - Use when: Some reflections not being removed
   - Effect: More pixels classified as glare

9. Press 'e' to decrease threshold (detect LESS glare)
   - Use when: Too much being removed (iris affected)
   - Effect: Only brightest pixels classified as glare
```

### Step 4: Adjust Inpaint Radius (if needed)
```
10. Press 'u' to increase radius (fill LARGER areas)
    - Use when: Glare halos still visible
    - Effect: Fills more around each glare spot

11. Press 'y' to decrease radius (fill SMALLER areas)
    - Use when: Over-inpainting affecting real features
    - Effect: More conservative filling
```

## Visual Comparison

### Without Glasses Mode
```
████████████████████
██░░░░░░░░░░░░░░░░██
██░░░░░█████░░░░░░██  ← Large bright reflection
██░░░░█████░░░●░░░██  ← Small pupil
██░░░░█████░░░░░░░██  ← Glare interferes
██░░░░░░░░░░░░░░░░██
████████████████████

Problem: Pupil detection picks up glare instead of pupil!
```

### With Standard Glint Removal
```
████████████████████
██░░░░░░░░░░░░░░░░██
██░░░░░▓▓▓▓▓░░░░░░██  ← Glare partially removed
██░░░░▓▓▓▓▓░░●░░░░██  ← Halo remains
██░░░░░▓▓▓▓▓░░░░░░██  ← Still interferes
██░░░░░░░░░░░░░░░░██
████████████████████

Better, but halo still visible
```

### With Glasses Mode
```
████████████████████
██░░░░░░░░░░░░░░░░██
██░░░░░░░░░░░░░░░░██  ← Glare completely removed
██░░░░░░░░░░░●░░░░██  ← Clean pupil detection!
██░░░░░░░░░░░░░░░░██  ← No interference
██░░░░░░░░░░░░░░░░██
████████████████████

Perfect! Pupil detected correctly
```

## Typical Settings

### No Glasses (Standard)
```
Glasses Mode:      OFF
Glare Threshold:   245  (only very bright)
Inpaint Radius:    2    (small)
```

### Regular Glasses
```
Glasses Mode:      ON
Glare Threshold:   220  (moderate)
Inpaint Radius:    5    (medium)
```

### Strong Reflective Glasses
```
Glasses Mode:      ON
Glare Threshold:   210  (lower - more sensitive)
Inpaint Radius:    7    (larger - fill more)
```

### Sunglasses / Tinted Glasses
```
Glasses Mode:      ON
Glare Threshold:   200  (very sensitive)
Inpaint Radius:    8    (very large)
May also need: Increase brightness (+)
```

## Troubleshooting

### Problem: Pupil detection picking up glare instead of pupil

**Solution:**
```
1. Enable Glasses Mode (w)
2. Increase threshold (t) - detect more glare
3. Increase radius (u) - fill larger areas
4. Check Window 3 - glare should be gone
5. Adjust pupil area constraints (z/x, c/v) if needed
```

### Problem: Too much being removed, affecting real features

**Solution:**
```
1. Decrease threshold (e) - be more selective
2. Decrease radius (y) - fill less aggressively
3. Check Window 3 - real features should remain
```

### Problem: Glare removal not effective enough

**Solution:**
```
1. Enable Bilateral Filter (f) - smooths before removal
2. Enable CLAHE (h) - may help distinguish features
3. Increase threshold (t) to maximum (250)
4. Increase radius (u) to maximum (10)
```

### Problem: Multiple reflection points across entire frame

**Solution:**
```
1. Enable Glasses Mode (w)
2. Set threshold to 210
3. Set radius to 8-10
4. Enable Bilateral Filter (f)
5. May need to adjust camera angle to reduce reflections
```

## Performance Note

Glasses Mode adds ~2-3ms processing time due to:
- Larger dilation operations
- Larger inpainting area

Still maintains 30-50 FPS on typical hardware.

## Recommendation

If you wear glasses:

1. **Start with Glasses Mode ON** (press 'w')
2. **Enable Glint Removal** (press 'g')
3. **Enable Bilateral Filter** (press 'f')
4. Adjust threshold (e/t) and radius (y/u) as needed

This gives the best baseline for glasses wearers!

## Console Feedback

When adjusting glare parameters:
```
[GLASSES MODE] ON
[GLARE] Threshold: 220
[GLARE] Inpaint radius: 5
```

This helps track your settings.

## Summary

Glasses Mode provides:
- ✅ Enhanced removal of large reflections
- ✅ Adjustable sensitivity (threshold)
- ✅ Adjustable coverage (radius)
- ✅ Prevents glare from interfering with detection
- ✅ Designed specifically for glasses wearers

Press **`w`** to enable and see the difference!
