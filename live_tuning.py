#!/usr/bin/env python3
"""
Live Eye Tracking Tuner
Runs both tuning tool and animation simultaneously so you can see changes in real-time.
"""

import os
import sys
import subprocess
import time
import threading
from pathlib import Path

# Configuration
CAMERA_ID = int(os.environ.get("CAMERA_ID", "0"))
WINDOW_W = int(os.environ.get("ANIMATION_WIDTH", "600"))
WINDOW_H = int(os.environ.get("ANIMATION_HEIGHT", "500"))
WINDOW_X = int(os.environ.get("ANIMATION_X", "1300"))  # Position on right side
WINDOW_Y = int(os.environ.get("ANIMATION_Y", "100"))

# Paths
THIS_DIR = Path(__file__).resolve().parent
LOGS_DIR = THIS_DIR / "logs"
SHARE_FILE = LOGS_DIR / "pupil_share.json"

# Create logs directory
LOGS_DIR.mkdir(exist_ok=True)

def find_python():
    """Find Python executable."""
    if os.name == 'nt':
        try:
            result = subprocess.run(['py', '--version'], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                return 'py'
        except:
            pass
    
    for cmd in ['python3', 'python']:
        try:
            result = subprocess.run([cmd, '--version'], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                return cmd
        except:
            pass
    
    raise RuntimeError("Python not found!")

def run_animation(python_exe):
    """Run the animation in a separate process."""
    print("[Animation] Starting...")
    
    # Set environment for windowed mode
    env = os.environ.copy()
    env['CAMERA_ID'] = str(CAMERA_ID)
    env['TRACKING_ENABLED'] = "1"
    env['TRACK_SHARE_FILE'] = str(SHARE_FILE)
    
    # Window configuration (NOT fullscreen)
    env['FULLSCREEN'] = "0"
    env['BORDERLESS'] = "0"  # Keep window decorations for easy moving
    env['WINDOW_W'] = str(WINDOW_W)
    env['WINDOW_H'] = str(WINDOW_H)
    env['WINDOW_X'] = str(WINDOW_X)
    env['WINDOW_Y'] = str(WINDOW_Y)
    
    # Run animation
    anim_script = THIS_DIR / "eye_tracking_animation.py"
    cmd = [python_exe, str(anim_script)]
    
    try:
        # Redirect output to avoid cluttering console
        with open(LOGS_DIR / "animation_live.out.log", 'w') as out_log:
            with open(LOGS_DIR / "animation_live.err.log", 'w') as err_log:
                subprocess.run(cmd, env=env, stdout=out_log, stderr=err_log)
    except Exception as e:
        print(f"[Animation] Error: {e}")

def main():
    """Main entry point."""
    print("=" * 70)
    print("LIVE EYE TRACKING TUNER")
    print("=" * 70)
    print()
    print("This will run TWO windows simultaneously:")
    print("  1. Tuning Tool (left) - Adjust parameters here")
    print("  2. Eye Animation (right) - See changes in real-time")
    print()
    print(f"Camera ID: {CAMERA_ID}")
    print(f"Share file: {SHARE_FILE}")
    print(f"Animation window: {WINDOW_W}x{WINDOW_H} at position ({WINDOW_X}, {WINDOW_Y})")
    print()
    print("=" * 70)
    print()
    
    # Find Python
    python_exe = find_python()
    print(f"Python: {python_exe}")
    
    # Check dependencies
    print("Checking dependencies...")
    try:
        result = subprocess.run(
            [python_exe, '-c', 'import cv2, numpy, tkinter; print("OK")'],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode != 0:
            print("ERROR: Missing dependencies!")
            print(f"Run: {python_exe} -m pip install opencv-python numpy")
            return 1
        print("✅ Dependencies OK")
    except Exception as e:
        print(f"ERROR: {e}")
        return 1
    
    print()
    print("=" * 70)
    print("INSTRUCTIONS:")
    print("=" * 70)
    print()
    print("1. TWO WINDOWS will open:")
    print("   - Tuning windows (OpenCV) on the LEFT")
    print("   - Animation window (Tkinter) on the RIGHT")
    print()
    print("2. Position the tuning windows so you can see both")
    print()
    print("3. Adjust parameters in the tuning tool:")
    print("   - Use keyboard controls (see console)")
    print("   - Watch the animation window respond in REAL-TIME")
    print()
    print("4. When satisfied, press 'q' or ESC in tuning tool")
    print("   - Animation will close automatically")
    print()
    print("TUNING CONTROLS:")
    print("  +/-  Threshold    h  Histogram EQ    Space  Toggle views")
    print("  z/x  Min area     g  Glint removal")
    print("  c/v  Max area     a  Auto threshold")
    print("  b/n  Circularity  p  Print settings")
    print()
    print("=" * 70)
    print()
    
    try:
        response = input("Ready to start? [Y/n]: ").strip().lower()
        if response and response not in ['y', 'yes']:
            print("Cancelled")
            return 0
    except KeyboardInterrupt:
        print("\nCancelled")
        return 0
    
    print()
    print("Starting in 2 seconds...")
    time.sleep(2)
    
    # Set environment for tuner
    env = os.environ.copy()
    env['TRACK_SHARE_FILE'] = str(SHARE_FILE)
    
    # Start animation in background thread
    print("[Animation] Launching in background...")
    anim_thread = threading.Thread(target=run_animation, args=(python_exe,), daemon=True)
    anim_thread.start()
    
    # Give animation time to start
    time.sleep(1.5)
    
    # Run tuner in foreground
    print("[Tuning] Launching tuning tool...")
    print()
    tuner_script = THIS_DIR / "tuning_tool.py"
    cmd = [python_exe, str(tuner_script), "-c", str(CAMERA_ID)]
    
    try:
        result = subprocess.run(cmd, env=env)
        return result.returncode
    except KeyboardInterrupt:
        print("\nStopped by user")
        return 0
    except Exception as e:
        print(f"Error: {e}")
        return 1
    finally:
        print()
        print("=" * 70)
        print("Session complete!")
        print("=" * 70)
        print()
        print("Note: Animation window should close automatically.")
        print("      If it doesn't, close it manually.")

if __name__ == '__main__':
    try:
        exit_code = main()
        print()
        input("Press Enter to exit...")
        sys.exit(exit_code)
    except Exception as e:
        print(f"\nFATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        input("\nPress Enter to exit...")
        sys.exit(1)
