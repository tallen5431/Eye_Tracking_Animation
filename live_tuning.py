#!/usr/bin/env python3
"""
Live Eye Tracking Tuner - Single Process

Runs both tuning tool and eye tracker animation using a SHARED camera
in a single process. This avoids the camera conflict that occurs when
two separate processes try to open the same device.

Usage:
    python live_tuning.py
    CAMERA_ID=1 python live_tuning.py
"""

import os
import sys
import time
from pathlib import Path

# Resolve imports
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)

import cv2

from pupil_tuner.camera import open_camera
from pupil_tuner.app import PupilTrackerTuner
from eye_tracking_animation import PupilHighlighter

# Configuration
CAMERA_ID = int(os.environ.get("CAMERA_ID", "0"))
ROTATION = int(os.environ.get("CAMERA_ROTATION", "270"))
LOGS_DIR = Path(THIS_DIR) / "logs"
LOGS_DIR.mkdir(exist_ok=True)


def main():
    print("=" * 60)
    print("LIVE EYE TRACKING TUNER (shared camera)")
    print("=" * 60)
    print()
    print(f"Camera ID: {CAMERA_ID}")
    print(f"Rotation:  {ROTATION}")
    print()

    # Open camera ONCE
    print("Opening camera...")
    try:
        cap, camera_id = open_camera(CAMERA_ID)
    except RuntimeError as e:
        print(f"ERROR: {e}")
        return 1

    print(f"Camera {camera_id} ready")
    print()

    # Create both pipelines sharing the same camera
    tuner = PupilTrackerTuner(camera_id=camera_id, cap=cap)
    tracker = PupilHighlighter(camera_id=camera_id, rotation=ROTATION, cap=cap)

    # Run tuner startup calibration (uses shared camera)
    tuner._run_startup_calibration()

    # Create Eye Tracker window
    cv2.namedWindow("Eye Tracker - Pipeline", cv2.WINDOW_NORMAL)

    print("=" * 60)
    print("CONTROLS (Tuner keys active):")
    print("  q/ESC  Quit           m  Toggle pupil/iris mode")
    print("  +/-    Threshold      h  Histogram EQ")
    print("  z/x    Min area       g  Glint removal")
    print("  c/v    Max area       a  Auto threshold")
    print("  9/0    Contrast       s  Save calibration")
    print("  ,/.    Brightness     Space  Toggle views")
    print("  r      Reset defaults")
    print("=" * 60)
    print()

    last_autosave = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                time.sleep(0.01)
                continue

            # Feed the SAME frame to both pipelines
            tuner.process_and_display(frame)
            tracker.process_and_display(frame.copy())

            # Auto-save calibration every 30s
            if time.time() - last_autosave > 30.0:
                try:
                    tuner.save_calibration()
                    last_autosave = time.time()
                except Exception:
                    pass

            # Single waitKey for all windows
            key = cv2.waitKey(1) & 0xFF
            if tuner.handle_key(key):
                break

    except KeyboardInterrupt:
        print("\nStopped by user")
    finally:
        print()
        print("Cleaning up...")
        try:
            tuner.save_calibration()
        except Exception:
            pass
        cap.release()
        cv2.destroyAllWindows()
        cv2.waitKey(1)
        print("Done")

    return 0


if __name__ == '__main__':
    sys.exit(main() or 0)
