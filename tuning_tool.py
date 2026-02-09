#!/usr/bin/env python3
"""
Pupil Tracker Tuning Tool (Modularized)

This file is intentionally thin. The pipeline lives in:
  pupil_tuner/pipeline.py
  pupil_tuner/scoring.py
  pupil_tuner/overlay.py

Run:
  python tuning_tool.py --camera 0
"""
import argparse
from pupil_tuner.app import PupilTrackerTuner

def main():
    parser = argparse.ArgumentParser(description="Pupil Tracker Tuning Tool (Modularized)")
    parser.add_argument("-c", "--camera", type=int, default=0, help="Camera index (default: 0)")
    args = parser.parse_args()

    tuner = PupilTrackerTuner(camera_id=args.camera)
    tuner.run()

if __name__ == "__main__":
    main()
