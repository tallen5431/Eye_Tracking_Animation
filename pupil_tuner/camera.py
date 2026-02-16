from __future__ import annotations
import platform
import cv2
import time

def open_camera(camera_id: int, width: int = 640, height: int = 480, fps: int = 30):
    """
    Open a camera device with improved reliability.

    Features:
    - Windows DirectShow preference for stability
    - Fallback scan if primary camera fails
    - Camera warm-up period (discard unstable initial frames)
    - Exponential backoff retry logic

    Returns: (cap, actual_camera_id)
    Raises: RuntimeError on failure
    """
    system = platform.system().lower()

    # Try primary camera with exponential backoff
    for attempt in range(3):
        # Prefer DirectShow on Windows for stability
        if system == "windows":
            cap = cv2.VideoCapture(camera_id, cv2.CAP_DSHOW)
        else:
            cap = cv2.VideoCapture(camera_id)

        if cap.isOpened():
            _apply_props(cap, width, height, fps)

            # Camera warm-up: discard first 10 frames (often dark/unstable)
            for _ in range(10):
                cap.read()
            time.sleep(0.1)  # Let camera stabilize

            ok, frame = cap.read()
            if ok and frame is not None:
                print(f"[INFO] Camera {camera_id} opened successfully")
                return cap, camera_id
            cap.release()

        # Exponential backoff before retry
        if attempt < 2:
            wait_time = 0.5 * (2 ** attempt)  # 0.5s, 1.0s
            print(f"[WARN] Camera {camera_id} attempt {attempt + 1} failed, retrying in {wait_time}s...")
            time.sleep(wait_time)

    # Fallback scan with same warm-up logic
    print(f"[WARN] Camera {camera_id} failed, scanning for alternatives...")
    for idx in range(0, 11):
        if system == "windows":
            cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        else:
            cap = cv2.VideoCapture(idx)

        if not cap.isOpened():
            cap.release()
            continue

        _apply_props(cap, width, height, fps)

        # Warm-up for fallback camera too
        for _ in range(10):
            cap.read()
        time.sleep(0.1)

        ok, frame = cap.read()
        if ok and frame is not None:
            return cap, idx
        cap.release()

    raise RuntimeError("Could not open any camera (tried indices 0-10).")

def _apply_props(cap, width: int, height: int, fps: int):
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
