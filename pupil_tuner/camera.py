from __future__ import annotations
import platform
import cv2

def open_camera(camera_id: int, width: int = 640, height: int = 480, fps: int = 30):
    """
    Open a camera device, with Windows DirectShow preference and fallback scan.

    Returns: (cap, actual_camera_id)
    Raises: RuntimeError on failure
    """
    system = platform.system().lower()

    # Prefer DirectShow on Windows for stability
    if system == "windows":
        cap = cv2.VideoCapture(camera_id, cv2.CAP_DSHOW)
    else:
        cap = cv2.VideoCapture(camera_id)

    if cap.isOpened():
        _apply_props(cap, width, height, fps)
        ok, _ = cap.read()
        if ok:
            return cap, camera_id
        cap.release()

    # Fallback scan
    for idx in range(0, 11):
        if system == "windows":
            cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        else:
            cap = cv2.VideoCapture(idx)

        if not cap.isOpened():
            cap.release()
            continue

        _apply_props(cap, width, height, fps)
        ok, _ = cap.read()
        if ok:
            return cap, idx
        cap.release()

    raise RuntimeError("Could not open any camera (tried indices 0-10).")

def _apply_props(cap, width: int, height: int, fps: int):
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
