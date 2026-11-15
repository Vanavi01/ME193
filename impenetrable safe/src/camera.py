"""Open the USB webcam the same way on the PC (Windows/DirectShow) and the UnoQ (Linux/V4L2).

Auto-exposure is a problem for this project: when a hand enters, the camera re-brightens
the whole image, so the same white paper looks different with and without a hand. Calibration
therefore picks a fixed manual exposure (choose_exposure) and every script reuses it."""
import sys

import cv2

from . import config


def _backend():
    if sys.platform.startswith("win"):
        return cv2.CAP_DSHOW
    if sys.platform.startswith("linux"):
        return cv2.CAP_V4L2
    return cv2.CAP_ANY


def exposure_candidates():
    if _backend() == cv2.CAP_DSHOW:
        return list(range(-10, 0))                     # DirectShow: log2 seconds
    return [20, 40, 60, 80, 120, 160, 250, 400, 600, 1000, 1500]   # V4L2: 100 us units


def set_manual_exposure(cap, value, settle=10):
    manual = 0.25 if _backend() == cv2.CAP_DSHOW else 1   # driver conventions differ
    ok = bool(cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, manual))
    cap.set(cv2.CAP_PROP_EXPOSURE, value)
    cap.set(cv2.CAP_PROP_AUTO_WB, 0)
    for _ in range(settle):
        cap.read()
    return ok


def choose_exposure(cap, roi, target=None):
    """Try each manual exposure on the EMPTY area; keep the one whose background brightness is
    closest to target without clipping. Returns (value, brightness) or (None, None)."""
    target = target or config.EXPOSURE_TARGET
    x, y, w, h = roi
    best = (None, None, 1e9)
    for e in exposure_candidates():
        if not set_manual_exposure(cap, e):
            return None, None
        g = cv2.cvtColor(read_frame(cap)[y:y + h, x:x + w], cv2.COLOR_BGR2GRAY)
        if (g >= 253).mean() > 0.01:
            continue
        if abs(g.mean() - target) < best[2]:
            best = (e, float(g.mean()), abs(g.mean() - target))
    if best[0] is not None:
        set_manual_exposure(cap, best[0])
    return best[0], best[1]


def open_camera(index, width=config.FRAME_W, height=config.FRAME_H, exposure=None, warmup=30):
    """Returns (cap, info). exposure=None keeps auto-exposure."""
    cap = cv2.VideoCapture(index, _backend())
    if not cap.isOpened():
        raise RuntimeError(f"camera {index} not available")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    for _ in range(warmup):
        cap.read()
    locked = set_manual_exposure(cap, exposure) if exposure is not None else False
    info = {"index": index,
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "exposure": exposure if locked else None,
            "exposure_locked": locked}
    return cap, info


def open_from_calibration(calib, index=None):
    return open_camera(calib["camera_index"] if index is None else index,
                       calib["width"], calib["height"], exposure=calib.get("exposure"))


def read_frame(cap):
    ok, frame = cap.read()
    if not ok or frame is None:
        raise RuntimeError("camera returned no frame (unplugged? hub power?)")
    return frame
