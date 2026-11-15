"""Single source of truth for turning a camera frame into a model input.
Used by collect, train, audit, the PC live demo AND the UnoQ, so they always match."""
import cv2
import numpy as np

from . import config


def get_roi():
    calib = config.load_calibration()
    if calib is None:
        raise RuntimeError("No camera calibration found. Run: python -m src.calibrate")
    return tuple(calib["roi"])


def crop_roi(frame, roi=None):
    x, y, w, h = roi if roi is not None else get_roi()
    return frame[y:y + h, x:x + w]


def trim_tape_edges(img, frac=0.3, max_trim=0.15):
    """Cut off tape strips along the crop edges (paper nudged a few px after calibration),
    so the model never sees the border and padding can't smear it into a thick band."""
    from .quality import red_mask          # tape-color mask
    m = red_mask(img) > 0
    h, w = m.shape
    t, b, l, r = 0, h, 0, w
    while b - t > (1 - max_trim) * h and m[b - 1, l:r].mean() > frac:
        b -= 1
    while b - t > (1 - 2 * max_trim) * h and m[t, l:r].mean() > frac:
        t += 1
    while r - l > (1 - max_trim) * w and m[t:b, r - 1].mean() > frac:
        r -= 1
    while r - l > (1 - 2 * max_trim) * w and m[t:b, l].mean() > frac:
        l += 1
    return img[t:b, l:r]


def pad_square(img):
    """Pad the short side (edge pixels repeated: paper stays paper, an arm keeps reaching
    the edge) so resizing to a square doesn't squash the fingers."""
    h, w = img.shape[:2]
    if h == w:
        return img
    d = abs(h - w)
    a, b = d // 2, d - d // 2
    t, bt, l, r = (a, b, 0, 0) if h < w else (0, 0, a, b)
    return cv2.copyMakeBorder(img, t, bt, l, r, cv2.BORDER_REPLICATE)


def preprocess_roi(roi_bgr, channels=None) -> np.ndarray:
    """BGR crop -> float32 (C, IMG_SIZE, IMG_SIZE) in [-1, 1].
    C = 3 (RGB, keeps skin / tape / white paper apart) or 1 (grayscale)."""
    channels = channels or config.IMG_CHANNELS
    s = config.IMG_SIZE
    if config.TRIM_TAPE_EDGES and roi_bgr.shape[2:] == (3,):
        roi_bgr = trim_tape_edges(roi_bgr)
    if config.PAD_TO_SQUARE:
        roi_bgr = pad_square(roi_bgr)
    if channels == 1:
        a = cv2.resize(cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY), (s, s),
                       interpolation=cv2.INTER_AREA)[None]
    else:
        a = cv2.resize(cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB), (s, s),
                       interpolation=cv2.INTER_AREA).transpose(2, 0, 1)
    return (a.astype(np.float32) / 255.0 - 0.5) / 0.5


def preprocess(frame, roi=None, channels=None) -> np.ndarray:
    return preprocess_roi(crop_roi(frame, roi), channels)


def model_view(roi_bgr, size=160) -> np.ndarray:
    """What the network actually sees, upscaled for display (BGR uint8)."""
    a = ((preprocess_roi(roi_bgr) * 0.5 + 0.5) * 255).clip(0, 255).astype(np.uint8)
    if a.shape[0] == 1:
        v = cv2.cvtColor(a[0], cv2.COLOR_GRAY2BGR)
    else:
        v = cv2.cvtColor(a.transpose(1, 2, 0), cv2.COLOR_RGB2BGR)
    return cv2.resize(v, (size, size), interpolation=cv2.INTER_NEAREST)
