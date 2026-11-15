"""Synthetic images for camera-free tests: white background, blue tape border, drawn hands."""
import cv2
import numpy as np

TAPE = (195, 70, 30)            # BGR of the blue painter's tape (HSV ~ (109, 215, 195))

SKIN = (140, 170, 215)          # BGR, medium
SKIN_DARK = (60, 85, 120)
SKIN_LIGHT = (180, 200, 230)    # close to white in gray; caught by saturation


def background(size=200, level=235, seed=0):
    rng = np.random.default_rng(seed)
    img = level + rng.normal(0, 2, (size, size, 3))
    return np.clip(img, 0, 255).astype(np.uint8)


def hand(n_fingers, size=200, seed=0, rotate=0.0, skin=SKIN, shift=(0, 0)):
    """Palm + arm from the bottom + n fingers fanned out."""
    img = background(size, seed=seed)
    cx, cy = size // 2 + shift[0], int(size * 0.65) + shift[1]
    cv2.ellipse(img, (cx, cy), (int(size * .16), int(size * .13)), 0, 0, 360, skin, -1)
    cv2.rectangle(img, (cx - int(size * .08), cy), (cx + int(size * .08), size), skin, -1)
    for i in range(n_fingers):
        ang = 0.0 if n_fingers == 1 else -70 + 140 * i / (n_fingers - 1)
        a = np.deg2rad(ang - 90 + rotate)
        start = (int(cx + np.cos(a) * size * .10), int(cy + np.sin(a) * size * .10))
        end = (int(cx + np.cos(a) * size * .36), int(cy + np.sin(a) * size * .36))
        cv2.line(img, start, end, skin, max(3, int(size * .05)))
    return img


def with_red_border(img, margin=3, thick=6):
    out = img.copy()
    h, w = out.shape[:2]
    cv2.rectangle(out, (margin, margin), (w - margin, h - margin), TAPE, thick)
    return out


def scene(square=(200, 120, 240, 240), thick=8, frame=(480, 640), tilt=0.0):
    """Full camera frame: white table with a red square outline."""
    img = np.full((*frame, 3), 235, np.uint8)
    x, y, w, h = square
    if tilt:
        box = cv2.boxPoints(((x + w / 2, y + h / 2), (w, h), tilt)).astype(np.int32)
        cv2.polylines(img, [box], True, TAPE, thick)
    else:
        cv2.rectangle(img, (x, y), (x + w, y + h), TAPE, thick)
    return img
