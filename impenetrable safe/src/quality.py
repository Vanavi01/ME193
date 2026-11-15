"""Image-quality checks shared by calibrate, collect and audit (pure functions, no camera)."""
from typing import Optional

import cv2
import numpy as np

from . import config


# ------------------------------------------------- border tape (blue; names say "red" - historic)
def red_mask(bgr) -> np.ndarray:
    """Pixels that look like the border tape (color band from config.TAPE_*)."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    return cv2.inRange(hsv, (config.TAPE_H_MIN, config.TAPE_S_MIN, config.TAPE_V_MIN),
                       (config.TAPE_H_MAX, 255, 255))


tape_mask = red_mask


def red_fraction(bgr) -> float:
    m = red_mask(bgr)
    return float(np.count_nonzero(m)) / m.size


def border_red_fraction(bgr, band_frac=config.BAND_FRAC) -> float:
    """Red pixels in the outer band of a crop (the border creeping in = camera moved)."""
    m = red_mask(bgr) > 0
    h, w = m.shape
    b = max(2, int(min(h, w) * band_frac))
    band = np.ones_like(m)
    band[b:h - b, b:w - b] = False
    return float(m[band].mean())


def detect_red_square(frame) -> Optional[dict]:
    """Find the red outline. Returns {"roi": inside-the-border box, "outer": outline box,
    "tilt_deg": rotation} or None if no complete hollow red square is visible."""
    return diagnose_red_square(frame)[0]


def diagnose_red_square(frame):
    """(detection or None, human-readable reason). Used by the live calibration view."""
    m = cv2.morphologyEx(red_mask(frame), cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    if np.count_nonzero(m) < 0.002 * m.size:
        return None, "no border tape visible - is the square under the camera?"
    contours, hier = cv2.findContours(m, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    hier = hier[0]
    outers = [i for i in range(len(contours)) if hier[i][3] == -1]
    ci = max(outers, key=lambda i: cv2.contourArea(contours[i]))
    c = contours[ci]
    x, y, w, h = cv2.boundingRect(c)
    H, W = frame.shape[:2]
    if w * h < config.MIN_SQUARE_AREA_FRAC * W * H:
        return None, "border too small (or only a scrap of tape visible)"
    holes = [j for j in range(len(contours)) if hier[j][3] == ci]
    hole_box = cv2.boundingRect(max((contours[j] for j in holes), key=cv2.contourArea)) \
        if holes else None
    if hole_box is not None and hole_box[2] * hole_box[3] <= 0.2 * w * h:
        hole_box = None
    edges = [name for name, hit in (("top", y <= 2), ("bottom", y + h >= H - 2),
                                    ("left", x <= 2), ("right", x + w >= W - 2)) if hit]
    if edges:
        # only the tape's OUTER edge may touch the frame: the inside must be fully visible
        hx, hy, hw, hh = hole_box if hole_box is not None else (0, 0, W, H)
        inside_ok = hole_box is not None and hx > 2 and hy > 2 and hx + hw < W - 2 and hy + hh < H - 2
        if not inside_ok:
            return None, f"square cut off at the {'/'.join(edges)} of the view - center it"
    if np.count_nonzero(m[y:y + h, x:x + w]) / (w * h) > 0.6:
        return None, "tape color is a solid area, not an outline"
    angle = cv2.minAreaRect(c)[2] % 90
    tilt = float(min(angle, 90 - angle))
    # start from the inner hole of the outline (exact for thick/uneven tape); if the
    # outline isn't closed (e.g. an arm across it) fall back to shrinking from the outside
    start = hole_box if hole_box is not None else (x, y, w, h)
    roi = _inset_until_clean(frame, start)
    if roi is None:
        return None, "can't find a clean area inside the border (something tape-colored inside?)"
    det = {"roi": roi, "outer": (x, y, w, h), "tilt_deg": tilt}
    if tilt > config.MAX_TILT_DEG:
        return det, f"found, but tilted {tilt:.1f} deg - rotate paper/camera"
    return det, "OK"


def _inset_until_clean(frame, box, step=2, max_frac=0.3, strip=3):
    """Shrink the box until no tape is inside. Each side moves on its own (the side whose
    edge strip has the most tape), so a slightly slanted/keystoned border only costs area
    on the side that slants. Tape that isn't at any edge (something inside) shrinks all sides."""
    x, y, w, h = box
    m = red_mask(frame) > 0
    x0, y0, x1, y1 = x, y, x + w, y + h
    limit = max_frac * min(w, h)
    while True:
        c = m[y0:y1, x0:x1]
        if c.size == 0 or max(x0 - x, y0 - y, x + w - x1, y + h - y1) > limit:
            return None
        if c.mean() < config.RED_CLEAN_FRAC:
            e = config.ROI_EXTRA_MARGIN
            return (x0 + e, y0 + e, x1 - x0 - 2 * e, y1 - y0 - 2 * e)
        sides = {"l": c[:, :strip].mean(), "r": c[:, -strip:].mean(),
                 "t": c[:strip].mean(), "b": c[-strip:].mean()}
        worst = max(sides, key=sides.get)
        if sides[worst] == 0:
            x0, y0, x1, y1 = x0 + step, y0 + step, x1 - step, y1 - step
        elif worst == "l":
            x0 += step
        elif worst == "r":
            x1 -= step
        elif worst == "t":
            y0 += step
        else:
            y1 -= step


def calibration_drift(frame, calib) -> list:
    """Is a saved calibration still valid for this frame (square EMPTY)? [] = yes."""
    issues = []
    det, reason = diagnose_red_square(frame)
    if det is None:
        issues.append(f"border not detected ({reason}) -> fix the view, then full recalibration")
    else:
        overlap = iou(det["outer"], calib["outer"])
        if overlap < config.MOVE_IOU_MIN:
            issues.append(f"camera or paper moved (overlap with saved {overlap:.0%}) -> realign with "
                          "--view, or run a full recalibration")
    x, y, w, h = calib.get("border_roi", calib["roi"])     # judge lighting on the paper only
    g = cv2.cvtColor(frame[y:y + h, x:x + w], cv2.COLOR_BGR2GRAY)
    if abs(g.mean() - calib["bg_mean"]) > config.LIGHT_DRIFT_MAX:
        issues.append(f"lighting changed (background {calib['bg_mean']:.0f} -> {g.mean():.0f}) "
                      "-> python -m src.calibrate --relight")
    if g.std() > config.BG_STD_MAX:
        issues.append(f"background uneven now (std {g.std():.0f}): shadow/glare, or something "
                      "in the square")
    return issues


def compare_calibrations(old, new) -> list:
    """Warnings when the crop geometry changed vs the calibration the data was collected with."""
    w0, h0 = old["roi"][2:]
    w1, h1 = new["roi"][2:]
    a0, a1 = w0 / h0, w1 / h1
    warns = []
    if (a0 > 1.1 and a1 < 0.9) or (a0 < 0.9 and a1 > 1.1):
        warns.append("crop orientation flipped (camera rotated ~90 deg?): hands now enter from a "
                     "different side than in the training images")
    elif abs(a1 / a0 - 1) > 0.10:
        warns.append(f"crop shape changed {a0:.2f} -> {a1:.2f}: images are stretched differently "
                     "than the training data; collect a few new images and retrain")
    # camera height alone doesn't matter (hand and square scale together and the crop is
    # resized to 64x64); what matters is how much of the crop a hand covers
    f0, f1 = old.get("open_hand_frac"), new.get("open_hand_frac")
    if f0 and f1 and not 0.75 <= f1 / f0 <= 1 / 0.75:
        warns.append(f"open hand now covers {f1:.0%} of the crop vs {f0:.0%} before: hands look "
                     f"{'bigger' if f1 > f0 else 'smaller'} than in the training images")
    return warns


def iou(a, b) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix = max(0, min(ax + aw, bx + bw) - max(ax, bx))
    iy = max(0, min(ay + ah, by + bh) - max(ay, by))
    inter = ix * iy
    union = aw * ah + bw * bh - inter
    return inter / union if union else 0.0


# ---------------------------------------------------------------- hand / blur
def hand_mask(roi, bg) -> np.ndarray:
    """Pixels that differ from the empty-background reference (gray OR saturation)."""
    if bg.shape != roi.shape:
        bg = cv2.resize(bg, (roi.shape[1], roi.shape[0]))
    g = cv2.GaussianBlur(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY), (5, 5), 0).astype(np.int16)
    gb = cv2.GaussianBlur(cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY), (5, 5), 0).astype(np.int16)
    s = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)[..., 1].astype(np.int16)
    sb = cv2.cvtColor(bg, cv2.COLOR_BGR2HSV)[..., 1].astype(np.int16)
    m = (np.abs(g - gb) > config.HAND_DIFF_THRESH) | ((s - sb) > config.HAND_SAT_THRESH)
    m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    return m.astype(bool)


SIDES = ("top", "bottom", "left", "right")


def edge_contact(mask, band=3) -> dict:
    """Fraction of each crop edge that the hand touches."""
    return {"top": float(mask[:band].any(axis=0).mean()),
            "bottom": float(mask[-band:].any(axis=0).mean()),
            "left": float(mask[:, :band].any(axis=1).mean()),
            "right": float(mask[:, -band:].any(axis=1).mean())}


def cut_off_sides(contact, arm_side) -> list:
    """Edges (other than where the arm comes in) that the hand runs past."""
    return [s for s in SIDES if s != arm_side and contact[s] > config.EDGE_CONTACT_MAX]


def sharpness(gray, mask=None) -> float:
    """Laplacian variance along the hand's outline (motion blur softens it).
    Falls back to the whole image if there is no hand."""
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    if mask is not None and mask.any():
        k = np.ones((5, 5), np.uint8)
        m = mask.astype(np.uint8)
        band = (cv2.dilate(m, k) > 0) & (cv2.erode(m, k) == 0)
        if band.sum() > 50:
            return float(lap[band].var())
    return float(lap.var())


def presence_blob(roi, bg, mask, box=None) -> float:
    """Largest solid hand region as a fraction of the paper area. Pixels on/near the red tape
    (in the frame or the reference) are ignored, so a slightly shifted paper or border can't
    look like a hand; only the area inside the border counts when `box` is given."""
    if bg.shape != roi.shape:
        bg = cv2.resize(bg, (roi.shape[1], roi.shape[0]))
    red = cv2.dilate(((red_mask(roi) > 0) | (red_mask(bg) > 0)).astype(np.uint8),
                     np.ones((21, 21), np.uint8)) > 0
    m = mask & ~red
    if box is not None:
        x, y, w, h = box
        m = m[y:y + h, x:x + w]
    n, _, stats, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8))
    return float(stats[1:, cv2.CC_STAT_AREA].max()) / m.size if n > 1 else 0.0


def _model_crop(roi):
    if not config.TRIM_TAPE_EDGES:
        return roi
    from .preprocess import trim_tape_edges
    return trim_tape_edges(roi)


def measure(roi, bg=None, box=None) -> dict:
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    m = {"mean": float(gray.mean()),
         "saturated": float((gray >= 253).mean()),
         "red_band": border_red_fraction(_model_crop(roi)),   # tape the model would still see
         "hand_frac": None, "hand_blob": None, "bg_level": None, "bg_ref": None, "edges": None}
    mask = None
    if bg is not None:
        mask = hand_mask(roi, bg)
        m["hand_frac"] = float(mask.mean())
        m["hand_blob"] = presence_blob(roi, bg, mask, box)
        m["edges"] = edge_contact(mask)
        bgg = cv2.cvtColor(bg if bg.shape == roi.shape else
                           cv2.resize(bg, (roi.shape[1], roi.shape[0])), cv2.COLOR_BGR2GRAY)
        m["bg_ref"] = float(bgg.mean())
        if (~mask).sum() > 0.05 * mask.size:
            m["bg_level"] = float(gray[~mask].mean())
    m["sharpness"] = sharpness(gray, mask)
    return m


def check_sample(roi, label, calib=None, bg=None):
    """Returns (issues, metrics). Empty issues list = good image for this label."""
    calib = calib or {}
    box = None
    if calib.get("border_roi") and calib.get("roi") and list(calib["border_roi"]) != list(calib["roi"]):
        bx, by, bw, bh = calib["border_roi"]          # inside-of-border, relative to the crop
        box = (bx - calib["roi"][0], by - calib["roi"][1], bw, bh)
        if roi.shape[:2] != (calib["roi"][3], calib["roi"][2]):
            box = None                                # stored image of another size
    m = measure(roi, bg, box)
    issues = []
    hand_min = calib.get("hand_frac_min", config.DEFAULT_HAND_FRAC_MIN)
    sharp_min = calib.get("sharpness_min", config.DEFAULT_SHARPNESS_MIN)
    if m["hand_frac"] is not None:
        if label > 0 and m["hand_blob"] < hand_min:     # one solid region, not scattered noise
            issues.append("no hand in square")
        if m["hand_frac"] > config.HAND_MAX_FRAC:
            issues.append("square mostly covered")
        if config.REJECT_CUT_OFF_FINGERS and label > 0:
            # arm side: from calibration, else the edge the hand touches most in this frame
            arm = calib.get("arm_side") or max(SIDES, key=lambda s: m["edges"][s])
            cut = cut_off_sides(m["edges"], arm)
            if cut:
                issues.append(f"fingers out of view at the {'/'.join(cut)}")
        lvl = m["bg_level"] if m["bg_level"] is not None else m["mean"]
        if abs(lvl - m["bg_ref"]) > config.LIGHT_DRIFT_MAX:
            issues.append("lighting changed since calibration")
    if label > 0 and m["sharpness"] < sharp_min:
        issues.append("blurry (hold still)")
    if m["mean"] < config.DARK_MEAN_MIN:
        issues.append("too dark")
    if m["saturated"] > config.SATURATED_MAX:
        issues.append("overexposed")
    if calib.get("crop_mode", "border") == "border" and m["red_band"] > config.BAND_RED_MAX:
        issues.append("border tape in crop (camera moved? blue sleeve?)")   # in "frame" mode it's meant to be there
    return issues, m


# ---------------------------------------------------------------- duplicates
def signature(roi) -> np.ndarray:
    g = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    return cv2.resize(g, (16, 16), interpolation=cv2.INTER_AREA).astype(np.float32)


def is_duplicate(a, b) -> bool:
    return float(np.abs(a - b).mean()) < config.DUP_DIFF
