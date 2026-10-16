'''
Step 3: find the paddle in a webcam frame and tell which side faces the camera.

Pure image functions (OpenCV + numpy, no camera, no MediaPipe) so they can be
unit tested on synthetic images. tracker.py feeds them real frames plus the
wrist/elbow positions from MediaPipe Pose.

How it works:
  1. MediaPipe gives the racket arm's wrist, elbow and knuckles (the other arm is
     ignored, so a controller held in the free hand can't be mistaken for the
     paddle). The paddle head sits beyond the wrist along the forearm, so we
     search a circle there (the "wrist gate": red shirts or black hair elsewhere
     in the frame are ignored), plus around where the paddle was last frame, so
     a bent wrist doesn't lose it. The hand itself (and whatever is in the fist:
     handle, motor) is cut out of the search, so only the paddle head counts.
  2. Inside that area, threshold red (HSV) and black (dark, unsaturated) pixels
     and keep the biggest blob of each.
  3. Red wins whenever there is enough of it (red is distinctive, and a dark
     jumper behind the paddle must not outvote it). Black needs a bigger,
     compact blob, because shadows and dark clothes are also black.
     Limitation: a red shirt right behind the paddle reads as red, so avoid red clothing.

Which side faces the camera = which rubber faces the ball = forehand/backhand.

All tunable numbers live in VisionConfig, saved to vision_config.json by
calibrate_vision.py. Without that file the defaults below are used.
'''

import json
import math
import os
from collections import deque
from dataclasses import asdict, dataclass, fields

import cv2
import numpy as np

CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'vision_config.json')

RED_HUE_MAX_HALF_WIDTH = 14     # calibrated red hue range is at most median +-14 (+4 margin)

GAME_X_RANGE = (-0.9, 0.9)
GAME_Y_RANGE = (0.8, -0.1)      # top of the image = high above the table


@dataclass
class VisionConfig:
    # Red: hue range (OpenCV hue 0-179); lo > hi means it wraps around 0, e.g. 170..10
    red_hue_lo: int = 170
    red_hue_hi: int = 10
    red_sat_min: int = 120
    red_val_min: int = 70
    # Black: dark and not very colourful
    black_sat_max: int = 120
    black_val_max: int = 60
    # Which arm holds the racket: 'right' or 'left' (the player's own arm), or 'auto'
    racket_hand: str = 'right'
    # Search area
    hand_exclude: float = 1.0   # size of the cut-out around the hand (0 = off)
    roi_ahead: float = 0.6      # search centre = wrist + roi_ahead * (wrist - elbow)
    roi_radius: float = 0.9     # search radius = roi_radius * forearm length
    roi_min_px: int = 40
    track_radius: float = 0.6   # extra circle around last frame's paddle, x search radius
    # Acceptance
    min_red_frac: float = 0.04      # blob area / search-circle area needed to count
    min_black_frac: float = 0.08
    min_black_solidity: float = 0.8     # area / convex hull: a paddle face is a compact oval
    kernel_px: int = 5              # noise removal; bigger = cleaner but loses small/far paddles
    # Position smoothing (One Euro filter): lower min_cutoff = steadier when still,
    # higher beta = less lag when moving fast
    min_cutoff: float = 1.5
    beta: float = 0.02
    # Player's reachable area in the mirrored image (0..1) -> whole table width/height
    u_lo: float = 0.15
    u_hi: float = 0.85
    v_lo: float = 0.15
    v_hi: float = 0.85
    # Which webcam: 'laptop', 'usb', a camera number, or part of its name (camera_select.py).
    # Chosen with find_camera.py.
    camera: object = 'laptop'
    # Webcam: 0 = automatic exposure; otherwise a manual exposure value (e.g. -6).
    # Shorter exposure = less motion blur on swings but a darker image.
    exposure: int = 0

    def save(self, path=CONFIG_PATH):
        with open(path, 'w') as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path=CONFIG_PATH):
        '''Defaults overridden by whatever the file has; missing file -> defaults.'''
        if not os.path.exists(path):
            return cls()
        with open(path) as f:
            data = json.load(f)
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})


@dataclass
class Hand:
    name: str           # 'left' or 'right' (as seen in the mirrored image)
    wrist: tuple        # (x, y) pixels
    elbow: tuple
    knuckle: tuple = None   # (x, y) pixels: middle of the index/pinky landmarks, if visible


@dataclass
class PaddleObservation:
    side: str           # 'red', 'black', or None if the face is not visible
    center: tuple       # (x, y) pixels: blob centre, or a best guess if no blob
    hand: str
    red_frac: float
    black_frac: float
    roi: tuple          # (cx, cy, radius) of the wrist search circle, for drawing


# ── colour ────────────────────────────────────────────────────────────────────

def hue_mask(hsv, lo, hi, sat_min, val_min):
    if lo <= hi:
        return cv2.inRange(hsv, (lo, sat_min, val_min), (hi, 255, 255))
    return (cv2.inRange(hsv, (lo, sat_min, val_min), (179, 255, 255)) |
            cv2.inRange(hsv, (0, sat_min, val_min), (hi, 255, 255)))


def color_masks(bgr, cfg=None):
    cfg = cfg or VisionConfig()
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    red = hue_mask(hsv, cfg.red_hue_lo, cfg.red_hue_hi, cfg.red_sat_min, cfg.red_val_min)
    black = cv2.inRange(hsv, (0, 0, 0), (179, cfg.black_sat_max, cfg.black_val_max))
    if cfg.kernel_px > 1:
        kernel = np.ones((cfg.kernel_px, cfg.kernel_px), np.uint8)
        red = cv2.morphologyEx(red, cv2.MORPH_OPEN, kernel)
        black = cv2.morphologyEx(black, cv2.MORPH_OPEN, kernel)
    return red, black


def calibrate_red(hsv_pixels, cfg):
    '''Set the red thresholds from sampled HSV pixels (N x 3) of the red rubber.
    Hue is shifted by 90 first so a range that wraps around 0 becomes continuous.'''
    h, s, v = hsv_pixels[:, 0].astype(int), hsv_pixels[:, 1], hsv_pixels[:, 2]
    shifted = (h + 90) % 180
    lo, hi = np.percentile(shifted, [3, 97])
    # Stray non-red pixels in the box (skin, background) must not stretch the range
    # into purple/blue: keep it within RED_HUE_MAX_HALF_WIDTH of the paddle's own hue.
    mid = np.median(shifted)
    lo, hi = max(lo, mid - RED_HUE_MAX_HALF_WIDTH), min(hi, mid + RED_HUE_MAX_HALF_WIDTH)
    margin = 4
    cfg.red_hue_lo = int((lo - margin - 90) % 180)
    cfg.red_hue_hi = int((hi + margin - 90) % 180)
    cfg.red_sat_min = int(max(40, np.percentile(s, 5) - 25))
    cfg.red_val_min = int(max(30, np.percentile(v, 5) - 25))
    return cfg


def calibrate_black(hsv_pixels, cfg):
    '''Set the black thresholds from sampled HSV pixels of the black rubber.'''
    s, v = hsv_pixels[:, 1], hsv_pixels[:, 2]
    cfg.black_val_max = int(min(140, np.percentile(v, 95) + 15))
    cfg.black_sat_max = int(min(255, np.percentile(s, 95) + 25))
    return cfg


def false_positive_rates(bgr_background, cfg):
    '''Fraction of a paddle-free frame that each mask wrongly marks (lower is better).'''
    red, black = color_masks(bgr_background, cfg)
    return np.count_nonzero(red) / red.size, np.count_nonzero(black) / black.size


def _hit_rate(hsv_pixels, cfg, kind):
    '''Fraction of sampled paddle pixels the (un-smoothed) threshold accepts.'''
    img = hsv_pixels.reshape(-1, 1, 3).astype(np.uint8)
    if kind == 'red':
        m = hue_mask(img, cfg.red_hue_lo, cfg.red_hue_hi, cfg.red_sat_min, cfg.red_val_min)
    else:
        m = cv2.inRange(img, (0, 0, 0), (179, cfg.black_sat_max, cfg.black_val_max))
    return np.count_nonzero(m) / m.size


def tighten_red_against_skin(cfg, red_hsv, skin_hsv, target=0.01, keep=0.7):
    '''Make red stricter until (almost) no sampled skin pixel counts as red, while
    still accepting at least `keep` of the paddle's red pixels. Each round takes the
    better of two moves: pull the orange end of the hue range in by 1 (skin is
    orange-red), or raise the saturation floor by 5 (skin is paler than red rubber).'''
    def rates(c):
        return _hit_rate(skin_hsv, c, 'red'), _hit_rate(red_hsv, c, 'red')

    def shrink_hue(c):
        if c.red_hue_lo > c.red_hue_hi:                 # wrapping range, e.g. 165..11
            if c.red_hue_hi == 0:
                c.red_hue_hi = 179                      # becomes 165..179 (no wrap)
            else:
                c.red_hue_hi -= 1
        elif c.red_hue_hi > c.red_hue_lo:
            c.red_hue_hi -= 1
        return c

    def raise_sat(c):
        c.red_sat_min = min(255, c.red_sat_min + 5)
        return c

    skin, _ = rates(cfg)
    while skin > target:
        options = []
        for move in (shrink_hue, raise_sat):
            c = move(VisionConfig(**asdict(cfg)))
            s, r = rates(c)
            if r >= keep and asdict(c) != asdict(cfg):
                options.append((s, -r, c))
        if not options:
            break
        skin, _, best = min(options, key=lambda o: (o[0], o[1]))
        for k, v in asdict(best).items():
            setattr(cfg, k, v)
    return cfg


def tighten_for_background(cfg, red_hsv, black_hsv, bgr_background, target=0.01, keep=0.7):
    '''Make the thresholds stricter until the paddle-free background is (almost)
    never marked, while still accepting at least `keep` of the paddle's own pixels.
    Red: raise the saturation floor. Black: lower the brightness ceiling.'''
    while (false_positive_rates(bgr_background, cfg)[0] > target and cfg.red_sat_min < 250):
        cfg.red_sat_min += 5
        if _hit_rate(red_hsv, cfg, 'red') < keep:
            cfg.red_sat_min -= 5
            break
    while (false_positive_rates(bgr_background, cfg)[1] > target and cfg.black_val_max > 10):
        cfg.black_val_max -= 5
        if _hit_rate(black_hsv, cfg, 'black') < keep:
            cfg.black_val_max += 5
            break
    return cfg


# ── finding the paddle ────────────────────────────────────────────────────────

def search_circle(hand, cfg=None):
    cfg = cfg or VisionConfig()
    wx, wy = hand.wrist
    ex, ey = hand.elbow
    fx, fy = wx - ex, wy - ey
    forearm = float(np.hypot(fx, fy))
    return wx + cfg.roi_ahead * fx, wy + cfg.roi_ahead * fy, max(cfg.roi_min_px, cfg.roi_radius * forearm)


def hand_circle(hand, cfg=None):
    '''(cx, cy, r) covering the hand and anything gripped in it: from the wrist to the
    knuckles, or, without knuckles, the first part of the forearm direction beyond the wrist.'''
    cfg = cfg or VisionConfig()
    wx, wy = hand.wrist
    fx, fy = wx - hand.elbow[0], wy - hand.elbow[1]
    forearm = float(np.hypot(fx, fy))
    if hand.knuckle is not None:
        kx, ky = hand.knuckle
        cx, cy = (wx + kx) / 2, (wy + ky) / 2
        r = max(1.3 * float(np.hypot(kx - wx, ky - wy)), 0.2 * forearm)
    else:
        cx, cy, r = wx + 0.2 * fx, wy + 0.2 * fy, 0.3 * forearm
    return cx, cy, cfg.hand_exclude * r


def biggest_blob(mask):
    '''Returns (area, (cx, cy), solidity) of the largest blob, or None.'''
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    c = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(c)
    if area <= 0:
        return None
    m = cv2.moments(c)
    hull = cv2.contourArea(cv2.convexHull(c))
    return area, (m['m10'] / m['m00'], m['m01'] / m['m00']), area / hull if hull else 0.0


def observe_hand(red_mask, black_mask, hand, cfg=None, last_center=None, last_offset=None):
    '''last_center: where the paddle was last frame (pixels), searched as well.
    last_offset: paddle - wrist last time a face was seen, used as the position
    guess when no face is visible (so the paddle doesn't jump).'''
    cfg = cfg or VisionConfig()
    cx, cy, r = search_circle(hand, cfg)
    gate = np.zeros(red_mask.shape, np.uint8)
    cv2.circle(gate, (int(cx), int(cy)), int(r), 255, -1)
    if last_center is not None:
        cv2.circle(gate, (int(last_center[0]), int(last_center[1])), int(cfg.track_radius * r), 255, -1)
    hx, hy, hr = hand_circle(hand, cfg)
    if hr > 0:
        cv2.circle(gate, (int(hx), int(hy)), int(hr), 0, -1)       # never the hand or what it grips
    circle_area = math.pi * r * r       # fractions are relative to the wrist circle

    red = biggest_blob(red_mask & gate)
    black = biggest_blob(black_mask & gate)
    red_frac = red[0] / circle_area if red else 0.0
    black_frac = black[0] / circle_area if black else 0.0

    if last_offset is not None:
        guess = (hand.wrist[0] + last_offset[0], hand.wrist[1] + last_offset[1])
    else:
        guess = (cx, cy)
    side, center = None, guess
    if red and red_frac >= cfg.min_red_frac:
        side, center = 'red', red[1]
    elif black and black_frac >= cfg.min_black_frac and black[2] >= cfg.min_black_solidity:
        side, center = 'black', black[1]
    return PaddleObservation(side, center, hand.name, red_frac, black_frac, (cx, cy, r))


def find_paddle(bgr, hands, masks=None, cfg=None, last_center=None, last_offset=None, last_hand=None):
    '''Look near each hand and return the best PaddleObservation, or None if no hands.
    Prefers a hand where a paddle face is visible, then the hand that had the
    paddle last frame (so it doesn't flicker between hands), then the most colour.'''
    cfg = cfg or VisionConfig()
    if cfg.racket_hand in ('left', 'right'):
        hands = [h for h in hands if h.name == cfg.racket_hand]
    if not hands:
        return None
    red, black = masks if masks is not None else color_masks(bgr, cfg)
    obs = []
    for h in hands:
        mine = h.name == last_hand
        obs.append(observe_hand(red, black, h, cfg,
                                last_center if mine else None, last_offset if mine else None))
    return max(obs, key=lambda o: (o.side is not None, o.hand == last_hand, max(o.red_frac, o.black_frac)))


# ── turning observations into game input ─────────────────────────────────────

def image_to_game(u, v, cfg=None):
    '''Mirrored image position (0..1) -> paddle (x, y) on the game's hit plane.'''
    cfg = cfg or VisionConfig()

    def lerp(t, src, dst):
        k = (t - src[0]) / (src[1] - src[0])
        k = max(0.0, min(1.0, k))
        return dst[0] + k * (dst[1] - dst[0])
    return lerp(u, (cfg.u_lo, cfg.u_hi), GAME_X_RANGE), lerp(v, (cfg.v_lo, cfg.v_hi), GAME_Y_RANGE)


def stroke_for_side(side, forehand_color='red'):
    if side is None:
        return None
    return 'forehand' if side == forehand_color else 'backhand'


class SideFilter:
    '''Debounce: the reported side only changes after it is seen in most of the
    last `window` frames, so a single bad frame can't flip forehand/backhand.'''

    def __init__(self, window=5, needed=3):
        self.needed = needed
        self.history = deque(maxlen=window)
        self.side = None

    def update(self, side):
        self.history.append(side)
        for candidate in ('red', 'black'):
            if candidate != self.side and self.history.count(candidate) >= self.needed:
                self.side = candidate
        if self.history.count(None) == len(self.history) == self.history.maxlen:
            self.side = None            # paddle face gone for the whole window
        return self.side


class OneEuroFilter:
    '''Adaptive smoothing for a 2D point (Casiez et al. 2012): heavy smoothing when
    the paddle is nearly still (kills jitter), light smoothing when it moves fast
    (kills lag). Works in whatever units it is fed, with timestamps in seconds.'''

    def __init__(self, min_cutoff=1.5, beta=0.02, d_cutoff=1.0):
        self.min_cutoff, self.beta, self.d_cutoff = min_cutoff, beta, d_cutoff
        self.value = None
        self._speed = (0.0, 0.0)
        self._t = None

    @staticmethod
    def _alpha(cutoff, dt):
        tau = 1.0 / (2 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def update(self, xy, t):
        if self.value is None or self._t is None or t <= self._t:
            self.value, self._t = tuple(xy), t
            return self.value
        dt = t - self._t
        self._t = t
        raw_speed = ((xy[0] - self.value[0]) / dt, (xy[1] - self.value[1]) / dt)
        a_d = self._alpha(self.d_cutoff, dt)
        self._speed = tuple(a_d * r + (1 - a_d) * p for r, p in zip(raw_speed, self._speed))
        cutoff = self.min_cutoff + self.beta * math.hypot(*self._speed)
        a = self._alpha(cutoff, dt)
        self.value = tuple(a * x + (1 - a) * p for x, p in zip(xy, self.value))
        return self.value
