'''
Fast paddle tracking: red rubber only, x position only, no MediaPipe.

Why: pose detection cost ~12 ms/frame and competed with the game for the CPU,
which made tracking feel slow and clunky. A red colour mask on a 320x240 frame
costs ~0.4 ms, so tracking now keeps up with the webcam (30 fps) easily.

    tracker = RedTracker(camera_index=0)
    tracker.start()
    state = tracker.get()     # RedState: paddle_x (metres on the table), seen, fps, frame
    tracker.stop()

Red thresholds come from vision_config.json (calibrate_vision.py), so the red
calibration done earlier still applies. RedTrackerCore.step() is pure (frame in,
state out) so it can be unit tested without a camera.
'''

import threading
import time
from dataclasses import dataclass, field

import cv2
import numpy as np

from paddle_vision import (GAME_X_RANGE, OneEuroFilter, SideFilter, VisionConfig,
                           biggest_blob, hue_mask)
from pose_logic import (Presence, StartGesture, hand_cutout, racket_gate,
                        skeleton_lines)

PROC_W, PROC_H = 320, 240       # processing size: 4x fewer pixels than 640x480
MIN_AREA_PX = 60                # red blob must be at least this big (at 320x240)
WINDOW_PX = 90                  # search near last position first (cheaper, ignores clutter)
LOST_AFTER_S = 0.4
POSE_FRESH_S = 0.5              # pose older than this is ignored
LEAD_S = 0.0                    # predict ahead (s); 0: the filter already has ~no lag, more overshoots
MIN_BLACK_AREA_PX = 150         # black face must be bigger than a red one (shadows/hair are black)
FOREARM_CUT = 0.45              # forearm cut-out thickness, x forearm length (dark sleeves)


@dataclass
class RedState:
    paddle_x: float = None      # game x (metres), None before first sighting
    seen: bool = False
    u: float = None             # raw mirrored image position 0..1 (for display/tests)
    fps: float = 0.0
    proc_ms: float = 0.0        # processing time of the last frame
    pose: object = None         # latest fresh PoseInfo (MediaPipe), or None
    pose_hz: float = 0.0        # how often pose results arrive
    present: bool = None        # player in view (None when pose is off)
    gate: tuple = None          # (u, v, half) racket-arm search box in use, or None
    start_count: int = 0        # number of 'both hands up' start gestures so far
    side: str = None            # paddle face toward the camera: 'red', 'black' or None (debounced)
    backhand: bool = False      # black side showing = backhand
    tags: list = None           # level_tags.TagSightings while watch_tags is on, else None
    frame: np.ndarray = field(default=None, repr=False)    # small RGB frame with overlay
    frame_id: int = 0


def u_to_game_x(u, cfg, x_range=GAME_X_RANGE):
    '''Webcam play area (u_lo..u_hi) -> game x over x_range (the whole table by default;
    the forehand drill passes just the forehand zone, for finer, steadier control).'''
    k = (u - cfg.u_lo) / (cfg.u_hi - cfg.u_lo)
    k = max(0.0, min(1.0, k))
    return x_range[0] + k * (x_range[1] - x_range[0])


def red_mask(small_bgr, cfg, hsv=None):
    hsv = cv2.cvtColor(small_bgr, cv2.COLOR_BGR2HSV) if hsv is None else hsv
    mask = hue_mask(hsv, cfg.red_hue_lo, cfg.red_hue_hi, cfg.red_sat_min, cfg.red_val_min)
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))


def black_face(hsv_crop, cfg, x0):
    '''(centre x, area) of a compact black blob (the black rubber) in a crop, or None.'''
    mask = cv2.inRange(hsv_crop, (0, 0, 0), (179, cfg.black_sat_max, cfg.black_val_max))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    blob = biggest_blob(mask)
    if blob is None:
        return None
    area, (cx, _), solidity = blob
    if area < MIN_BLACK_AREA_PX or solidity < cfg.min_black_solidity:
        return None
    return x0 + cx, area


def forearm_segment(pose, racket_hand):
    '''Elbow and wrist in pixels, plus forearm length, or None.'''
    if pose is None:
        return None
    w, e = pose.get(f'{racket_hand}_wrist'), pose.get(f'{racket_hand}_elbow')
    if w is None or e is None:
        return None
    wp = (int(w[0] * PROC_W), int(w[1] * PROC_H))
    ep = (int(e[0] * PROC_W), int(e[1] * PROC_H))
    return ep, wp, float(np.hypot(wp[0] - ep[0], wp[1] - ep[1]))


def largest_blob_x(mask, x0=0):
    '''(centre x, area) of the biggest red blob, or None. x0 = offset of a cropped mask.'''
    n, _, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n <= 1:
        return None
    i = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    area = int(stats[i, cv2.CC_STAT_AREA])
    if area < MIN_AREA_PX:
        return None
    return x0 + float(centroids[i][0]), area


class RedTrackerCore:

    def __init__(self, cfg=None, x_range=GAME_X_RANGE):
        self.cfg = cfg or VisionConfig()
        self.x_range = x_range
        self.filter = OneEuroFilter(min_cutoff=2.0, beta=0.05)
        self.state = RedState()
        self._last_px = None
        self._last_seen_t = -1e9
        self.presence = Presence()
        self.gesture = StartGesture()
        self.sides = SideFilter(window=5, needed=3)

    def step(self, mirrored_bgr, t, draw=True, pose=None, pose_enabled=False):
        '''pose: latest PoseInfo from AsyncPose (may be a few frames old), or None.
        pose_enabled: pose is running, so presence is meaningful.'''
        t_start = time.perf_counter()
        small = cv2.resize(mirrored_bgr, (PROC_W, PROC_H), interpolation=cv2.INTER_AREA)
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        mask = red_mask(small, self.cfg, hsv)

        fresh_pose = pose if pose is not None and t - pose.t < POSE_FRESH_S else None
        gate = racket_gate(fresh_pose, self.cfg.racket_hand)
        cutout = hand_cutout(fresh_pose, self.cfg.racket_hand, PROC_W, PROC_H)
        if cutout is not None:                  # the hand is never the paddle, even if skin looks red
            cx, cy, r = cutout
            cv2.circle(mask, (int(cx), int(cy)), int(r), 0, -1)

        found = None
        face = None
        if gate is not None:                    # pose known: only near the racket arm
            gu, gv, half = gate
            x0, x1 = max(0, int((gu - half) * PROC_W)), min(PROC_W, int((gu + half) * PROC_W))
            y0, y1 = max(0, int((gv - half) * PROC_H)), min(PROC_H, int((gv + half) * PROC_H))
            if x1 > x0 and y1 > y0:
                found = largest_blob_x(mask[y0:y1, x0:x1], x0)
                if found is not None:
                    face = 'red'
                else:                           # no red: is the black side (backhand) showing?
                    crop = hsv[y0:y1, x0:x1].copy()
                    if cutout is not None:      # never the hand...
                        cv2.circle(crop, (int(cutout[0]) - x0, int(cutout[1]) - y0), int(cutout[2]),
                                   (0, 0, 255), -1)
                    arm = forearm_segment(fresh_pose, self.cfg.racket_hand)
                    if arm is not None:         # ...or a dark sleeve on the forearm
                        (ex, ey), (wx, wy), length = arm
                        cv2.line(crop, (ex - x0, ey - y0), (wx - x0, wy - y0), (0, 0, 255),
                                 max(4, int(FOREARM_CUT * length)))
                    found = black_face(crop, self.cfg, x0)
                    if found is not None:
                        face = 'black'
        else:
            if self._last_px is not None:       # 1) near where the paddle was
                x0 = max(0, int(self._last_px) - WINDOW_PX)
                x1 = min(PROC_W, int(self._last_px) + WINDOW_PX)
                found = largest_blob_x(mask[:, x0:x1], x0)
            if found is None:                   # 2) anywhere in the frame
                found = largest_blob_x(mask)
            if found is not None:
                face = 'red'                    # without pose only red is trusted

        s = self.state
        if found is not None:
            px, _ = found
            self._last_px = px
            self._last_seen_t = t
            fx, _ = self.filter.update((px, 0.0), t)
            speed = self.filter._speed[0]         # px/s, already smoothed by the filter
            u = (fx + speed * LEAD_S) / PROC_W
            s.u = min(1.0, max(0.0, u))
            s.paddle_x = u_to_game_x(s.u, self.cfg, self.x_range)
        s.side = self.sides.update(face)
        s.backhand = s.side == 'black'
        s.pose = fresh_pose
        s.gate = gate
        s.present = self.presence.update(fresh_pose, t) if pose_enabled else None
        if self.gesture.update(fresh_pose, t):
            s.start_count += 1
        s.seen = t - self._last_seen_t < LOST_AFTER_S
        if not s.seen:
            self._last_px = None
        if draw:
            s.frame = self._overlay(small, mask, found, s.pose, gate, cutout)
            s.frame_id += 1
        s.proc_ms = (time.perf_counter() - t_start) * 1000
        return s

    def _overlay(self, small, mask, found, pose=None, gate=None, cutout=None):
        view = small.copy()
        view[mask > 0] = (0.5 * view[mask > 0] + (0, 0, 127)).astype(np.uint8)    # tint red pixels
        if pose is not None:                                                     # MediaPipe skeleton
            for a, b in skeleton_lines(pose, PROC_W, PROC_H):
                cv2.line(view, a, b, (90, 255, 120), 2)
            for u, v in pose.points.values():
                cv2.circle(view, (int(u * PROC_W), int(v * PROC_H)), 3, (255, 255, 255), -1)
        if cutout is not None:                                                   # ignored: the hand
            cv2.circle(view, (int(cutout[0]), int(cutout[1])), int(cutout[2]), (200, 200, 200), 1)
        if gate is not None:                                                     # racket-arm search box
            gu, gv, half = gate
            cv2.rectangle(view, (int((gu - half) * PROC_W), int((gv - half) * PROC_H)),
                          (int((gu + half) * PROC_W), int((gv + half) * PROC_H)), (0, 220, 255), 1)
        if found is not None:
            x = int(found[0])
            cv2.line(view, (x, 0), (x, PROC_H), (60, 60, 60) if self.state.backhand else (255, 255, 255), 2)
        for u in (self.cfg.u_lo, self.cfg.u_hi):                                  # play-area edges
            cv2.line(view, (int(u * PROC_W), 0), (int(u * PROC_W), PROC_H), (0, 220, 255), 1)
        return cv2.cvtColor(view, cv2.COLOR_BGR2RGB)


def open_fast_camera(index, cfg):
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError(f'Could not open camera {index}')
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 60)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)         # always the newest frame, not a queued old one
    if cfg.exposure:
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
        cap.set(cv2.CAP_PROP_EXPOSURE, cfg.exposure)
    return cap


class RedTracker:

    def __init__(self, camera_index=None, cfg=None, use_pose=True, x_range=GAME_X_RANGE):
        '''camera_index: 'laptop' / 'usb' / a number / part of a name; None = the camera saved
        in vision_config.json (find_camera.py). Resolved to a device number here.'''
        from camera_select import pick
        self.use_pose = use_pose
        self.pose = None
        self.core = RedTrackerCore(cfg or VisionConfig.load(), x_range)
        self.camera_index = pick(self.core.cfg.camera if camera_index is None else camera_index)
        self._lock = threading.Lock()
        self._state = RedState()
        self._running = False
        self._error = None
        self._thread = None
        self.tag_detector = None    # a level_tags.LevelTagDetector: also read AprilTags...
        self.watch_tags = False     # ...but only while this is on (start / results screens)

    def start(self):
        self._cap = open_fast_camera(self.camera_index, self.core.cfg)
        if self.use_pose:
            from pose_async import AsyncPose
            self.pose = AsyncPose()
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def get(self):
        with self._lock:
            if self._error:
                raise RuntimeError('tracking thread crashed') from self._error
            return self._state

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)
        self._cap.release()
        if self.pose:
            self.pose.close()

    def _loop(self):
        last = last_pose_t = time.perf_counter()
        fps = pose_hz = 0.0
        last_n = 0
        try:
            while self._running:
                ok, raw = self._cap.read()
                if not ok:
                    time.sleep(0.005)
                    continue
                now = time.perf_counter()
                small = cv2.resize(raw, (PROC_W, PROC_H), interpolation=cv2.INTER_AREA)
                latest_pose = None
                if self.pose:
                    self.pose.submit(small, now)    # returns at once; MediaPipe works in its own thread
                    latest_pose = self.pose.latest()
                state = self.core.step(cv2.flip(small, 1), now, pose=latest_pose,
                                       pose_enabled=self.pose is not None)
                fps = 0.9 * fps + 0.1 / max(now - last, 1e-3)
                last = now
                state.fps = fps
                state.tags = None
                if self.watch_tags and self.tag_detector:   # full-size frame: tags readable further away
                    from level_tags import draw_on_preview
                    state.tags = self.tag_detector.detect(raw)
                    if state.frame is not None:
                        draw_on_preview(state.frame, state.tags, raw.shape[1])
                if self.pose:
                    n = self.pose.result_count()
                    pose_hz = 0.9 * pose_hz + 0.1 * (n - last_n) / max(now - last_pose_t, 1e-3)
                    last_n, last_pose_t = n, now
                    state.pose_hz = pose_hz
                snapshot = RedState(**vars(state))
                with self._lock:
                    self._state = snapshot
        except Exception as e:
            with self._lock:
                self._error = e
