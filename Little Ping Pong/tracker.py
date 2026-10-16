'''
Step 3: webcam + MediaPipe Pose + paddle detection, running in a background
thread so the game keeps its 60 fps.

    tracker = VisionTracker(camera_index=0)
    tracker.start()
    state = tracker.get()     # latest TrackerState: paddle position, side, stroke, frame
    tracker.stop()

The per-frame logic lives in TrackerCore.step(), which takes a frame and the
hands MediaPipe found, so it can be unit tested without a camera or a model.
Settings come from vision_config.json (made by calibrate_vision.py).
'''

import os
import threading
import time
from dataclasses import dataclass, field

import cv2
import numpy as np

from paddle_vision import (Hand, OneEuroFilter, SideFilter, VisionConfig,
                           color_masks, find_paddle, hand_circle,
                           image_to_game, stroke_for_side)

MODEL = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models', 'pose_landmarker_lite.task')
LEFT_ELBOW, RIGHT_ELBOW, LEFT_WRIST, RIGHT_WRIST = 13, 14, 15, 16
LEFT_PINKY, RIGHT_PINKY, LEFT_INDEX, RIGHT_INDEX = 17, 18, 19, 20
MIN_VISIBILITY = 0.5
LOST_AFTER_S = 0.5      # paddle counts as lost after this long without hands


@dataclass
class TrackerState:
    paddle: tuple = None        # (x, y) on the game's hit plane, or None before first sighting
    side: str = None            # 'red' / 'black' / None (debounced)
    stroke: str = None          # 'forehand' / 'backhand' / None
    hand: str = None            # 'left' / 'right'
    seen: bool = False          # hands found recently
    fps: float = 0.0
    red_frac: float = 0.0       # raw detection strength this frame (for tuning)
    black_frac: float = 0.0
    frame: np.ndarray = field(default=None, repr=False)    # RGB frame with overlay, for display
    frame_id: int = 0
    masks: tuple = field(default=None, repr=False)         # (red, black) masks, for tuning tools


def hands_from_landmarks(landmarks, width, height, mirror=False):
    '''MediaPipe pose landmarks (normalised) -> list of Hand in pixels, skipping hidden arms.
    Names are the player's own arms. mirror=True: landmarks came from the un-mirrored
    camera image and are converted to the mirrored image the game shows.'''
    def px(lm):
        x = 1.0 - lm.x if mirror else lm.x
        return int(x * width), int(lm.y * height)

    def visible(*lms):
        return min(getattr(lm, 'visibility', 1.0) for lm in lms) >= MIN_VISIBILITY

    hands = []
    for name, wrist_i, elbow_i, index_i, pinky_i in (
            ('left', LEFT_WRIST, LEFT_ELBOW, LEFT_INDEX, LEFT_PINKY),
            ('right', RIGHT_WRIST, RIGHT_ELBOW, RIGHT_INDEX, RIGHT_PINKY)):
        w, e = landmarks[wrist_i], landmarks[elbow_i]
        if not visible(w, e):
            continue
        knuckle = None
        if len(landmarks) > max(index_i, pinky_i) and visible(landmarks[index_i], landmarks[pinky_i]):
            (ix, iy), (qx, qy) = px(landmarks[index_i]), px(landmarks[pinky_i])
            knuckle = ((ix + qx) // 2, (iy + qy) // 2)
        hands.append(Hand(name, px(w), px(e), knuckle))
    return hands


class TrackerCore:

    def __init__(self, forehand_color='red', cfg=None):
        self.forehand_color = forehand_color
        self.cfg = cfg or VisionConfig()
        self.sides = SideFilter()
        self.smoother = OneEuroFilter(self.cfg.min_cutoff, self.cfg.beta)
        self.state = TrackerState()
        self._last_seen_t = -1e9
        self._last_center = None    # paddle position last frame (pixels)
        self._last_offset = None    # paddle - wrist, last time a face was visible
        self._last_hand = None

    def set_config(self, cfg):
        '''Apply new settings live (used by the calibration tool's sliders).'''
        self.cfg = cfg
        self.smoother.min_cutoff, self.smoother.beta = cfg.min_cutoff, cfg.beta

    def step(self, bgr, hands, t, keep_masks=False):
        '''Process one (already mirrored) frame. Returns the new TrackerState.'''
        h, w = bgr.shape[:2]
        masks = color_masks(bgr, self.cfg)
        obs = find_paddle(bgr, hands, masks, self.cfg, self._last_center, self._last_offset, self._last_hand)
        s = self.state

        if obs is not None:
            self._last_seen_t = t
            if obs.side:
                wrist = next(hd.wrist for hd in hands if hd.name == obs.hand)
                self._last_offset = (obs.center[0] - wrist[0], obs.center[1] - wrist[1])
            elif obs.hand != self._last_hand:
                self._last_offset = None
            self._last_center = obs.center
            self._last_hand = obs.hand
            px, py = self.smoother.update(obs.center, t)
            s.paddle = image_to_game(px / w, py / h, self.cfg)
            s.hand = obs.hand
            s.side = self.sides.update(obs.side)
            s.red_frac, s.black_frac = obs.red_frac, obs.black_frac
        else:
            s.side = self.sides.update(None)
            s.red_frac = s.black_frac = 0.0
        s.seen = t - self._last_seen_t < LOST_AFTER_S
        if not s.seen:
            self._last_center = None    # player left: start fresh when they come back
        s.stroke = stroke_for_side(s.side, self.forehand_color)
        s.masks = masks if keep_masks else None
        s.frame = cv2.cvtColor(draw_overlay(bgr.copy(), hands, obs, s, self.cfg), cv2.COLOR_BGR2RGB)
        s.frame_id += 1
        return s


def draw_overlay(img, hands, obs, state, cfg=None):
    cfg = cfg or VisionConfig()
    for hand in hands:
        racket = cfg.racket_hand in ('auto', hand.name)
        color = (255, 255, 255) if racket else (120, 120, 120)
        cv2.line(img, hand.elbow, hand.wrist, color, 2 if racket else 1)
        cv2.circle(img, hand.wrist, 6 if racket else 4, color, -1)
        if racket:
            hx, hy, hr = hand_circle(hand, cfg)
            if hr > 0:
                cv2.circle(img, (int(hx), int(hy)), int(hr), (150, 150, 150), 1)   # ignored: the hand
            cv2.putText(img, 'racket', (hand.elbow[0] + 8, hand.elbow[1]), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 255, 255), 1)
    if obs is not None:
        cx, cy, r = obs.roi
        cv2.circle(img, (int(cx), int(cy)), int(r), (0, 220, 255), 2)
        color = {'red': (40, 40, 230), 'black': (30, 30, 30)}.get(obs.side, (160, 160, 160))
        cv2.circle(img, (int(obs.center[0]), int(obs.center[1])), 10, color, -1)
        cv2.circle(img, (int(obs.center[0]), int(obs.center[1])), 10, (255, 255, 255), 2)
    label = state.stroke.upper() if state.stroke else ('NO PADDLE FACE' if state.seen else 'NO PLAYER')
    cv2.putText(img, label, (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 4)
    cv2.putText(img, label, (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    return img


# ── camera + pose model (shared by the tracker thread and calibrate_vision.py) ──

def open_camera(index, cfg):
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError(f'Could not open camera {index}')
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    apply_exposure(cap, cfg.exposure)
    return cap


def apply_exposure(cap, exposure):
    '''0 = automatic. Otherwise manual exposure (DirectShow units: -1 long ... -10 very short).
    Not every webcam honours this.'''
    if exposure == 0:
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)
    else:
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
        cap.set(cv2.CAP_PROP_EXPOSURE, exposure)


class PoseRunner:
    '''MediaPipe Pose in VIDEO mode: frame in, list of Hand out.'''

    def __init__(self):
        import mediapipe as mp                          # imported here so tests don't need it
        from mediapipe.tasks.python import BaseOptions, vision
        self._mp = mp
        options = vision.PoseLandmarkerOptions(base_options=BaseOptions(model_asset_path=MODEL),
                                               running_mode=vision.RunningMode.VIDEO, num_poses=1)
        self._landmarker = vision.PoseLandmarker.create_from_options(options)
        self._t0 = time.perf_counter()
        self._last_ms = -1

    def hands(self, raw_bgr):
        '''raw_bgr: the camera frame BEFORE mirroring. Pose runs on the real image so
        MediaPipe's left/right are the player's own arms (on a mirrored image they swap);
        the returned pixel positions are for the mirrored frame.'''
        rgb = cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2RGB)
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        ms = max(self._last_ms + 1, int((time.perf_counter() - self._t0) * 1000))
        self._last_ms = ms
        result = self._landmarker.detect_for_video(image, ms)
        h, w = raw_bgr.shape[:2]
        return hands_from_landmarks(result.pose_landmarks[0], w, h, mirror=True) if result.pose_landmarks else []

    def close(self):
        self._landmarker.close()


class VisionTracker:

    def __init__(self, camera_index=0, forehand_color='red', keep_masks=False, cfg=None, racket_hand=None):
        self.camera_index = camera_index
        self.keep_masks = keep_masks
        self.cfg = cfg or VisionConfig.load()
        if racket_hand:
            self.cfg.racket_hand = racket_hand
        self.core = TrackerCore(forehand_color, self.cfg)
        self._lock = threading.Lock()
        self._state = TrackerState()
        self._running = False
        self._thread = None
        self._error = None

    def start(self):
        self._cap = open_camera(self.camera_index, self.cfg)
        self._pose = PoseRunner()
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def get(self):
        with self._lock:
            if self._error:
                raise RuntimeError('vision thread crashed') from self._error
            return self._state

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)
        self._cap.release()
        self._pose.close()

    def _loop(self):
        last = time.perf_counter()
        fps = 0.0
        try:
            while self._running:
                ok, raw = self._cap.read()
                if not ok:
                    time.sleep(0.01)
                    continue
                hands = self._pose.hands(raw)
                frame = cv2.flip(raw, 1)                # mirror: move right -> paddle moves right
                now = time.perf_counter()
                state = self.core.step(frame, hands, now, keep_masks=self.keep_masks)
                fps = 0.9 * fps + 0.1 / max(now - last, 1e-3)
                last = now
                state.fps = fps
                snapshot = TrackerState(**vars(state))  # copy so the game never sees a half-written state
                with self._lock:
                    self._state = snapshot
        except Exception as e:                          # surface crashes to the game thread
            with self._lock:
                self._error = e
