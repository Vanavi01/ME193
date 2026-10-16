'''
AprilTags choose the difficulty: show a tag to the webcam and its level is picked.

No training is needed: every AprilTag's pattern encodes an ID number, which OpenCV's
built-in detector (cv2.aruco, family tag36h11, the same family as the AprilTags
project) reads directly from the frame. The only thing to set up is which ID means
which level, kept in level_tags.json:

    {"0": 1, "1": 2, "2": 3}      tag 0 = Normal, tag 1 = Hard, tag 2 = Pro

live_level_tags.py can change that by holding a tag up and pressing 1/2/3, and
`python live_level_tags.py --make-tags` writes printable tags 0-2.

A level only counts once its tag has been seen in STABLE_FRAMES frames in a row, so
a tag glimpsed for one frame (or a misread) never switches the level.
'''

import json
import os
from dataclasses import dataclass

import cv2
import numpy as np

from game import LEVELS

TAG_FAMILY = cv2.aruco.DICT_APRILTAG_36h11
TAGS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'level_tags.json')

LEVEL_NAMES = {n: lv.name for n, lv in LEVELS.items()}   # 1 Normal, 2 Hard, 3 Pro
DEFAULT_TAGS = {0: 1, 1: 2, 2: 3}                    # tag ID -> level
STABLE_FRAMES = 5                                    # ~0.2 s at 30 fps


@dataclass
class TagSighting:
    tag_id: int
    corners: np.ndarray     # 4x2 pixel corners
    level: int | None       # None = a tag with no level assigned

    @property
    def label(self):
        return LEVEL_NAMES[self.level] if self.level else f'tag {self.tag_id} (no level)'

    @property
    def center(self):
        return self.corners.mean(axis=0)

    @property
    def size_px(self):
        return float(np.linalg.norm(self.corners[0] - self.corners[2]))


def load_tags(path=TAGS_PATH):
    '''Tag ID -> level from level_tags.json (the defaults if it doesn't exist).'''
    try:
        with open(path) as f:
            return {int(k): int(v) for k, v in json.load(f).items()}
    except FileNotFoundError:
        return dict(DEFAULT_TAGS)


def save_tags(tags, path=TAGS_PATH):
    with open(path, 'w') as f:
        json.dump({str(k): v for k, v in sorted(tags.items())}, f, indent=2)


def assign(tags, tag_id, level):
    '''Give tag_id the level; any other tag that had that level loses it, so each level
    has exactly one tag.'''
    out = {t: lv for t, lv in tags.items() if lv != level and t != tag_id}
    out[tag_id] = level
    return out


class LevelTagDetector:

    def __init__(self, tags=None):
        self.tags = load_tags() if tags is None else tags
        params = cv2.aruco.DetectorParameters()
        self._detector = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(TAG_FAMILY), params)

    def detect(self, frame_bgr):
        '''All tags in the frame, biggest (closest) first.'''
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY) if frame_bgr.ndim == 3 else frame_bgr
        corners, ids, _ = self._detector.detectMarkers(gray)
        if ids is None:
            return []
        found = [TagSighting(int(i), c.reshape(4, 2), self.tags.get(int(i)))
                 for c, i in zip(corners, ids.flatten())]
        return sorted(found, key=lambda s: -s.size_px)


class LevelChooser:
    '''Turns per-frame sightings into a level choice that only changes after the same
    level tag has been seen STABLE_FRAMES frames in a row.'''

    def __init__(self, stable_frames=STABLE_FRAMES):
        self.stable_frames = stable_frames
        self.level = None           # the chosen level (stays after the tag is put away)
        self._candidate = None
        self._count = 0

    def update(self, sightings):
        '''Feed one frame's sightings; returns the new level if it just changed, else None.'''
        levels = [s.level for s in sightings if s.level]
        seen = levels[0] if levels else None        # the biggest tag with a level wins
        if seen is None or seen != self._candidate:
            self._candidate, self._count = seen, (1 if seen else 0)
        else:
            self._count += 1
        if seen and self._count >= self.stable_frames and seen != self.level:
            self.level = seen
            return seen
        return None

    @property
    def progress(self):
        '''0..1 how far the current tag is towards changing the level (0 if it wouldn't).'''
        if self._candidate is None or self._candidate == self.level:
            return 0.0
        return min(1.0, self._count / self.stable_frames)


def choose_level(game, chooser, sightings):
    '''Feed one frame's tags; sets the game's level when a steady tag asks for a different
    one. Returns the new level, else None. Synced to game.level_num first, so a tag still
    works after the level was changed with the 1/2/3 keys.'''
    chooser.level = game.level_num
    new = chooser.update(sightings)
    if new:
        game.set_level(new)
    return new


def draw_on_preview(view_rgb, sightings, src_w):
    '''Outline tags (found in the full-size, unmirrored frame src_w wide) on the small,
    mirrored webcam preview; green = a level tag, grey = no level.'''
    h, w = view_rgb.shape[:2]
    k = w / src_w
    for s in sightings:
        pts = np.column_stack([w - s.corners[:, 0] * k, s.corners[:, 1] * k]).astype(np.int32)
        color = (0, 220, 0) if s.level else (160, 160, 160)
        cv2.polylines(view_rgb, [pts], True, color, 2)
        if s.level:
            cv2.putText(view_rgb, LEVEL_NAMES[s.level].upper(), (int(pts[:, 0].min()), max(12, int(pts[:, 1].min()) - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return view_rgb


def tag_image(tag_id, px=400, border=1):
    '''A printable tag: black-and-white, with a white quiet zone (needed for detection).'''
    dictionary = cv2.aruco.getPredefinedDictionary(TAG_FAMILY)
    cell = px // 10                                  # 36h11 = 8x8 cells with its black border
    img = cv2.aruco.generateImageMarker(dictionary, tag_id, cell * 8)
    return cv2.copyMakeBorder(img, cell * (1 + border), cell * (1 + border),
                              cell * (1 + border), cell * (1 + border),
                              cv2.BORDER_CONSTANT, value=255)
