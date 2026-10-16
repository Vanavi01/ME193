'''Skin read as red made the paddle jump between the hand and the paddle.
Fixes: (1) calibration learns skin and tightens red, (2) pose cuts the racket hand out.'''

import unittest

import cv2
import numpy as np

from paddle_vision import VisionConfig, _hit_rate, tighten_red_against_skin
from pose_logic import hand_cutout, pose_from_landmarks
from red_tracker import PROC_H, PROC_W, RedTrackerCore
from tests.test_pose_logic import fake_landmarks
from tests.test_red_tracker import H, W

# The user's saved red settings when the bug happened
USER_CFG = dict(red_hue_lo=165, red_hue_hi=11, red_sat_min=70, red_val_min=84)
RED = (40, 30, 200)         # red rubber, BGR  (HSV ~178, 217, 200)
SKIN = (120, 150, 205)      # medium skin, BGR (HSV ~11, 106, 205) -> inside the user's red range
WALL = (170, 175, 165)

# Pixel layout (640x480, mirrored): racket arm up on the right, paddle above the hand.
WRIST, KNUCKLE, ELBOW, PADDLE = (430, 330), (436, 282), (410, 420), (448, 220)


def hsv_cloud(h, s, v, n=300, seed=0):
    rng = np.random.default_rng(seed)
    return np.stack([(h + rng.normal(0, 1.5, n)) % 180,
                     np.clip(s + rng.normal(0, 20, n), 0, 255),
                     np.clip(v + rng.normal(0, 20, n), 0, 255)], 1).astype(np.uint8)


def scene(tilted_paddle=True, hand=True):
    img = np.full((H, W, 3), WALL, np.uint8)
    axes = (40, 10) if tilted_paddle else (40, 48)          # tilted: thin sliver of red
    cv2.ellipse(img, PADDLE, axes, 0, 0, 360, RED, -1)
    if hand:
        mid = ((WRIST[0] + KNUCKLE[0]) // 2, (WRIST[1] + KNUCKLE[1]) // 2)
        cv2.ellipse(img, mid, (20, 26), 0, 0, 360, SKIN, -1)
    return img


def arm_pose(t=0.0, knuckles=True):
    def raw(p):                                             # mirrored pixels -> raw normalised
        return (1 - p[0] / W, p[1] / H)
    o = {'right_wrist': raw(WRIST), 'right_elbow': raw(ELBOW)}
    if knuckles:
        o['right_index'] = raw((KNUCKLE[0] + 6, KNUCKLE[1]))
        o['right_pinky'] = raw((KNUCKLE[0] - 6, KNUCKLE[1] + 4))
    lms = fake_landmarks(o)
    from pose_logic import LANDMARKS
    from types import SimpleNamespace
    for name in ('right_index', 'right_pinky'):
        if name in o:
            x, y = o[name]
            lms[LANDMARKS[name]] = SimpleNamespace(x=x, y=y, visibility=0.95)
    return pose_from_landmarks(lms, t)


class TestCalibrationLearnsSkin(unittest.TestCase):

    def test_users_settings_count_skin_as_red(self):
        cfg = VisionConfig(**USER_CFG)
        self.assertGreater(_hit_rate(hsv_cloud(11, 106, 205), cfg, 'red'), 0.5)

    def test_skin_step_excludes_skin_and_keeps_the_paddle(self):
        cfg = VisionConfig(**USER_CFG)
        red = hsv_cloud(178, 200, 190)
        skins = np.vstack([hsv_cloud(9, 71, 235, seed=1), hsv_cloud(11, 106, 205, seed=2),
                           hsv_cloud(11, 120, 180, seed=3), hsv_cloud(10, 128, 120, seed=4)])
        tighten_red_against_skin(cfg, red, skins)
        self.assertLess(_hit_rate(skins, cfg, 'red'), 0.02)
        self.assertGreater(_hit_rate(red, cfg, 'red'), 0.7)

    def test_already_good_settings_are_left_alone(self):
        cfg = VisionConfig(red_hue_lo=170, red_hue_hi=3, red_sat_min=140, red_val_min=80)
        before = dict(vars(cfg))
        tighten_red_against_skin(cfg, hsv_cloud(178, 200, 190), hsv_cloud(11, 106, 205))
        self.assertEqual(vars(cfg), before)


class TestHandCutout(unittest.TestCase):

    def test_covers_wrist_to_knuckles(self):
        x, y, r = hand_cutout(arm_pose(), 'right', PROC_W, PROC_H)
        s = PROC_W / W
        self.assertAlmostEqual(x, (WRIST[0] + KNUCKLE[0]) / 2 * s, delta=2)
        self.assertAlmostEqual(y, (WRIST[1] + KNUCKLE[1]) / 2 * s, delta=2)
        self.assertGreater(r, 26 * s)                       # covers the drawn hand

    def test_does_not_reach_the_paddle(self):
        x, y, r = hand_cutout(arm_pose(), 'right', PROC_W, PROC_H)
        s = PROC_W / W
        nearest_paddle_y = (PADDLE[1] + 10) * s
        self.assertLess(r, y - nearest_paddle_y)

    def test_falls_back_to_forearm_without_knuckles_and_none_without_wrist(self):
        self.assertIsNotNone(hand_cutout(arm_pose(knuckles=False), 'right', PROC_W, PROC_H))
        self.assertIsNone(hand_cutout(None, 'right', PROC_W, PROC_H))
        self.assertIsNone(hand_cutout(arm_pose(), 'left', PROC_W, PROC_H)
                          if False else hand_cutout(pose_from_landmarks(fake_landmarks(hidden=('left_wrist',)), 0),
                                                    'left', PROC_W, PROC_H))


class TestNoJumpingToTheHand(unittest.TestCase):

    def track(self, cfg, pose_fn, frames=20, **scene_kw):
        core = RedTrackerCore(cfg)
        us = []
        for i in range(frames):
            t = i / 30
            s = core.step(scene(**scene_kw), t, pose=pose_fn(t) if pose_fn else None)
            if s.u is not None:
                us.append(s.u)
        return us

    def test_the_bug_without_the_fixes(self):
        '''User's settings, no pose: the tilted paddle loses to the (bigger) red-looking hand.'''
        us = self.track(VisionConfig(**USER_CFG), None)
        hand_u = (WRIST[0] + KNUCKLE[0]) / 2 / W
        self.assertAlmostEqual(us[-1], hand_u, delta=0.02)

    def test_pose_cutout_keeps_tracking_on_the_paddle(self):
        us = self.track(VisionConfig(**USER_CFG), arm_pose)
        self.assertTrue(us)
        for u in us:
            self.assertAlmostEqual(u, PADDLE[0] / W, delta=0.02)

    def test_calibrated_red_ignores_the_hand_even_without_pose(self):
        cfg = VisionConfig(**USER_CFG)
        tighten_red_against_skin(cfg, hsv_cloud(178, 217, 200), hsv_cloud(11, 106, 205))
        us = self.track(cfg, None)
        self.assertTrue(us)
        for u in us:
            self.assertAlmostEqual(u, PADDLE[0] / W, delta=0.02)


if __name__ == '__main__':
    unittest.main()
