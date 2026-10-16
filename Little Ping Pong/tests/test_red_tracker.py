'''Fast red-only, x-only tracking (red_tracker.py).'''

import time
import unittest

import cv2
import numpy as np

from paddle_vision import GAME_X_RANGE, VisionConfig, calibrate_red
from red_tracker import LOST_AFTER_S, RedTrackerCore

H, W = 480, 640
RED = (40, 30, 210)     # BGR
WALL = (170, 175, 165)
DT = 1 / 30


def frame(paddle_u=None, color=RED, radius=40):
    img = np.full((H, W, 3), WALL, np.uint8)
    img = np.clip(img.astype(int) + np.random.default_rng(0).integers(-8, 9, img.shape), 0, 255).astype(np.uint8)
    if paddle_u is not None:
        cv2.ellipse(img, (int(paddle_u * W), 220), (radius, int(radius * 1.2)), 0, 0, 360, color, -1)
    return img


def run(core, us, t0=0.0, **kw):
    state = None
    for i, u in enumerate(us):
        state = core.step(frame(u, **kw), t0 + i * DT)
    return state


class TestRedTracker(unittest.TestCase):

    def test_centre_of_image_is_centre_of_table(self):
        s = run(RedTrackerCore(), [0.5] * 5)
        self.assertTrue(s.seen)
        self.assertAlmostEqual(s.paddle_x, 0.0, delta=0.03)

    def test_play_area_edges_map_to_table_edges(self):
        cfg = VisionConfig()
        self.assertAlmostEqual(run(RedTrackerCore(), [cfg.u_lo] * 8).paddle_x, GAME_X_RANGE[0], delta=0.04)
        self.assertAlmostEqual(run(RedTrackerCore(), [cfg.u_hi] * 8).paddle_x, GAME_X_RANGE[1], delta=0.04)
        self.assertEqual(run(RedTrackerCore(), [0.02] * 8).paddle_x, GAME_X_RANGE[0])     # clamped

    def test_follows_a_fast_sweep_with_little_lag(self):
        '''Sweep across the play area in 0.5 s (a quick step to reach a wide ball).'''
        us = list(np.linspace(0.2, 0.8, 16))
        s = run(RedTrackerCore(), us)
        self.assertLess(abs(s.u - us[-1]), 0.04)        # within ~1 frame of motion

    def test_finds_paddle_after_a_big_jump(self):
        core = RedTrackerCore()
        run(core, [0.25] * 5)
        s = run(core, [0.8] * 6, t0=1.0)               # outside the search window -> full-frame search
        self.assertAlmostEqual(s.u, 0.8, delta=0.03)

    def test_tiny_red_speck_is_ignored(self):
        s = run(RedTrackerCore(), [0.5] * 5, radius=3)
        self.assertFalse(s.seen)
        self.assertIsNone(s.paddle_x)

    def test_paddle_out_of_view_holds_position_then_reports_lost(self):
        core = RedTrackerCore()
        x = run(core, [0.6] * 5).paddle_x
        s = run(core, [None] * 3, t0=0.2)
        self.assertTrue(s.seen)
        s = run(core, [None] * 3, t0=0.2 + LOST_AFTER_S)
        self.assertFalse(s.seen)
        self.assertEqual(s.paddle_x, x)

    def test_uses_the_red_calibration(self):
        dull = (95, 90, 150)                            # low-saturation red: default thresholds miss it
        self.assertFalse(run(RedTrackerCore(), [0.5] * 5, color=dull).seen)
        patch = cv2.cvtColor(frame(0.5, color=dull)[210:230, 310:330], cv2.COLOR_BGR2HSV).reshape(-1, 3)
        cfg = calibrate_red(patch, VisionConfig())
        self.assertTrue(run(RedTrackerCore(cfg), [0.5] * 5, color=dull).seen)

    def test_fast_enough(self):
        '''The old pose pipeline took ~15 ms/frame; this must stay well under 5 ms.'''
        core = RedTrackerCore()
        img = frame(0.5)
        core.step(img, 0.0)
        t = time.perf_counter()
        for i in range(60):
            core.step(img, i * DT)
        per_frame_ms = (time.perf_counter() - t) / 60 * 1000
        self.assertLess(per_frame_ms, 5.0)


if __name__ == '__main__':
    unittest.main()
