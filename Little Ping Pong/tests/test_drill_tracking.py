'''Drill tracking: the webcam play area maps onto just the forehand zone, not the whole table.'''

import random
import unittest

import numpy as np

from drill import FOREHAND_X_RANGE, Drill, tracking_x_range
from paddle_vision import GAME_X_RANGE, VisionConfig
from physics import PLAYER_HIT_Z, predict_crossing
from red_tracker import RedTrackerCore, u_to_game_x
from tests.test_red_tracker import frame


def arrivals(hand='right', n=60):
    xs = []
    for s in range(n):
        d = Drill(hand=hand, rng=random.Random(s))
        d.start()
        xs.append(predict_crossing(d.ball, PLAYER_HIT_Z)[1].x)
    return min(xs), max(xs)


def image_share(x_lo, x_hi, x_range, cfg=VisionConfig()):
    '''Fraction of the webcam image width the paddle travels to cover game x_lo..x_hi.'''
    per_x = (cfg.u_hi - cfg.u_lo) / (x_range[1] - x_range[0])
    return (x_hi - x_lo) * per_x


class TestForehandMapping(unittest.TestCase):

    def test_every_ball_lands_inside_the_tracked_zone_with_room_to_spare(self):
        lo, hi = arrivals()
        reach = Drill().level.reach_m
        self.assertLess(FOREHAND_X_RANGE[0], lo - 0.15)
        self.assertGreater(FOREHAND_X_RANGE[1], hi + 0.15)
        self.assertGreater(reach, 0)

    def test_paddle_uses_much_more_of_the_image_than_whole_table_mapping(self):
        lo, hi = arrivals()
        whole_table = image_share(lo, hi, GAME_X_RANGE)
        forehand = image_share(lo, hi, FOREHAND_X_RANGE)
        self.assertLess(whole_table, 0.2)          # the problem: ~17% of the image
        self.assertGreater(forehand, 0.35)          # the fix: ~2.4x more
        self.assertGreater(forehand / whole_table, 2.3)

    def test_left_handed_zone_is_mirrored(self):
        self.assertEqual(tracking_x_range('left'), (-FOREHAND_X_RANGE[1], -FOREHAND_X_RANGE[0]))
        lo, hi = arrivals('left')
        zl = tracking_x_range('left')
        self.assertTrue(zl[0] < lo and hi < zl[1])

    def test_main_game_mapping_is_unchanged(self):
        cfg = VisionConfig()
        self.assertEqual(u_to_game_x(cfg.u_lo, cfg), GAME_X_RANGE[0])
        self.assertEqual(u_to_game_x(cfg.u_hi, cfg), GAME_X_RANGE[1])
        self.assertEqual(u_to_game_x(cfg.u_hi, cfg, FOREHAND_X_RANGE), FOREHAND_X_RANGE[1])

    def test_webcam_jitter_moves_the_drill_paddle_less(self):
        '''Same tracker, same shaky input (+-2 px): the paddle shakes ~2.4x less in the drill.'''
        def paddle_jitter(x_range):
            core = RedTrackerCore(x_range=x_range)
            xs = []
            for i in range(60):
                s = core.step(frame(0.65 + (0.003 if i % 2 else -0.003)), i / 30)
                xs.append(s.paddle_x)
            return float(np.std(xs[10:]))
        whole, drill = paddle_jitter(GAME_X_RANGE), paddle_jitter(FOREHAND_X_RANGE)
        self.assertGreater(whole / drill, 2.3)


if __name__ == '__main__':
    unittest.main()
