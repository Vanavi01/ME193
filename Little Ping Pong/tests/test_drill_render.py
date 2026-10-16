'''Forehand drill screen: every state draws, fast, and the camera looks from the forehand side.'''

import os
import random
import time
import unittest

os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')

import pygame  # noqa: E402

import drill_play  # noqa: E402
from drill import FEED_X, Drill  # noqa: E402
from physics import PLAYER_HIT_Z, TABLE_W  # noqa: E402
from red_tracker import RedTrackerCore  # noqa: E402
from tests.test_drill import ball_of, next_ball, play_ball  # noqa: E402
from tests.test_red_tracker import frame  # noqa: E402


class TestDrillScreen(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        pygame.init()
        cls.screen = pygame.display.set_mode((drill_play.WIDTH, drill_play.HEIGHT))

    @classmethod
    def tearDownClass(cls):
        pygame.quit()

    def test_every_state_draws(self):
        r = drill_play.DrillRenderer(self.screen)
        r.draw(Drill(rng=random.Random(0)), 0.0, 'keyboard')            # ready
        d = ball_of('chop')
        for _ in range(20):
            d.update(1 / 60, 0.3)
        r.draw(d, 0.5, 'IMU')                                           # ball incoming + call banner
        d.pause()
        r.draw(d, 0.0, 'IMU')                                           # paused
        d = Drill(rng=random.Random(1), balls=2)
        d.start()
        for _ in range(2):
            next_ball(d)
            play_ball(d, 'flat')
        self.assertEqual(d.state, 'done')
        r.draw(d, 0.0, 'IMU')                                           # results

    def test_with_webcam_feed_and_fast(self):
        r = drill_play.DrillRenderer(self.screen)
        core = RedTrackerCore()
        for i in range(3):
            r.vision = core.step(frame(0.7), i / 30)
        d = ball_of('smash')
        r.draw(d, 0.3, 'IMU')
        t = time.perf_counter()
        for _ in range(30):
            r.draw(d, 0.3, 'IMU')
        self.assertLess((time.perf_counter() - t) / 30 * 1000, 1000 / 60 / 2)

    def test_tag_prompt_draws_on_start_and_results(self):
        r = drill_play.DrillRenderer(self.screen)
        r.tags_on, r.tag_progress = True, 0.5
        r.draw(Drill(rng=random.Random(0)), 0.0, 'IMU')
        r.level_flash_t = pygame.time.get_ticks() / 1000
        d = Drill(rng=random.Random(1), balls=2)
        d.start()
        for _ in range(2):
            next_ball(d)
            play_ball(d, 'flat')
        r.draw(d, 0.0, 'IMU')

    def test_forehand_side_is_on_the_right_of_the_screen(self):
        cam = drill_play.drill_camera(1)
        hit_zone_x, _, _ = cam.project(0.4, 0.38, PLAYER_HIT_Z)
        backhand_side_x, _, _ = cam.project(-TABLE_W / 2, 0.0, 0.0)
        self.assertGreater(hit_zone_x, drill_play.WIDTH * 0.6)
        self.assertLess(backhand_side_x, drill_play.WIDTH * 0.4)
        lefty = drill_play.drill_camera(-1)                          # mirrored for left-handers
        self.assertLess(lefty.project(-0.4, 0.38, PLAYER_HIT_Z)[0], drill_play.WIDTH * 0.4)

    def test_mouse_maps_through_the_skewed_camera(self):
        cam = drill_play.drill_camera(1)
        sx, sy, _ = cam.project(FEED_X, 0.38, PLAYER_HIT_Z)
        x, y = cam.unproject_to_z(sx, sy, PLAYER_HIT_Z)
        self.assertAlmostEqual(x, FEED_X, places=6)
        self.assertAlmostEqual(y, 0.38, places=6)


if __name__ == '__main__':
    unittest.main()
