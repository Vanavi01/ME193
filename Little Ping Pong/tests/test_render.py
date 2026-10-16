'''Game screen: draws without errors and fast enough for a smooth 60 fps (headless).'''

import os
import random
import time
import unittest

os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')

import pygame  # noqa: E402

import play  # noqa: E402
from game import X_ONLY_PADDLE_Y, Game  # noqa: E402
from red_tracker import RedTrackerCore  # noqa: E402
from tests.test_red_tracker import frame  # noqa: E402

FRAME_BUDGET_MS = 1000 / 60


class TestRender(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        pygame.init()
        cls.screen = pygame.display.set_mode((play.WIDTH, play.HEIGHT))

    @classmethod
    def tearDownClass(cls):
        pygame.quit()

    def make(self, vision=True):
        r = play.Renderer(self.screen)
        g = Game(rng=random.Random(1))
        g.start()
        if vision:
            core = RedTrackerCore()
            for i in range(5):
                r.vision = core.step(frame(0.6), i / 30)
        for _ in range(20):
            g.update(1 / 60, (r.vision.paddle_x, None) if vision else (0.0, 0.3))
        return r, g

    def test_draws_in_well_under_a_frame(self):
        '''Was 10.6 ms (full-screen alpha surfaces + fonts rebuilt every frame).'''
        r, g = self.make()
        r.draw(g, 0.5, 'IMU')
        t = time.perf_counter()
        for _ in range(60):
            r.draw(g, 0.5, 'IMU')
        ms = (time.perf_counter() - t) / 60 * 1000
        self.assertLess(ms, FRAME_BUDGET_MS / 3)

    def test_x_only_paddle_is_drawn_at_the_fixed_height(self):
        r, g = self.make()
        self.assertEqual(g.paddle_y, X_ONLY_PADDLE_Y)
        r.draw(g, 0.0, 'keyboard')

    def test_mouse_mode_still_draws(self):
        r, g = self.make(vision=False)
        r.draw(g, 0.0, 'keyboard')


if __name__ == '__main__':
    unittest.main()
