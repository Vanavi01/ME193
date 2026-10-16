'''Backhand: black side toward the camera (inside the racket-arm gate) -> backhand,
paddle drawn black, and tracking continues on the black face.'''

import os
import unittest

os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import pygame  # noqa: E402

import play  # noqa: E402
from game import Game  # noqa: E402
from paddle_vision import VisionConfig  # noqa: E402
from red_tracker import RedTrackerCore  # noqa: E402
from tests.test_red_tracker import H, W  # noqa: E402
from tests.test_skin import ELBOW, PADDLE, WALL, WRIST, arm_pose  # noqa: E402

RED = (40, 30, 200)
BLACK = (25, 25, 25)
SKIN = (150, 180, 220)          # light skin: not red with the default settings


def scene(face=None, paddle_x=PADDLE[0], sleeve=False, hair=False):
    img = np.full((H, W, 3), WALL, np.uint8)
    if sleeve:                                              # dark sleeve along the forearm
        cv2.line(img, ELBOW, WRIST, BLACK, 26)
    if hair:                                                # thin dark streak in the search area
        cv2.line(img, (paddle_x - 90, PADDLE[1] - 40), (paddle_x + 90, PADDLE[1] - 10), BLACK, 6)
    if face:
        cv2.ellipse(img, (paddle_x, PADDLE[1]), (40, 48), 0, 0, 360, RED if face == 'red' else BLACK, -1)
    cv2.circle(img, WRIST, 16, SKIN, -1)
    return img


def run(core, n, t0=0.0, use_pose=True, **kw):
    s = None
    for i in range(n):
        t = t0 + i / 30
        s = core.step(scene(**kw), t, pose=arm_pose(t) if use_pose else None)
    return s


class TestBackhandDetection(unittest.TestCase):

    def test_black_face_is_backhand_and_still_tracked(self):
        s = run(RedTrackerCore(), 5, face='black')
        self.assertEqual(s.side, 'black')
        self.assertTrue(s.backhand)
        self.assertTrue(s.seen)
        self.assertAlmostEqual(s.u, PADDLE[0] / W, delta=0.02)

    def test_tracking_follows_the_black_face_when_it_moves(self):
        core = RedTrackerCore()
        run(core, 5, face='black')
        s = run(core, 8, t0=0.2, face='black', paddle_x=PADDLE[0] - 30)
        self.assertAlmostEqual(s.u, (PADDLE[0] - 30) / W, delta=0.02)

    def test_red_face_is_not_backhand(self):
        s = run(RedTrackerCore(), 5, face='red')
        self.assertEqual(s.side, 'red')
        self.assertFalse(s.backhand)

    def test_switching_needs_a_few_frames(self):
        core = RedTrackerCore()
        run(core, 5, face='red')
        self.assertFalse(run(core, 2, t0=0.2, face='black').backhand)     # 2 frames: not yet
        self.assertTrue(run(core, 2, t0=0.3, face='black').backhand)      # 4 of last 5: yes
        self.assertTrue(run(core, 1, t0=0.4, face='red').backhand)        # one red frame: no flip

    def test_dark_sleeve_is_not_a_backhand(self):
        s = run(RedTrackerCore(), 6, sleeve=True)
        self.assertFalse(s.backhand)
        self.assertFalse(s.seen)

    def test_thin_dark_streak_is_not_a_backhand(self):
        self.assertFalse(run(RedTrackerCore(), 6, hair=True).backhand)

    def test_black_is_ignored_without_pose(self):
        '''Without pose there is no racket-arm gate, and black alone is too risky (hair, clothes).'''
        s = run(RedTrackerCore(), 6, use_pose=False, face='black')
        self.assertFalse(s.backhand)
        self.assertFalse(s.seen)

    def test_uses_the_calibrated_black_threshold(self):
        grey_black = dict(face='black')
        cfg = VisionConfig(black_val_max=10)                # far too strict for this rubber
        self.assertFalse(run(RedTrackerCore(cfg), 5, **grey_black).backhand)


class TestBackhandDisplay(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        pygame.init()
        cls.screen = pygame.display.set_mode((play.WIDTH, play.HEIGHT))

    @classmethod
    def tearDownClass(cls):
        pygame.quit()

    def draw(self, face):
        r = play.Renderer(self.screen)
        g = Game()
        g.start()
        core = RedTrackerCore()
        for i in range(5):
            r.vision = core.step(scene(face=face), i / 30, pose=arm_pose(i / 30))
        r.note_side(r.vision.backhand)
        g.update(1 / 60, (r.vision.paddle_x, None))
        r.draw(g, 0.0, 'IMU')
        cx, cy, _ = r.p(g.paddle[0], g.paddle_y, play.PLAYER_HIT_Z)
        return np.array(self.screen.get_at((int(cx), int(cy)))[:3]), r, g

    def test_paddle_turns_black_on_backhand(self):
        fore, _, _ = self.draw('red')
        back, _, _ = self.draw('black')
        self.assertGreater(fore[0], fore[1] + 60)           # red rubber
        self.assertLess(back.max(), 120)                     # dark rubber

    def test_hit_during_backhand_says_backhand(self):
        g = Game()
        g.stroke = 'backhand'
        g.start()
        g._last_swing_strength = 0.5
        g._player_hit()
        self.assertEqual(g.message, 'Nice backhand!')


if __name__ == '__main__':
    unittest.main()
