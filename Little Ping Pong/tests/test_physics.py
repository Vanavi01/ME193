'''Step 2b tests: ball physics and the 2.5D camera.'''

import random
import unittest

from camera import Camera
from physics import (CPU_HIT_Z, NET_H, NET_Z, PLAYER_HIT_Z, RESTITUTION,
                     TABLE_L, TABLE_W, Ball, aim, predict_crossing, step)

DT = 1 / 240


def fly_until_bounce(ball, max_time=4.0):
    '''Returns (ball at the first bounce, height when it passed the net).'''
    net_height = None
    t = 0.0
    while t < max_time:
        before_z = ball.z
        if step(ball, DT) == 'bounce':
            return ball, net_height
        if (before_z - NET_Z) * (ball.z - NET_Z) <= 0 and net_height is None:
            net_height = ball.y
        t += DT
    return None, net_height


class TestStep(unittest.TestCase):

    def test_bounce_on_table_keeps_restitution(self):
        b = Ball(0, 0.021, 1.0, vy=-3.0)
        self.assertEqual(step(b, DT), 'bounce')
        self.assertAlmostEqual(b.vy, RESTITUTION * (3.0 + 9.81 * DT), places=6)

    def test_no_bounce_off_the_table(self):
        b = Ball(TABLE_W, 0.021, 1.0, vy=-3.0)      # beside the table
        self.assertIsNone(step(b, DT))
        self.assertLess(b.vy, 0)


class TestAim(unittest.TestCase):

    def test_lands_on_target(self):
        start = (0.2, 0.3, CPU_HIT_Z)
        b = Ball(*start)
        b.vx, b.vy, b.vz = aim(start, -0.4, 0.6, 0.6)
        landed, _ = fly_until_bounce(b)
        self.assertAlmostEqual(landed.x, -0.4, delta=0.03)
        self.assertAlmostEqual(landed.z, 0.6, delta=0.03)

    def test_always_clears_net_and_lands_on_far_side(self):
        rng = random.Random(0)
        for _ in range(200):
            # CPU -> player, and player -> CPU (including low, late contact points)
            if rng.random() < 0.5:
                start = (rng.uniform(-0.7, 0.7), rng.uniform(0.15, 0.5), CPU_HIT_Z)
                tx, tz = rng.uniform(-0.55, 0.55), rng.uniform(0.35, 1.0)
            else:
                start = (rng.uniform(-0.9, 0.9), rng.uniform(0.02, 0.6), rng.uniform(-0.6, PLAYER_HIT_Z))
                tx, tz = rng.uniform(-0.6, 0.6), rng.uniform(TABLE_L - 0.75, TABLE_L - 0.25)
            b = Ball(*start)
            b.vx, b.vy, b.vz = aim(start, tx, tz, rng.uniform(0.5, 0.75))
            landed, net_y = fly_until_bounce(b)
            self.assertIsNotNone(landed, start)
            self.assertGreater(net_y, NET_H, f'hit the net from {start}')
            self.assertAlmostEqual(landed.z, tz, delta=0.05)


class TestPredictCrossing(unittest.TestCase):

    def test_matches_simulation(self):
        start = (0.0, 0.3, CPU_HIT_Z)
        b = Ball(*start)
        b.vx, b.vy, b.vz = aim(start, 0.3, 0.7, 0.6)
        t_pred, at = predict_crossing(b, PLAYER_HIT_Z, dt=DT)
        t = 0.0
        while b.z > PLAYER_HIT_Z:
            step(b, DT)
            t += DT
        self.assertAlmostEqual(t, t_pred, delta=DT)
        self.assertAlmostEqual(b.x, at.x, places=6)
        self.assertAlmostEqual(b.y, at.y, places=6)

    def test_ball_that_falls_away_never_crosses(self):
        b = Ball(5.0, 0.3, 1.0, vz=-0.1)            # off the table, dropping slowly
        self.assertIsNone(predict_crossing(b, PLAYER_HIT_Z - 3))


class TestCamera(unittest.TestCase):

    def setUp(self):
        self.cam = Camera(1280, 720)

    def test_centre_line_is_screen_centre(self):
        for z in (0, NET_Z, TABLE_L):
            self.assertAlmostEqual(self.cam.project(0, 0, z)[0], 640)

    def test_far_things_are_smaller_and_higher(self):
        near, far = self.cam.project(0, 0, 0), self.cam.project(0, 0, TABLE_L)
        self.assertGreater(near[2], far[2])
        self.assertGreater(near[1], far[1])          # table recedes up the screen

    def test_whole_table_is_on_screen(self):
        for x in (-TABLE_W / 2, TABLE_W / 2):
            for z in (0, TABLE_L):
                sx, sy, _ = self.cam.project(x, 0, z)
                self.assertTrue(0 <= sx <= 1280 and 0 <= sy <= 720, (x, z, sx, sy))

    def test_unproject_round_trip(self):
        for x, y in ((0, 0), (0.5, 0.3), (-0.8, 0.6), (0.2, -0.2)):
            sx, sy, _ = self.cam.project(x, y, PLAYER_HIT_Z)
            ux, uy = self.cam.unproject_to_z(sx, sy, PLAYER_HIT_Z)
            self.assertAlmostEqual(ux, x, places=6)
            self.assertAlmostEqual(uy, y, places=6)


if __name__ == '__main__':
    unittest.main()
