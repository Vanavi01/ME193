'''Spin in the game: topspin = lower arc + faster; chop (backspin) = similar arc, shorter, slower.'''

import os
import random
import unittest

import numpy as np

os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')

from game import SPIN_EARLY_S, SPIN_LATE_S, SPINS, Game  # noqa: E402
from physics import BALL_R, CPU_HIT_Z, GRAVITY, NET_H, NET_Z, TABLE_L, Ball, aim, step  # noqa: E402
from strokes import (FEATURE_POST_S, LiveStrokeClassifier, NearestCentroid,  # noqa: E402
                     features)
from tests.test_game import FRAME, new_game, perfect, run_until  # noqa: E402
from tests.test_strokes import dataset, swing_stream  # noqa: E402


def fly(ball, dt=1 / 240):
    '''Time to first bounce, highest point, bounce z, and forward speed just after the bounce.'''
    t = top = 0.0
    while t < 4:
        vz_before = ball.vz
        if step(ball, dt) == 'bounce':
            return t, top, ball.z, ball.vz / vz_before
        top = max(top, ball.y)
        t += dt
    return None


def shot(kind, start=(0.1, 0.25, -0.25), target=(0.2, 2.2), flight=0.6):
    s = SPINS[kind]
    b = Ball(*start)
    b.set_spin(None if kind == 'flat' else kind, GRAVITY * s.gravity, s.bounce)
    b.vx, b.vy, b.vz = aim(start, target[0], target[1] + s.depth_m, flight * s.flight, b.g)
    return fly(b)


class TestSpinPhysics(unittest.TestCase):

    def test_aim_lands_on_target_under_any_gravity(self):
        for g in (GRAVITY * 0.65, GRAVITY, GRAVITY * 1.4):
            b = Ball(0, 0.3, -0.2, g=g)
            b.vx, b.vy, b.vz = aim((0, 0.3, -0.2), 0.3, 2.2, 0.6, g)
            _, _, z, _ = fly(b)
            self.assertAlmostEqual(z, 2.2, delta=0.03)

    def test_topspin_is_lower_and_faster(self):
        t_flat, top_flat, _, _ = shot('flat')
        t_top, top_top, _, _ = shot('topspin')
        self.assertLess(t_top, t_flat)
        self.assertLess(top_top, top_flat - 0.02)

    def test_chop_has_a_similar_arc_but_is_shorter_and_slower(self):
        t_flat, top_flat, z_flat, _ = shot('flat')
        t_chop, top_chop, z_chop, _ = shot('backspin')
        self.assertGreater(t_chop, t_flat)
        self.assertLess(z_chop, z_flat - 0.3)
        self.assertAlmostEqual(top_chop, top_flat, delta=0.25 * top_flat)

    def test_bounce_kicks_on_with_topspin_and_checks_up_with_backspin(self):
        self.assertAlmostEqual(shot('topspin')[3], SPINS['topspin'].bounce)
        self.assertAlmostEqual(shot('backspin')[3], SPINS['backspin'].bounce)
        self.assertAlmostEqual(shot('flat')[3], 1.0)

    def test_every_spin_clears_the_net(self):
        rng = random.Random(0)
        for kind in SPINS:
            for _ in range(40):
                start = (rng.uniform(-0.8, 0.8), rng.uniform(0.05, 0.5), rng.uniform(-0.5, -0.2))
                s = SPINS[kind]
                b = Ball(*start, g=GRAVITY * s.gravity)
                b.vx, b.vy, b.vz = aim(start, rng.uniform(-0.6, 0.6), rng.uniform(1.9, 2.5) + s.depth_m,
                                       rng.uniform(0.4, 0.65) * s.flight, b.g)
                net_y = None
                while b.z < NET_Z and b.y > -1:
                    step(b, 1 / 240)
                    net_y = b.y
                self.assertGreater(net_y, NET_H, (kind, start))


def hit_with(kind, seed=0, delay_s=0.1):
    '''Hit the serve, then deliver the classification delay_s (real) later, like the IMU does.'''
    g = new_game(seed=seed)
    run_until(g, lambda g: g.heading == 'to_cpu', swing_at=perfect(0.5))
    for _ in range(int(delay_s / FRAME)):
        g.update(FRAME, g.paddle)
    g.apply_spin(kind)
    return g


def to_cpu(g):
    '''Real time until the CPU hits the ball back, and where the ball first bounced.'''
    t0, bounce_z, vy = g.time, None, g.ball.vy
    while g.heading == 'to_cpu' and g.state == 'rally':
        g.update(FRAME, g.paddle)
        if bounce_z is None and vy < 0 < g.ball.vy:
            bounce_z = g.ball.z
        vy = g.ball.vy
    return g.time - t0, bounce_z


class TestSpinInGame(unittest.TestCase):

    def test_spin_arriving_after_the_hit_reshapes_the_shot(self):
        t = {k: np.mean([to_cpu(hit_with(k, seed))[0] for seed in range(6)]) for k in SPINS}
        z = {k: np.mean([to_cpu(hit_with(k, seed))[1] for seed in range(6)]) for k in SPINS}
        self.assertLess(t['topspin'], t['flat'])
        self.assertGreater(t['backspin'], t['flat'])
        self.assertLess(z['backspin'], z['flat'] - 0.3)

    def test_message_ball_spin_and_event(self):
        g = hit_with('topspin')
        self.assertEqual((g.message, g.ball.spin), ('Topspin!', 'topspin'))
        self.assertIn(('spin', 'topspin'), g.drain_events())
        g = hit_with('backspin')
        self.assertEqual((g.message, g.ball.spin), ('Chop!', 'backspin'))

    def test_flat_keeps_the_normal_message(self):
        g = hit_with('flat')
        self.assertIn(g.message, ('Nice!', 'Smash!'))
        self.assertIsNone(g.ball.spin)

    def test_classification_before_the_ball_arrives_is_used_by_the_hit(self):
        g = new_game()
        def early(g, remaining):                       # swing peaks a bit early; classified before the hit
            if 0.08 < remaining < 0.08 + 1.5 * FRAME and g._next_spin is None:
                g.apply_spin('topspin')
            return perfect(0.5)(g, remaining)
        run_until(g, lambda g: g.heading == 'to_cpu', swing_at=early)
        self.assertEqual(g.ball.spin, 'topspin')

    def test_stale_early_classification_is_not_used(self):
        g = new_game()
        g.apply_spin('backspin')                        # long before the ball arrives
        run_until(g, lambda g: g.heading == 'to_cpu', swing_at=perfect(0.5))
        self.assertIsNone(g.ball.spin)

    def test_late_classification_is_ignored(self):
        g = hit_with('topspin', delay_s=SPIN_LATE_S + 0.1)
        self.assertIsNone(g.ball.spin)

    def test_only_one_spin_per_shot(self):
        g = hit_with('topspin')
        g.apply_spin('backspin')
        self.assertEqual(g.ball.spin, 'topspin')

    def test_backhand_shots_get_no_spin(self):
        g = new_game()
        g.stroke = 'backhand'
        run_until(g, lambda g: g.heading == 'to_cpu', swing_at=perfect(0.5))
        g.apply_spin('topspin')
        self.assertIsNone(g.ball.spin)

    def test_cpu_returns_flat_and_takes_short_chops(self):
        g = hit_with('backspin')
        to_cpu(g)
        self.assertEqual((g.state, g.heading), ('rally', 'to_player'))     # not a 'Missed!'
        self.assertIsNone(g.ball.spin)
        self.assertEqual(g.ball.g, GRAVITY)

    def test_cpu_takes_a_chop_that_dies_on_its_side(self):
        '''A very short, slow chop loses forward speed on every bounce and might never reach
        the CPU's end: the CPU must still play it (on the drop after it bounces).'''
        g = hit_with('backspin')
        b = g.ball
        b.x, b.y, b.z, b.vx, b.vy, b.vz = 0.0, 0.3, NET_Z + 0.2, 0.0, 0.5, 0.25    # just over the net, crawling
        t0 = g.time
        while g.heading == 'to_cpu' and g.time - t0 < 3:
            g.update(FRAME, g.paddle)
        self.assertEqual(g.heading, 'to_player')
        self.assertLess(g.time - t0, 1.5)

    def test_long_rally_with_random_spins_never_breaks(self):
        rng = random.Random(4)
        for level in (1, 2, 3):
            g = new_game(level, seed=level)
            hits = 0
            while g.streak < 15 and g.state == 'rally':
                run_until(g, lambda g: g.heading == 'to_cpu' or g.state != 'rally', swing_at=perfect(0.5))
                if g.state != 'rally':
                    break
                g.apply_spin(rng.choice(list(SPINS)))
                run_until(g, lambda g: g.heading == 'to_player' or g.state != 'rally')
            self.assertEqual(g.streak, 15, f'level {level}: {g.message}')


class TestLiveClassifier(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        sw = dataset(8)
        cls.model = NearestCentroid().fit([features(s.samples) for s in sw], [s.label for s in sw])

    def test_classifies_once_the_follow_through_has_arrived(self):
        c = LiveStrokeClassifier(self.model)
        stream = swing_stream('topspin', 0.0, np.random.default_rng(7))
        peak_t = 1.1
        results = []
        for s in stream:
            c.add(s)
            if abs(s.t - peak_t) < 0.008:
                c.on_swing(peak_t)
            r = c.poll()
            if r:
                results.append((s.t, r[0]))
        self.assertEqual(len(results), 1)                       # once per swing
        t_ready, label = results[0]
        self.assertEqual(label, 'topspin')
        self.assertAlmostEqual(t_ready, peak_t + FEATURE_POST_S, delta=0.03)

    def test_no_history_no_guess(self):
        c = LiveStrokeClassifier(self.model)
        stream = swing_stream('flat', 0.0, np.random.default_rng(1))
        for s in stream[60:80]:                                 # connected mid-swing
            c.add(s)
        c.on_swing(1.1)
        for s in stream[80:]:
            c.add(s)
        self.assertIsNone(c.poll())


class TestSpinDrawing(unittest.TestCase):

    def test_spinning_ball_draws_a_trail(self):
        import pygame
        import play
        pygame.init()
        screen = pygame.display.set_mode((play.WIDTH, play.HEIGHT))
        r = play.Renderer(screen)
        g = hit_with('topspin')
        for _ in range(10):
            g.update(FRAME, g.paddle)
        r.draw(g, 0.0, 'IMU')
        pygame.quit()


if __name__ == '__main__':
    unittest.main()
