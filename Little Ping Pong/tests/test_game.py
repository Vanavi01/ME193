'''Step 2b tests: rally rules, hit detection (location + swing timing) and the streak.'''

import random
import unittest

from game import (HIT_WINDOW_S, LEVELS, POINT_PAUSE_S, RALLY_SPEEDUP_MAX,
                  X_ONLY_PADDLE_Y, Game)
from physics import CPU_HIT_Z, NET_Z, PLAYER_HIT_Z, predict_crossing

FRAME = 1 / 60


def new_game(level=1, seed=0):
    g = Game(level=level, rng=random.Random(seed))
    g.start()
    return g


def crossing(g):
    '''(real seconds until the ball reaches the player, ball at that point).'''
    t, at = predict_crossing(g.ball, PLAYER_HIT_Z)
    return t / g.speed, at


def run_until(g, condition, paddle=None, max_s=10.0, swing_at=None, x_only=False):
    '''Step frames until condition(g). paddle=None -> perfectly placed paddle.
    swing_at(g, real_s_to_crossing) -> swing strength or None.'''
    t = 0.0
    while t < max_s:
        if condition(g):
            return True
        p = paddle
        if g.heading == 'to_player' and g._crossing is None:
            remaining, at = crossing(g)
            if p is None:
                p = (at.x, at.y)
            if swing_at:
                s = swing_at(g, remaining)
                if s is not None:
                    g.swing(s)
        if p is not None and x_only:
            p = (p[0], None)
        g.update(FRAME, p if p is not None else g.paddle)
        t += FRAME
    return False


def perfect(strength=0.6):
    '''Swing just as the ball arrives (once per incoming ball).'''
    def swing_at(g, remaining):
        if remaining < 1.5 * FRAME and g._last_swing_t < g.time - 1:
            return strength
        return None
    return swing_at


def kinds(events):
    return [e[0] for e in events]


class TestStart(unittest.TestCase):

    def test_nothing_moves_before_start(self):
        g = Game(rng=random.Random(0))
        z = g.ball.z
        for _ in range(60):
            g.update(FRAME, (0, 0.2))
        self.assertEqual(g.state, 'ready')
        self.assertEqual(g.ball.z, z)

    def test_serve_comes_to_player_and_bounces_on_their_side(self):
        g = new_game()
        self.assertEqual(g.heading, 'to_player')
        bounced_z = []
        while g.ball.z > 0:
            vy = g.ball.vy
            g.update(FRAME, (0, 0.2))
            if vy < 0 < g.ball.vy:
                bounced_z.append(g.ball.z)
        self.assertEqual(len(bounced_z), 1)
        self.assertLess(bounced_z[0], NET_Z)


class TestHits(unittest.TestCase):

    def test_perfect_player_keeps_a_long_rally(self):
        for level in LEVELS:
            g = new_game(level)
            ok = run_until(g, lambda g: g.streak >= 10 or g.state != 'rally',
                           swing_at=perfect(), max_s=120)
            self.assertTrue(ok)
            self.assertEqual(g.streak, 10, f'level {level}: {g.message}')
            self.assertEqual(g.best, 10)

    def test_hit_events(self):
        g = new_game()
        run_until(g, lambda g: g.streak == 2, swing_at=perfect(0.9))
        events = kinds(g.drain_events())
        self.assertEqual(events.count('hit'), 2)
        self.assertEqual(events.count('new_best'), 2)
        self.assertIn('cpu_hit', events)
        self.assertEqual(g.drain_events(), [])        # drained

    def test_hit_records_the_stroke(self):
        g = new_game()
        g.stroke = 'backhand'
        run_until(g, lambda g: g.streak == 1, swing_at=perfect(0.5))
        self.assertEqual(g.message, 'Nice backhand!')
        self.assertIn(('hit', 0.5, 'backhand'), g.drain_events())

    def test_late_swing_inside_window_still_hits(self):
        '''Bluetooth delay: the swing arrives just after the ball reached the paddle.'''
        g = new_game()
        run_until(g, lambda g: g._crossing is not None)
        for _ in range(int(0.75 * HIT_WINDOW_S / FRAME)):
            g.update(FRAME, g.paddle)
        g.swing(0.6)
        g.update(FRAME, g.paddle)
        self.assertEqual(g.streak, 1)
        self.assertEqual(g.heading, 'to_cpu')

    def test_return_lands_on_cpu_side(self):
        g = new_game()
        run_until(g, lambda g: g.heading == 'to_cpu', swing_at=perfect())
        bounced_z = []
        while g.heading == 'to_cpu' and g.ball.z < CPU_HIT_Z:
            vy = g.ball.vy
            g.update(FRAME, g.paddle)
            if vy < 0 < g.ball.vy:
                bounced_z.append(g.ball.z)
        self.assertEqual(len(bounced_z), 1)
        self.assertGreater(bounced_z[0], NET_Z)

    def test_harder_swing_returns_faster(self):
        def time_to_cpu(strength):
            g = new_game(seed=3)
            run_until(g, lambda g: g.heading == 'to_cpu', swing_at=perfect(strength))
            start = g.time
            run_until(g, lambda g: g.heading == 'to_player')
            return g.time - start
        self.assertLess(time_to_cpu(1.0), time_to_cpu(0.1))


class TestMisses(unittest.TestCase):

    def test_no_swing_is_a_miss(self):
        g = new_game()
        run_until(g, lambda g: g.state == 'point')
        self.assertEqual(g.message, 'No swing!')
        self.assertEqual(g.streak, 0)

    def test_paddle_out_of_position_is_a_miss_even_with_a_swing(self):
        g = new_game()
        _, at = crossing(g)
        far = (at.x + 0.5, at.y)
        run_until(g, lambda g: g.state == 'point', paddle=far, swing_at=perfect())
        self.assertEqual(g.message, 'Out of reach!')

    def test_reach_is_level_dependent(self):
        for level in LEVELS:
            reach = LEVELS[level].reach_m
            for offset, expect_hit in ((reach * 0.8, True), (reach * 1.2, False)):
                g = new_game(level)
                _, at = crossing(g)
                run_until(g, lambda g: g.state != 'rally' or g.heading == 'to_cpu',
                          paddle=(at.x + offset, at.y), swing_at=perfect())
                self.assertEqual(g.heading == 'to_cpu', expect_hit, (level, offset))

    def test_swing_too_early_is_a_miss(self):
        g = new_game()
        early = HIT_WINDOW_S + 0.15
        run_until(g, lambda g: g.state == 'point',
                  swing_at=lambda g, remaining: 0.6 if abs(remaining - early) < FRAME / 2 else None)
        self.assertEqual(g.message, 'No swing!')

    def test_miss_resets_streak_but_keeps_best_and_serves_again(self):
        g = new_game()
        run_until(g, lambda g: g.streak == 3, swing_at=perfect())
        run_until(g, lambda g: g.state == 'point')          # stop swinging
        self.assertEqual((g.streak, g.best), (0, 3))
        self.assertIn(('miss', 'No swing!'), g.drain_events())
        run_until(g, lambda g: g.state == 'rally', max_s=POINT_PAUSE_S + 0.1)
        self.assertEqual(g.heading, 'to_player')
        self.assertEqual(g.best, 3)

    def test_restart_resets_streak_but_keeps_best(self):
        g = new_game()
        run_until(g, lambda g: g.streak == 2, swing_at=perfect())
        g.start()
        self.assertEqual((g.streak, g.best), (0, 2))


class TestLevels(unittest.TestCase):

    def test_higher_level_ball_arrives_sooner(self):
        times = [crossing(new_game(level, seed=5))[0] for level in sorted(LEVELS)]
        self.assertEqual(times, sorted(times, reverse=True))
        self.assertGreater(times[0], 0.9)     # Normal: just under a second to react
        self.assertLess(times[-1], 0.75)      # Pro: fast

    def test_rally_gets_faster_then_caps_and_a_miss_resets_it(self):
        g = new_game()
        base = g.speed
        run_until(g, lambda g: g.streak == 3, swing_at=perfect())
        self.assertAlmostEqual(g.speed, base * 1.09)
        run_until(g, lambda g: g.streak == 15, swing_at=perfect(), max_s=120)
        self.assertAlmostEqual(g.speed, base * (1 + RALLY_SPEEDUP_MAX))
        run_until(g, lambda g: g.state == 'point')
        self.assertAlmostEqual(g.speed, base)


class TestXOnly(unittest.TestCase):
    '''Vision mode tracks x only: paddle = (x, None).'''

    def test_height_does_not_matter(self):
        g = new_game()
        _, at = crossing(g)
        run_until(g, lambda g: g.state != 'rally' or g.heading == 'to_cpu',
                  paddle=(at.x, None), swing_at=perfect())
        self.assertEqual(g.streak, 1)

    def test_left_right_reach_still_matters(self):
        g = new_game()
        _, at = crossing(g)
        run_until(g, lambda g: g.state == 'point', paddle=(at.x + 1.5 * g.level.reach_m, None),
                  swing_at=perfect())
        self.assertEqual(g.message, 'Out of reach!')

    def test_paddle_height_is_fixed_for_the_whole_rally(self):
        '''Bug fix: the paddle used to jump to each new shot's height on every CPU hit.'''
        g = new_game()
        heights = set()

        def done(g):
            if g.paddle[1] is None:                         # frames played with x-only input
                heights.add(g.paddle_y)
            return g.streak >= 6
        self.assertTrue(run_until(g, done, swing_at=perfect(), max_s=60, x_only=True))
        g.update(1 / 60, (0.0, None))
        heights.add(g.paddle_y)
        self.assertEqual(heights, {X_ONLY_PADDLE_Y})
        g.update(1 / 60, (0.0, 0.4))
        self.assertEqual(g.paddle_y, 0.4)                   # mouse mode: tracked height wins


if __name__ == '__main__':
    unittest.main()
