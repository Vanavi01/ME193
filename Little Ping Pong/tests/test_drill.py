'''Forehand drill (smash / chop practice) rules.'''

import random
import unittest

from drill import (DRILL_PACE, JUDGE_TIMEOUT_S, POINTS, SMASH_MIN, STREAK_BONUS,
                   Drill, is_correct)
from physics import PLAYER_HIT_Z, TABLE_L, predict_crossing

FRAME = 1 / 60


def new_drill(level=1, seed=0, **kw):
    d = Drill(level, rng=random.Random(seed), **kw)
    d.start()
    return d


def arrival(d):
    t, at = predict_crossing(d.ball, PLAYER_HIT_Z)
    return t / d.speed, at


def play_ball(d, kind=None, strength=0.8, stroke_delay=0.1, paddle_dx=0.0, swing=True, stroke_first=False):
    '''Play the current incoming ball: paddle under it, swing on arrival, stroke type
    reported stroke_delay later (like the IMU). Runs until the ball is finished.'''
    _, at = arrival(d)
    x = at.x + paddle_dx
    swung_at = None
    start_ball = d.fed
    for _ in range(int(6 / FRAME)):
        if d.heading == 'to_player' and d._crossing is None and swung_at is None and swing:
            remaining, _ = arrival(d)
            if remaining < 1.5 * FRAME:
                d.swing(strength)
                swung_at = d.time
                if kind and stroke_first:
                    d.set_stroke(kind)
        if kind and swung_at is not None and not stroke_first and d.time - swung_at >= stroke_delay:
            d.set_stroke(kind)
            kind = None
        d.update(FRAME, x)
        if d.state in ('gap', 'done') or d.fed != start_ball:
            return d
    return d


def next_ball(d):
    while d.state == 'gap':
        d.update(FRAME, 0.3)


def ball_of(call, level=1):
    '''A drill whose first ball is the given call.'''
    for seed in range(100):
        d = new_drill(level, seed)
        if d.call == call:
            return d
    raise AssertionError


class TestRules(unittest.TestCase):

    def test_what_counts_as_correct(self):
        self.assertTrue(is_correct('smash', 'flat', 0.9))
        self.assertTrue(is_correct('smash', 'topspin', SMASH_MIN))
        self.assertFalse(is_correct('smash', 'flat', SMASH_MIN - 0.1))      # too soft
        self.assertFalse(is_correct('smash', 'backspin', 1.0))
        self.assertTrue(is_correct('chop', 'backspin', 0.2))
        self.assertFalse(is_correct('chop', 'flat', 0.9))
        self.assertTrue(is_correct('hit', 'flat', 0.2))                    # a normal hit: any strength
        self.assertTrue(is_correct('hit', 'topspin', 0.9))
        self.assertFalse(is_correct('hit', 'backspin', 0.5))

    def test_feeds_go_to_the_forehand_side_and_differ_by_call(self):
        hs = {'smash': [], 'hit': [], 'chop': []}
        for seed in range(60):
            d = new_drill(seed=seed)
            _, at = arrival(d)
            self.assertGreater(at.x, 0.1)                       # right-hander's forehand side
            hs[d.call].append(at.y)
        self.assertTrue(all(hs.values()))                       # all three calls come up
        self.assertGreater(min(hs['smash']), max(hs['hit']) + 0.1)   # smash balls sit up high,
        self.assertGreater(min(hs['hit']), max(hs['chop']))          # normal balls mid, chop balls low

    def test_calls_are_mixed_evenly(self):
        from collections import Counter
        calls = Counter(new_drill(seed=s).call for s in range(300))
        for c in ('smash', 'hit', 'chop'):
            self.assertAlmostEqual(calls[c] / 300, 1 / 3, delta=0.08)

    def test_left_handed_player_gets_balls_on_the_left(self):
        _, at = arrival(new_drill(hand='left'))
        self.assertLess(at.x, -0.1)


class TestScoring(unittest.TestCase):

    def test_correct_smash_scores_and_builds_a_streak(self):
        d = ball_of('smash')
        play_ball(d, 'flat', strength=0.9)
        self.assertEqual((d.score, d.streak, d.message), (POINTS, 1, 'Great smash!'))
        self.assertIn(('judged', 'smash', 'flat', True, POINTS), d.drain_events())

    def test_streak_bonus(self):
        d = new_drill(seed=3)
        for i in range(3):
            next_ball(d)
            play_ball(d, 'backspin' if d.call == 'chop' else 'flat', strength=0.9)
        self.assertEqual(d.streak, 3)
        self.assertEqual(d.score, 3 * POINTS + STREAK_BONUS * (0 + 1 + 2))

    def test_normal_hit(self):
        d = ball_of('hit')
        play_ball(d, 'flat', strength=0.3)
        self.assertEqual((d.streak, d.message), (1, 'Good hit!'))
        d = ball_of('hit')
        play_ball(d, 'backspin')
        self.assertEqual((d.streak, d.message), (0, 'That was a chop!'))

    def test_correct_chop(self):
        d = ball_of('chop')
        play_ball(d, 'backspin', strength=0.4)
        self.assertEqual((d.streak, d.message), (1, 'Great chop!'))
        self.assertEqual(d.ball.spin, 'backspin')               # the return floats like a chop

    def test_wrong_stroke_scores_nothing_and_resets_the_streak(self):
        d = ball_of('chop')
        d.streak = 4
        play_ball(d, 'topspin')
        self.assertEqual((d.score, d.streak), (0, 0))
        self.assertEqual(d.message, 'That was a topspin!')
        self.assertEqual(d.results[-1], ('chop', 'wrong'))

    def test_soft_smash(self):
        d = ball_of('smash')
        play_ball(d, 'flat', strength=0.3)
        self.assertEqual(d.message, 'Too soft!')

    def test_no_stroke_type_is_judged_flat(self):
        '''No classifier (or it was too slow): after JUDGE_TIMEOUT_S the hit counts as flat.'''
        d = ball_of('smash')
        play_ball(d, None, strength=0.9)
        self.assertEqual(d.results[-1], ('smash', 'correct'))
        d2 = ball_of('chop')
        play_ball(d2, None)
        self.assertEqual(d2.results[-1], ('chop', 'wrong'))

    def test_stroke_type_arriving_before_the_hit_counts(self):
        d = ball_of('chop')
        play_ball(d, 'backspin', stroke_first=True)
        self.assertEqual(d.results[-1], ('chop', 'correct'))

    def test_late_stroke_type_is_ignored(self):
        d = ball_of('chop')
        play_ball(d, 'backspin', stroke_delay=JUDGE_TIMEOUT_S + 0.1)
        self.assertEqual(d.results[-1], ('chop', 'wrong'))      # judged flat at the timeout


class TestMissesAndRound(unittest.TestCase):

    def test_no_swing_and_too_far(self):
        d = new_drill()
        play_ball(d, swing=False)
        self.assertEqual((d.message, d.results[-1][1]), ('No swing!', 'miss'))
        next_ball(d)
        play_ball(d, 'flat', paddle_dx=d.level.reach_m * 1.5)
        self.assertEqual(d.message, 'Too far!')

    def test_next_ball_after_the_levels_gap(self):
        for level, gap in [(1, 1.8), (2, 1.0), (3, 0.5)]:
            d = new_drill(level)
            play_ball(d, 'flat')
            self.assertEqual((d.state, d.fed), ('gap', 1))
            t = d.time
            next_ball(d)
            self.assertEqual((d.state, d.fed), ('rally', 2))
            self.assertAlmostEqual(d.time - t, gap, delta=2 * FRAME, msg=d.level.name)


class TestPace(unittest.TestCase):
    '''The level sets how fast every ball flies and how long between serves.'''

    @staticmethod
    def arrival_s(level, seed):
        '''Real seconds from the feed until the ball reaches the player.'''
        d = new_drill(level, seed)
        t = 0.0
        while d.ball.z > PLAYER_HIT_Z:
            d.update(FRAME, 0.4)
            t += FRAME
        return d.call, t

    def test_harder_levels_are_faster_and_serve_sooner(self):
        self.assertLess(DRILL_PACE[1].speed, DRILL_PACE[2].speed)
        self.assertLess(DRILL_PACE[2].speed, DRILL_PACE[3].speed)
        self.assertGreater(DRILL_PACE[1].gap_s, DRILL_PACE[2].gap_s)
        self.assertGreater(DRILL_PACE[2].gap_s, DRILL_PACE[3].gap_s)

    def test_every_call_arrives_sooner_on_harder_levels(self):
        for seed in range(30):                          # same seed = same ball at every level
            (call, normal), (_, hard), (_, pro) = (self.arrival_s(lv, seed) for lv in (1, 2, 3))
            self.assertGreater(normal, hard + 0.15, call)
            self.assertGreater(hard, pro + 0.12, call)

    def test_reaction_time_stays_playable(self):
        times = [self.arrival_s(lv, seed)[1] for lv in (1, 2, 3) for seed in range(30)]
        self.assertGreater(min(times), 0.55)            # Pro: still over half a second
        self.assertLess(max(times), 1.8)                # Normal: not sluggish

    def test_returns_also_fly_at_the_levels_speed(self):
        flights = []
        for level in (1, 3):
            d = ball_of('hit', level)
            play_ball(d, 'flat')                        # runs until the return leaves the table
            flights.append(d.time - d._hit['t'])
        self.assertGreater(flights[0], flights[1] * 1.4)    # 1.30 / 0.75 = 1.7x quicker on Pro

    def test_level_change_between_balls_applies_to_the_next(self):
        d = new_drill(1)
        play_ball(d, 'flat')
        d.set_level(3)                                  # tag shown / key pressed during the gap
        t = d.time
        next_ball(d)
        self.assertAlmostEqual(d.time - t, 1.8, delta=2 * FRAME)    # this gap was already set...
        self.assertEqual(d.speed, DRILL_PACE[3].speed)              # ...the ball is Pro speed

    def test_returned_ball_leaves_over_the_net(self):
        d = ball_of('smash')
        play_ball(d, 'flat')
        self.assertTrue(d.state in ('gap', 'done'))
        self.assertGreater(d.ball.z, TABLE_L * 0.5)

    def test_round_ends_with_results(self):
        d = new_drill(seed=1, balls=6)
        for _ in range(6):
            next_ball(d)
            play_ball(d, 'backspin' if d.call == 'chop' else 'flat', strength=0.9)
        self.assertEqual(d.state, 'done')
        self.assertEqual(len(d.results), 6)
        self.assertEqual(d.accuracy(), 1.0)
        self.assertIn(('round_over',), d.drain_events())
        d.update(FRAME, 0.0)                                    # nothing more happens
        self.assertEqual(d.fed, 6)

    def test_accuracy_per_call(self):
        d = new_drill(balls=10)
        d.results = [('smash', 'correct'), ('smash', 'miss'), ('chop', 'correct'), ('chop', 'correct')]
        self.assertEqual(d.accuracy('smash'), 0.5)
        self.assertEqual(d.accuracy('chop'), 1.0)
        self.assertEqual(d.accuracy(), 0.75)

    def test_pause_freezes_everything(self):
        d = new_drill()
        d.update(FRAME, 0.3)
        d.pause()
        z = d.ball.z
        for _ in range(60):
            d.update(FRAME, 0.3)
        d.swing(0.9)
        self.assertEqual(d.ball.z, z)
        d.resume()
        d.update(FRAME, 0.3)
        self.assertLess(d.ball.z, z)

    def test_every_level_a_perfect_player_gets_full_marks(self):
        for level in (1, 2, 3):
            d = new_drill(level, seed=level, balls=8)
            while d.state != 'done':
                next_ball(d)
                play_ball(d, 'backspin' if d.call == 'chop' else 'flat', strength=0.9)
            self.assertEqual(d.accuracy(), 1.0, level)
            self.assertEqual(d.best, 8)


if __name__ == '__main__':
    unittest.main()
