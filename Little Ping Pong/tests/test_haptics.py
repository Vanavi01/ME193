'''Step 2c tests: haptic patterns, the non-blocking motor worker, and swing suppression.'''

import threading
import time
import unittest

import legoeducation as le

from haptics import (PATTERNS, SETTLE_S, Haptics, pattern_duration_s,
                     pattern_for_event, pattern_steps)
from swing import SwingDetector, detect_swings
from tests.test_swing import pulse, still


class FakeMotor:
    '''Records motor_run_for_time calls with the time they arrived.'''

    def __init__(self, fail=False):
        self.calls = []
        self.fail = fail
        self.done = threading.Event()

    def motor_run_for_time(self, ms, *, direction, motor, speed, blocking):
        if self.fail:
            raise RuntimeError('BLE hiccup')
        self.calls.append((time.perf_counter(), ms, direction, motor, speed, blocking))


def wait_for(motor, n, timeout=2.0):
    end = time.perf_counter() + timeout
    while len(motor.calls) < n and time.perf_counter() < end:
        time.sleep(0.005)


class TestPatterns(unittest.TestCase):

    def test_one_pattern_for_everything_regardless_of_strength(self):
        self.assertEqual(list(PATTERNS), ['buzz'])
        self.assertEqual(pattern_steps('buzz', 0.1), pattern_steps('buzz', 1.0))
        self.assertEqual(pattern_steps('buzz'), [(1, 45, 110), (0, 0, 90), (1, 45, 110)])   # the old 'miss'

    def test_patterns_are_short(self):
        '''Feedback must be over well before the next ball arrives (~0.7 s at Pro).'''
        for name in PATTERNS:
            self.assertLess(pattern_duration_s(name), 0.4, name)

    def test_events_map_to_patterns(self):
        self.assertEqual(pattern_for_event(('hit', 0.5, None)), ('buzz', None))
        self.assertEqual(pattern_for_event(('hit', 0.9, 'backhand')), ('buzz', None))
        self.assertEqual(pattern_for_event(('miss', 'No swing!')), ('buzz', None))
        self.assertIsNone(pattern_for_event(('cpu_hit',)))
        self.assertIsNone(pattern_for_event(('new_best', 4)))


class TestHapticsWorker(unittest.TestCase):

    def test_play_returns_immediately_and_motor_runs_in_background(self):
        motor = FakeMotor()
        h = Haptics(motor)
        t = time.perf_counter()
        h.play('buzz')
        self.assertLess(time.perf_counter() - t, 0.01)      # the game loop is never held up
        wait_for(motor, 2)
        h.close()
        self.assertEqual([(c[1], c[4]) for c in motor.calls], [(110, 45), (110, 45)])
        self.assertTrue(all(c[5] is False for c in motor.calls))           # non-blocking BLE commands
        self.assertTrue(all(c[3] == le.MOTOR_BOTH for c in motor.calls))
        self.assertTrue(all(c[2] == le.MOTOR_MOVE_DIRECTION_CLOCKWISE for c in motor.calls))

    def test_steps_are_spaced_by_their_duration(self):
        motor = FakeMotor()
        h = Haptics(motor)
        h.play('buzz')                          # 110 ms on, 90 ms pause, 110 ms on
        wait_for(motor, 2)
        h.close()
        gap = motor.calls[1][0] - motor.calls[0][0]
        self.assertAlmostEqual(gap, 0.200, delta=0.05)

    def test_quiet_window_covers_the_pattern(self):
        h = Haptics(FakeMotor())
        t = time.perf_counter()
        h.play('buzz')
        self.assertAlmostEqual(h.quiet_until() - t, pattern_duration_s('buzz') + SETTLE_S, delta=0.02)
        h.close()

    def test_disabled_does_nothing(self):
        motor = FakeMotor()
        h = Haptics(motor, enabled=False)
        h.play('buzz')
        time.sleep(0.1)
        h.close()
        self.assertEqual(motor.calls, [])
        self.assertEqual(h.quiet_until(), 0.0)

    def test_motor_errors_never_reach_the_game(self):
        h = Haptics(FakeMotor(fail=True))
        h.play('buzz')                          # must not raise
        time.sleep(0.15)
        h.close()


class TestSwingSuppression(unittest.TestCase):

    def test_motor_shake_right_after_a_hit_is_not_a_swing(self):
        '''Real swing at t=1 s; the hit registers when the ball arrives (up to 0.2 s later),
        plus Bluetooth delay, so the motor shakes the gyro at ~1.45 s: after the detector's
        own 0.25 s refractory period has run out.'''
        samples = still(0, 1) + pulse(1.0, 1500) + still(1.15, 0.3) + pulse(1.45, 900, width_s=0.12) + still(1.6, 1)
        self.assertEqual(len(detect_swings(samples)), 2)    # without suppression: a fake second swing
        d = SwingDetector()
        events = []
        for s in samples:
            e = d.update(s)
            if e:
                events.append(e)
                d.suppress_until(s.t + 0.45)                # what play.py does after triggering haptics
        self.assertEqual(len(events), 1)

    def test_real_swings_count_again_after_the_quiet_window(self):
        d = SwingDetector()
        d.suppress_until(0.5)
        events = [e for e in map(d.update, still(0, 0.6) + pulse(0.6, 1500)) if e]
        self.assertEqual(len(events), 1)


if __name__ == '__main__':
    unittest.main()
