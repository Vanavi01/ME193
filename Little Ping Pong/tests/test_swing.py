'''Step 2a tests: swing detection.   Run all tests with:  python -m unittest'''

import csv
import math
import os
import unittest

from paddle_imu import ImuSample
from swing import FULL_DPS, START_DPS, SwingDetector, detect_swings, strength_from_dps

DT = 1 / 64     # the motor streams at ~64 Hz
DATA = os.path.join(os.path.dirname(__file__), 'data')


def sample(t, gz=0.0, gx=0.0, az=1024):
    return ImuSample(t, 0, 0, az, gx, 0, gz, 0, 0, 0)


def pulse(t0, peak_dps, width_s=0.15, axis='z'):
    '''One swing: a half-sine of rotation lasting width_s, starting at t0.'''
    out = []
    t = t0
    while t < t0 + width_s:
        w = peak_dps * math.sin(math.pi * (t - t0) / width_s)
        out.append(sample(t, gz=w) if axis == 'z' else sample(t, gx=w))
        t += DT
    return out


def still(t0, seconds, noise=5):
    n = int(seconds / DT)
    return [sample(t0 + i * DT, gz=noise * (-1) ** i) for i in range(n)]


def load_recording(name):
    with open(os.path.join(DATA, name)) as f:
        rows = csv.DictReader(f)
        return [ImuSample(*(float(r[k]) for k in ImuSample.FIELDS)) for r in rows]


class TestStrength(unittest.TestCase):

    def test_strength_scale(self):
        self.assertEqual(strength_from_dps(START_DPS), 0.0)
        self.assertEqual(strength_from_dps(FULL_DPS), 1.0)
        self.assertEqual(strength_from_dps(5000), 1.0)
        self.assertAlmostEqual(strength_from_dps((START_DPS + FULL_DPS) / 2), 0.5)


class TestSwingDetector(unittest.TestCase):

    def test_holding_still_is_not_a_swing(self):
        self.assertEqual(detect_swings(still(0, 3)), [])

    def test_slow_rotation_is_not_a_swing(self):
        self.assertEqual(detect_swings(pulse(0, 400, width_s=0.8)), [])

    def test_one_swing_gives_one_event(self):
        events = detect_swings(still(0, 1) + pulse(1, 1500) + still(1.2, 1))
        self.assertEqual(len(events), 1)
        e = events[0]
        self.assertAlmostEqual(e.peak_dps, 1500, delta=50)
        self.assertGreater(e.strength, 0.5)
        self.assertAlmostEqual(e.t, 1.075, delta=2 * DT)      # peak is mid-pulse

    def test_event_reported_soon_after_peak(self):
        '''The game must hear about the swing within a couple of samples of the peak.'''
        detector = SwingDetector()
        for s in pulse(0, 1500):
            if detector.update(s):
                self.assertLess(s.t - 0.075, 3 * DT)
                return
        self.fail('no swing reported')

    def test_harder_swing_is_stronger(self):
        soft = detect_swings(pulse(0, 900))[0]
        hard = detect_swings(pulse(0, 1900))[0]
        self.assertLess(soft.strength, hard.strength)

    def test_axis_is_unit_vector_along_rotation(self):
        e = detect_swings(pulse(0, 1500, axis='x'))[0]
        self.assertAlmostEqual(e.axis[0], 1.0, places=3)
        self.assertAlmostEqual(math.hypot(*e.axis), 1.0, places=6)

    def test_two_separate_swings(self):
        events = detect_swings(pulse(0, 1500) + still(0.15, 0.5) + pulse(0.65, 1500))
        self.assertEqual(len(events), 2)

    def test_jitter_right_after_swing_is_ignored(self):
        '''A second burst inside the refractory window is the same swing wobbling.'''
        events = detect_swings(pulse(0, 1500) + pulse(0.16, 800, width_s=0.08))
        self.assertEqual(len(events), 1)

    def test_bluetooth_gap_during_swing(self):
        '''Drop samples around the peak (real gaps reach ~90 ms): still one swing.'''
        samples = [s for s in pulse(0, 1500) if not 0.05 < s.t < 0.11]
        self.assertEqual(len(detect_swings(samples)), 1)


class TestRecordedSwings(unittest.TestCase):
    '''Regression test on a real recording: held still, rotated slowly twice,
    then swung (one swing at 6.3 s was followed by the paddle swinging back).'''

    def test_recording(self):
        events = detect_swings(load_recording('swings_4.csv'))
        self.assertEqual([round(e.t, 1) for e in events], [6.3, 6.7, 7.7, 9.9])
        self.assertTrue(all(e.peak_dps > 900 for e in events))
        # the return swing rotates the opposite way to the swing before it
        self.assertLess(events[0].axis[2] * events[1].axis[2], 0)


if __name__ == '__main__':
    unittest.main()
