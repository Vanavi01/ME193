'''Spin experiment tests: swing windows, features, collector, classifier, honest evaluation, files.'''

import io
import math
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stdout

import numpy as np

import train_strokes
from paddle_imu import ImuSample
from strokes import (FEATURE_NAMES, POST_S, PRE_S, NearestCentroid, StrokeCollector, Swing,
                     _angle_change, cut_window, features, leave_one_out, load_model,
                     load_swings, save_model, save_swings, separation_scores)

DT = 1 / 64
# Made-up but plausible differences between the types (sensor frame of the motor on the paddle):
TYPES = {
    #            gravity before swing (face tilt)   rotation axis at the peak   pitch change
    'topspin':  ((0.0, 0.50, 0.87),                 (0.9, 0.4, 0.2),            +250),
    'backspin': ((0.0, -0.50, 0.87),                (-0.8, 0.5, 0.3),           -250),
    'flat':     ((0.0, 0.0, 1.0),                   (0.1, 0.2, 1.0),            0),
}


def swing_stream(label, t0, rng, peak_dps=1400, with_return=True):
    '''1 s still + a swing peaking at t0+1.1 (+ a smaller return swing) + still.'''
    g, axis, d_pitch = TYPES[label]
    g = np.array(g) + rng.normal(0, 0.05, 3)
    axis = np.array(axis) / np.linalg.norm(axis) + rng.normal(0, 0.08, 3)
    peak = peak_dps * rng.uniform(0.8, 1.2)
    out = []
    for i in range(int(2.2 / DT)):
        t = t0 + i * DT
        w = 0.0
        if 1.0 <= t - t0 < 1.2:
            w = peak * math.sin(math.pi * (t - t0 - 1.0) / 0.2)
        elif with_return and 1.45 <= t - t0 < 1.6:
            w = -0.6 * peak * math.sin(math.pi * (t - t0 - 1.45) / 0.15)
        gyro = w * axis + rng.normal(0, 4, 3)
        acc = g * 1024 + rng.normal(0, 15, 3) + (abs(w) / 2 * axis if w else 0)
        frac = min(1.0, max(0.0, (t - t0 - 1.0) / 0.2))
        out.append(ImuSample(t, *map(int, acc), *map(int, gyro), 0, int(d_pitch * frac), 0))
    return out


def dataset(n_per_type=10, seed=0):
    rng = np.random.default_rng(seed)
    swings = []
    for label in TYPES:
        for _ in range(n_per_type):
            stream = swing_stream(label, 0.0, rng)
            window = cut_window(stream, 1.1)
            swings.append(Swing(label, window))
    return swings


class TestWindowsAndFeatures(unittest.TestCase):

    def test_window_is_centred_on_the_peak(self):
        w = cut_window(swing_stream('flat', 5.0, np.random.default_rng(0)), 6.1)
        self.assertAlmostEqual(w[0].t, -PRE_S, delta=DT)
        self.assertAlmostEqual(w[-1].t, POST_S, delta=DT)

    def test_incomplete_window_is_rejected(self):
        stream = swing_stream('flat', 0.0, np.random.default_rng(0))
        self.assertIsNone(cut_window(stream, 0.1))              # starts before the recording
        self.assertIsNone(cut_window(stream[:70], 1.1))        # recording ends mid-swing

    def test_angle_change_wraps(self):
        self.assertEqual(_angle_change(-1790, 1790), 20)       # 179 -> -179 deg is +2 deg, not -358
        self.assertEqual(_angle_change(100, 300), -200)

    def test_features_capture_tilt_and_axis(self):
        f = dict(zip(FEATURE_NAMES, features(dataset(1)[0].samples)))     # a topspin
        self.assertEqual(len(f), len(FEATURE_NAMES))
        self.assertTrue(all(math.isfinite(v) for v in f.values()))
        self.assertGreater(f['tilt_y'], 0.3)                    # closed face
        self.assertGreater(f['axis_x'], 0.5)
        self.assertGreater(f['d_pitch'], 1.5)


class TestStrokeCollector(unittest.TestCase):

    def test_one_window_per_stroke_and_return_swings_ignored(self):
        rng = np.random.default_rng(1)
        stream = []
        for k, label in enumerate(['topspin', 'backspin', 'flat']):
            stream += swing_stream(label, k * 2.2, rng)
        c = StrokeCollector()
        windows = [w for w in map(c.update, stream) if w is not None]
        self.assertEqual(len(windows), 3)                       # not 6: the return swings are skipped
        for w in windows:
            peak_i = int(np.argmax([math.hypot(s.gx, s.gy, s.gz) for s in w]))
            self.assertAlmostEqual(w[peak_i].t, 0.0, delta=2 * DT)


class TestClassifier(unittest.TestCase):

    def test_separable_types_are_classified_honestly_well(self):
        sw = dataset(10)
        X = [features(s.samples) for s in sw]
        acc, confusion = leave_one_out(X, [s.label for s in sw])
        self.assertGreaterEqual(acc, 0.9)
        self.assertEqual(sum(confusion.values()), 30)

    def test_random_labels_score_near_chance(self):
        '''Sanity check that leave-one-out isn't leaking: shuffled labels can't be learned.'''
        sw = dataset(10)
        X = [features(s.samples) for s in sw]
        y = list(np.random.default_rng(3).permutation([s.label for s in sw]))
        acc, _ = leave_one_out(X, y)
        self.assertLess(acc, 0.6)

    def test_prediction_confidence_and_save_load(self):
        sw = dataset(8)
        X = [features(s.samples) for s in sw]
        m = NearestCentroid().fit(X, [s.label for s in sw])
        new = cut_window(swing_stream('backspin', 0, np.random.default_rng(9)), 1.1)
        label, conf = m.predict(features(new))
        self.assertEqual(label, 'backspin')
        self.assertTrue(0 < conf <= 1)
        path = os.path.join(tempfile.mkdtemp(), 'm.json')
        save_model(path, m)
        self.assertEqual(load_model(path).predict(features(new)), (label, conf))

    def test_tilt_and_axis_are_among_the_most_useful_measurements(self):
        sw = dataset(10)
        scores = separation_scores([features(s.samples) for s in sw], [s.label for s in sw])
        top = sorted(scores, key=scores.get, reverse=True)[:6]
        self.assertTrue({'tilt_y', 'axis_x', 'axis_z', 'd_pitch'} & set(top))


class TestFilesAndTraining(unittest.TestCase):

    def test_save_load_round_trip_across_files(self):
        d = tempfile.mkdtemp()
        sw = dataset(2)
        save_swings(os.path.join(d, 'a.csv'), sw[:3])
        save_swings(os.path.join(d, 'b.csv'), sw[3:])
        back = load_swings(os.path.join(d, '*.csv'))
        self.assertEqual(sorted(s.label for s in back), sorted(s.label for s in sw))
        self.assertTrue(np.allclose(features(back[0].samples), features(sw[0].samples), atol=1e-6))

    def test_train_script_end_to_end(self):
        d = tempfile.mkdtemp()
        save_swings(os.path.join(d, 's.csv'), dataset(8))
        model_path = os.path.join(d, 'model.json')
        old_argv, old_path = sys.argv, train_strokes.MODEL_PATH
        sys.argv, train_strokes.MODEL_PATH = ['train_strokes.py', '--data', os.path.join(d, '*.csv')], model_path
        out = io.StringIO()
        try:
            with redirect_stdout(out):
                train_strokes.main()
        finally:
            sys.argv, train_strokes.MODEL_PATH = old_argv, old_path
        self.assertIn('Leave-one-out accuracy', out.getvalue())
        self.assertIn('Reliable', out.getvalue())
        self.assertTrue(os.path.exists(model_path))


if __name__ == '__main__':
    unittest.main()
