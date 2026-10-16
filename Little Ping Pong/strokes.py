'''
Spin experiment (forehand, red side only): can the paddle IMU tell hit types apart?

Pure logic, unit tested; no hardware here. Used by:
    record_strokes.py   record labelled swings  -> recordings/strokes/*.csv
    train_strokes.py    honest accuracy check   -> models/stroke_model.json
    live_strokes.py     print the detected hit type for each swing

Each swing is cut into a window around its peak and turned into a few
measurements (features), then classified with a nearest-centroid model: a swing
gets the label of the most similar "average swing". Accuracy is measured with
leave-one-out (each swing is predicted by a model that never saw it), so it is
not flattered by memorising the training swings.
'''

import csv
import glob
import json
import math
import os
from dataclasses import dataclass

import numpy as np

from paddle_imu import ACCEL_PER_G, ImuSample
from swing import END_DPS, SwingDetector, gyro_mag

STROKE_TYPES = ('topspin', 'backspin', 'flat')
PRE_S = 0.35            # window before the swing peak (includes the backswing)
POST_S = 0.15           # and after it (follow-through)
GRAVITY_S = (-0.35, -0.22)  # part of the window used to read the paddle's tilt before the swing
COOLDOWN_S = 0.6        # ignore swings this soon after a recorded one (the paddle swinging back)
FEATURE_POST_S = 0.10   # features use the window only up to here: same accuracy on the user's
                        # data as 0.15 s (100%), and the spin is known 50 ms sooner in the game

FEATURE_NAMES = (
    'axis_x', 'axis_y', 'axis_z',               # rotation axis at the peak
    'tilt_x', 'tilt_y', 'tilt_z',               # gravity direction before the swing = face tilt
    'axis_dot_gravity',                         # rotating about a vertical vs horizontal axis
    'acc_x', 'acc_y', 'acc_z',                  # acceleration direction at the peak
    'd_pitch', 'd_roll',                        # orientation change over the swing (x10 deg)
    'peak_kdps', 'duration_s',                  # how fast / how long
    'rot_x', 'rot_y', 'rot_z',                  # total rotation over the window (x100 deg)
)


@dataclass
class Swing:
    label: str
    samples: list           # ImuSample with t relative to the peak (peak at t=0)


# ── windows and features ─────────────────────────────────────────────────────

def cut_window(samples, t_peak):
    '''Samples within [t_peak - PRE_S, t_peak + POST_S], re-timed so the peak is t=0.
    None if the window isn't fully covered (recording started/ended mid-swing).'''
    if not samples or samples[0].t > t_peak - PRE_S + 0.05 or samples[-1].t < t_peak + POST_S - 0.05:
        return None
    out = []
    for s in samples:
        if t_peak - PRE_S <= s.t <= t_peak + POST_S:
            out.append(ImuSample(s.t - t_peak, *(getattr(s, f) for f in ImuSample.FIELDS[1:])))
    return out


def _angle_change(a_end, a_start):
    '''Change between two tenth-of-degree angles, wrapped to -1800..1800.'''
    return (a_end - a_start + 1800) % 3600 - 1800


def _unit(v):
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-9 else np.zeros(3)


def features(window):
    '''Fixed-length feature vector (FEATURE_NAMES) for one swing window
    (only the part up to FEATURE_POST_S after the peak is used).'''
    window = [s for s in window if s.t <= FEATURE_POST_S + 1e-9]
    t = np.array([s.t for s in window])
    acc = np.array([[s.ax, s.ay, s.az] for s in window], float) / ACCEL_PER_G
    gyr = np.array([[s.gx, s.gy, s.gz] for s in window], float)
    mag = np.linalg.norm(gyr, axis=1)

    i_peak = int(np.argmin(np.abs(t)))
    axis = _unit(gyr[i_peak])
    pre = (t >= GRAVITY_S[0]) & (t <= GRAVITY_S[1])
    tilt = _unit(acc[pre].mean(axis=0) if pre.any() else acc[0])
    acc_peak = _unit(acc[i_peak])
    d_pitch = _angle_change(window[-1].pitch, window[0].pitch) / 100       # x10 deg
    d_roll = _angle_change(window[-1].roll, window[0].roll) / 100
    dt = np.diff(t, prepend=t[0])
    rot = (gyr * dt[:, None]).sum(axis=0) / 100                            # x100 deg
    duration = float(dt[mag > END_DPS].sum())
    return np.array([*axis, *tilt, float(axis @ tilt), *acc_peak, d_pitch, d_roll,
                     mag.max() / 1000, duration, *rot])


# ── collecting swings from a live stream ──────────────────────────────────────

class StrokeCollector:
    '''Feed IMU samples; returns a swing window once POST_S after each swing peak.
    Swings within COOLDOWN_S of the previous one (the paddle swinging back) are skipped.'''

    def __init__(self):
        self.detector = SwingDetector()
        self.buffer = []
        self._pending = None            # peak time waiting for its follow-through samples
        self._last_peak = -math.inf

    def update(self, s):
        self.buffer.append(s)
        horizon = s.t - (PRE_S + POST_S + 1.0)
        if self.buffer[0].t < horizon:
            self.buffer = [b for b in self.buffer if b.t >= horizon]
        event = self.detector.update(s)
        if event and self._pending is None and event.t - self._last_peak >= COOLDOWN_S:
            self._pending = event.t
        if self._pending is not None and s.t >= self._pending + POST_S:
            t_peak, self._pending = self._pending, None
            self._last_peak = t_peak
            return cut_window(self.buffer, t_peak)
        return None


class LiveStrokeClassifier:
    '''In the game: keep recent IMU samples; when the swing detector reports a swing,
    classify it as soon as FEATURE_POST_S of follow-through has arrived.'''

    def __init__(self, model):
        self.model = model
        self.buffer = []
        self._pending = None

    def add(self, s):
        self.buffer.append(s)
        horizon = s.t - (PRE_S + POST_S + 0.5)
        if self.buffer[0].t < horizon:
            self.buffer = [b for b in self.buffer if b.t >= horizon]

    def on_swing(self, t_peak):
        self._pending = t_peak

    def poll(self):
        '''(label, confidence) once the pending swing can be classified, else None.'''
        if self._pending is None or not self.buffer or self.buffer[-1].t < self._pending + FEATURE_POST_S:
            return None
        t_peak, self._pending = self._pending, None
        window = [ImuSample(s.t - t_peak, *(getattr(s, f) for f in ImuSample.FIELDS[1:]))
                  for s in self.buffer if t_peak - PRE_S <= s.t <= t_peak + FEATURE_POST_S]
        if len(window) < 8 or window[0].t > -PRE_S + 0.1:
            return None                     # not enough history (e.g. right after connecting)
        return self.model.predict(features(window))


# ── classifier ────────────────────────────────────────────────────────────────

class NearestCentroid:
    '''Standardise features, then pick the class whose average swing is closest.'''

    def fit(self, X, y):
        X = np.asarray(X, float)
        self.mean = X.mean(axis=0)
        self.std = np.maximum(X.std(axis=0), 1e-3)
        Z = (X - self.mean) / self.std
        self.labels = sorted(set(y))
        self.centroids = np.array([Z[np.array(y) == c].mean(axis=0) for c in self.labels])
        return self

    def predict(self, x):
        '''(label, confidence 0..1): confidence = how much closer the best class is than the runner-up.'''
        z = (np.asarray(x, float) - self.mean) / self.std
        d = np.linalg.norm(self.centroids - z, axis=1)
        order = np.argsort(d)
        if len(order) == 1:
            return self.labels[order[0]], 1.0
        best, second = d[order[0]], d[order[1]]
        return self.labels[order[0]], float(1 - best / second) if second > 0 else 0.0

    def to_dict(self):
        return {'labels': self.labels, 'mean': self.mean.tolist(), 'std': self.std.tolist(),
                'centroids': self.centroids.tolist(), 'features': list(FEATURE_NAMES)}

    @classmethod
    def from_dict(cls, d):
        m = cls()
        m.labels = d['labels']
        m.mean, m.std, m.centroids = (np.array(d[k]) for k in ('mean', 'std', 'centroids'))
        return m


def leave_one_out(X, y):
    '''Honest accuracy: each swing is predicted by a model trained on all the others.
    Returns (accuracy, confusion dict {(true, predicted): count}).'''
    X, y = np.asarray(X, float), list(y)
    confusion = {}
    correct = 0
    for i in range(len(y)):
        keep = [j for j in range(len(y)) if j != i]
        if len({y[j] for j in keep}) < 2:
            continue
        pred, _ = NearestCentroid().fit(X[keep], [y[j] for j in keep]).predict(X[i])
        confusion[(y[i], pred)] = confusion.get((y[i], pred), 0) + 1
        correct += pred == y[i]
    return correct / len(y), confusion


def separation_scores(X, y):
    '''Per feature: spread between the type averages / spread within a type (higher = more useful).'''
    X, y = np.asarray(X, float), np.array(y)
    labels = sorted(set(y))
    means = np.array([X[y == c].mean(axis=0) for c in labels])
    within = np.array([X[y == c].var(axis=0) for c in labels]).mean(axis=0)
    return dict(zip(FEATURE_NAMES, means.var(axis=0) / np.maximum(within, 1e-9)))


# ── files ─────────────────────────────────────────────────────────────────────

def save_swings(path, swings):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['swing', 'label', *ImuSample.FIELDS])
        for i, sw in enumerate(swings):
            for s in sw.samples:
                w.writerow([i, sw.label, *s.row()])


def load_swings(pattern):
    swings = []
    for path in sorted(glob.glob(pattern)):
        groups = {}
        with open(path) as f:
            for r in csv.DictReader(f):
                key = (path, r['swing'])
                groups.setdefault(key, (r['label'], []))[1].append(
                    ImuSample(float(r['t']), *(int(float(r[k])) for k in ImuSample.FIELDS[1:])))
        swings += [Swing(label, samples) for label, samples in groups.values()]
    return swings


def save_model(path, model):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'w') as f:
        json.dump(model.to_dict(), f, indent=2)


def load_model(path):
    with open(path) as f:
        return NearestCentroid.from_dict(json.load(f))
