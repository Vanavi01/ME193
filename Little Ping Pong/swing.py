'''
Step 2a: swing detection from the paddle IMU.

A swing is a burst of fast rotation. Thresholds come from the step 1 recordings:
    held still          |gyro| < 10 deg/s
    slow rotation       |gyro| 100-400 deg/s
    real swings         |gyro| 900-1900 deg/s (sensor saturates near 2000)

The detector reports a swing at its *peak* (as soon as the rotation starts
slowing down), not at its end, so the game hears about it as early as possible.

Usage:
    detector = SwingDetector()
    for sample in paddle.drain():
        event = detector.update(sample)
        if event:
            print(event.strength)
'''

import math
from dataclasses import dataclass

from paddle_imu import ACCEL_PER_G

START_DPS = 600         # |gyro| above this starts a swing
END_DPS = 300           # |gyro| below this ends it
FULL_DPS = 2000         # |gyro| that counts as strength 1.0
PEAK_DROP = 0.85        # report once |gyro| falls below this fraction of the peak
REFRACTORY_S = 0.25     # ignore new swings this long after one ends


@dataclass
class SwingEvent:
    t: float            # time of the peak (same clock as the samples)
    peak_dps: float     # peak |gyro|
    peak_g: float       # peak |accel| during the swing, in g
    strength: float     # 0..1, from peak_dps
    axis: tuple         # unit rotation axis (x, y, z) at the peak; later used for spin


def gyro_mag(s):
    return math.sqrt(s.gx * s.gx + s.gy * s.gy + s.gz * s.gz)


def accel_g(s):
    return math.sqrt(s.ax * s.ax + s.ay * s.ay + s.az * s.az) / ACCEL_PER_G


def strength_from_dps(dps):
    return max(0.0, min(1.0, (dps - START_DPS) / (FULL_DPS - START_DPS)))


class SwingDetector:

    def __init__(self, start_dps=START_DPS, end_dps=END_DPS, refractory_s=REFRACTORY_S):
        self.start_dps = start_dps
        self.end_dps = end_dps
        self.refractory_s = refractory_s
        self._active = False
        self._reported = False
        self._ready_at = -math.inf
        self._peak = None       # (dps, sample) at the peak so far
        self._peak_g = 0.0

    def suppress_until(self, t):
        """Ignore any swing starting before time t (sample clock), e.g. while the motor
        on the paddle is giving haptic feedback and shaking the IMU."""
        self._ready_at = max(self._ready_at, t)
        self._active = False

    def update(self, s):
        """Feed one ImuSample. Returns a SwingEvent when a swing peaks, else None."""
        dps = gyro_mag(s)

        if not self._active:
            if dps >= self.start_dps and s.t >= self._ready_at:
                self._active = True
                self._reported = False
                self._peak = (dps, s)
                self._peak_g = accel_g(s)
            return None

        self._peak_g = max(self._peak_g, accel_g(s))
        if dps > self._peak[0]:
            self._peak = (dps, s)

        event = None
        if not self._reported and dps < PEAK_DROP * self._peak[0]:
            event = self._make_event()
            self._reported = True

        if dps < self.end_dps:
            if not self._reported:      # the swing ended before it visibly slowed
                event = self._make_event()
            self._active = False
            self._ready_at = s.t + self.refractory_s
        return event

    def _make_event(self):
        dps, s = self._peak
        return SwingEvent(
            t=s.t,
            peak_dps=dps,
            peak_g=self._peak_g,
            strength=strength_from_dps(dps),
            axis=(s.gx / dps, s.gy / dps, s.gz / dps),
        )


def detect_swings(samples, **kwargs):
    """Run a detector over a whole recording and return all SwingEvents."""
    detector = SwingDetector(**kwargs)
    return [e for e in map(detector.update, samples) if e]
