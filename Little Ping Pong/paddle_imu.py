'''
Paddle IMU: connects to the LEGO Education Double Motor mounted on the paddle
and records every IMU sample it streams over Bluetooth.

Usage:
    from paddle_imu import PaddleIMU

    paddle = PaddleIMU()
    paddle.connect()              # first Double Motor found (or pass card_serial=...)
    paddle.start_recording()
    ...                           # swing!
    samples = paddle.stop_recording()
    paddle.disconnect()

All IMU values are the raw int16 numbers the motor sends. Units, measured from
the step 1 recordings (record_imu.py prints the numbers this is based on):
    accelerometer   ~1024 raw = 1 g        (see ACCEL_PER_G)
    gyroscope       degrees per second     (saturates near 2000 deg/s)
    yaw/pitch/roll  tenths of a degree     (+-1800 = +-180 deg)
Stream rate is ~64 Hz with occasional Bluetooth gaps of up to ~90 ms.
'''

import collections
import threading
import time
from dataclasses import dataclass

import legoeducation as le

# Fastest rate the library allows (15 ms ≈ 66 Hz). The library default is
# 100 ms (10 Hz), which is too slow to see a ping-pong swing.
FASTEST_NOTIFICATION_MS = 15

ACCEL_PER_G = 1024


@dataclass
class ImuSample:
    t: float            # seconds since recording started (time the sample arrived on the PC)
    ax: int             # accelerometer X/Y/Z (raw)
    ay: int
    az: int
    gx: int             # gyroscope X/Y/Z (deg/s)
    gy: int
    gz: int
    yaw: int            # orientation (tenths of a degree)
    pitch: int
    roll: int

    FIELDS = ('t', 'ax', 'ay', 'az', 'gx', 'gy', 'gz', 'yaw', 'pitch', 'roll')

    def row(self):
        return [getattr(self, f) for f in self.FIELDS]


class PaddleIMU(le.DoubleMotor):

    def __init__(self):
        super().__init__()
        # The library calls this after every notification, once imu_device is updated.
        self.set_notification_callback(self._on_notification)
        self._lock = threading.Lock()
        self._last_imu = None
        self._recording = False
        self._t0 = time.perf_counter()
        self._samples = []
        self._unread = collections.deque(maxlen=1000)   # for drain()
        self.latest = None   # most recent ImuSample, available even when not recording

    # ── connection ───────────────────────────────────────────────────────────

    def connect(self, card_serial=None, card_color=None, notification_ms=FASTEST_NOTIFICATION_MS,
                attempts=3, scan=None):
        """Connect to OUR Double Motor: card_serial (--serial), else the serial saved by
        find_motor.py (motor.json), else the only motor in range. Refuses to guess when
        several motors are around (classroom). scan: injectable motor scan, for tests."""
        from motor_select import choose_serial, explain, load_saved_serial
        from motor_select import scan as scan_motors
        saved = load_saved_serial()
        motors = [] if (card_serial or saved) else (scan or scan_motors)()
        serial = choose_serial(card_serial, saved, motors)
        last_error = None
        for attempt in range(attempts):
            try:
                super().connect(card_color=card_color, card_serial=serial,
                                device_notification_delay=notification_ms)
            except Exception as e:
                text = str(e).lower()
                if 'firmware' in text or 'updat' in text:
                    raise ConnectionError(explain(e, serial)) from e        # retrying won't help
                last_error = e
            if self.connected:
                self.serial = serial
                return
            time.sleep(1.5)                     # not found yet / "not ready": give it a moment
        raise ConnectionError(explain(last_error or 'not found in range', serial))

    # ── recording ────────────────────────────────────────────────────────────

    def start_recording(self):
        with self._lock:
            self._samples = []
            self._t0 = time.perf_counter()
            self._recording = True

    def stop_recording(self):
        """Stop recording and return the list of ImuSample."""
        with self._lock:
            self._recording = False
            return list(self._samples)

    def sample_count(self):
        with self._lock:
            return len(self._samples)

    def sample_time(self, perf_t):
        """Convert a time.perf_counter() value to the clock used in ImuSample.t."""
        return perf_t - self._t0

    def drain(self):
        """Return every sample that arrived since the last drain() call (for a game loop)."""
        with self._lock:
            new = list(self._unread)
            self._unread.clear()
            return new

    def _on_notification(self, _notification):
        # Runs on the BLE thread: keep it short. A notification can carry only
        # motor data, so only record when the library parsed a *new* IMU message.
        imu = self.imu_device
        if imu is self._last_imu or imu.accelerometerX != imu.accelerometerX:  # unchanged or NaN
            return
        self._last_imu = imu
        sample = ImuSample(
            time.perf_counter() - self._t0,
            imu.accelerometerX, imu.accelerometerY, imu.accelerometerZ,
            imu.gyroscopeX, imu.gyroscopeY, imu.gyroscopeZ,
            imu.yaw, imu.pitch, imu.roll,
        )
        self.latest = sample
        with self._lock:
            self._unread.append(sample)
            if self._recording:
                self._samples.append(sample)
