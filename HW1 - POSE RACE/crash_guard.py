'''
Crash protection using the LEGO Education Color Sensor (same `legoeducation`
library as the drive motors, wrapped by lelib.colorSensor). Mount the sensor
facing forward/down: while the car rolls over open floor it reads one steady
color, and a wall or object under/in front of it reads as a different color
(different paint, cardboard, carpet edge, etc.) or spikes reflectivity because
the surface is suddenly much closer. That color change is our proximity cue -
we don't have a real distance sensor, so "the color changed" stands in for
"something is too close."

Debounced the same way drive.py debounces gestures (see CLAUDE.md): a state
change must hold for CRASH_HOLD_FRAMES consecutive frames before it's trusted,
so one noisy reading at a shadow or seam doesn't trip the guard.

Wire-in contract: call update() once per frame and treat its return value as
"a crash is active" - the caller (drive.py) forces FORWARD/LEFT/RIGHT to STOP
while active, but must NOT touch BACKWARD/STOP, so the driver can still back
off the wall. Once the sensor sees the baseline floor color again (i.e. the
car has backed clear), active drops back to False on its own.
'''
from lelib import colorSensor

BASELINE_FRAMES = 15     # frames of floor color averaged before guarding starts
CRASH_HOLD_FRAMES = 3    # consecutive frames a state must hold before it's trusted


class CrashGuard:
    def __init__(self, card_serial=None):
        self.card_serial = card_serial
        self.sensor = colorSensor()
        self.baseline = None
        self._baseline_samples = []
        self._candidate = None
        self._streak = 0
        self.active = False

    def connect(self):
        print(f"Connecting to color sensor (crash guard)...")
        self.sensor.connect(card_serial=self.card_serial)
        print("Color sensor connected.")

    def disconnect(self):
        self.sensor.disconnect()

    def update(self):
        """Call once per frame. Returns whether a crash is currently active."""
        color = self.sensor.detect_color()

        if self.baseline is None:
            self._baseline_samples.append(color)
            if len(self._baseline_samples) >= BASELINE_FRAMES:
                self.baseline = max(set(self._baseline_samples), key=self._baseline_samples.count)
            return self.active

        candidate = color != self.baseline
        if candidate == self._candidate:
            self._streak += 1
        else:
            self._candidate, self._streak = candidate, 1

        if self._streak >= CRASH_HOLD_FRAMES:
            self.active = candidate

        return self.active
