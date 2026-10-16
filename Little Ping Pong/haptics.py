'''
Step 2c: haptic feedback through the Double Motor on the paddle.

A short burst of motor rotation gives a "thock" you can feel; a quick
forward-then-back kick feels sharper (reaction torque). For a stronger rumble,
fix an off-centre LEGO piece (e.g. a short beam) on one motor output.

    haptics = Haptics(paddle)            # paddle = connected PaddleIMU / DoubleMotor
    haptics.play('buzz')                 # returns at once; a worker thread drives the motor
    haptics.close()

The motor shakes the IMU it is mounted next to, so the game must ignore swings
while a pattern plays: use haptics.quiet_until() with SwingDetector.suppress_until().
'''

import queue
import threading
import time

# A pattern is a list of (direction, speed %, milliseconds); direction 0 = pause.
# One pattern for every event (user's choice: the strongest-feeling one, formerly 'miss').
FORWARD, BACK, PAUSE = 1, -1, 0
PATTERNS = {
    'buzz': [(FORWARD, 45, 110), (PAUSE, 0, 90), (FORWARD, 45, 110)],
}
SETTLE_S = 0.12             # IMU keeps ringing a little after the motor stops


def pattern_steps(name, strength=None):
    '''strength is accepted for compatibility but no longer changes the pattern.'''
    return list(PATTERNS[name])


def pattern_duration_s(name):
    return sum(ms for _, _, ms in PATTERNS[name]) / 1000


def pattern_for_event(event):
    '''Game event tuple -> (pattern name, strength) or None. Hits and misses feel the same.'''
    if event[0] in ('hit', 'miss'):
        return 'buzz', None
    return None


class Haptics:

    def __init__(self, motor, enabled=True):
        self.motor = motor
        self.enabled = enabled
        self._queue = queue.Queue(maxsize=1)
        self._quiet_until = 0.0
        self._lock = threading.Lock()
        self._running = True
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def play(self, name, strength=None):
        '''Queue a pattern. If one is already waiting it is replaced (newest feedback wins).'''
        if not self.enabled:
            return
        steps = pattern_steps(name, strength)
        with self._lock:
            self._quiet_until = max(self._quiet_until,
                                    time.perf_counter() + pattern_duration_s(name) + SETTLE_S)
        try:
            self._queue.get_nowait()
        except queue.Empty:
            pass
        self._queue.put_nowait(steps)

    def quiet_until(self):
        '''perf_counter() time until which the IMU is shaken by the motor.'''
        with self._lock:
            return self._quiet_until

    def close(self):
        self._running = False
        self._queue.put(None)
        self._thread.join(timeout=1)

    def _worker(self):
        import legoeducation as le
        directions = {FORWARD: le.MOTOR_MOVE_DIRECTION_CLOCKWISE, BACK: le.MOTOR_MOVE_DIRECTION_COUNTERCLOCKWISE}
        while self._running:
            steps = self._queue.get()
            if steps is None:
                return
            for direction, speed, ms in steps:
                if direction != PAUSE:
                    try:
                        self.motor.motor_run_for_time(ms, direction=directions[direction], motor=le.MOTOR_BOTH,
                                                      speed=speed, blocking=False)
                    except Exception as e:      # feedback must never crash the game
                        print(f'haptics: {e}')
                time.sleep(ms / 1000)
