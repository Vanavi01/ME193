'''
Turns a drive command + speed into throttled BLE motor commands for the
LEGO Education Double Motor. See CLAUDE.md for why the flips and the
send throttle exist.
'''

from lelib import doubleMotor

MIN_SPEED, MAX_SPEED = 10, 100
SPEED_STEP = 10
TURN_RATIO = 0.0     # inner wheel as a fraction of speed. 0 = arc, -1 = spin

RIGHT_FLIP = -1
LEFT_FLIP = 1
SEND_INTERVAL = 0.1


def speeds_for(cmd, spd):
    inner = int(spd * TURN_RATIO)
    return {
        "FORWARD":  ( spd,  spd),
        "BACKWARD": (-spd, -spd),
        "LEFT":     ( inner,  spd),
        "RIGHT":    ( spd,  inner),
        "STOP":     (0, 0),
    }[cmd]


class CarMotor:
    def __init__(self, card_serial):
        self.card_serial = card_serial
        self.dm = doubleMotor()
        self._last_send = 0.0
        self._last_cmd = None

    def connect(self):
        print(f"Connecting to double motor {self.card_serial}...")
        self.dm.connect(card_serial=self.card_serial)
        print("Connected. q to quit.")

    def reset_send_cache(self):
        """Force the next send() through even if the speed tuple is unchanged."""
        self._last_cmd = None

    def send(self, command, speed, now):
        left, right = speeds_for(command, speed)
        if now - self._last_send > SEND_INTERVAL and (left, right) != self._last_cmd:
            self.dm.set_speed_left(LEFT_FLIP * left)
            self.dm.set_speed_right(RIGHT_FLIP * right)
            self.dm.run_left()
            self.dm.run_right()
            self._last_send, self._last_cmd = now, (left, right)

    def stop(self):
        self.dm.stop()

    def disconnect(self):
        self.dm.disconnect()
