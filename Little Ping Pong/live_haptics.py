'''
Step 2c hardware test: feel each haptic pattern and measure how much it shakes the IMU.

    python live_haptics.py               (--serial 0994 to pick your motor)

Hold the paddle still (as at the end of a stroke). Each pattern plays 3 times.
For each one it prints the peak gyro reading during the pulse and how many FAKE
swings the detector would see without suppression. In the game those are
ignored automatically (play.py suppresses swings while the motor runs), so this
is just to show it was needed, and how strong each pattern really is.

Tip: if the pulses feel weak, fix an off-centre LEGO piece (e.g. a short beam)
on one motor output so it wobbles: much stronger rumble.
'''

import argparse
import time

from haptics import PATTERNS, SETTLE_S, Haptics, pattern_duration_s
from paddle_imu import PaddleIMU
from swing import SwingDetector, gyro_mag


def main():
    parser = argparse.ArgumentParser(description='Haptic feedback test.')
    parser.add_argument('--serial', default=None)
    args = parser.parse_args()

    paddle = PaddleIMU()
    print('Connecting to Double Motor...')
    paddle.connect(card_serial=args.serial)
    haptics = Haptics(paddle)
    print('Connected. Hold the paddle still.\n')
    time.sleep(1.0)
    try:
        for name in PATTERNS:
            strength = None
            peaks, fakes = [], 0
            for _ in range(3):
                paddle.drain()
                detector = SwingDetector()                  # no suppression: count fake swings
                haptics.play(name, strength)
                end = time.perf_counter() + pattern_duration_s(name) + SETTLE_S + 0.2
                peak = 0.0
                while time.perf_counter() < end:
                    for s in paddle.drain():
                        peak = max(peak, gyro_mag(s))
                        fakes += detector.update(s) is not None
                    time.sleep(0.005)
                peaks.append(peak)
                time.sleep(0.6)
            print(f'{name:6s} {pattern_duration_s(name) * 1000:4.0f} ms   peak gyro during pulse '
                  f'{max(peaks):5.0f} deg/s   fake swings without suppression: {fakes}/3')
        print('\nA swing starts at 600 deg/s; any pattern with fake swings needs the suppression '
              'that play.py applies (it does, for all of them).')
    finally:
        haptics.close()
        paddle.disconnect()


if __name__ == '__main__':
    main()
