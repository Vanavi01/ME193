'''
Step 2a hardware test: swing the paddle and watch swings get detected live.

    python live_swing.py
    python live_swing.py --serial 0994

Expected: holding still or turning the paddle slowly prints nothing; each real
swing prints one SWING line with a strength bar. Ctrl+C to quit.
'''

import argparse
import time

from paddle_imu import PaddleIMU
from swing import SwingDetector


def main():
    parser = argparse.ArgumentParser(description='Live swing detection test.')
    parser.add_argument('--serial', default=None, help='Connection Card serial (optional)')
    args = parser.parse_args()

    paddle = PaddleIMU()
    print('Connecting to Double Motor...')
    paddle.connect(card_serial=args.serial)
    print('Connected. Swing the paddle! (Ctrl+C to quit)\n')

    detector = SwingDetector()
    count = 0
    try:
        while True:
            for s in paddle.drain():
                e = detector.update(s)
                if e:
                    count += 1
                    bar = '#' * round(e.strength * 20)
                    print(f'SWING {count:3d}  strength {e.strength:4.2f} [{bar:<20}]  '
                          f'{e.peak_dps:5.0f} deg/s  {e.peak_g:4.1f} g  '
                          f'axis ({e.axis[0]:+.2f} {e.axis[1]:+.2f} {e.axis[2]:+.2f})')
            time.sleep(0.01)
    except KeyboardInterrupt:
        pass
    finally:
        paddle.disconnect()


if __name__ == '__main__':
    main()
