'''
Spin experiment, step 1: record labelled FOREHAND swings (red side facing the ball).

    python record_strokes.py                        # 10 each of topspin, backspin, flat
    python record_strokes.py --count 15 --serial 0994
    python record_strokes.py --types topspin flat   # just some types

For each type: press Enter, then swing that stroke the given number of times,
pausing ~1 s between swings (the paddle swinging back afterwards is ignored).
Then keep or redo the set. Everything is saved to recordings/strokes/strokes_<time>.csv;
run it again on another day / in other clothes to add more data (train uses all files).

    topspin   closed face, brush UP the back of the ball (low -> high)
    backspin  open face, chop DOWN under the ball (high -> low)
    flat      face square, drive straight through (a normal hit)
'''

import argparse
import os
import time
from datetime import datetime

from paddle_imu import PaddleIMU
from strokes import STROKE_TYPES, StrokeCollector, Swing, save_swings
from swing import gyro_mag

HOW = {
    'topspin': 'closed face, brush UP the back of the ball (low -> high)',
    'backspin': 'open face, chop DOWN under the ball (high -> low)',
    'flat': 'face square, drive straight through',
}


def record_type(paddle, label, count):
    collector = StrokeCollector()
    paddle.drain()
    got = []
    while len(got) < count:
        for s in paddle.drain():
            window = collector.update(s)
            if window is not None:
                got.append(Swing(label, window))
                peak = max(gyro_mag(w) for w in window)
                print(f'   {label} {len(got):2d}/{count}   peak {peak:5.0f} deg/s')
        time.sleep(0.005)
    return got


def main():
    parser = argparse.ArgumentParser(description='Record labelled forehand swings.')
    parser.add_argument('--types', nargs='+', default=list(STROKE_TYPES))
    parser.add_argument('--count', type=int, default=10)
    parser.add_argument('--serial', default=None)
    args = parser.parse_args()

    path = os.path.join('recordings', 'strokes', f'strokes_{datetime.now():%Y%m%d_%H%M%S}.csv')
    paddle = PaddleIMU()
    print('Connecting to Double Motor...')
    paddle.connect(card_serial=args.serial)
    print('Connected. Forehand only, red side facing the ball.\n')
    swings = []
    try:
        for label in args.types:
            while True:
                input(f'{label.upper()}: {HOW.get(label, "")}\n  Press Enter, then do {args.count} swings...')
                batch = record_type(paddle, label, args.count)
                if input('  Keep these? [Y/n] ').strip().lower() != 'n':
                    swings += batch
                    break
            save_swings(path, swings)               # save after every type, in case of Ctrl+C
    finally:
        paddle.disconnect()
    print(f'\nSaved {len(swings)} swings to {path}\nNext: python train_strokes.py')


if __name__ == '__main__':
    main()
