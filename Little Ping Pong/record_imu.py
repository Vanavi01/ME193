'''
Step 1 test: connect to the Double Motor and record its IMU data to a CSV.

    python record_imu.py                    # 10 s, first Double Motor found
    python record_imu.py --seconds 20
    python record_imu.py --serial 1234      # pick a motor by its Connection Card

Suggested routine while recording (it makes the summary useful):
    1. First 2 s: hold the paddle completely still.   -> gravity in raw accel units
    2. Slowly rotate it about 90 degrees, then stop.   -> gyro scale vs yaw/pitch/roll
    3. Then swing it like a forehand a few times.     -> what a real swing looks like

Afterwards, view the recording with:  python plot_imu.py
'''

import argparse
import csv
import os
import time
from datetime import datetime

import numpy as np

from paddle_imu import ImuSample, PaddleIMU

STILL_SECONDS = 2.0


def save_csv(samples, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(ImuSample.FIELDS)
        writer.writerows(s.row() for s in samples)


def print_summary(samples):
    if len(samples) < 2:
        print('\nToo few samples to summarise. Is the motor streaming?')
        return
    d = np.array([s.row() for s in samples], dtype=float)
    t = d[:, 0]
    accel, gyro, angles = d[:, 1:4], d[:, 4:7], d[:, 7:10]
    dt = np.diff(t)

    print('\n-- Summary ---------------------------------------')
    print(f'samples:          {len(t)} over {t[-1] - t[0]:.2f} s')
    print(f'sample rate:      {(len(t) - 1) / (t[-1] - t[0]):.1f} Hz '
          f'(interval median {np.median(dt) * 1000:.1f} ms, max gap {dt.max() * 1000:.0f} ms)')

    still = t < t[0] + STILL_SECONDS
    if still.sum() >= 5:
        g_raw = np.linalg.norm(accel[still], axis=1).mean()
        gyro_noise = gyro[still].std(axis=0)
        print(f'gravity (still):  |accel| = {g_raw:.0f} raw  -> 1 g ~ {g_raw:.0f} raw units')
        print(f'gyro at rest:     mean {np.round(gyro[still].mean(axis=0), 1)}  std {np.round(gyro_noise, 1)}')

    print(f'peak |accel|:     {np.linalg.norm(accel, axis=1).max():.0f} raw')
    print(f'peak |gyro|:      {np.linalg.norm(gyro, axis=1).max():.0f} raw')

    # Gyro integrated over time vs the change in yaw/pitch/roll reported by the motor.
    # If the gyro is in deg/s these should roughly match (ignore axes that wrapped past +-180).
    integrated = (gyro[:-1] * dt[:, None]).sum(axis=0)
    print(f'sum(gyro*dt) (x,y,z):  {np.round(integrated, 0)}')
    print(f'change(yaw,pitch,roll):{np.round(angles[-1] - angles[0], 0)}')
    print('  -> compare magnitudes to find the gyro scale (e.g. x10 apart means 0.1 deg/s units)')


def main():
    parser = argparse.ArgumentParser(description='Record Double Motor IMU data.')
    parser.add_argument('--seconds', type=float, default=10.0)
    parser.add_argument('--serial', default=None, help='Connection Card serial (optional)')
    parser.add_argument('--out', default=None, help='CSV path (default: recordings/imu_<time>.csv)')
    args = parser.parse_args()
    out = args.out or os.path.join('recordings', f'imu_{datetime.now():%Y%m%d_%H%M%S}.csv')

    paddle = PaddleIMU()
    print('Connecting to Double Motor...')
    paddle.connect(card_serial=args.serial)
    print('Connected.')

    try:
        print(f'\nRecording {args.seconds:.0f} s. Hold STILL for the first {STILL_SECONDS:.0f} s, '
              'then rotate ~90 deg, then swing.\n')
        paddle.start_recording()
        end = time.perf_counter() + args.seconds
        while time.perf_counter() < end:
            s = paddle.latest
            if s:
                a = (s.ax ** 2 + s.ay ** 2 + s.az ** 2) ** 0.5
                g = (s.gx ** 2 + s.gy ** 2 + s.gz ** 2) ** 0.5
                print(f'\r n={paddle.sample_count():5d}  |accel|={a:7.0f}  |gyro|={g:7.0f}  '
                      f'yaw={s.yaw:5d} pitch={s.pitch:5d} roll={s.roll:5d}   ', end='', flush=True)
            time.sleep(0.1)
        samples = paddle.stop_recording()
    finally:
        paddle.disconnect()

    save_csv(samples, out)
    print(f'\n\nSaved {len(samples)} samples to {out}')
    print_summary(samples)


if __name__ == '__main__':
    main()
