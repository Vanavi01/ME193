'''
Plot an IMU recording made by record_imu.py.

    python plot_imu.py                          # newest file in recordings/
    python plot_imu.py recordings/imu_xxx.csv
'''

import glob
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

from paddle_imu import ImuSample
from swing import detect_swings


def main():
    if len(sys.argv) > 1:
        path = sys.argv[1]
    else:
        files = glob.glob(os.path.join('recordings', 'imu_*.csv'))
        if not files:
            sys.exit('No recordings found. Run record_imu.py first.')
        path = max(files, key=os.path.getmtime)

    d = np.genfromtxt(path, delimiter=',', names=True)
    t = d['t']

    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(11, 8))
    fig.suptitle(os.path.basename(path))

    for name in ('ax', 'ay', 'az'):
        axes[0].plot(t, d[name], label=name)
    axes[0].plot(t, np.sqrt(d['ax'] ** 2 + d['ay'] ** 2 + d['az'] ** 2), 'k', lw=1, label='|accel|')
    axes[0].set_ylabel('accelerometer (1024 = 1 g)')

    for name in ('gx', 'gy', 'gz'):
        axes[1].plot(t, d[name], label=name)
    axes[1].plot(t, np.sqrt(d['gx'] ** 2 + d['gy'] ** 2 + d['gz'] ** 2), 'k', lw=1, label='|gyro|')
    axes[1].set_ylabel('gyroscope (deg/s)')

    for name in ('yaw', 'pitch', 'roll'):
        axes[2].plot(t, d[name], label=name)
    axes[2].set_ylabel('orientation (0.1 deg)')
    axes[2].set_xlabel('time (s)')

    for ax in axes:
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(alpha=0.3)

    # Sample markers along the bottom expose dropped samples / BLE gaps.
    axes[2].plot(t, np.full_like(t, axes[2].get_ylim()[0]), '|', color='gray', ms=6)

    # Mark every swing the step 2a detector finds.
    samples = [ImuSample(*(row[f] for f in ImuSample.FIELDS)) for row in d]
    for e in detect_swings(samples):
        for ax in axes:
            ax.axvline(e.t, color='red', alpha=0.4, lw=2)
        axes[1].annotate(f'{e.strength:.2f}', (e.t, e.peak_dps), color='red', fontsize=8)

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
