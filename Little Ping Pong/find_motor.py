'''
List the LEGO Double Motors in Bluetooth range and save which one is yours.

    python find_motor.py               # list motors (serial + card colour)
    python find_motor.py --use 0994    # save yours: every script then connects to it
    python find_motor.py --test        # also connect to the saved motor and check the IMU streams

Saved in motor.json. --serial N on any script still overrides it for one run.
'''

import argparse
import time

from motor_select import load_saved_serial, save_serial, scan


def main():
    parser = argparse.ArgumentParser(description='Find and choose your Double Motor.')
    parser.add_argument('--use', default=None, help='save this Connection Card serial as yours')
    parser.add_argument('--test', action='store_true', help='connect to the saved motor and stream for 3 s')
    args = parser.parse_args()

    if args.use:
        save_serial(args.use)
        print(f'Saved: scripts will connect to Double Motor {args.use}.')

    print('Scanning for Double Motors (~5 s)...')
    motors = scan()
    saved = load_saved_serial()
    if not motors:
        print('  none found: is it switched on (light blinking)? Is another program connected to it?')
    for m in motors:
        mark = '   <- yours (saved)' if saved and m.serial == saved else ''
        print(f'  {m.label()}{mark}')
    if saved and motors and not any(m.serial == saved for m in motors):
        print(f'  (your saved motor {saved} is not advertising: off, out of range, or connected elsewhere)')
    if not saved:
        print('No motor saved yet: python find_motor.py --use <serial>')

    if args.test and saved:
        from paddle_imu import PaddleIMU
        p = PaddleIMU()
        t = time.perf_counter()
        p.connect()
        print(f'Connected to {saved} in {time.perf_counter() - t:.1f} s; streaming...')
        p.drain()
        time.sleep(3)
        n = len(p.drain())
        p.disconnect()
        print(f'{n} IMU samples in 3 s ({n / 3:.0f} Hz) -> {"OK" if n > 120 else "too few: check the motor"}')


if __name__ == '__main__':
    main()
