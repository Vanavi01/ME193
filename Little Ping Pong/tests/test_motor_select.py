'''Connecting to OUR Double Motor when several are in range (classroom).'''

import os
import tempfile
import unittest
from unittest import mock

import legoeducation as le

import motor_select
from motor_select import (MotorInfo, choose_serial, explain, load_saved_serial,
                          parse_motor, save_serial)
from paddle_imu import PaddleIMU

# What the scan found on 2026-10-13 (4 motors in range):
SCAN_NAMES = [
    ('51:2A:CF:10:AE:1C', '⬜️    Double Motor'),
    ('7E:59:1A:3C:0D:75', '\U0001f7e9 0994 Double Motor'),
    ('76:DC:D2:96:66:0E', '⬜️    Double Motor'),
    ('68:A2:45:D1:5A:1B', '\U0001f7ea 6235 Double Motor'),
]
CLASSROOM = [parse_motor(a, n) for a, n in SCAN_NAMES]


class TestParsing(unittest.TestCase):

    def test_real_advertised_names(self):
        self.assertEqual([(m.serial, m.color) for m in CLASSROOM],
                         [(None, 'none'), ('0994', 'green'), (None, 'none'), ('6235', 'purple')])
        self.assertIn('0994 (green card)', CLASSROOM[1].label())
        self.assertIn('no card', CLASSROOM[0].label())
        self.assertEqual(parse_motor('aa', '❓ 3664 Double Motor').label(), '3664 (card)')   # unknown colour symbol


class TestChoosing(unittest.TestCase):

    def test_requested_beats_saved(self):
        self.assertEqual(choose_serial('6235', '0994', CLASSROOM), '6235')

    def test_saved_serial_is_used(self):
        self.assertEqual(choose_serial(None, '0994', CLASSROOM), '0994')

    def test_refuses_to_guess_in_a_classroom(self):
        with self.assertRaises(ConnectionError) as cm:
            choose_serial(None, None, CLASSROOM)
        msg = str(cm.exception)
        self.assertIn('4 Double Motors', msg)
        self.assertIn('0994 (green card)', msg)
        self.assertIn('find_motor.py --use', msg)

    def test_single_motor_is_fine_without_a_serial(self):
        self.assertEqual(choose_serial(None, None, [CLASSROOM[1]]), '0994')

    def test_no_motor_at_all(self):
        with self.assertRaises(ConnectionError):
            choose_serial(None, None, [])

    def test_save_and_load(self):
        path = os.path.join(tempfile.mkdtemp(), 'motor.json')
        self.assertIsNone(load_saved_serial(path))
        save_serial('0994', path)
        self.assertEqual(load_saved_serial(path), '0994')

    def test_advice(self):
        self.assertIn('code.legoeducation.com', explain(Exception('needs a firmware update'), '0994'))
        self.assertIn('close any other program', explain(Exception('timeout'), '0994'))


class TestPaddleConnect(unittest.TestCase):
    '''PaddleIMU.connect with the LEGO library's connect replaced by a fake.'''

    def connect_with(self, behaviour, saved=None, requested=None, scan=()):
        calls = []

        def fake_connect(self_, card_color=None, card_serial=None, device_notification_delay=None):
            calls.append(card_serial)
            behaviour(self_, len(calls))
        p = PaddleIMU()
        with mock.patch.object(le.DoubleMotor, 'connect', fake_connect), \
             mock.patch.object(motor_select, 'load_saved_serial', lambda: saved), \
             mock.patch('time.sleep'):
            try:
                p.connect(card_serial=requested, scan=lambda: list(scan))
                return p, calls, None
            except ConnectionError as e:
                return p, calls, e

    @staticmethod
    def ok(self_, n):
        self_.connected = True

    def test_uses_the_saved_serial(self):
        p, calls, err = self.connect_with(self.ok, saved='0994')
        self.assertIsNone(err)
        self.assertEqual((calls, p.serial), (['0994'], '0994'))

    def test_classroom_without_saved_serial_never_connects(self):
        p, calls, err = self.connect_with(self.ok, scan=CLASSROOM)
        self.assertIsInstance(err, ConnectionError)
        self.assertEqual(calls, [])                         # didn't grab someone else's motor

    def test_retries_when_not_found_at_first(self):
        def found_second_time(self_, n):
            self_.connected = n >= 2
        p, calls, err = self.connect_with(found_second_time, saved='0994')
        self.assertIsNone(err)
        self.assertEqual(calls, ['0994', '0994'])

    def test_gives_advice_after_repeated_failures(self):
        def boom(self_, n):
            raise RuntimeError('Device not ready')
        p, calls, err = self.connect_with(boom, saved='0994')
        self.assertEqual(len(calls), 3)
        self.assertIn('Double Motor 0994', str(err))
        self.assertIn('close any other program', str(err))

    def test_firmware_problem_is_not_retried(self):
        def old_firmware(self_, n):
            raise RuntimeError('Your LEGO Education hardware needs a firmware update.')
        p, calls, err = self.connect_with(old_firmware, saved='0994')
        self.assertEqual(len(calls), 1)
        self.assertIn('code.legoeducation.com', str(err))


if __name__ == '__main__':
    unittest.main()
