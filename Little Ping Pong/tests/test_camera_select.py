'''Choosing the webcam by kind ('laptop' / 'usb') or name, not by a number that shifts.'''

import unittest

from camera_select import CameraDevice, resolve_camera

# This laptop on 2026-10-13, USB webcam unplugged (cv2-enumerate-cameras, DirectShow order):
TODAY = [
    CameraDevice(0, 'Integrated Camera', True),
    CameraDevice(1, 'Pixel 10 Pro (Windows Virtual Camera)', False),
    CameraDevice(2, 'Camera (NVIDIA Broadcast)', False),
    CameraDevice(3, 'OBS Virtual Camera', False),
]
# With the USB webcam plugged in, the numbers shift:
WITH_USB = [
    CameraDevice(0, 'Pixel 10 Pro (Windows Virtual Camera)', False),
    CameraDevice(1, 'Integrated Camera', True),
    CameraDevice(2, 'Full HD webcam', True),
    CameraDevice(3, 'Camera (NVIDIA Broadcast)', False),
    CameraDevice(4, 'Integrated IR Camera', True),
]


class TestResolveCamera(unittest.TestCase):

    def test_laptop_camera_is_found_by_kind_whatever_its_number(self):
        self.assertEqual(resolve_camera('laptop', TODAY), (0, ''))
        self.assertEqual(resolve_camera('laptop', WITH_USB), (1, ''))
        self.assertEqual(resolve_camera('Integrated', WITH_USB)[0], 1)

    def test_usb_webcam(self):
        self.assertEqual(resolve_camera('usb', WITH_USB), (2, ''))
        self.assertEqual(resolve_camera('external', WITH_USB)[0], 2)

    def test_usb_unplugged_falls_back_to_the_laptop_camera(self):
        index, note = resolve_camera('usb', TODAY)
        self.assertEqual(index, 0)
        self.assertIn('laptop', note)

    def test_never_picks_the_ir_camera_or_a_virtual_camera(self):
        ir_first = [CameraDevice(0, 'Integrated IR Camera', True), CameraDevice(1, 'Integrated Camera', True)]
        self.assertEqual(resolve_camera('laptop', ir_first)[0], 1)
        only_virtual = [CameraDevice(0, 'OBS Virtual Camera', False)]
        with self.assertRaises(ValueError):
            resolve_camera('usb', only_virtual)
        with self.assertRaises(ValueError):
            resolve_camera('laptop', only_virtual)

    def test_numbers_and_names(self):
        self.assertEqual(resolve_camera(3, TODAY), (3, ''))
        self.assertEqual(resolve_camera('2', TODAY), (2, ''))
        self.assertEqual(resolve_camera('full hd', WITH_USB)[0], 2)          # any part of the name
        with self.assertRaises(ValueError):
            resolve_camera('logitech', TODAY)

    def test_kinds(self):
        c = CameraDevice(0, 'Integrated IR Camera', True)
        self.assertTrue(c.builtin and c.infrared)
        self.assertFalse(CameraDevice(0, 'Full HD webcam', True).builtin)


if __name__ == '__main__':
    unittest.main()
