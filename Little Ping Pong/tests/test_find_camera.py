'''Choosing the webcam: live-camera detection, picking, and the saved setting.'''

import os
import tempfile
import unittest

from find_camera import CameraInfo, pick_camera
from paddle_vision import VisionConfig
from red_tracker import RedTracker

# What this laptop reported on 2026-10-12 (USB "Full HD webcam" plugged in):
THIS_LAPTOP = [
    CameraInfo(0, 640, 480, 10.0, 8.0),       # virtual camera: dark, 10 fps
    CameraInfo(1, 640, 480, 30.5, 1.0),       # built-in camera: black (cover closed?)
    CameraInfo(2, 640, 480, 30.5, 142.0),     # USB Full HD webcam: live
    CameraInfo(3, 640, 480, 1.0, 7.0),        # virtual camera
]


class TestPickCamera(unittest.TestCase):

    def test_picks_the_live_usb_webcam_on_this_laptop(self):
        self.assertEqual(pick_camera(THIS_LAPTOP).index, 2)

    def test_dark_or_slow_cameras_are_not_live(self):
        self.assertFalse(CameraInfo(0, 640, 480, 30, 3).live)       # black picture
        self.assertFalse(CameraInfo(0, 640, 480, 5, 120).live)      # bright but 5 fps
        self.assertTrue(CameraInfo(0, 640, 480, 30, 120).live)

    def test_prefers_faster_then_sharper(self):
        cams = [CameraInfo(0, 640, 480, 30, 100), CameraInfo(1, 1280, 720, 30, 100), CameraInfo(2, 1920, 1080, 15, 100)]
        self.assertEqual(pick_camera(cams).index, 1)

    def test_none_when_nothing_is_live(self):
        self.assertIsNone(pick_camera([CameraInfo(0, 640, 480, 30, 2)]))
        self.assertIsNone(pick_camera([]))


class TestSavedCamera(unittest.TestCase):

    def test_camera_setting_round_trips_and_defaults_to_the_laptop(self):
        path = os.path.join(tempfile.mkdtemp(), 'v.json')
        self.assertEqual(VisionConfig.load(path).camera, 'laptop')
        for value in (2, 'usb', 'laptop'):
            VisionConfig(camera=value).save(path)
            self.assertEqual(VisionConfig.load(path).camera, value)

    def test_tracker_uses_the_saved_camera_unless_told_otherwise(self):
        cfg = VisionConfig(camera=2)
        self.assertEqual(RedTracker(cfg=cfg).camera_index, 2)
        self.assertEqual(RedTracker(1, cfg=cfg).camera_index, 1)
        self.assertEqual(RedTracker(0, cfg=cfg).camera_index, 0)     # 0 is a real choice, not "unset"


if __name__ == '__main__':
    unittest.main()
