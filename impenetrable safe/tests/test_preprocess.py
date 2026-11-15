import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from src import config, preprocess
from tests import synth


class PreprocessTests(unittest.TestCase):
    def test_shape_dtype_range(self):
        x = preprocess.preprocess_roi(synth.hand(3))
        self.assertEqual(x.shape, (config.IMG_CHANNELS, config.IMG_SIZE, config.IMG_SIZE))
        self.assertEqual(x.dtype, np.float32)
        self.assertGreaterEqual(x.min(), -1.0)
        self.assertLessEqual(x.max(), 1.0)

    def test_white_maps_near_one_and_hand_is_darker(self):
        white = preprocess.preprocess_roi(np.full((200, 200, 3), 255, np.uint8))
        self.assertTrue(np.allclose(white, 1.0))
        self.assertLess(preprocess.preprocess_roi(synth.hand(5)).mean(), 0.9)

    def test_non_square_roi_still_64x64(self):
        x = preprocess.preprocess_roi(np.full((180, 220, 3), 200, np.uint8))
        self.assertEqual(x.shape, (config.IMG_CHANNELS, config.IMG_SIZE, config.IMG_SIZE))

    def test_wide_crop_padded_not_stretched(self):
        """A 2:1 crop with a centered circle: after padding the circle stays round."""
        import cv2
        img = np.full((100, 200, 3), 235, np.uint8)
        cv2.circle(img, (100, 50), 30, (60, 85, 120), -1)
        a = preprocess.preprocess_roi(img, channels=1)[0] < 0      # dark = circle
        ys, xs = np.nonzero(a)
        w, h = xs.max() - xs.min() + 1, ys.max() - ys.min() + 1
        self.assertLessEqual(abs(w - h), 1)
        with mock.patch.object(config, "PAD_TO_SQUARE", False):     # old behavior squashes it
            a = preprocess.preprocess_roi(img, channels=1)[0] < 0
            ys, xs = np.nonzero(a)
            self.assertGreater((ys.max() - ys.min()) / (xs.max() - xs.min()), 1.6)

    def test_tape_strip_at_edge_trimmed(self):
        img = np.full((273, 408, 3), 235, np.uint8)
        img[-7:] = synth.TAPE                                 # paper nudged: tape in bottom rows
        t = preprocess.trim_tape_edges(img)
        self.assertEqual(t.shape[:2], (266, 408))
        x = preprocess.preprocess_roi(img)
        self.assertGreater(x.min(), 0.8)                       # no blue band left, just paper

    def test_hand_not_trimmed(self):
        img = synth.hand(4, size=200)
        self.assertEqual(preprocess.trim_tape_edges(img).shape, img.shape)

    def test_pad_square_shapes(self):
        self.assertEqual(preprocess.pad_square(np.zeros((291, 459, 3), np.uint8)).shape, (459, 459, 3))
        self.assertEqual(preprocess.pad_square(np.zeros((300, 120, 3), np.uint8)).shape, (300, 300, 3))

    def test_color_keeps_red_tape_and_skin_apart(self):
        """Colors measured on the old red-tape setup (HSV): tape (175, 220, 170), skin (6, 150, 200).
        Color separates them better than gray (mostly via the green channel)."""
        import cv2

        def patch(h, s, v):
            bgr = cv2.cvtColor(np.uint8([[[h, s, v]]]), cv2.COLOR_HSV2BGR)[0, 0]
            return np.full((64, 64, 3), bgr, np.uint8)
        tape, skin = patch(175, 220, 170), patch(6, 150, 200)
        g = abs(preprocess.preprocess_roi(tape, channels=1).mean()
                - preprocess.preprocess_roi(skin, channels=1).mean())
        c = np.abs(preprocess.preprocess_roi(tape, channels=3).mean(axis=(1, 2))
                   - preprocess.preprocess_roi(skin, channels=3).mean(axis=(1, 2))).max()
        self.assertGreater(c, g)

    def test_grayscale_still_available(self):
        x = preprocess.preprocess_roi(synth.hand(3), channels=1)
        self.assertEqual(x.shape, (1, config.IMG_SIZE, config.IMG_SIZE))

    def test_model_view_is_color_image(self):
        v = preprocess.model_view(synth.hand(2), size=100)
        self.assertEqual(v.shape, (100, 100, 3))

    def test_crop_uses_explicit_roi(self):
        frame = np.arange(480 * 640 * 3, dtype=np.uint32).reshape(480, 640, 3)
        c = preprocess.crop_roi(frame, (10, 20, 30, 40))
        self.assertEqual(c.shape[:2], (40, 30))
        self.assertTrue(np.array_equal(c, frame[20:60, 10:40]))

    def test_crop_uses_calibration_file(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "calibration.json"
            p.write_text('{"roi": [5, 6, 7, 8]}')
            with mock.patch.object(config, "CALIB_PATH", p):
                self.assertEqual(preprocess.get_roi(), (5, 6, 7, 8))
                c = preprocess.crop_roi(np.zeros((100, 100, 3), np.uint8))
                self.assertEqual(c.shape[:2], (8, 7))

    def test_missing_calibration_is_a_clear_error(self):
        with mock.patch.object(config, "CALIB_PATH", Path("does/not/exist.json")):
            with self.assertRaisesRegex(RuntimeError, "calibrate"):
                preprocess.get_roi()


if __name__ == "__main__":
    unittest.main()
