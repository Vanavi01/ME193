import unittest
from unittest import mock

import cv2
import numpy as np

from src import config
from src.quality import (check_sample, detect_red_square, diagnose_red_square, hand_mask, iou,
                         is_duplicate, red_fraction, red_mask, sharpness, signature)
from tests import synth


def crop(img, roi):
    x, y, w, h = roi
    return img[y:y + h, x:x + w]


class RedSquareTests(unittest.TestCase):
    def test_finds_area_inside_border(self):
        sq = (200, 120, 240, 240)
        det = detect_red_square(synth.scene(sq))
        self.assertIsNotNone(det)
        x, y, w, h = det["roi"]
        self.assertGreater(x, sq[0])
        self.assertGreater(y, sq[1])
        self.assertLess(x + w, sq[0] + sq[2])
        self.assertLess(y + h, sq[1] + sq[3])
        self.assertGreater(w, 0.8 * sq[2])                     # didn't shrink too much
        self.assertLess(red_fraction(crop(synth.scene(sq), det["roi"])), config.RED_CLEAN_FRAC)
        self.assertLess(det["tilt_deg"], 1.0)

    def test_none_on_plain_white(self):
        self.assertIsNone(detect_red_square(np.full((480, 640, 3), 235, np.uint8)))

    def test_none_on_solid_red_blob(self):
        img = np.full((480, 640, 3), 235, np.uint8)
        cv2.rectangle(img, (200, 120), (440, 360), synth.TAPE, -1)
        self.assertIsNone(detect_red_square(img))

    def test_reports_tilt(self):
        det = detect_red_square(synth.scene(tilt=15))
        self.assertIsNotNone(det)
        self.assertGreater(det["tilt_deg"], config.MAX_TILT_DEG)

    def test_square_cut_off_by_frame_edge_rejected(self):
        img = synth.scene(square=(200, 300, 240, 240))       # bottom runs off a 480-high frame
        det, reason = diagnose_red_square(img)
        self.assertIsNone(det)
        self.assertIn("bottom", reason)

    def test_tape_touching_edge_ok_if_inside_fully_visible(self):
        img = np.full((480, 640, 3), 235, np.uint8)
        cv2.rectangle(img, (80, 20), (639, 420), synth.TAPE, -1)          # outer tape hits the right edge
        cv2.rectangle(img, (140, 75), (595, 365), (235, 235, 235), -1)    # inside fully in view
        det, reason = diagnose_red_square(img)
        self.assertEqual(reason, "OK")
        x, y, w, h = det["roi"]
        self.assertLess(x + w, 600)

    def test_bottom_strip_only_rejected(self):
        img = np.full((480, 640, 3), 235, np.uint8)            # what camera 0 saw before
        cv2.rectangle(img, (0, 420), (640, 480), synth.TAPE, -1)
        self.assertIsNone(diagnose_red_square(img)[0])

    def test_reasons(self):
        self.assertIn("no border tape", diagnose_red_square(np.full((480, 640, 3), 235, np.uint8))[1])
        self.assertEqual(diagnose_red_square(synth.scene())[1], "OK")
        self.assertIn("tilted", diagnose_red_square(synth.scene(tilt=15))[1])

    def test_thick_uneven_tape_uses_inner_hole(self):
        img = np.full((480, 640, 3), 235, np.uint8)
        red = synth.TAPE
        cv2.rectangle(img, (20, 35), (620, 470), red, -1)          # outer edge of the tape
        cv2.rectangle(img, (75, 95), (555, 410), (235, 235, 235), -1)  # inner hole 480x315
        cv2.rectangle(img, (200, 95), (440, 100), red, -1)         # overlapping tape step
        det, reason = diagnose_red_square(img)
        self.assertEqual(reason, "OK")
        x, y, w, h = det["roi"]
        self.assertGreater(w, 480 - 25)                             # close to the real hole
        self.assertGreater(h, 315 - 25)
        self.assertLess(red_fraction(crop(img, det["roi"])), config.RED_CLEAN_FRAC)

    def test_tape_color_vs_skin_and_paper(self):
        """HSV measured on this camera: blue tape (109, 178, 195), paper (3, 34, 235),
        skin ~ (6, 150, 200). Only the tape may count as border."""
        def px(h, s, v):
            return cv2.cvtColor(np.uint8([[[h, s, v]]]), cv2.COLOR_HSV2BGR)
        self.assertEqual(red_mask(px(109, 178, 195))[0, 0], 255)
        for hsv in [(3, 34, 235), (6, 150, 200), (8, 120, 110), (178, 140, 130), (15, 90, 160)]:
            self.assertEqual(red_mask(px(*hsv))[0, 0], 0, hsv)

    def test_iou(self):
        self.assertAlmostEqual(iou((0, 0, 10, 10), (0, 0, 10, 10)), 1.0)
        self.assertAlmostEqual(iou((0, 0, 10, 10), (20, 20, 5, 5)), 0.0)
        self.assertAlmostEqual(iou((0, 0, 10, 10), (5, 0, 10, 10)), 50 / 150)


class HandAndSharpnessTests(unittest.TestCase):
    def setUp(self):
        self.bg = synth.background(seed=99)

    def frac(self, img):
        return hand_mask(img, self.bg).mean()

    def test_empty_background_is_not_a_hand(self):
        self.assertLess(self.frac(synth.background(seed=1)), 0.01)

    def test_hand_detected_and_grows_with_fingers(self):
        f1, f5 = self.frac(synth.hand(1)), self.frac(synth.hand(5))
        self.assertGreater(f1, 0.08)
        self.assertGreater(f5, f1)
        self.assertLess(f5, 0.7)

    def test_dark_and_light_skin_detected(self):
        medium = self.frac(synth.hand(3))
        self.assertGreater(self.frac(synth.hand(3, skin=synth.SKIN_DARK)), 0.8 * medium)
        self.assertGreater(self.frac(synth.hand(3, skin=synth.SKIN_LIGHT)), 0.5 * medium)

    def test_blur_lowers_sharpness(self):
        sharp = synth.hand(3)
        blurry = cv2.GaussianBlur(sharp, (21, 21), 0)
        s_sharp = sharpness(cv2.cvtColor(sharp, cv2.COLOR_BGR2GRAY), hand_mask(sharp, self.bg))
        s_blur = sharpness(cv2.cvtColor(blurry, cv2.COLOR_BGR2GRAY), hand_mask(blurry, self.bg))
        self.assertLess(s_blur, 0.35 * s_sharp)


class CheckSampleTests(unittest.TestCase):
    def setUp(self):
        self.bg = synth.background(seed=99)
        sharp = synth.hand(3)
        s = sharpness(cv2.cvtColor(sharp, cv2.COLOR_BGR2GRAY), hand_mask(sharp, self.bg))
        self.calib = {"sharpness_min": 0.35 * s, "hand_frac_min": 0.03}

    def issues(self, img, label):
        return " | ".join(check_sample(img, label, self.calib, self.bg)[0])

    def test_good_hand_passes(self):
        for n in range(1, 6):
            self.assertEqual(self.issues(synth.hand(n, seed=n), n), "", f"label {n}")

    def test_empty_square_labeled_as_digit_rejected(self):
        self.assertIn("no hand", self.issues(synth.background(seed=3), 3))

    def test_empty_square_as_zero_allowed(self):
        self.assertNotIn("no hand", self.issues(synth.background(seed=3), 0))

    def test_border_tape_in_crop_rejected(self):
        """Camera moved a lot: tape deep inside the crop (more than the auto-trim removes)."""
        img = synth.hand(2)
        img[-45:] = synth.TAPE
        self.assertIn("border tape", self.issues(img, 2))

    def test_thin_tape_strip_at_edge_is_trimmed_not_flagged(self):
        img = synth.hand(2)
        img[-6:] = synth.TAPE
        self.assertNotIn("border tape", self.issues(img, 2))

    def test_dark_image_rejected(self):
        dark = (synth.hand(2).astype(np.float32) * 0.2).astype(np.uint8)
        msg = self.issues(dark, 2)
        self.assertIn("too dark", msg)
        self.assertIn("lighting changed", msg)

    def test_blurry_rejected(self):
        blurry = cv2.GaussianBlur(synth.hand(3), (21, 21), 0)
        self.assertIn("blurry", self.issues(blurry, 3))

    def test_covered_square_rejected(self):
        covered = np.full_like(self.bg, 90)
        self.assertIn("covered", self.issues(covered, 5))

    @mock.patch.object(config, "REJECT_CUT_OFF_FINGERS", True)
    def test_fingers_out_of_view_rejected(self):
        """Arm side is worked out per frame (arm enters from the bottom in synth images)."""
        ok = synth.hand(3)
        cut = synth.hand(3, shift=(0, -70))                 # fingertips run off the top
        self.assertEqual(check_sample(ok, 3, self.calib, self.bg)[0], [])
        self.assertIn("out of view at the top", " | ".join(check_sample(cut, 3, self.calib, self.bg)[0]))

    def test_out_of_view_allowed_by_default(self):
        cut = synth.hand(3, shift=(0, -70))
        self.assertNotIn("out of view", " | ".join(check_sample(cut, 3, self.calib, self.bg)[0]))

    def test_red_border_allowed_in_frame_mode(self):
        img = synth.with_red_border(synth.hand(2))
        calib = {**self.calib, "crop_mode": "frame"}
        self.assertNotIn("border tape", " | ".join(check_sample(img, 2, calib, self.bg)[0]))

    def test_metrics_returned(self):
        _, m = check_sample(synth.hand(4), 4, self.calib, self.bg)
        for k in ("hand_frac", "sharpness", "bg_level", "red_band"):
            self.assertIn(k, m)


class DuplicateTests(unittest.TestCase):
    def test_same_image_is_duplicate(self):
        a = synth.hand(2, seed=1)
        self.assertTrue(is_duplicate(signature(a), signature(synth.hand(2, seed=2))))

    def test_moved_hand_is_not_duplicate(self):
        a, b = synth.hand(2), synth.hand(2, shift=(25, 0))
        self.assertFalse(is_duplicate(signature(a), signature(b)))


if __name__ == "__main__":
    unittest.main()
