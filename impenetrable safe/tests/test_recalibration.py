"""Recalibration logic: noticing that the camera moved/rotated or the lighting changed."""
import unittest

import numpy as np

from src.quality import calibration_drift, compare_calibrations, detect_red_square
from tests import synth


def calib_for(frame):
    det = detect_red_square(frame)
    x, y, w, h = det["roi"]
    return {"roi": list(det["roi"]), "outer": list(det["outer"]),
            "bg_mean": float(frame[y:y + h, x:x + w].mean()), "open_hand_frac": 0.35}


class DriftTests(unittest.TestCase):
    def setUp(self):
        self.frame = synth.scene(square=(100, 80, 420, 300))
        self.calib = calib_for(self.frame)

    def test_same_scene_is_valid(self):
        self.assertEqual(calibration_drift(self.frame, self.calib), [])

    def test_moved_camera_detected(self):
        moved = synth.scene(square=(170, 120, 420, 300))
        self.assertTrue(any("moved" in i for i in calibration_drift(moved, self.calib)))

    def test_border_out_of_view_detected(self):
        gone = np.full_like(self.frame, 235)
        self.assertTrue(any("not detected" in i for i in calibration_drift(gone, self.calib)))

    def test_frame_mode_judges_lighting_on_paper_only(self):
        # crop = whole frame (incl. red tape); lighting/evenness use the inside of the border
        calib = {**self.calib, "border_roi": self.calib["roi"], "roi": [0, 0, 640, 480]}
        self.assertEqual(calibration_drift(self.frame, calib), [])

    def test_shifted_empty_page_is_not_a_hand(self):
        """Regression: a slightly moved paper/border made blank pages pass as 4s and 5s."""
        from src.quality import check_sample
        calib = {**self.calib, "border_roi": self.calib["roi"], "roi": [0, 0, 640, 480],
                 "crop_mode": "frame", "hand_frac_min": 0.06}
        shifted = synth.scene(square=(106, 84, 420, 300))           # paper nudged 6 px / 4 px
        issues, m = check_sample(shifted, 5, calib, self.frame)
        self.assertIn("no hand in square", issues)
        self.assertLess(m["hand_blob"], 0.01)

    def test_darker_lighting_detected(self):
        dark = (self.frame.astype(np.float32) * 0.6).astype(np.uint8)
        self.assertTrue(any("lighting" in i for i in calibration_drift(dark, self.calib)))


class CompareCalibrationTests(unittest.TestCase):
    base = {"roi": [0, 0, 480, 315], "open_hand_frac": 0.35}

    def test_small_change_no_warning(self):
        self.assertEqual(compare_calibrations(self.base, {"roi": [5, 5, 470, 310],
                                                          "open_hand_frac": 0.33}), [])

    def test_rotation_90_warns(self):
        w = compare_calibrations(self.base, {"roi": [0, 0, 315, 480], "open_hand_frac": 0.35})
        self.assertTrue(any("rotated" in x for x in w))

    def test_shape_change_warns(self):
        w = compare_calibrations(self.base, {"roi": [0, 0, 400, 380], "open_hand_frac": 0.35})
        self.assertTrue(any("shape changed" in x for x in w))

    def test_camera_height_alone_is_fine(self):
        # everything scales together: smaller crop, same hand coverage
        self.assertEqual(compare_calibrations(self.base, {"roi": [0, 0, 360, 236],
                                                          "open_hand_frac": 0.36}), [])

    def test_hand_coverage_change_warns(self):
        w = compare_calibrations(self.base, {"roi": [0, 0, 480, 315], "open_hand_frac": 0.55})
        self.assertTrue(any("bigger" in x for x in w))


if __name__ == "__main__":
    unittest.main()
