"""Guided automatic capture: random prompt order, bursts, steadiness, movement, balance."""
import unittest
import unittest.mock
from collections import Counter

from src import config
from src.autocapture import AutoCapture
from tests import synth

CALIB = {"hand_frac_min": 0.06, "sharpness_min": 10.0, "crop_mode": "frame"}
DT = 0.05


class Driver:
    def __init__(self, targets=None, counts=None, burst=3, seed=0):
        targets = targets or {c: 6 for c in config.CLASSES}
        self.ac = AutoCapture(targets, counts or {}, CALIB, synth.background(seed=99),
                              burst=burst, seed=seed)
        self.t = 100.0
        self.events = []
        self.k = 0

    def feed(self, img, n=1):
        res = None
        for _ in range(n):
            self.t += DT
            res = self.ac.step(img, self.t)
            self.events.append(res)
        return res

    def pose(self, label):
        """A clearly different position each time (big shift/rotation)."""
        self.k += 1
        k = self.k
        return synth.hand(label, seed=k, shift=(30 * (k % 3) - 30, 20 * (k % 2) - 10),
                          rotate=12 * (k % 4) - 18)

    def hold(self, label=None, n=None):
        """Show the requested count (or `label`) still, long enough to be captured."""
        label = self.ac.label if label is None else label
        self.t = max(self.t, self.ac.switch_until)
        return self.feed(self.pose(label), n or config.STEADY_FRAMES + int(config.AUTO_INTERVAL_S / DT) + 2)

    def captures(self):
        return [e for e in self.events if e["event"] == "capture"]


class AutoCaptureTests(unittest.TestCase):
    def test_needs_a_steady_hand(self):
        d = Driver()
        self.assertEqual(d.feed(synth.hand(d.ac.label))["reason"], "hold still")

    def test_moving_hand_is_not_captured(self):
        d = Driver()
        for k in range(20):
            d.feed(synth.hand(5, shift=(50 * (k % 2) - 25, 20 * (k % 2))))
        self.assertEqual(d.captures(), [])

    def test_same_pose_asks_to_move_more(self):
        d = Driver()
        img = d.pose(d.ac.label)
        d.feed(img, n=15)
        self.assertEqual(len(d.captures()), 1)
        self.assertEqual(d.feed(img, n=20)["reason"], "move your hand more")

    def test_switches_after_a_burst_and_pauses(self):
        d = Driver(burst=3)
        first = d.ac.label
        for _ in range(3):
            d.hold()
        caps = d.captures()
        self.assertEqual([c["label"] for c in caps], [first] * 3)
        self.assertIn("next", caps[-1])
        self.assertNotEqual(d.ac.label, first)                 # a DIFFERENT count next
        self.assertEqual(d.feed(d.pose(d.ac.label), n=3)["reason"], "switching")

    def test_order_is_mixed_and_ends_balanced(self):
        d = Driver(targets={c: 6 for c in config.CLASSES}, burst=3, seed=1)
        for _ in range(200):
            if d.ac.done:
                break
            d.hold()
        self.assertTrue(d.ac.done)
        labels = [c["label"] for c in d.captures()]
        self.assertEqual(Counter(labels), {c: 6 for c in config.CLASSES})
        runs = [labels[i] for i in range(len(labels)) if i == 0 or labels[i] != labels[i - 1]]
        self.assertGreaterEqual(len(runs), 12)                # each count visited >= twice
        self.assertNotEqual(runs[:6], sorted(runs[:6]))       # not simply 0,1,2,3,4,5

    def test_digit_needs_a_hand(self):
        d = Driver(counts={0: 6})
        self.assertNotEqual(d.ac.label, 0)
        d.t = d.ac.switch_until
        self.assertEqual(d.feed(synth.background(seed=1), n=15)["reason"], "no hand in square")

    def test_empty_pictures_limited_for_zero(self):
        targets = {c: (20 if c == 0 else 0) for c in config.CLASSES}
        d = Driver(targets=targets, burst=50)
        for k in range(config.AUTO_EMPTY_MAX + 3):
            d.feed(synth.background(seed=k), n=config.STEADY_FRAMES + 10)
        self.assertEqual(d.ac.empties, config.AUTO_EMPTY_MAX)
        self.assertIn("FIST", d.events[-1]["reason"])

    def test_resume_only_asks_for_unfinished(self):
        d = Driver(counts={0: 6, 1: 6, 2: 6, 3: 6, 4: 6, 5: 2})
        self.assertEqual(d.ac.label, 5)

    def test_done_after_all_counts(self):
        d = Driver(counts={c: 6 for c in config.CLASSES})
        self.assertTrue(d.ac.done)
        self.assertEqual(d.feed(synth.hand(2))["event"], "done")

    def test_undo_asks_for_that_count_again(self):
        d = Driver(burst=1)
        first = d.ac.label
        d.hold()
        self.assertNotEqual(d.ac.label, first)
        d.ac.undo(first)
        self.assertEqual((d.ac.label, d.ac.counts[first]), (first, 0))

    def test_soft_issues_do_not_block(self):
        d = Driver(targets={c: (6 if c == 3 else 0) for c in config.CLASSES})
        with unittest.mock.patch.object(config, "REJECT_CUT_OFF_FINGERS", True):
            d.feed(synth.hand(3, shift=(0, -70)), n=config.STEADY_FRAMES + 8)
        caps = d.captures()
        self.assertEqual(len(caps), 1)
        self.assertTrue(any("out of view" in w for w in caps[0]["warnings"]))

    def test_pause(self):
        d = Driver()
        d.ac.paused = True
        self.assertIn("paused", d.hold()["reason"])
        self.assertEqual(d.captures(), [])


if __name__ == "__main__":
    unittest.main()
