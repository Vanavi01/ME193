"""Honest test-set scoring and the class-balanced loss."""
import shutil
import tempfile
import unittest
from pathlib import Path

import cv2
import torch
import torch.nn as nn

from src import config
from src.evaluate import evaluate_model, load_test_items, report
from src.train import class_weights, load_items
from tests import synth


class ConstantNet(nn.Module):
    """Always answers `answer` (lets us check the scoring by hand)."""

    def __init__(self, answer):
        super().__init__()
        self.w = nn.Parameter(torch.zeros(1))
        self.answer = answer

    def forward(self, x):
        out = torch.zeros(x.shape[0], len(config.CLASSES))
        out[:, self.answer] = 10.0
        return out + self.w


class EvaluateTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        for session in ("victor_20261112-1000", "alex_20261112-1100"):
            for c in config.CLASSES:
                d = self.root / "_test" / session / str(c)
                d.mkdir(parents=True)
                for k in range(2):
                    cv2.imwrite(str(d / f"{k}.png"), synth.hand(c, seed=k))
        (self.root / "victor" / "3").mkdir(parents=True)
        cv2.imwrite(str(self.root / "victor" / "3" / "1.png"), synth.hand(3))

    def tearDown(self):
        shutil.rmtree(self.root)

    def test_test_sessions_never_used_for_training(self):
        self.assertEqual({who for who, _, _ in load_items(self.root)}, {"victor"})

    def test_loads_all_test_sessions(self):
        items = load_test_items(self.root / "_test")
        self.assertEqual(len(items), 2 * 6 * 2)
        self.assertEqual({s for s, _, _ in items}, {"victor_20261112-1000", "alex_20261112-1100"})

    def test_scoring(self):
        r = evaluate_model(ConstantNet(3), load_test_items(self.root / "_test"))
        self.assertAlmostEqual(r["acc"], 1 / 6)
        self.assertEqual(r["per_class"][3], 1.0)
        self.assertEqual(r["per_class"][0], 0.0)
        self.assertEqual(len(r["mistakes"]), 20)
        self.assertIn("TEST accuracy", report(r))


class ClassWeightTests(unittest.TestCase):
    def test_rare_class_weighs_more(self):
        items = [(None, 0)] * 10 + [(None, 1)] * 10 + [(None, 3)] * 20 + \
                [(None, c) for c in (2, 4, 5) for _ in range(10)]
        w = class_weights(items)
        self.assertAlmostEqual(float(w[3]) * 2, float(w[0]), places=5)


if __name__ == "__main__":
    unittest.main()
