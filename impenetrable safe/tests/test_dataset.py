import shutil
import tempfile
import unittest
from pathlib import Path

import cv2
import torch

from src import config
from src.audit import audit
from src.train import HandDataset, load_items, split_items
from tests import synth


def make_dataset(root: Path, persons=("alice", "bob"), per_class=3, skip_class=None):
    for pi, person in enumerate(persons):
        for label in config.CLASSES:
            if label == skip_class:
                continue
            d = root / person / str(label)
            d.mkdir(parents=True)
            for k in range(per_class):
                img = synth.hand(label, seed=100 * pi + k, shift=(8 * k - 8, 4 * k), rotate=10 * k)
                cv2.imwrite(str(d / f"{1000 + k}.png"), img)
    (root / "_calibration").mkdir()
    cv2.imwrite(str(root / "_calibration" / "background.png"), synth.background())


class DatasetTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.root)

    def test_load_items_skips_underscore_dirs(self):
        make_dataset(self.root)
        items = load_items(self.root)
        self.assertEqual(len(items), 2 * 6 * 3)
        self.assertEqual({who for who, _, _ in items}, {"alice", "bob"})

    def test_held_out_person_has_no_leakage(self):
        make_dataset(self.root)
        train, val = split_items(load_items(self.root), "bob")
        self.assertTrue(all("bob" not in str(p) for p, _ in train))
        self.assertTrue(all("bob" in str(p) for p, _ in val))
        self.assertEqual(len(val), 18)

    def test_unknown_val_person_is_an_error(self):
        make_dataset(self.root)
        with self.assertRaises(ValueError):
            split_items(load_items(self.root), "nobody")

    def test_dataset_tensor_shape_and_range(self):
        make_dataset(self.root)
        items = [(p, l) for _, p, l in load_items(self.root)]
        for augment in (False, True):
            x, y = HandDataset(items, augment)[5]
            self.assertEqual(tuple(x.shape), (config.IMG_CHANNELS, config.IMG_SIZE, config.IMG_SIZE))
            self.assertEqual(x.dtype, torch.float32)
            self.assertTrue(-1.0 <= x.min() and x.max() <= 1.0)
            self.assertIn(y, config.CLASSES)


class AuditTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        self.bg = synth.background()

    def tearDown(self):
        shutil.rmtree(self.root)

    def test_counts_and_passes_when_complete(self):
        make_dataset(self.root)
        r = audit(self.root, None, self.bg, min_total=6, sheets_dir=None)
        self.assertEqual(r["fails"], [])
        self.assertEqual(r["totals"], {c: 6 for c in config.CLASSES})

    def test_fails_on_missing_class(self):
        make_dataset(self.root, skip_class=4)
        r = audit(self.root, None, self.bg, min_total=6, sheets_dir=None)
        self.assertTrue(any("class 4" in f for f in r["fails"]))

    def test_warns_on_single_person(self):
        make_dataset(self.root, persons=("alice",))
        r = audit(self.root, None, self.bg, min_total=1, sheets_dir=None)
        self.assertTrue(any("1 person" in w for w in r["warns"]))

    def test_finds_duplicates(self):
        make_dataset(self.root)
        d = self.root / "alice" / "2"
        src = sorted(d.glob("*.png"))[0]
        for k in range(5):
            shutil.copy(src, d / f"{src.stem}_{k}.png")   # sorts right after the original
        r = audit(self.root, None, self.bg, min_total=1, sheets_dir=None)
        self.assertGreaterEqual(r["duplicates"], 5)

    def test_writes_contact_sheets(self):
        make_dataset(self.root)
        out = self.root / "_audit"
        audit(self.root, None, self.bg, min_total=1, sheets_dir=out)
        for c in config.CLASSES:
            self.assertTrue((out / f"contact_{c}.png").exists())


if __name__ == "__main__":
    unittest.main()
