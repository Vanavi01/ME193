import unittest

import numpy as np
import torch
import torch.nn as nn

from src import config
from src.model import FingerNet
from src.preprocess import preprocess_roi
from tests import synth


class ModelTests(unittest.TestCase):
    def test_output_shape(self):
        out = FingerNet(len(config.CLASSES))(
            torch.zeros(4, config.IMG_CHANNELS, config.IMG_SIZE, config.IMG_SIZE))
        self.assertEqual(tuple(out.shape), (4, 6))

    def test_can_overfit_tiny_synthetic_set(self):
        """If the model can't memorize 24 easy images, the pipeline is broken."""
        torch.manual_seed(0)
        xs, ys = [], []
        for n in config.CLASSES:
            for k in range(4):
                img = synth.hand(n, seed=k, rotate=8 * k - 12, shift=(6 * k - 9, 0))
                xs.append(preprocess_roi(img))
                ys.append(n)
        x = torch.from_numpy(np.stack(xs))
        y = torch.tensor(ys)
        net = FingerNet(len(config.CLASSES))
        opt = torch.optim.Adam(net.parameters(), lr=3e-3)
        loss_fn = nn.CrossEntropyLoss()
        first = loss_fn(net(x), y).item()
        for _ in range(150):
            opt.zero_grad()
            loss = loss_fn(net(x), y)
            loss.backward()
            opt.step()
        acc = (net(x).argmax(1) == y).float().mean().item()
        self.assertLess(loss.item(), 0.5 * first)
        self.assertGreaterEqual(acc, 0.9)


class CheckpointTests(unittest.TestCase):
    def test_round_trip_keeps_input_format(self):
        import tempfile
        from pathlib import Path
        from src.model import load_checkpoint, save_checkpoint
        net = FingerNet()
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "m.pt"
            save_checkpoint(net, p)
            loaded = load_checkpoint(p)
            x = torch.randn(2, net.in_ch, config.IMG_SIZE, config.IMG_SIZE)
            net.eval()
            self.assertTrue(torch.allclose(net(x), loaded(x)))

    def test_old_grayscale_model_refused_for_color_config(self):
        import tempfile
        from pathlib import Path
        from unittest import mock
        from src.model import load_checkpoint
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "old.pt"
            torch.save(FingerNet(in_ch=1).state_dict(), p)        # legacy plain state_dict
            with mock.patch.object(config, "IMG_CHANNELS", 3):
                with self.assertRaisesRegex(RuntimeError, "retrain"):
                    load_checkpoint(p)
            with mock.patch.object(config, "IMG_CHANNELS", 1):
                self.assertEqual(load_checkpoint(p).in_ch, 1)


if __name__ == "__main__":
    unittest.main()
