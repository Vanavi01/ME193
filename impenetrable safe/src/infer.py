"""Frame -> (digit, confidence). Same class is used on the PC and the UnoQ."""
from typing import Tuple

import torch

from . import config
from .model import load_checkpoint
from .preprocess import preprocess


class Predictor:
    def __init__(self, path=config.MODEL_PATH):
        self.net = load_checkpoint(path)

    @torch.no_grad()
    def probabilities(self, frame):
        """Softmax probability for each finger count 0-5 (sums to 1)."""
        x = torch.from_numpy(preprocess(frame))[None]
        return torch.softmax(self.net(x), dim=1)[0].numpy()

    def predict(self, frame) -> Tuple[int, float]:
        p = self.probabilities(frame)
        return int(p.argmax()), float(p.max())
