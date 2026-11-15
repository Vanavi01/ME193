"""FingerNet: 3 conv layers of growing width, pooled after each, features from ALL
three scales concatenated into one linear layer -> 6 logits (0-5 fingers).
Train with CrossEntropyLoss (it applies softmax internally)."""
import torch
import torch.nn as nn

from . import config


def conv_block(c_in, c_out):
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
    )


class FingerNet(nn.Module):
    def __init__(self, n_classes=6, widths=(8, 16, 32), pooled=4, in_ch=None):
        super().__init__()
        self.in_ch = in_ch or config.IMG_CHANNELS
        self.widths = tuple(widths)
        c1, c2, c3 = widths
        self.b1 = conv_block(self.in_ch, c1)   # 64 -> 32
        self.b2 = conv_block(c1, c2)           # 32 -> 16
        self.b3 = conv_block(c2, c3)           # 16 -> 8
        self.squash = nn.AdaptiveMaxPool2d(pooled)   # every scale -> pooled x pooled
        self.fc = nn.Linear(pooled * pooled * (c1 + c2 + c3), n_classes)

    def forward(self, x):
        f1 = self.b1(x)
        f2 = self.b2(f1)
        f3 = self.b3(f2)
        feats = [self.squash(f).flatten(1) for f in (f1, f2, f3)]
        return self.fc(torch.cat(feats, dim=1))


def save_checkpoint(net, path):
    """Weights + the input format they expect, so a model can't be fed the wrong images."""
    torch.save({"state_dict": {k: v.cpu() for k, v in net.state_dict().items()},
                "in_ch": net.in_ch, "widths": list(net.widths), "img_size": config.IMG_SIZE}, path)


def load_checkpoint(path):
    ck = torch.load(path, map_location="cpu")
    if "state_dict" not in ck:                         # old file: plain grayscale weights
        ck = {"state_dict": ck, "in_ch": 1, "widths": [8, 16, 32], "img_size": 64}
    if ck["in_ch"] != config.IMG_CHANNELS or ck["img_size"] != config.IMG_SIZE:
        raise RuntimeError(f"model expects {ck['in_ch']}-channel {ck['img_size']}px images but "
                           f"config uses {config.IMG_CHANNELS}-channel {config.IMG_SIZE}px; retrain "
                           "(python -m src.train) or change config")
    net = FingerNet(len(config.CLASSES), tuple(ck["widths"]), in_ch=ck["in_ch"])
    net.load_state_dict(ck["state_dict"])
    net.eval()
    return net


if __name__ == "__main__":
    net = FingerNet()
    print(net(torch.zeros(2, net.in_ch, 64, 64)).shape,
          sum(p.numel() for p in net.parameters()), "parameters")
