"""Train FingerNet on the PC.
  python -m src.train --val-person alex      # hold out a whole person (honest accuracy)
  python -m src.train                        # random 20% split (optimistic; warns)
Saves models/finger_net.pt plus loss/accuracy plots and a confusion matrix."""
import argparse
import random

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from . import config
from .evaluate import evaluate_model, load_test_items, report
from .model import FingerNet, load_checkpoint, save_checkpoint
from .preprocess import preprocess_roi


def class_weights(items):
    """Balanced loss: each count matters equally even if it has fewer pictures."""
    counts = np.bincount([l for _, l in items], minlength=len(config.CLASSES)).astype(np.float32)
    w = counts.sum() / (len(counts) * np.maximum(counts, 1))
    return torch.tensor(w, dtype=torch.float32)


def device():
    """GPU if this PyTorch build can use one, else CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class HandDataset(Dataset):
    """Images are read and preprocessed ONCE and kept in memory (reading 640x480 PNGs every
    epoch was the slowest part of training); only the random augmentation runs per epoch."""

    def __init__(self, items, augment):
        self.items, self.augment = items, augment
        # fill=1.0 = white in normalized space, matches the white background
        self.affine = transforms.RandomAffine(
            config.AUG_ROTATE, translate=(config.AUG_TRANSLATE, config.AUG_TRANSLATE),
            scale=config.AUG_SCALE, fill=1.0)
        self.cache = [torch.from_numpy(preprocess_roi(cv2.imread(str(p)))) for p, _ in items]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        t, label = self.cache[i], self.items[i][1]
        if self.augment:
            t = self.affine(t)
            t = (t * random.uniform(0.8, 1.2) + random.uniform(-0.1, 0.1)).clamp(-1, 1)
        return t, label


def load_items(data_dir=None):
    """[(person, path, label)]; folders starting with "_" (calibration, audit) are skipped."""
    data_dir = data_dir or config.DATA_DIR
    items = []
    for person in sorted(p for p in data_dir.iterdir() if p.is_dir() and not p.name.startswith("_")):
        for label in config.CLASSES:
            for img in sorted((person / str(label)).glob("*.png")):
                items.append((person.name, img, label))
    return items


def split_items(items, val_person=None, seed=0):
    """Hold out one whole person (honest), or a random 20% (optimistic) if none given."""
    if val_person:
        if val_person not in {who for who, _, _ in items}:
            raise ValueError(f"no data for person '{val_person}'")
        train = [(p, l) for who, p, l in items if who != val_person]
        val = [(p, l) for who, p, l in items if who == val_person]
        return train, val
    print("WARNING: random split; frames from one person leak into validation.")
    items = items[:]
    random.Random(seed).shuffle(items)
    cut = int(0.8 * len(items))
    return [(p, l) for _, p, l in items[:cut]], [(p, l) for _, p, l in items[cut:]]


def evaluate(net, loader, loss_fn):
    net.eval()
    dev = next(net.parameters()).device
    total, correct, loss_sum, preds, labels = 0, 0, 0.0, [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(dev), y.to(dev)
            out = net(x)
            loss_sum += loss_fn(out, y).item() * len(y)
            p = out.argmax(1)
            correct += (p == y).sum().item()
            total += len(y)
            preds += p.tolist()
            labels += y.tolist()
    return loss_sum / total, correct / total, preds, labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val-person")
    ap.add_argument("--epochs", type=int, default=config.EPOCHS)
    ap.add_argument("--all", action="store_true",
                    help="final model: train on EVERY picture for a fixed number of epochs and "
                         "save the last one (no validation; judge it with a --test session)")
    args = ap.parse_args()

    if args.all:
        train = [(p, l) for _, p, l in load_items()]
        val = train                               # training accuracy only, for the log
    else:
        train, val = split_items(load_items(), args.val_person)
    print(f"{len(train)} train / {len(val)} val images")

    dev = device()
    print(f"training on {dev}" + (f" ({torch.cuda.get_device_name(0)})" if dev.type == "cuda" else
                                  f" ({torch.get_num_threads()} threads)"))
    tl = DataLoader(HandDataset(train, True), batch_size=config.BATCH_SIZE, shuffle=True)
    vl = DataLoader(HandDataset(val, False), batch_size=config.BATCH_SIZE)
    net = FingerNet(len(config.CLASSES)).to(dev)
    print(f"input: {net.in_ch}-channel {config.IMG_SIZE}x{config.IMG_SIZE}")
    loss_fn = nn.CrossEntropyLoss(weight=class_weights(train).to(dev))
    opt = torch.optim.Adam(net.parameters(), lr=config.LR)

    hist = {"train_loss": [], "val_loss": [], "val_acc": []}
    best = 0.0
    config.MODELS_DIR.mkdir(exist_ok=True)
    for epoch in range(args.epochs):
        net.train()
        run, n = 0.0, 0
        for x, y in tl:
            x, y = x.to(dev), y.to(dev)
            opt.zero_grad()
            loss = loss_fn(net(x), y)
            loss.backward()          # backprop: dLoss/dWeight for every weight
            opt.step()               # gradient descent step
            run += loss.item() * len(y)
            n += len(y)
        vloss, vacc, preds, labels = evaluate(net, vl, loss_fn)
        hist["train_loss"].append(run / n)
        hist["val_loss"].append(vloss)
        hist["val_acc"].append(vacc)
        print(f"epoch {epoch + 1:3d}  train {run / n:.4f}  val {vloss:.4f}  acc {vacc:.3f}")
        if args.all and epoch == args.epochs - 1:
            best = vacc
            save_checkpoint(net, config.MODEL_PATH)
        elif not args.all and vacc >= best:
            best = vacc
            save_checkpoint(net, config.MODEL_PATH)

    cm = np.zeros((6, 6), int)
    for t, p in zip(labels, preds):
        cm[t, p] += 1
    print("confusion matrix (rows=true, cols=pred), last epoch:\n", cm)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(hist["train_loss"], label="train")
    ax[0].plot(hist["val_loss"], label="val")
    ax[0].set(title="Loss", xlabel="epoch")
    ax[0].legend()
    ax[1].plot(hist["val_acc"])
    ax[1].set(title="Validation accuracy", xlabel="epoch")
    fig.savefig(config.MODELS_DIR / "training_curves.png", dpi=150)
    print(f"best val acc {best:.3f}; saved {config.MODEL_PATH}")
    test_items = load_test_items()
    if test_items:                                  # the number that actually matters
        print()
        print(report(evaluate_model(load_checkpoint(config.MODEL_PATH), test_items)))
    else:
        print("no TEST session yet (python -m src.collect --test): val accuracy is optimistic")


if __name__ == "__main__":
    main()
