"""Honest accuracy: the model on separately recorded TEST sessions (data/_test), which are
never used for training.
  python -m src.collect --person victor --test     # record a test session
  python -m src.evaluate                            # score the current model on all of them
Writes data/_audit/test_mistakes.png (wrong answers, what the model saw)."""
from collections import defaultdict

import cv2
import numpy as np
import torch

from . import config
from .model import load_checkpoint
from .preprocess import model_view, preprocess_roi


def load_test_items(test_dir=None):
    """[(session, path, label)] from data/_test/<person>_<session>/<label>/*.png"""
    test_dir = test_dir or config.TEST_DIR_PATH
    if not test_dir.exists():
        return []
    return [(s.name, p, c) for s in sorted(d for d in test_dir.iterdir() if d.is_dir())
            for c in config.CLASSES for p in sorted((s / str(c)).glob("*.png"))]


@torch.no_grad()
def evaluate_model(net, items):
    net.eval()
    dev = next(net.parameters()).device
    cm = np.zeros((len(config.CLASSES),) * 2, int)
    per_session = defaultdict(lambda: [0, 0])
    mistakes = []
    for session, path, label in items:
        img = cv2.imread(str(path))
        x = torch.from_numpy(preprocess_roi(img))[None].to(dev)
        probs = torch.softmax(net(x), 1)[0].cpu().numpy()
        pred = int(probs.argmax())
        cm[label, pred] += 1
        per_session[session][0] += int(pred == label)
        per_session[session][1] += 1
        if pred != label:
            mistakes.append((path, label, pred, float(probs.max())))
    total = cm.sum()
    return {"acc": float(np.trace(cm) / total) if total else 0.0, "n": int(total), "cm": cm,
            "per_class": {c: (cm[c, c] / cm[c].sum() if cm[c].sum() else None)
                          for c in config.CLASSES},
            "per_session": {s: a / n for s, (a, n) in per_session.items()},
            "mistakes": mistakes}


def report(r):
    lines = [f"TEST accuracy (new sessions, never trained on): {r['acc']:.1%} on {r['n']} pictures"]
    lines.append("  per count: " + "  ".join(
        f"{c}:{a:.0%}" for c, a in r["per_class"].items() if a is not None))
    for s, a in r["per_session"].items():
        lines.append(f"  session {s}: {a:.0%}")
    lines.append("  confusion (rows = shown, cols = predicted):")
    lines += ["    " + " ".join(f"{v:3d}" for v in row) for row in r["cm"]]
    return "\n".join(lines)


def mistakes_sheet(mistakes, n=24, cols=6, cell=150):
    if not mistakes:
        return None
    ims = []
    for path, label, pred, conf in mistakes[:n]:
        v = model_view(cv2.imread(str(path)), size=cell)
        cv2.putText(v, f"{label}->{pred} {conf:.0%}", (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (0, 0, 255), 2)
        ims.append(v)
    while len(ims) % cols:
        ims.append(np.zeros_like(ims[0]))
    return np.vstack([np.hstack(ims[i:i + cols]) for i in range(0, len(ims), cols)])


def main():
    items = load_test_items()
    if not items:
        raise SystemExit("No test sessions yet. Record one: python -m src.collect --test")
    r = evaluate_model(load_checkpoint(config.MODEL_PATH), items)
    print(report(r))
    sheet = mistakes_sheet(r["mistakes"])
    if sheet is not None:
        config.AUDIT_DIR.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(config.AUDIT_DIR / "test_mistakes.png"), sheet)
        print(f"wrong answers: {config.AUDIT_DIR / 'test_mistakes.png'}")


if __name__ == "__main__":
    main()
