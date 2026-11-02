"""Model 1: single-feature (peak |accel| on ONE axis per window) decision tree.

Usage: python train_model1.py [data_dir] [axis]
  axis: ax | ay | az  (omit to compare all three by leave-one-session-out accuracy)
Writes model1.json next to this script (copy it into unoq_app/python/).
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.tree import DecisionTreeClassifier

FS = 100
WINDOW, HOP = 2 * FS, FS
AXES = ["ax_g", "ay_g", "az_g"]
# bike and air_bike are merged into "bike" for the 4-class task; remove this
# mapping to keep them separate (document whichever you choose).
LABEL_MAP = {"air_bike": "bike"}

here = os.path.dirname(os.path.abspath(__file__))
data_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(here, "..", "data")
axis_arg = sys.argv[2] + "_g" if len(sys.argv) > 2 else None


def peaks(df, axis):
    x = df[axis].abs().to_numpy()
    return np.array([x[i:i + WINDOW].max() for i in range(0, len(x) - WINDOW + 1, HOP)])


rows = {a: [] for a in AXES}  # per axis: list of (peak, label, session)
for path in sorted(glob.glob(os.path.join(data_dir, "*.csv"))):
    if os.path.getsize(path) == 0:
        print(f"SKIPPING {os.path.basename(path)}: file is empty")
        continue
    df = pd.read_csv(path)
    if (df[AXES].abs().sum(axis=1) == 0).mean() > 0.5:
        print(f"SKIPPING {os.path.basename(path)}: mostly all-zero IMU rows (sensor asleep)")
        continue
    # Label and session come from the file name (<session>_<label>.csv), not from the
    # label/session columns, so renaming a file is enough to relabel a recording.
    session, raw_label = os.path.splitext(os.path.basename(path))[0].split("_", 1)
    session = session.upper()
    label = LABEL_MAP.get(raw_label, raw_label)
    if raw_label != df["label"].iloc[0]:
        print(f"  note: {os.path.basename(path)} column says '{df['label'].iloc[0]}', using file name")
    dt = np.diff(df["t_ms"].to_numpy())
    print(f"{os.path.basename(path)}: {len(df)} rows, median dt={np.median(dt):.0f} ms, "
          f"max gap={dt.max()} ms")
    for a in AXES:
        for p in peaks(df, a):
            rows[a].append((p, label, session))

if not rows[AXES[0]]:
    sys.exit(f"no CSVs found in {data_dir}")


print("\nPeak |accel| per window (g): min / median / max, by label")
for a in AXES:
    for lab in sorted({r[1] for r in rows[a]}):
        v = np.array([r[0] for r in rows[a] if r[1] == lab])
        print(f"  {a} {lab:8s} n={len(v):4d}  {v.min():.2f} / {np.median(v):.2f} / {v.max():.2f}")

print("\nSessions per label:")
for lab in sorted({r[1] for r in rows[AXES[0]]}):
    ss = sorted({r[2] for r in rows[AXES[0]] if r[1] == lab})
    flag = "" if len(ss) >= 2 else "   <-- needs >= 2 sessions or it can never be held out"
    print(f"  {lab:8s} {len(ss)} sessions {ss}{flag}")

n_classes = len({r[1] for r in rows[AXES[0]]})
n_sessions = len({r[2] for r in rows[AXES[0]]})
if n_classes < 2 or n_sessions < 2:
    sys.exit(f"\nNeed >= 2 classes and >= 2 sessions to train/evaluate "
             f"(have {n_classes} classes, {n_sessions} sessions). Record more data.")


def fit(train):
    X = np.array([[r[0]] for r in train])
    y = [r[1] for r in train]
    # at most 4 classes -> 3 thresholds; min_samples_leaf avoids chasing noise
    return DecisionTreeClassifier(max_leaf_nodes=4, min_samples_leaf=5, random_state=0).fit(X, y)


def loso(data):
    """Leave-one-session-out predictions."""
    sessions = sorted({r[2] for r in data})
    y_true, y_pred = [], []
    for s in sessions:
        tr = [r for r in data if r[2] != s]
        te = [r for r in data if r[2] == s]
        if len({r[1] for r in tr}) < 2:
            continue
        clf = fit(tr)
        y_true += [r[1] for r in te]
        y_pred += list(clf.predict(np.array([[r[0]] for r in te])))
    return y_true, y_pred


print("\nLeave-one-session-out macro-F1 per axis:")
scores = {}
for a in AXES:
    yt, yp = loso(rows[a])
    scores[a] = f1_score(yt, yp, average="macro") if yt else 0.0
    print(f"  {a}: {scores[a]:.3f}")

axis = axis_arg or max(scores, key=scores.get)
print(f"\nUsing axis: {axis}")
yt, yp = loso(rows[axis])
labels = sorted(set(yt))
print(classification_report(yt, yp, labels=labels, zero_division=0))
print("confusion matrix (rows=true, cols=pred), order:", labels)
print(confusion_matrix(yt, yp, labels=labels))

# Final model on all data -> sorted thresholds for the on-board classifier.
clf = fit(rows[axis])
t = clf.tree_
thr = sorted(t.threshold[t.feature >= 0])
# label of each interval (-inf,thr0], (thr0,thr1], ... (thr_last, inf)
edges = [-np.inf] + thr + [np.inf]
ordered = []
for lo, hi in zip(edges[:-1], edges[1:]):
    mid = (max(lo, 0) + hi) / 2 if np.isfinite(hi) else lo + 1
    ordered.append(str(clf.predict([[mid]])[0]))
# Merge neighbouring intervals with the same label (the tree can split one class
# several times); only boundaries where the predicted label changes are real thresholds.
thr_m, lab_m = [], [ordered[0]]
for t_, l_ in zip(thr, ordered[1:]):
    if l_ != lab_m[-1]:
        thr_m.append(t_)
        lab_m.append(l_)
model = {"axis": AXES.index(axis) + 1, "axis_name": axis,
         "thresholds": [round(float(x), 3) for x in thr_m], "labels": lab_m}
out = os.path.join(here, "model1.json")
with open(out, "w") as f:
    json.dump(model, f, indent=2)
print("\nWrote", out, model)
