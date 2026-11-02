"""Model 2: all six IMU channels -> window features -> classical classifier (no neural nets).

Usage: python train_model2.py [data_dir]
Compares a decision tree, a random forest and an SVM with leave-one-session-out evaluation,
and also scores Model 1 (single-axis peak) on the same folds for a side-by-side comparison.
Exports the better of decision tree / random forest (the ones the board can run in pure
Python) to model2.json. Copy that file to model2_app/python/.
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(here, "..", "model2_app", "python"))
import features  # noqa: E402
import model_runtime  # noqa: E402

WINDOW, HOP = 200, 100
COLS = ["ax_g", "ay_g", "az_g", "gx_dps", "gy_dps", "gz_dps"]
LABEL_MAP = {"air_bike": "bike"}  # keep identical to train_model1.py
MIN_WINDOWS_PER_FILE = 3          # skip recordings too short to mean anything

data_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(here, "..", "data")

X, y, groups, ay_peak = [], [], [], []
for path in sorted(glob.glob(os.path.join(data_dir, "*.csv"))):
    name = os.path.basename(path)
    if os.path.getsize(path) == 0:
        print(f"SKIPPING {name}: empty")
        continue
    df = pd.read_csv(path)
    if (df[COLS[:3]].abs().sum(axis=1) == 0).mean() > 0.5:
        print(f"SKIPPING {name}: mostly all-zero rows (sensor asleep)")
        continue
    session, raw = os.path.splitext(name)[0].split("_", 1)
    session, label = session.upper(), LABEL_MAP.get(raw, raw)
    arr = df[COLS].to_numpy()
    starts = list(range(0, len(arr) - WINDOW + 1, HOP))
    if len(starts) < MIN_WINDOWS_PER_FILE:
        print(f"SKIPPING {name}: only {len(starts)} windows")
        continue
    for s in starts:
        w = arr[s:s + WINDOW]
        X.append(features.window_features([tuple(r) for r in w]))
        y.append(label)
        groups.append(session)
        ay_peak.append(float(np.abs(w[:, 1]).max()))
    print(f"{name}: {len(starts)} windows -> {label}")

X, y, groups, ay_peak = np.array(X), np.array(y), np.array(groups), np.array(ay_peak)
classes = sorted(set(y))
print(f"\n{len(y)} windows, {X.shape[1]} features, classes {classes}")
for c in classes:
    print(f"  {c:6s} {int((y == c).sum()):4d} windows, sessions {sorted(set(groups[y == c]))}")


def make(name):
    if name == "tree":
        return DecisionTreeClassifier(max_depth=5, min_samples_leaf=5, class_weight="balanced",
                                      random_state=0)
    if name == "forest":
        return RandomForestClassifier(n_estimators=30, max_depth=5, min_samples_leaf=3,
                                      class_weight="balanced", random_state=0)
    if name == "svm":
        return make_pipeline(StandardScaler(), SVC(kernel="rbf", C=3, class_weight="balanced"))


def loso(name):
    pred = np.empty(len(y), dtype=object)
    for g in sorted(set(groups)):
        te = groups == g
        tr = ~te
        if name == "model1":  # single feature: peak |ay| in the window
            clf = DecisionTreeClassifier(max_leaf_nodes=len(classes), min_samples_leaf=5,
                                         random_state=0).fit(ay_peak[tr, None], y[tr])
            pred[te] = clf.predict(ay_peak[te, None])
        else:
            pred[te] = make(name).fit(X[tr], y[tr]).predict(X[te])
    return pred


print("\nLeave-one-session-out (every window predicted by a model that never saw its session):")
results = {}
for name in ["model1", "tree", "forest", "svm"]:
    p = loso(name)
    results[name] = p
    print(f"  {name:7s} macro-F1 {f1_score(y, p, average='macro'):.3f}   "
          f"accuracy {(p == y).mean():.3f}")

best = max(["tree", "forest"], key=lambda n: f1_score(y, results[n], average="macro"))
print(f"\nBest deployable model: {best}")
print(classification_report(y, results[best], labels=classes, zero_division=0))
print("confusion matrix (rows=true, cols=pred), order:", classes)
print(confusion_matrix(y, results[best], labels=classes))

# ---- final fit on all data, export, and verify the exported model matches sklearn ----
clf = make(best).fit(X, y)
estimators = [clf] if best == "tree" else list(clf.estimators_)
model = {"type": best, "classes": [str(c) for c in clf.classes_],
         "feature_names": features.FEATURE_NAMES, "trees": []}
for est in estimators:
    t = est.tree_
    val = t.value[:, 0, :]
    val = val / val.sum(axis=1, keepdims=True)
    model["trees"].append({"left": t.children_left.tolist(), "right": t.children_right.tolist(),
                           "feature": t.feature.tolist(), "threshold": t.threshold.tolist(),
                           "value": val.round(5).tolist()})

sk = clf.predict(X)
rt = np.array([model_runtime.predict(model, x.tolist())[0] for x in X])
print(f"\nExport check: pure-Python runtime agrees with sklearn on {(sk == rt).mean():.1%} of windows")

imp = clf.feature_importances_
print("Top features:")
for i in np.argsort(imp)[::-1][:8]:
    print(f"  {features.FEATURE_NAMES[i]:12s} {imp[i]:.3f}")

out = os.path.join(here, "model2.json")
with open(out, "w") as f:
    json.dump(model, f)
print(f"\nWrote {out} ({os.path.getsize(out) / 1024:.0f} KB)")
