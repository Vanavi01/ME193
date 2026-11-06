"""Pure-Python inference for the exported decision tree / random forest (model2.json)."""


def _tree_proba(tree, x):
    i = 0
    left, right, feat, thr = tree["left"], tree["right"], tree["feature"], tree["threshold"]
    while left[i] != -1:
        i = left[i] if x[feat[i]] <= thr[i] else right[i]
    return tree["value"][i]


def predict(model, x):
    """Return (label, per-class probability list)."""
    k = len(model["classes"])
    acc = [0.0] * k
    for tree in model["trees"]:
        p = _tree_proba(tree, x)
        for j in range(k):
            acc[j] += p[j]
    total = len(model["trees"])
    probs = [a / total for a in acc]
    return model["classes"][max(range(k), key=lambda j: probs[j])], probs
