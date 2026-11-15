"""Check the whole dataset before training.
  python -m src.audit
Prints a report (also saved to data/_audit/report.txt), writes contact sheets of what the
model sees (data/_audit/contact_<label>.png), exits 1 if anything must be fixed."""
import random
import sys
from collections import Counter

import cv2
import numpy as np

from . import config
from .preprocess import model_view
from .quality import check_sample, is_duplicate, signature


def contact_sheet(paths, n=64, cols=8, cell=96, seed=0):
    pick = random.Random(seed).sample(paths, min(n, len(paths)))
    rows = max(1, -(-len(pick) // cols))
    sheet = np.full((rows * cell, cols * cell, 3), 255, np.uint8)
    for i, p in enumerate(pick):
        img = cv2.imread(str(p))
        if img is None:
            continue
        r, c = divmod(i, cols)
        sheet[r * cell:(r + 1) * cell, c * cell:(c + 1) * cell] = model_view(img, size=cell)
    return sheet


def audit(data_dir=None, calib=None, bg=None, min_total=None, sheets_dir=None):
    data_dir = data_dir or config.DATA_DIR
    min_total = config.MIN_TOTAL_PER_CLASS if min_total is None else min_total
    fails, warns, lines = [], [], []
    persons = sorted(p for p in data_dir.iterdir() if p.is_dir() and not p.name.startswith("_")) \
        if data_dir.exists() else []
    if not persons:
        return {"fails": ["no data found"], "warns": [], "lines": [], "totals": {},
                "counts": {}, "duplicates": 0, "flagged": [], "outliers": []}

    expected = (calib["roi"][3], calib["roi"][2]) if calib else None
    counts = {p.name: {c: 0 for c in config.CLASSES} for p in persons}
    paths = {c: [] for c in config.CLASSES}
    fracs = {c: [] for c in config.CLASSES}
    issue_counts, flagged, dups, total = Counter(), [], 0, 0

    for p in persons:
        for c in config.CLASSES:
            prev = None
            for f in sorted((p / str(c)).glob("*.png")):
                img = cv2.imread(str(f))
                total += 1
                if img is None:
                    fails.append(f"unreadable image: {f}")
                    continue
                counts[p.name][c] += 1
                paths[c].append(f)
                issues, m = check_sample(img, c, calib, bg)
                if expected:   # size alone is fine (resized to 64x64); shape changes stretch
                    a_img, a_cal = img.shape[1] / img.shape[0], expected[1] / expected[0]
                    if abs(a_img / a_cal - 1) > 0.10:
                        issues.append("crop shape differs from current calibration")
                sig = signature(img)
                if prev is not None and is_duplicate(sig, prev):
                    dups += 1
                    issues.append("near-duplicate of previous")
                prev = sig
                for i in issues:
                    issue_counts[i] += 1
                if issues:
                    flagged.append((f, issues))
                if m["hand_frac"] is not None:
                    fracs[c].append((m["hand_frac"], f))

    totals = {c: sum(counts[p][c] for p in counts) for c in config.CLASSES}
    for c, n in totals.items():
        if n < min_total:
            fails.append(f"class {c}: only {n} images (need >= {min_total})")
    nonzero = [n for n in totals.values() if n > 0]
    if nonzero and max(nonzero) / min(nonzero) > config.IMBALANCE_MAX:
        warns.append(f"class imbalance {max(nonzero) / min(nonzero):.1f}x (max {config.IMBALANCE_MAX}x)")
    if len(persons) < 2:
        warns.append("only 1 person: no held-out-person validation; rubric wants different hands")
    if total and dups / total > config.DUP_WARN_FRAC:
        warns.append(f"{dups / total:.0%} near-duplicates: move your hand between saves")
    if total and len(flagged) / total > config.FLAG_WARN_FRAC:
        warns.append(f"{len(flagged)} images ({len(flagged) / total:.0%}) fail quality checks")

    # hand-size outliers per class: often mislabeled images
    outliers = []
    for c, vals in fracs.items():
        if len(vals) < 10:
            continue
        v = np.array([x for x, _ in vals])
        med = np.median(v)
        mad = np.median(np.abs(v - med)) * 1.4826 + 1e-6
        outliers += [(f, c, x) for x, f in vals if abs(x - med) / mad > 3.5]
    if outliers:
        warns.append(f"{len(outliers)} hand-size outliers (check for mislabels, listed below)")

    # report
    lines.append("person      " + "".join(f"{c:>6}" for c in config.CLASSES) + "   total")
    for name, row in counts.items():
        lines.append(f"{name:<12}" + "".join(f"{row[c]:>6}" for c in config.CLASSES)
                     + f"{sum(row.values()):>8}")
    lines.append(f"{'TOTAL':<12}" + "".join(f"{totals[c]:>6}" for c in config.CLASSES)
                 + f"{sum(totals.values()):>8}")
    if any(fracs.values()):
        lines.append("median hand area per class (should roughly grow 1->5): " + "  ".join(
            f"{c}:{np.median([x for x, _ in fracs[c]]):.2f}" for c in config.CLASSES if fracs[c]))
    if issue_counts:
        lines.append("quality issues: " + ", ".join(f"{k} x{n}" for k, n in issue_counts.most_common()))
    for f, issues in flagged[:20]:
        lines.append(f"  flagged {f.relative_to(data_dir)}: {'; '.join(issues)}")
    for f, c, x in outliers[:20]:
        lines.append(f"  outlier {f.relative_to(data_dir)}: class {c}, hand area {x:.2f}")

    if sheets_dir is not None:
        sheets_dir.mkdir(parents=True, exist_ok=True)
        for c in config.CLASSES:
            if paths[c]:
                cv2.imwrite(str(sheets_dir / f"contact_{c}.png"), contact_sheet(paths[c]))

    return {"fails": fails, "warns": warns, "lines": lines, "totals": totals, "counts": counts,
            "duplicates": dups, "flagged": flagged, "outliers": outliers}


def main():
    calib = config.load_calibration()
    bg = cv2.imread(str(config.BACKGROUND_PATH)) if config.BACKGROUND_PATH.exists() else None
    if calib is None or bg is None:
        print("WARNING: no calibration/background found; hand and lighting checks skipped.")
    r = audit(config.DATA_DIR, calib, bg, sheets_dir=config.AUDIT_DIR)
    out = r["lines"] + [""] + [f"FAIL: {x}" for x in r["fails"]] + [f"WARN: {x}" for x in r["warns"]]
    out.append("RESULT: " + ("FAIL" if r["fails"] else "WARN" if r["warns"] else "PASS"))
    text = "\n".join(out)
    print(text)
    config.AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    (config.AUDIT_DIR / "report.txt").write_text(text)
    print(f"\ncontact sheets (what the model sees): {config.AUDIT_DIR}")
    sys.exit(1 if r["fails"] else 0)


if __name__ == "__main__":
    main()
