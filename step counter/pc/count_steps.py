"""The final step tracker: count steps in an ankle recording.

Two methods: --method peak (default) is a gyroscope peak detector; --method net is the PyTorch network
described below. The peak detector was more accurate on every recording tried (see the README), so it
is the default; the network is kept because it is the assignment's neural-network step counter.


Usage:
    python count_steps.py data/S19_walk.csv            # replay the recording in a window, count live
    python count_steps.py data/S19_walk.csv --no-plot  # just print the count
    python count_steps.py data/S19_walk.csv --save out.png

Input is any ankle CSV from the board logger (columns t_ms, ax_g, ay_g, az_g, gx_dps, gy_dps, gz_dps).
Only the ankle sensor is needed: the saved model (pc/step_model.pt, written by train_steps.py) is the
ankle-only network, the best of the input sets that were tried.

The network outputs a step rate for every 10 ms; the running count is the sum of that rate so far.
It sees about 0.6 s ahead and behind each sample, so a real-time version would lag by about 0.6 s.
This is a replay of a recorded file, not a live feed from the board.

Accuracy: see results/uncertainty_report.md. On recordings it had not seen, its count was off by
about 14% on average (95% CI 8-23%), and a plain peak detector did better. Treat it as a demo.
The five recordings in data/ were all used to train the saved model, so counting one of them is not a
test; the program warns you. Record a new walk (and count the steps by hand) for a fair check.
"""
import argparse
import os
import time

import matplotlib
import numpy as np
import pandas as pd
import torch

import prepare_steps as ps
import train_steps as ts

here = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(here, "..", "data")


def load_ankle_csv(path):
    df = pd.read_csv(path)
    t = (df["t_ms"].to_numpy(float) - df["t_ms"].iloc[0]) / 1000.0
    grid = np.arange(0, t[-1], 1 / ts.FS)
    x = np.column_stack([np.interp(grid, t, df[c].to_numpy(float)) for c in ts.ANKLE])
    return grid, x


def count(path, model_path=ts.MODEL_PATH):
    """-> (time grid, per-network rate curves (n_models, T), signal for plotting)."""
    ck = torch.load(model_path, weights_only=False)
    cols = ck["channels"]
    if cols != ts.ANKLE:
        raise SystemExit(f"{model_path} uses channels {cols}; this tracker needs the ankle-only model. "
                         "Run: python train_steps.py --configs ankle --main ankle")
    t, x = load_ankle_csv(path)
    feats = ((x - ck["mean"]) / ck["std"]).T.astype(np.float32)
    rates = []
    for st in ck["states"]:
        m = ts.StepNet(len(cols), ck["width"])
        m.load_state_dict(st)
        rates.append(ts.predict(m, feats))
    return t, np.array(rates), x


def peak_steps(t, x, end_stomps=True):
    """Peak-detector step count (no neural network, no training data).

    The strapped leg's gyroscope swings once per stride: find those swings, and every swing stands
    for two steps (this leg, then the other). The walk is the stretch between the 3 start stomps and
    the 3 end stomps, found from the ankle accelerometer alone; the stomps are leg lifts, not steps.
    Set end_stomps=False for a recording that ended without the end stomps.
    -> (swing sample indices, walk start s, walk end s)
    """
    starts, ends = ps.find_stomps(t, x[:, :3])
    w0 = starts[-1] + 0.5 if len(starts) else t[0]
    w1 = ends[0] - 0.5 if (end_stomps and len(ends)) else t[-1]
    walk = (t >= w0) & (t <= w1)
    pk, _, _ = ps.swings(t, x[:, 3:6], walk)
    return ps.drop_isolated(pk, "L"), w0, w1


def trained_on(path, model_path=ts.MODEL_PATH):
    """True if this recording was in the final model's training data (so its count is not a test)."""
    name = os.path.basename(path).split("_")[0].upper()
    return name in torch.load(model_path, weights_only=False)["sessions"]


def known_counts(path):
    """Step labels and hand count for a recording that is in data/steps_log.csv, if any."""
    name = os.path.basename(path).split("_")[0].upper()
    out = {}
    lab = os.path.join(DATA, "steps", f"{name}.csv")
    if os.path.exists(lab):
        out["labels"] = int(pd.read_csv(lab)["step"].sum())
    log = os.path.join(DATA, "steps_log.csv")
    if os.path.exists(log):
        row = pd.read_csv(log)
        row = row[row["session"].str.upper() == name]
        if len(row) and not pd.isna(row["true_steps"].iloc[0]):
            out["hand"] = float(row["true_steps"].iloc[0])
    return out


def draw(t, x, cum, i, win=8.0):
    """Frame at sample i: last `win` seconds of ankle gyro, and the running count."""
    lo = max(0, i - int(win * ts.FS))
    return t[lo:i + 1], x[lo:i + 1, 4], cum[:i + 1]


def run_peak(args):
    t, x = load_ankle_csv(args.recording)
    pk, w0, w1 = peak_steps(t, x, end_stomps=not args.no_end_stomps)
    total = 2 * len(pk)
    known = known_counts(args.recording)
    print(f"{os.path.basename(args.recording)}: {t[-1]:.0f} s, walk {w0:.1f}-{w1:.1f} s")
    print(f"  peak detector: {len(pk)} leg swings -> {total} steps")
    if "hand" in known:
        print(f"  hand count (walking steps): {known['hand']:g}  ({100 * (total - known['hand']) / known['hand']:+.1f}%)")
    print("  on the 5 recordings tried it was off by 3.5% on average (worst 9%); the settings were chosen while "
          "looking at those recordings, so a new walk is the real test.")
    if args.no_plot:
        return
    marks = np.zeros(len(t))
    marks[pk] = 2
    cum = np.cumsum(marks)
    if args.save:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(10, 6))
    big = fig.text(0.5, 0.95, "", ha="center", va="top", fontsize=26, weight="bold")
    gy = x[:, 4]

    def render(i):
        a1.clear(); a2.clear()
        lo = max(0, i - 8 * ts.FS)
        a1.plot(t[lo:i + 1], gy[lo:i + 1], color="tab:blue", lw=1)
        shown = pk[(pk >= lo) & (pk <= i)]
        a1.plot(t[shown], gy[shown], "rv")
        a1.set_xlim(max(0, t[i] - 8), max(8, t[i]))
        a1.set_ylabel("ankle gyro y (deg/s)")
        a2.plot(t[:i + 1], cum[:i + 1], color="tab:green", lw=2)
        a2.set_xlim(0, t[-1]); a2.set_ylim(0, total * 1.1 + 1)
        a2.set_xlabel("time (s)"); a2.set_ylabel("steps")
        big.set_text(f"steps: {cum[i]:.0f}")

    if args.save:
        render(len(t) - 1)
        fig.savefig(args.save, dpi=120)
        print("saved", args.save)
        return
    plt.ion()
    t0 = time.time()
    i = 0
    while i < len(t) - 1 and plt.fignum_exists(fig.number):
        i = min(int((time.time() - t0) * args.speed * ts.FS), len(t) - 1)
        render(i)
        plt.pause(0.03)
    if plt.fignum_exists(fig.number):
        render(len(t) - 1)
        plt.ioff()
        plt.show()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("recording", help="ankle CSV, e.g. data/S19_walk.csv")
    ap.add_argument("--method", choices=["peak", "net"], default="net",
                    help="net = the PyTorch network (default); peak = gyro peak detector (more accurate)")
    ap.add_argument("--no-end-stomps", action="store_true",
                    help="peak method: the recording has no end stomps (walk runs to the end)")
    ap.add_argument("--no-plot", action="store_true")
    ap.add_argument("--save", help="write the final plot to this PNG instead of opening a window")
    ap.add_argument("--speed", type=float, default=2.0, help="replay speed (1 = real time)")
    args = ap.parse_args()

    if args.method == "peak":
        return run_peak(args)
    t, rates, x = count(args.recording)
    counts = rates.sum(1) / ts.FS
    k = len(counts)
    sd = counts.std(ddof=1) if k > 1 else 0.0
    from scipy import stats
    half = stats.t.ppf(0.975, k - 1) * sd if k > 1 else float("nan")
    known = known_counts(args.recording)
    print(f"{os.path.basename(args.recording)}: {t[-1]:.0f} s")
    print(f"  neural network:   {counts.mean():.1f} steps  (+- {sd:.1f} sd over {k} networks; "
          f"one network would land in {counts.mean() - half:.0f} to {counts.mean() + half:.0f})")
    if "labels" in known:
        print(f"  peak-detector labels: {known['labels']} steps")
    if "hand" in known:
        print(f"  hand count (walking steps): {known['hand']:g}")
    if trained_on(args.recording):
        print("  WARNING: this recording was used to train the network, so this count is NOT a fair test.")
    print("  on recordings it had never seen, its count was off by about 14% on average "
          "(95% CI 8-23%): see results/uncertainty_report.md")
    if args.no_plot:
        return

    rate = rates.mean(0)
    cum = np.cumsum(rate) / ts.FS
    if args.save:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(10, 6), gridspec_kw={"height_ratios": [1, 1]})
    big = fig.text(0.5, 0.95, "", ha="center", va="top", fontsize=26, weight="bold")

    def render(i):
        tt, sig, c = draw(t, x, cum, i)
        a1.clear(); a2.clear()
        a1.plot(tt, sig, color="tab:blue", lw=1)
        a1.set_ylabel("ankle gyro y (deg/s)")
        a1.set_xlim(max(0, t[i] - 8), max(8, t[i]))
        a2.plot(t[:i + 1], c, color="tab:green", lw=2, label="network count")
        if "labels" in known:
            a2.axhline(known["labels"], color="k", ls="--", lw=0.8, label="peak-detector labels")
        a2.set_xlim(0, t[-1])
        a2.set_ylim(0, max(cum[-1], known.get("labels", 0)) * 1.1 + 1)
        a2.set_xlabel("time (s)")
        a2.set_ylabel("steps")
        a2.legend(loc="upper left")
        big.set_text(f"steps: {cum[i]:.0f}")

    if args.save:
        render(len(t) - 1)
        fig.savefig(args.save, dpi=120)
        print("saved", args.save)
        return
    plt.ion()
    t0 = time.time()
    i = 0
    while i < len(t) - 1 and plt.fignum_exists(fig.number):
        i = min(int((time.time() - t0) * args.speed * ts.FS), len(t) - 1)
        render(i)
        plt.pause(0.03)
    if plt.fignum_exists(fig.number):
        render(len(t) - 1)
        plt.ioff()
        plt.show()


if __name__ == "__main__":
    main()
