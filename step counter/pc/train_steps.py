"""Neural-net step counter: train a 1-D CNN in PyTorch on data/steps/*.csv and measure how good it is.

Usage: python train_steps.py [--quick]      (--quick = few iterations/seeds, to check the pipeline)

Input: the 100 Hz files written by prepare_steps.py (ankle IMU + every phone sensor, plus the step
labels). Sessions and their hand counts come from data/steps_log.csv.

The net reads a stretch of sensor data and outputs, for every 10 ms sample, a step *rate* (steps per
second). Its target is the step labels smeared into small Gaussian bumps that each add up to exactly
one step, so the sum of the output over any stretch of time is the number of steps in it, and the
bumps show when each step happened. The net is fully convolutional, so it runs on a recording of any
length.

Evaluation is leave-one-recording-out: every recording is counted by models that never saw it.
Each fold trains SEEDS models from different random starts (an ensemble); their spread is the
model's own uncertainty. This is repeated for three input sets (ankle only, phone only, both) and
compared with the classic peak detector (prepare_steps.py's labels) and the phone's pedometer.

Writes results/ (counts.csv, ablation.csv, *.png) and pc/step_model.pt (ensemble trained on
every recording, for the live counter).
"""
import argparse
import os
import time
from concurrent.futures import ProcessPoolExecutor

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

FS = 100
ANKLE = ["ax_g", "ay_g", "az_g", "gx_dps", "gy_dps", "gz_dps"]
SIGMA_S = 0.1          # width of each step's target bump
CROP = 512             # training crop length (samples, ~5 s)
BATCH = 32
LR = 2e-3
COUNT_WEIGHT = 0.05    # extra loss on the step count of each crop
P_DROP = 0.1

here = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(here, "..", "data")
STEPS = os.path.join(DATA, "steps")
OUT = os.path.join(here, "..", "results")
MODEL_PATH = os.path.join(here, "step_model.pt")



# ---------------------------------------------------------------- data

def target_rate(n, steps):
    """Step labels -> rate (steps/s) made of Gaussian bumps that each integrate to one step."""
    rate = np.zeros(n)
    half = int(4 * SIGMA_S * FS)
    k = np.exp(-0.5 * (np.arange(-half, half + 1) / (SIGMA_S * FS)) ** 2)
    k = k / k.sum() * FS
    for i in np.flatnonzero(steps):
        lo, hi = max(i - half, 0), min(i + half + 1, n)
        rate[lo:hi] += k[lo - (i - half):hi - (i - half)]
    return rate


def load_sessions():
    log = pd.read_csv(os.path.join(DATA, "steps_log.csv"))
    summary = pd.read_csv(os.path.join(STEPS, "summary.csv")).set_index("session")
    sessions = []
    for _, r in log.iterrows():
        path = os.path.join(STEPS, f"{r['session']}.csv")
        if not os.path.exists(path):
            print(f"skipping {r['session']}: run prepare_steps.py first")
            continue
        df = pd.read_csv(path)
        labeled = int(df["step"].sum())
        hand = r["true_steps"]
        sessions.append({
            "session": r["session"], "activity": r["activity"], "df": df,
            "rate": target_rate(len(df), df["step"].to_numpy()),
            "labeled": labeled,
            "truth": labeled if pd.isna(hand) else float(hand),
            "truth_from": "labels" if pd.isna(hand) else "hand count",
            "pedometer": summary["phone_pedometer"].get(r["session"], np.nan),
        })
    return sessions


def channel_sets(sessions):
    phone = [c for c in sessions[0]["df"].columns
             if c.startswith("ph_") and all(c in s["df"] for s in sessions)]
    # Phone motion without absolute angle: the phone sits at a different angle in the pocket every
    # time, so orientation, gravity and total acceleration (which contains gravity) can tell the net
    # which recording it is looking at instead of what a step looks like.
    motion = [c for c in phone if c.startswith(("ph_accelerometer_", "ph_gyroscope_"))]
    return {"both": ANKLE + phone, "ankle": ANKLE, "phone": phone,
            "both_motion": ANKLE + motion, "phone_motion": motion}


LABELS = {"both": "net: ankle+phone (all channels)", "ankle": "net: ankle only",
          "phone": "net: phone only (all channels)", "both_motion": "net: ankle+phone motion",
          "phone_motion": "net: phone motion only"}


def normalizer(sessions, cols):
    x = np.concatenate([s["df"][cols].to_numpy(float) for s in sessions])
    return np.nanmean(x, axis=0), np.nanstd(x, axis=0) + 1e-6


def features(df, cols, mean, std):
    """(channels, time) float32; missing phone samples become 0 = the training mean."""
    x = (df[cols].to_numpy(float) - mean) / std
    return np.nan_to_num(x).T.astype(np.float32)


# ---------------------------------------------------------------- model

class StepNet(nn.Module):
    """Dilated 1-D CNN: each output sample sees about 1.2 s of every input channel."""

    def __init__(self, n_in, width=32):
        super().__init__()
        layers, c = [], n_in
        for d in (1, 2, 4, 8):
            layers += [nn.Conv1d(c, width, 9, padding=4 * d, dilation=d), nn.ReLU(),
                       nn.Dropout(P_DROP)]
            c = width
        self.body = nn.Sequential(*layers)
        self.head = nn.Conv1d(width, 1, 1)

    def forward(self, x):                  # x: (batch, channels, time)
        return nn.functional.softplus(self.head(self.body(x))).squeeze(1)   # steps/s, >= 0


def train(train_sets, n_in, iters, seed, log_every=0):
    """train_sets: list of (features (C, T), target rate (T,)). Returns model and loss history."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = StepNet(n_in)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, iters)
    lengths = np.array([x.shape[1] for x, _ in train_sets], float)
    history = []
    for it in range(iters):
        xb, yb = [], []
        for k in rng.choice(len(train_sets), BATCH, p=lengths / lengths.sum()):
            x, y = train_sets[k]
            s = rng.integers(0, x.shape[1] - CROP)
            xc = x[:, s:s + CROP] * rng.uniform(0.9, 1.1, (x.shape[0], 1))    # sensor gain jitter
            xb.append(xc + rng.normal(0, 0.05, xc.shape))                       # sensor noise
            yb.append(y[s:s + CROP])
        xb = torch.tensor(np.stack(xb), dtype=torch.float32)
        yb = torch.tensor(np.stack(yb), dtype=torch.float32)

        pred = model(xb)                                                   # forward pass
        loss = ((pred - yb) ** 2).mean() \
            + COUNT_WEIGHT * ((pred.sum(1) - yb.sum(1)) / FS).pow(2).mean()
        opt.zero_grad()
        loss.backward()                                                    # back-propagation
        opt.step()                                                         # weight update
        sched.step()
        history.append(loss.item())
        if log_every and (it + 1) % log_every == 0:
            print(f"    iter {it + 1:5d}  loss {np.mean(history[-log_every:]):.4f}")
    return model, history


@torch.no_grad()
def predict(model, x):
    model.eval()
    return model(torch.tensor(x[None])).numpy()[0]


# ---------------------------------------------------------------- evaluation

def boot_ci(values, n=10000, seed=0):
    """95% bootstrap confidence interval of the mean, resampling recordings."""
    rng = np.random.default_rng(seed)
    v = np.asarray(values)
    means = v[rng.integers(0, len(v), (n, len(v)))].mean(1)
    return np.percentile(means, [2.5, 97.5])


# Every training run is independent, so they run in parallel worker processes, one thread each
# (on this PC that is ~8x faster than one process using all cores).
_W = {}


def _init_worker():
    torch.set_num_threads(1)
    _W["sessions"] = load_sessions()
    _W["sets"] = channel_sets(_W["sessions"])


def _job(job):
    """(config, held-out index or None for the final model, seed, iters)."""
    name, test_i, seed, iters = job
    sessions, cols = _W["sessions"], _W["sets"][name]
    train_s = [s for j, s in enumerate(sessions) if j != test_i]
    mean, std = normalizer(train_s, cols)
    data = [(features(s["df"], cols, mean, std), s["rate"]) for s in train_s]
    model, history = train(data, len(cols), iters, seed)
    if test_i is None:
        return model.state_dict(), history
    return predict(model, features(sessions[test_i]["df"], cols, mean, std))


def summarize(sessions, rates_by_fold, seeds):
    rows, curves = [], {}
    for i, test in enumerate(sessions):
        rates = np.array(rates_by_fold[i])
        counts = rates.sum(1) / FS
        curves[test["session"]] = rates
        rows.append({"session": test["session"], "activity": test["activity"],
                     "truth": test["truth"], "truth_from": test["truth_from"],
                     "pred": counts.mean(), "pred_sd": counts.std(ddof=1) if seeds > 1 else 0.0,
                     "labeled": test["labeled"], "pedometer": test["pedometer"]})
    df = pd.DataFrame(rows)
    df["err"] = df["pred"] - df["truth"]
    df["pct_err"] = 100 * df["err"] / df["truth"]
    return df, curves


def pct_summary(df, col):
    ape = 100 * (df[col] - df["truth"]).abs() / df["truth"]
    ape = ape.dropna()
    lo, hi = boot_ci(ape)
    return ape.mean(), lo, hi


# ---------------------------------------------------------------- figures

def plot_counts(df, path):
    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = np.arange(len(df))
    w = 0.2
    ax.bar(x - 1.5 * w, df["truth"], w, label="truth (hand count, or labels for S18)", color="0.35")
    ax.bar(x - 0.5 * w, df["pred"], w, yerr=2 * df["pred_sd"], capsize=3,
           label="neural net (±2 sd over 5 seeds)", color="tab:blue")
    ax.bar(x + 0.5 * w, df["labeled"], w, label="peak detector", color="tab:green")
    ax.bar(x + 1.5 * w, df["pedometer"], w, label="phone pedometer", color="tab:orange")
    ax.set_xticks(x, [f"{s}\n{a}" for s, a in zip(df["session"], df["activity"])])
    ax.set_ylabel("steps")
    ax.set_title("Step count per held-out recording (leave-one-recording-out)")
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_timelines(sessions, curves, path):
    fig, axs = plt.subplots(len(sessions), 1, figsize=(11, 2.4 * len(sessions)), sharex=False)
    for ax, s in zip(np.atleast_1d(axs), sessions):
        t = s["df"]["t_s"].to_numpy()
        r = curves[s["session"]]
        cum = np.cumsum(r, axis=1) / FS
        ax.fill_between(t, cum.min(0), cum.max(0), alpha=0.3, color="tab:blue")
        ax.plot(t, cum.mean(0), color="tab:blue", label="net (held out), range over seeds")
        ax.plot(t, np.cumsum(s["df"]["step"].to_numpy()), color="k", lw=0.8, ls="--",
                label="step labels")
        ax2 = ax.twinx()
        ax2.plot(t, r.mean(0), color="tab:red", lw=0.5, alpha=0.6)
        ax2.set_ylabel("rate (steps/s)", color="tab:red", fontsize=8)
        ax.set_title(f"{s['session']} {s['activity']}", fontsize=9)
        ax.set_ylabel("steps")
        ax.legend(fontsize=7, loc="upper left")
    np.atleast_1d(axs)[-1].set_xlabel("time (s)")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def plot_ablation(abl, path):
    fig, ax = plt.subplots(figsize=(7, 3.8))
    x = np.arange(len(abl))
    ax.bar(x, abl["mape"], yerr=[abl["mape"] - abl["ci_lo"], abl["ci_hi"] - abl["mape"]],
           capsize=4, color=["tab:blue" if m.startswith("net") else
                             "tab:green" if m.startswith("peak") else "tab:orange" for m in abl["method"]])
    ax.set_xticks(x, [m.replace("net: ", "") for m in abl["method"]], fontsize=7, rotation=20, ha="right")
    ax.set_ylabel("mean absolute % error")
    ax.set_title("Error per method, 95% bootstrap CI over recordings")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_loss(history, path):
    h = pd.Series(history).rolling(50, min_periods=1).mean()
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.plot(h)
    ax.set_yscale("log")
    ax.set_xlabel("training iteration (one batch, one back-propagation step)")
    ax.set_ylabel("loss (50-iteration average)")
    ax.set_title("Training the final model on all recordings")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--configs", nargs="+", default=["both_motion", "phone_motion"],
                    help="input sets to evaluate this run (others keep their rows from run1_all_channels/)")
    ap.add_argument("--main", default="both_motion",
                    help="input set used for counts.csv, the timelines and the saved final model")
    args = ap.parse_args()
    iters, seeds = (100, 2) if args.quick else (1000, 5)
    os.makedirs(OUT, exist_ok=True)

    sessions = load_sessions()
    sets = channel_sets(sessions)
    print(f"{len(sessions)} recordings, {sum(len(s['df']) for s in sessions) / FS / 60:.1f} min, "
          f"{sum(s['labeled'] for s in sessions)} labeled steps")
    print(f"channels: ankle {len(sets['ankle'])}, phone {len(sets['phone'])}  "
          f"({iters} iterations x {seeds} seeds per fold)")

    configs = args.configs
    if args.main not in configs:
        configs = [args.main] + configs
    jobs = [(name, i, seed, iters) for name in configs for i in range(len(sessions))
            for seed in range(seeds)]
    jobs += [(args.main, None, seed, iters) for seed in range(seeds)]   # final models, all data
    workers = max(1, (os.cpu_count() or 2) - 1)
    print(f"{len(jobs)} training runs on {workers} worker processes...")
    t0 = time.time()
    with ProcessPoolExecutor(workers, initializer=_init_worker) as ex:
        out = list(ex.map(_job, jobs))
    print(f"done in {(time.time() - t0) / 60:.1f} min")

    results, curves = {}, {}
    for name in configs:
        by_fold = {i: [o for j, o in zip(jobs, out) if j[0] == name and j[1] == i]
                   for i in range(len(sessions))}
        results[name], curves[name] = summarize(sessions, by_fold, seeds)
    final = [o for j, o in zip(jobs, out) if j[1] is None]
    curves_both = curves[args.main]

    main_df = results[args.main]
    main_df.round(2).to_csv(os.path.join(OUT, "counts.csv"), index=False)
    abl = []
    old_path = os.path.join(OUT, "ablation.csv")          # input sets evaluated earlier keep their rows
    if not os.path.exists(old_path):
        old_path = os.path.join(OUT, "run1_all_channels", "ablation.csv")
    if os.path.exists(old_path):
        rename = {"net: ankle+phone": LABELS["both"], "net: phone only": LABELS["phone"]}
        for _, r in pd.read_csv(old_path).iterrows():
            lab = rename.get(r["method"], r["method"])
            if lab.startswith("net") and lab not in [LABELS[c] for c in configs]:
                abl.append({**r.to_dict(), "method": lab})
    for name in configs:
        m, lo, hi = pct_summary(results[name], "pred")
        abl.append({"method": LABELS[name], "mape": m, "ci_lo": lo, "ci_hi": hi,
                    "mean_signed_err": results[name]["err"].mean(),
                    "mean_seed_sd": results[name]["pred_sd"].mean()})
    order = list(LABELS.values())
    abl.sort(key=lambda r: order.index(r["method"]))
    for col, label in [("labeled", "peak detector"), ("pedometer", "phone pedometer")]:
        m, lo, hi = pct_summary(main_df, col)
        abl.append({"method": label, "mape": m, "ci_lo": lo, "ci_hi": hi,
                    "mean_signed_err": (main_df[col] - main_df["truth"]).mean(),
                    "mean_seed_sd": np.nan})
    abl = pd.DataFrame(abl)
    abl.round(2).to_csv(os.path.join(OUT, "ablation.csv"), index=False)

    print("\nPer recording ({LABELS[args.main]}, held out):")
    print(main_df[["session", "activity", "truth", "truth_from", "pred", "pred_sd", "err",
                   "pct_err", "labeled", "pedometer"]].round(1).to_string(index=False))
    print("\nMean absolute % error, 95% bootstrap CI over recordings:")
    print(abl.round(1).to_string(index=False))

    plot_counts(main_df, os.path.join(OUT, "counts.png"))
    plot_timelines(sessions, curves_both, os.path.join(OUT, "timelines.png"))
    plot_ablation(abl, os.path.join(OUT, "ablation.png"))

    # Final ensemble, trained on every recording, saved for the live counter.
    cols = sets[args.main]
    mean, std = normalizer(sessions, cols)
    plot_loss(final[0][1], os.path.join(OUT, "training_loss.png"))
    torch.save({"channels": cols, "mean": mean, "std": std, "fs": FS, "width": 32,
                "states": [st for st, _ in final], "sessions": [s["session"] for s in sessions]},
               MODEL_PATH)
    n_params = sum(p.numel() for p in StepNet(len(cols)).parameters())
    print(f"Saved {len(final)}-model ensemble ({n_params} weights each) -> {MODEL_PATH}")
    print(f"Results in {os.path.abspath(OUT)}")


if __name__ == "__main__":
    main()
