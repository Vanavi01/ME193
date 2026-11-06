"""Uncertainty analysis of the step counter: how far off is it, and how sure can we be of that?

Usage: python analyze_uncertainty.py [--seeds 5]

Reads results/counts.csv and results/ablation.csv (written by train_steps.py) and writes
results/uncertainty.png and results/uncertainty_report.md.

Four separate sources of uncertainty are kept apart, because they answer different questions:
  1. Training noise    - the same recording counted by 5 nets trained from different random starts
                         (spread of the ensemble). Says how much the answer depends on luck.
  2. Generalization    - error on recordings the net never saw (leave-one-recording-out), with a
                         bootstrap 95% confidence interval over recordings. Says how good the counter
                         will be on a new walk. This is the headline number.
  3. Ground truth      - the hand counts are only good to about +-2 steps, so smaller errors than
                         that cannot be told apart from counting error.
  4. Too few recordings - with 5 recordings the interval on the headline number is wide, and
                         differences between methods are only meaningful when it excludes zero.
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

HAND_TOLERANCE = 2.0   # the hand counts are good to about +-2 steps

here = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(here, "..", "results")


def boot(values, n=20000, seed=0):
    rng = np.random.default_rng(seed)
    v = np.asarray(values, float)
    m = v[rng.integers(0, len(v), (n, len(v)))].mean(1)
    return np.percentile(m, [2.5, 97.5])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5, help="models per ensemble in train_steps.py")
    args = ap.parse_args()

    df = pd.read_csv(os.path.join(OUT, "counts.csv"))
    abl = pd.read_csv(os.path.join(OUT, "ablation.csv"))
    n, k = len(df), args.seeds
    tcrit = stats.t.ppf(0.975, k - 1) if k > 1 else np.nan

    df["err"] = df["pred"] - df["truth"]
    df["ape"] = 100 * df["err"].abs() / df["truth"]
    df["ens_lo"] = df["pred"] - tcrit * df["pred_sd"] / np.sqrt(k)     # 95% CI of the ensemble mean
    df["ens_hi"] = df["pred"] + tcrit * df["pred_sd"] / np.sqrt(k)
    df["single_lo"] = df["pred"] - tcrit * df["pred_sd"]               # where one more net would land
    df["single_hi"] = df["pred"] + tcrit * df["pred_sd"]
    df["det_err"] = df["labeled"] - df["truth"]
    df["ped_err"] = df["pedometer"] - df["truth"]
    df["det_ape"] = 100 * df["det_err"].abs() / df["truth"]
    df["ped_ape"] = 100 * df["ped_err"].abs() / df["truth"]
    hand = df["truth_from"] == "hand count"

    # ---- headline numbers
    mape = df["ape"].mean()
    lo, hi = boot(df["ape"])
    bias, rmse = df["err"].mean(), np.sqrt((df["err"] ** 2).mean())
    tot_true, tot_pred = df["truth"].sum(), df["pred"].sum()
    within = (df["err"].abs() <= HAND_TOLERANCE)[hand].sum()
    sd_to_err = df["pred_sd"].mean() / df["err"].abs().mean()

    # ---- paired comparison with the baselines (positive = net is worse)
    paired = {}
    for name, col in (("peak detector", "det_ape"), ("phone pedometer", "ped_ape")):
        d = df["ape"] - df[col]
        paired[name] = (d.mean(), *boot(d))

    # ---- by activity
    by_act = df.groupby(df["activity"].where(df["activity"] == "stop_go", "walk")).agg(
        recordings=("session", "count"), mape=("ape", "mean"), bias=("err", "mean"),
        seed_sd=("pred_sd", "mean"))

    # ---- figure
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.6), gridspec_kw={"width_ratios": [1.5, 1]})
    ax = axs[0]
    x = np.arange(n)
    ax.axhspan(-HAND_TOLERANCE, HAND_TOLERANCE, color="0.9", label=f"hand-count error (±{HAND_TOLERANCE:g})")
    ax.axhline(0, color="k", lw=0.8)
    ax.errorbar(x - 0.15, df["err"], yerr=[df["err"] - (df["single_lo"] - df["truth"]),
                                           (df["single_hi"] - df["truth"]) - df["err"]],
                fmt="o", color="tab:blue", capsize=4, label="neural net (bar: 95% range of one net)")
    ax.plot(x, df["det_err"], "s", color="tab:green", label="peak detector")
    ax.plot(x + 0.15, df["ped_err"], "^", color="tab:orange", label="phone pedometer")
    ax.set_xticks(x, [f"{s}\n{a}" for s, a in zip(df["session"], df["activity"])], fontsize=8)
    ax.set_ylabel("predicted − true steps")
    ax.set_title("Signed error per held-out recording")
    ax.legend(fontsize=7, loc="best")

    ax = axs[1]
    abl = abl.copy()
    y = np.arange(len(abl))[::-1]
    ax.errorbar(abl["mape"], y, xerr=[abl["mape"] - abl["ci_lo"], abl["ci_hi"] - abl["mape"]],
                fmt="o", color="tab:blue", capsize=4)
    ax.set_yticks(y, abl["method"], fontsize=8)
    ax.set_xlabel("mean absolute % error (95% bootstrap CI)")
    ax.set_title("Which inputs and baselines")
    ax.set_xlim(left=0)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "uncertainty.png"), dpi=130)
    plt.close(fig)

    # ---- report
    def f(v, d=1):
        return f"{v:.{d}f}"

    lines = ["# Step counter: uncertainty analysis", "",
             f"Leave-one-recording-out on {n} recordings ({int(tot_true)} walking steps). Every recording "
             f"is counted by networks that never saw it; each recording is counted by an ensemble of {k} "
             "networks started from different random weights.", "",
             "## Headline", "",
             f"- **Mean absolute error: {f(mape)}% (95% CI {f(lo)}% to {f(hi)}%)**, bootstrap over the {n} recordings.",
             f"- Bias (mean predicted − true): {f(bias)} steps; RMSE {f(rmse)} steps.",
             f"- All recordings together: {f(tot_pred, 0)} predicted vs {f(tot_true, 0)} true "
             f"({f(100 * (tot_pred / tot_true - 1))}%).",
             f"- Within the hand-count error (±{HAND_TOLERANCE:g} steps): {within} of {int(hand.sum())} "
             "recordings that have hand counts.", "",
             "## Per recording", "",
             "| session | activity | truth | net (mean ± sd) | 95% range of one net | error | % error | peak detector | pedometer |",
             "|---|---|---|---|---|---|---|---|---|"]
    for _, r in df.iterrows():
        mark = "*" if r["truth_from"] != "hand count" else ""
        lines.append(f"| {r['session']} | {r['activity']} | {f(r['truth'])}{mark} | {f(r['pred'])} ± {f(r['pred_sd'])} "
                     f"| {f(r['single_lo'])} to {f(r['single_hi'])} | {r['err']:+.1f} | {r['err'] / r['truth'] * 100:+.1f}% "
                     f"| {r['labeled']:g} ({r['det_err']:+.1f}) | {r['pedometer']:g} ({r['ped_err']:+.1f}) |")
    lines += ["", "\\* truth is the script's step labels, not a hand count (the hand count was loose).", "",
              "## Method comparison", "",
              "| method | mean abs % error | 95% CI | mean signed error (steps) |", "|---|---|---|---|"]
    for _, r in abl.iterrows():
        lines.append(f"| {r['method']} | {f(r['mape'])}% | {f(r['ci_lo'])}% to {f(r['ci_hi'])}% | {r['mean_signed_err']:+.1f} |")
    lines += ["", "Difference in mean absolute % error, net minus baseline (positive = net is worse), "
              "paired over recordings:", ""]
    for name, (m, a, b) in paired.items():
        verdict = "net is better" if b < 0 else ("net is worse" if a > 0 else "no detectable difference")
        lines.append(f"- vs {name}: {m:+.1f} points (95% CI {a:+.1f} to {b:+.1f}): {verdict}.")
    lines += ["", "## By activity", "", "| group | recordings | mean abs % error | bias (steps) | mean seed sd |",
              "|---|---|---|---|---|"]
    for g, r in by_act.iterrows():
        lines.append(f"| {g} | {int(r['recordings'])} | {f(r['mape'])}% | {r['bias']:+.1f} | {f(r['seed_sd'])} |")

    lines += ["", "## How to read the uncertainty", "",
              f"1. **Training noise vs. error.** Nets trained from different random starts disagree by about "
              f"{f(df['pred_sd'].mean())} steps (mean sd), while the typical error is {f(df['err'].abs().mean())} "
              f"steps (ratio {f(sd_to_err, 2)}). " + (
                  "Training noise is small next to the error, so averaging more nets would not fix it; the "
                  "error comes from the nets generalizing imperfectly to a recording they have not seen."
                  if sd_to_err < 0.5 else
                  "Training noise is a large part of the error, so averaging more nets (a bigger ensemble) "
                  "should help."),
              f"2. **The headline interval has width {f(hi - lo)} points** ({f(lo)}% to {f(hi)}%), resampling "
              f"only {n} recordings. More recordings narrow it; more seeds do not.",
              f"3. **The ground truth is not exact.** Hand counts are good to about ±{HAND_TOLERANCE:g} steps "
              f"(about {f(100 * HAND_TOLERANCE / df['truth'].mean())}% of a typical recording), so error below that "
              "cannot be told apart from counting error. S18's hand count was loose, so it is scored against "
              "the labels instead; that is circular and flatters the peak detector on that recording.",
              "4. **The labels come from the peak detector.** Training labels were made by the same swing-peak "
              "method that serves as the baseline, so the net learns to imitate it. A net that matches the "
              "peak detector cannot beat it by much; the hand counts are what keep this honest.",
              "5. **The input set was chosen after seeing results, so the headline is a little optimistic.** "
              "Five input sets were tried (see the method table) and the best one on these same recordings, "
              "ankle only, is the headline. Adding the phone made the network worse: with all 18 phone channels "
              "the error was about 3.6 times higher. A likely cause is that orientation and gravity depend on "
              "how the phone sits in the pocket, so with only four training recordings the net can memorize "
              "which recording it is looking at. Keeping only gravity-free phone motion helped (about a third "
              "lower error) but not enough to beat ankle-only. Only new recordings would give an unbiased check.",
              "6. **Held-out recordings differ from each other** (speed, stops, stomps in the middle), "
              "so some folds are hard: a fold that holds out the only recording of a given kind leaves the "
              "net nothing to learn that behaviour from."]
    with open(os.path.join(OUT, "uncertainty_report.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines).encode("ascii", "replace").decode())   # console may be cp1252
    print(f"\nWrote {os.path.abspath(OUT)}: uncertainty_report.md, uncertainty.png")


if __name__ == "__main__":
    main()
