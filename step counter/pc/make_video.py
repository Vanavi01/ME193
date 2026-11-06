"""Explainer video: how the step counter works, from sensors to PyTorch to back-propagation.

Usage: python make_video.py [--stills] [--quick] [--out results/step_counter_explained.mp4]
  --stills  render one frame per scene to results/video_stills/ (to check the layout quickly)
  --quick   train the illustration network for fewer iterations

Everything on screen is real: the traces are your S19 recording, the activations, losses, gradients
and weights come from a network trained inside this script (a short run on all five recordings,
for illustration; the networks that are *scored* are the ones train_steps.py trains), and the final
scenes read results/counts.csv and results/ablation.csv. Run train_steps.py first.
"""
import argparse
import os
import subprocess
import sys

import imageio_ffmpeg
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.patches  # noqa: E402
import matplotlib.ticker  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

import train_steps as ts  # noqa: E402

W, H, FPS = 1280, 720, 24
BG, PANEL, TEXT, MUTED = "#0e1117", "#161b22", "#e6edf3", "#8b949e"
BLUE, ORANGE, GREEN, RED, PURPLE, YELLOW = "#58a6ff", "#f0883e", "#3fb950", "#f85149", "#bc8cff", "#d29922"
DEMO_SESSION = "S19"
SNAPS = [0, 5, 10, 20, 40, 80, 150, 250, 400]       # iterations at which the network is photographed
LAYERS = ["Conv 1", "Conv 2", "Conv 3", "Conv 4", "Head"]

here = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(here, "..", "results")
plt.rcParams["font.family"] = "DejaVu Sans"


# ---------------------------------------------------------------- real data and a real (short) training run

def build(quick):
    torch.set_num_threads(max(1, (os.cpu_count() or 2) // 2))
    sessions = ts.load_sessions()
    cols = ts.channel_sets(sessions)["ankle"]
    mean, std = ts.normalizer(sessions, cols)
    data = [(ts.features(s["df"], cols, mean, std), s["rate"]) for s in sessions]
    demo = next(s for s in sessions if s["session"] == DEMO_SESSION)
    x_full = ts.features(demo["df"], cols, mean, std)
    # a 5.12 s crop in the middle of steady walking
    c0 = int(20 * ts.FS)
    xc = x_full[:, c0:c0 + ts.CROP]
    yc = demo["rate"][c0:c0 + ts.CROP]

    iters = 60 if quick else SNAPS[-1]
    snaps = [i for i in SNAPS if i <= iters] + ([iters] if iters not in SNAPS else [])
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    model = ts.StepNet(len(cols))
    opt = torch.optim.Adam(model.parameters(), lr=ts.LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, iters)
    lengths = np.array([x.shape[1] for x, _ in data], float)
    gy = cols.index("gy_dps")
    convs = [model.body[0], model.body[3], model.body[6], model.body[9], model.head]
    loss_hist, gnorm, kernels, preds = [], [], [], {}
    for it in range(iters + 1):
        if it in snaps:
            model.eval()
            with torch.no_grad():
                preds[it] = model(torch.tensor(xc[None]))[0].numpy().copy()
            model.train()
        if it == iters:
            break
        xb, yb = [], []
        for k in rng.choice(len(data), ts.BATCH, p=lengths / lengths.sum()):
            x, y = data[k]
            s = rng.integers(0, x.shape[1] - ts.CROP)
            xcrop = x[:, s:s + ts.CROP] * rng.uniform(0.9, 1.1, (x.shape[0], 1))
            xb.append(xcrop + rng.normal(0, 0.05, xcrop.shape))
            yb.append(y[s:s + ts.CROP])
        xb = torch.tensor(np.stack(xb), dtype=torch.float32)
        yb = torch.tensor(np.stack(yb), dtype=torch.float32)
        pred = model(xb)
        loss = ((pred - yb) ** 2).mean() + ts.COUNT_WEIGHT * ((pred.sum(1) - yb.sum(1)) / ts.FS).pow(2).mean()
        opt.zero_grad()
        loss.backward()
        gnorm.append([c.weight.grad.norm().item() for c in convs])
        kernels.append(model.body[0].weight[:, gy, :].detach().numpy().copy())     # (32 filters, 9 taps)
        opt.step()
        sched.step()
        loss_hist.append(loss.item())
        if it % 50 == 0:
            print(f"  illustration run: iter {it}  loss {loss.item():.4f}", flush=True)
    kernels.append(model.body[0].weight[:, gy, :].detach().numpy().copy())
    kernels = np.array(kernels)
    filt = int(np.abs(kernels[-1]).sum(1).argmax())

    # real activations of the trained net on the crop
    acts, h = [], torch.tensor(xc[None])
    model.eval()
    with torch.no_grad():
        for layer in model.body:
            h = layer(h)
            if isinstance(layer, nn.ReLU):
                acts.append(h[0].numpy().copy())
        out = nn.functional.softplus(model.head(h))[0, 0].numpy()

    counts = pd.read_csv(os.path.join(OUT_DIR, "counts.csv")) if os.path.exists(os.path.join(OUT_DIR, "counts.csv")) else None
    abl = pd.read_csv(os.path.join(OUT_DIR, "ablation.csv")) if os.path.exists(os.path.join(OUT_DIR, "ablation.csv")) else None
    return dict(sessions=sessions, cols=cols, demo=demo, x_full=x_full, xc=xc, yc=yc, c0=c0,
                loss=np.array(loss_hist), gnorm=np.array(gnorm), kernels=kernels, filt=filt,
                snaps=snaps, preds=preds, acts=acts, out=out, counts=counts, abl=abl,
                n_params=sum(p.numel() for p in model.parameters()), iters=iters)


# ---------------------------------------------------------------- drawing helpers

def ease(p):
    p = float(np.clip(p, 0, 1))
    return p * p * (3 - 2 * p)


def reveal(p, a, b):
    return float(np.clip((p - a) / (b - a), 0, 1))


def frame(title, caption):
    fig = plt.figure(figsize=(W / 100, H / 100), dpi=100, facecolor=BG)
    fig.text(0.04, 0.925, title, color=TEXT, fontsize=26, weight="bold", va="center")
    fig.text(0.04, 0.045, caption, color=MUTED, fontsize=15, va="center")
    return fig


def axes(fig, rect, title=None, xlabel=None, ylabel=None):
    ax = fig.add_axes(rect, facecolor=PANEL)
    for s in ax.spines.values():
        s.set_color("#30363d")
    ax.tick_params(colors=MUTED, labelsize=10)
    if title:
        ax.set_title(title, color=TEXT, fontsize=13, loc="left")
    if xlabel:
        ax.set_xlabel(xlabel, color=MUTED, fontsize=11)
    if ylabel:
        ax.set_ylabel(ylabel, color=MUTED, fontsize=11)
    return ax


def box(fig, x, y, w, h, text, color=BLUE, alpha=1.0, fs=13, tc=TEXT):
    ax = fig.add_axes([x, y, w, h])
    ax.axis("off")
    ax.add_patch(plt.Rectangle((0, 0), 1, 1, transform=ax.transAxes, facecolor=color, alpha=0.18 * alpha + 0.0,
                               edgecolor=color, lw=2 * alpha))
    ax.text(0.5, 0.5, text, color=tc, ha="center", va="center", fontsize=fs, alpha=max(alpha, 0.0), weight="bold")
    return ax


def arrow(fig, x0, y0, x1, y1, color=MUTED, lw=2.5, alpha=1.0):
    fig.add_artist(matplotlib.patches.FancyArrowPatch(
        (x0, y0), (x1, y1), transform=fig.transFigure, arrowstyle="-|>", mutation_scale=18,
        color=color, lw=lw, alpha=alpha))


def txt(fig, x, y, s, color=TEXT, fs=14, alpha=1.0, **kw):
    fig.text(x, y, s, color=color, fontsize=fs, alpha=alpha, **kw)


# ---------------------------------------------------------------- scenes (p = progress 0..1 inside the scene)

def scene_title(D, p):
    fig = frame("Counting steps with a neural network",
                "Ankle sensor  ->  1-D convolutional network (PyTorch)  ->  steps per second  ->  step count")
    d = D["demo"]["df"]
    t = d["t_s"].to_numpy()
    for k, (x0, name, col, sub, sig) in enumerate([
            (0.05, "Ankle strap", BLUE, "Arduino Uno Q + MPU6050\n6 channels, 100 Hz", "gy_dps"),
            (0.05, "Phone in pocket", ORANGE, "Sensor Logger app\nlabels + tested as input", "ph_gyroscope_x")]):
        y0 = 0.58 - k * 0.27
        a = reveal(p, 0.05 + 0.2 * k, 0.25 + 0.2 * k)
        box(fig, x0, y0, 0.2, 0.2, f"{name}\n", col, a, fs=15)
        txt(fig, x0 + 0.1, y0 + 0.045, sub, MUTED, 11, a, ha="center", va="center")
        ax = axes(fig, [0.30, y0 + 0.01, 0.24, 0.18])
        ax.axis("off")
        n = int(600 * reveal(p, 0.15 + 0.2 * k, 0.8))
        i0 = int(1500 + 600 * reveal(p, 0.4, 1.0))
        seg = np.nan_to_num(d[sig].to_numpy()[i0:i0 + 600])
        ax.plot(seg[:max(n, 2)], color=col, lw=1.4)
        ax.set_xlim(0, 600)
        ax.set_ylim(np.nanmin(seg) * 1.1, np.nanmax(seg) * 1.1)
    a = reveal(p, 0.45, 0.65)
    arrow(fig, 0.56, 0.56, 0.62, 0.56, MUTED, alpha=a)
    box(fig, 0.63, 0.43, 0.17, 0.26, "PyTorch\n1-D CNN\n(%s weights)" % f"{D['n_params']:,}", PURPLE, a, fs=14)
    a = reveal(p, 0.7, 0.9)
    arrow(fig, 0.81, 0.56, 0.85, 0.56, MUTED, alpha=a)
    box(fig, 0.86, 0.45, 0.11, 0.22, "steps\nper second\n\nsum = count", GREEN, a, fs=12)
    return fig


def scene_data(D, p):
    d = D["demo"]["df"]
    t = d["t_s"].to_numpy()
    t0 = 9 + 14 * ease(p)
    lo, hi = t0 - 6, t0 + 1
    m = (t >= lo) & (t <= hi)
    fig = frame("1. The data: one label per leg swing",
                "Each leg swing shows up as a peak in that leg's gyroscope. The peaks (and your hand counts) are the labels.")
    n_steps = 0
    for k, (sig, col, name, foot) in enumerate([("gy_dps", BLUE, "Ankle gyroscope (left leg)", "L"),
                                                 ("ph_gyroscope_x", ORANGE, "Phone gyroscope (right leg)", "R")]):
        ax = axes(fig, [0.07, 0.55 - k * 0.30, 0.88, 0.22], name, ylabel="deg/s")
        ax.plot(t[m], np.nan_to_num(d[sig].to_numpy()[m]), color=col, lw=1.4)
        st = t[(d["foot"] == foot).to_numpy() & (d["step"] == 1).to_numpy()]
        shown = st[(st >= lo) & (st <= t0)]
        sig_v = np.interp(shown, t, np.nan_to_num(d[sig].to_numpy()))
        ax.plot(shown, sig_v, "v", color=GREEN, ms=11)
        ax.axvline(t0, color=TEXT, lw=1.5, alpha=0.7)
        ax.set_xlim(lo, hi)
    n_steps = int(((t <= t0) & (d["step"].to_numpy() == 1) & (d["in_walk"].to_numpy() == 1)).sum())
    txt(fig, 0.95, 0.815, f"steps counted: {n_steps}", GREEN, 22, ha="right", va="center", weight="bold")
    txt(fig, 0.07, 0.12, "Checked against your hand counts: the labels are within about 5 steps on every recording (S18's hand count was loose).",
        MUTED, 12)
    return fig


def scene_input(D, p):
    X, cols = D["x_full"], D["cols"]
    c0 = int(15 * ts.FS)
    span = 1500
    start = c0 + int((span - ts.CROP) * ease(p))
    fig = frame(f"2. What the network sees: 5 s of {len(D['cols'])} channels",
                "Training examples are random 5 s crops. The answer for each crop: how many steps happened, and when.")
    ax = axes(fig, [0.07, 0.40, 0.88, 0.42], f"{len(cols)} ankle channels (3 acceleration + 3 gyro), normalized; red = high, blue = low")
    ax.imshow(X[:, c0:c0 + span], aspect="auto", cmap="RdBu_r", vmin=-2.5, vmax=2.5,
              extent=[c0 / ts.FS, (c0 + span) / ts.FS, len(cols), 0])
    for r, nm in enumerate(["ax", "ay", "az", "gx", "gy", "gz"]):
        ax.text(c0 / ts.FS + 0.1, r + 0.5, nm, color="k", fontsize=10, weight="bold", va="center")
    ax.add_patch(plt.Rectangle((start / ts.FS, 0), ts.CROP / ts.FS, len(cols), fill=False, ec=YELLOW, lw=3))
    ax.set_yticks([])
    ax2 = axes(fig, [0.07, 0.10, 0.88, 0.22], "Target for the highlighted window: a bump of area 1 for every step (steps per second)")
    rate = D["demo"]["rate"][start:start + ts.CROP]
    tt = np.arange(ts.CROP) / ts.FS
    ax2.fill_between(tt, rate, color=GREEN, alpha=0.5)
    ax2.plot(tt, rate, color=GREEN)
    ax2.set_xlim(0, ts.CROP / ts.FS)
    ax2.set_ylim(0, max(D["demo"]["rate"].max() * 1.05, 1))
    ax2.text(0.99, 0.85, f"steps in this window = area = {rate.sum() / ts.FS:.1f}", transform=ax2.transAxes,
             ha="right", color=GREEN, fontsize=14, weight="bold")
    return fig


def scene_network(D, p):
    fig = frame("3. The network: 4 convolution layers + a head",
                "Each layer slides a small learned filter along time. Dilation 1, 2, 4, 8 lets one output see about 1.2 s.")
    rows = [(f"Input\n{len(D['cols'])} ch", D["xc"], "RdBu_r", (-2.5, 2.5)),
            ("Conv1 d=1\n32 ch", D["acts"][0], "magma", None), ("Conv2 d=2\n32 ch", D["acts"][1], "magma", None),
            ("Conv3 d=4\n32 ch", D["acts"][2], "magma", None), ("Conv4 d=8\n32 ch", D["acts"][3], "magma", None)]
    top, hgt, gap = 0.84, 0.088, 0.02
    for i, (name, arr, cm, lim) in enumerate(rows):
        a = reveal(p, i * 0.14, i * 0.14 + 0.14)
        y = top - i * (hgt + gap) - hgt
        ax = fig.add_axes([0.14, y, 0.46, hgt], facecolor=PANEL)
        ax.imshow(arr, aspect="auto", cmap=cm, vmin=None if lim is None else lim[0], vmax=None if lim is None else lim[1],
                  alpha=a)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_color("#30363d")
        txt(fig, 0.135, y + hgt / 2, name, TEXT, 11, a, ha="right", va="center")
        if i < 4:
            arrow(fig, 0.37, y - 0.003, 0.37, y - gap + 0.003, MUTED, 2, a)
    a = reveal(p, 0.72, 0.95)
    y = top - 5 * (hgt + gap) - hgt
    ax = axes(fig, [0.14, y, 0.46, hgt])
    ax.plot(D["out"], color=GREEN, lw=1.5, alpha=a)
    ax.fill_between(np.arange(len(D["out"])), D["out"], color=GREEN, alpha=0.3 * a)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlim(0, len(D["out"]))
    txt(fig, 0.135, y + hgt / 2, "Head + softplus\nsteps/s", TEXT, 11, a, ha="right", va="center")
    code = ("class StepNet(nn.Module):\n"
            "  def __init__(self, n_in, width=32):\n"
            "    for d in (1, 2, 4, 8):\n"
            "      Conv1d(c, width, 9,\n"
            "             padding=4*d, dilation=d)\n"
            "      ReLU(); Dropout(0.1)\n"
            "    self.head = Conv1d(width, 1, 1)\n"
            "\n"
            "  def forward(self, x):\n"
            "    return softplus(\n"
            "        self.head(self.body(x)))")
    a = reveal(p, 0.15, 0.4)
    ax = fig.add_axes([0.64, 0.44, 0.33, 0.40])
    ax.axis("off")
    ax.add_patch(plt.Rectangle((0, 0), 1, 1, transform=ax.transAxes, fc=PANEL, ec="#30363d"))
    ax.text(0.04, 0.96, code, color=TEXT, fontsize=10.5, family="monospace", va="top", alpha=a)
    txt(fig, 0.64, 0.37, f"{D['n_params']:,} weights in total.", TEXT, 14, a)
    txt(fig, 0.64, 0.32, "Softplus keeps the output >= 0:\nyou cannot take negative steps.", MUTED, 12, a, va="top")
    txt(fig, 0.64, 0.20, "Activations shown: the trained net on the\nwindow from the previous scene.", MUTED, 10.5, a, va="top")
    return fig


def scene_loss(D, p):
    fig = frame("4. How wrong is it? The loss",
                "A network that has not learned yet outputs noise. The loss turns the mismatch into one number.")
    pred, y = D["preds"][0], D["yc"]
    tt = np.arange(ts.CROP) / ts.FS
    ax = axes(fig, [0.07, 0.34, 0.88, 0.50], "Untrained network (blue) vs. the target (green)", xlabel="seconds", ylabel="steps per second")
    a = reveal(p, 0.0, 0.25)
    ax.fill_between(tt, y, color=GREEN, alpha=0.35)
    ax.plot(tt, y, color=GREEN, lw=2, label="target")
    ax.plot(tt, pred, color=BLUE, lw=2, alpha=a, label="network output")
    sh = reveal(p, 0.25, 0.5)
    ax.fill_between(tt, y, pred, color=RED, alpha=0.45 * sh, label="error")
    ax.legend(loc="upper right", fontsize=11, facecolor=PANEL, edgecolor="#30363d", labelcolor=TEXT)
    mse = float(((pred - y) ** 2).mean())
    cnt_p, cnt_t = pred.sum() / ts.FS, y.sum() / ts.FS
    a = reveal(p, 0.5, 0.75)
    txt(fig, 0.07, 0.24, r"$L=\mathrm{mean}\,(\hat y-y)^2\;+\;0.05\,(\mathrm{predicted\ count}-\mathrm{true\ count})^2$",
        TEXT, 19, a)
    a2 = reveal(p, 0.75, 0.95)
    txt(fig, 0.07, 0.13, f"mean squared error {mse:.3f}      counted {cnt_p:.1f} steps, truth {cnt_t:.1f}", YELLOW, 16, a2)
    return fig


def scene_backprop(D, p):
    fig = frame("5. Back-propagation: who is to blame?",
                "The error is passed backward, layer by layer, and each weight learns how much it contributed.")
    names = ["Input", "Conv 1", "Conv 2", "Conv 3", "Conv 4", "Head", "Loss"]
    xs = np.linspace(0.075, 0.855, len(names))
    y0 = 0.62
    fwd = reveal(p, 0.0, 0.25)
    bwd = reveal(p, 0.3, 0.8)
    for i, (x, n) in enumerate(zip(xs, names)):
        col = GREEN if n == "Loss" else (MUTED if n == "Input" else BLUE)
        box(fig, x, y0, 0.085, 0.12, n, col, 1.0, fs=12)
        if i < len(names) - 1:
            arrow(fig, x + 0.088, y0 + 0.075, xs[i + 1] - 0.003, y0 + 0.075, MUTED, 2, 0.4 + 0.6 * fwd)
    txt(fig, 0.5, 0.80, "forward pass: data flows right", MUTED, 13, fwd, ha="center")
    gn = D["gnorm"][0]
    gn_n = gn / gn.max()
    pos = {5: 5, 4: 4, 3: 3, 2: 2, 1: 1}      # layer index (1..5) -> box index
    ax = axes(fig, [0.065, 0.22, 0.885, 0.22])
    ax.set_xlim(0.065, 0.95)
    ax.set_ylim(0, 1.25)
    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_title("size of the gradient dL/dW for each layer's weights (this run, first training step)", color=TEXT, fontsize=12, loc="left")
    n_back = bwd * 6
    for li in range(5):
        box_i = 5 - li                       # Head=5 ... Conv1=1, drawn in the order backprop reaches them
        reached = np.clip(n_back - li, 0, 1)
        h = gn_n[box_i - 1] * reached
        ax.bar(xs[box_i] + 0.0425, h, width=0.055, color=RED, alpha=0.9)
        if reached > 0.99:
            ax.text(xs[box_i] + 0.0425, h + 0.05, f"{gn[box_i - 1]:.3g}", color=TEXT, ha="center", fontsize=10)
    for i in range(len(names) - 1, 0, -1):    # backward arrows, from the loss toward the input
        r = np.clip(n_back - (len(names) - 1 - i), 0, 1)
        if r > 0:
            arrow(fig, xs[i] - 0.003, y0 + 0.035, xs[i] - 0.003 - (xs[i] - xs[i - 1] - 0.091) * r, y0 + 0.035, RED, 2.5, 0.95)
    txt(fig, 0.5, 0.52, "backward pass: the blame flows left", RED, 13, reveal(p, 0.3, 0.45), ha="center")
    a = reveal(p, 0.8, 1.0)
    txt(fig, 0.07, 0.15, r"$\frac{\partial L}{\partial w_1}=\frac{\partial L}{\partial \hat y}\cdot\frac{\partial \hat y}{\partial h_4}\cdot\frac{\partial h_4}{\partial h_3}\cdot\frac{\partial h_3}{\partial h_2}\cdot\frac{\partial h_2}{\partial h_1}\cdot\frac{\partial h_1}{\partial w_1}$",
        TEXT, 18, a)
    txt(fig, 0.07, 0.085, "chain rule, applied layer by layer.  In PyTorch this whole step is one line:  loss.backward()",
        YELLOW, 13, a)
    return fig


def scene_training(D, p):
    fig = frame("6. Learning: nudge every weight, repeat",
                "Each step: forward, loss, backward, then w <- w - (step size) x (gradient). Adam adapts the step size per weight.")
    snaps = D["snaps"]
    cur = ease(p) * (len(snaps) - 1)
    k0 = int(np.floor(cur))
    k1 = min(k0 + 1, len(snaps) - 1)
    frac = cur - k0
    it_now = snaps[k0] + (snaps[k1] - snaps[k0]) * frac
    n_loss = len(D["loss"])
    ax = axes(fig, [0.07, 0.56, 0.42, 0.28], "Loss", xlabel="training iteration")
    sm = pd.Series(D["loss"]).rolling(5, min_periods=1).mean().to_numpy()
    ax.plot(np.arange(n_loss), sm, color=MUTED, lw=1, alpha=0.4)
    upto = int(np.clip(it_now, 1, n_loss))
    ax.plot(np.arange(upto), sm[:upto], color=BLUE, lw=2.5)
    ax.plot([upto - 1], [sm[upto - 1]], "o", color=YELLOW, ms=9)
    ax.set_yscale("log")
    ax.set_yticks([0.5, 1, 2, 4])
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.tick_params(which="both", colors=MUTED, labelsize=10)
    ax.set_xlim(0, n_loss)
    ax2 = axes(fig, [0.07, 0.15, 0.42, 0.28], "Network output on the same window", xlabel="seconds")
    tt = np.arange(ts.CROP) / ts.FS
    pa, pb = D["preds"][snaps[k0]], D["preds"][snaps[k1]]
    ax2.fill_between(tt, D["yc"], color=GREEN, alpha=0.3)
    ax2.plot(tt, D["yc"], color=GREEN, lw=1.5, label="target")
    ax2.plot(tt, (1 - frac) * pa + frac * pb, color=BLUE, lw=2, label="network")
    ax2.set_ylim(-0.1, max(D["yc"].max() * 1.4, 1))
    ax2.legend(loc="upper right", fontsize=10, facecolor=PANEL, edgecolor="#30363d", labelcolor=TEXT)
    ax3 = axes(fig, [0.58, 0.47, 0.37, 0.37], "One layer-1 filter reading the ankle gyro (9 weights)", xlabel="tap")
    ki = int(np.clip(it_now, 0, len(D["kernels"]) - 1))
    kern = D["kernels"][:, D["filt"], :]
    ax3.bar(np.arange(9), kern[ki], color=PURPLE)
    lim = np.abs(kern).max() * 1.15
    ax3.set_ylim(-lim, lim)
    ax3.axhline(0, color=MUTED, lw=0.8)
    txt(fig, 0.58, 0.34, f"iteration {int(it_now)} of {D['iters']}", YELLOW, 20, weight="bold")
    txt(fig, 0.58, 0.27, r"$w\leftarrow w-\eta\,\frac{\partial L}{\partial w}$", TEXT, 24)
    txt(fig, 0.58, 0.16, "Repeated hundreds of times on random 5 s crops.\n(Short illustration run on all recordings.)", MUTED, 12, va="center")
    return fig


def scene_testing(D, p):
    c = D["counts"]
    fig = frame("7. Testing on a walk it has never seen",
                "Leave-one-recording-out: train on four recordings, count the fifth, repeat. Network bars: +-2 sd over 5 networks.")
    if c is None:
        txt(fig, 0.5, 0.5, "run train_steps.py first", TEXT, 20, ha="center")
        return fig
    n = len(c)
    k = min(int(p * (n + 1)), n)
    for i in range(n):
        held = (i == min(k, n - 1)) and k < n
        done = i < k
        col = ORANGE if held else (GREEN if done else BLUE)
        box(fig, 0.07 + i * 0.178, 0.72, 0.15, 0.09, c["session"][i], col, 1.0 if (held or not done) else 0.7, fs=13)
        txt(fig, 0.145 + i * 0.178, 0.70, "held out" if held else ("counted" if done else "training"), col, 10.5, ha="center", va="top")
    ax = axes(fig, [0.07, 0.15, 0.88, 0.43], ylabel="steps")
    x = np.arange(n)
    for j, (col, lab, off, key) in enumerate([("0.55", "truth (hand count; S18: labels)", -0.3, "truth"), (BLUE, "neural network", -0.1, "pred"),
                                              (GREEN, "peak detector", 0.1, "labeled"), (ORANGE, "phone pedometer", 0.3, "pedometer")]):
        vals = np.array([c[key][i] if i < k else 0 for i in range(n)], float)
        ax.bar(x + off, vals, width=0.19, color=col, label=lab)
        if key == "pred":
            ax.errorbar(x[:k] + off, vals[:k], yerr=2 * c["pred_sd"].to_numpy()[:k], fmt="none", ecolor=TEXT, capsize=4)
    ax.set_xticks(x, [f"{s}\n{a}" for s, a in zip(c["session"], c["activity"])], fontsize=9)
    ax.legend(loc="upper left", fontsize=10, facecolor=PANEL, edgecolor="#30363d", labelcolor=TEXT, ncol=4)
    ax.set_ylim(0, float(c[["truth", "pred", "labeled", "pedometer"]].max().max()) * 1.35)
    return fig


def scene_results(D, p):
    a_ = D["abl"]
    fig = frame("8. How sure can we be?", "Error bars are 95% bootstrap confidence intervals over the recordings.")
    if a_ is None:
        txt(fig, 0.5, 0.5, "run train_steps.py first", TEXT, 20, ha="center")
        return fig
    ax = axes(fig, [0.34, 0.30, 0.60, 0.54], xlabel="mean absolute % error (lower is better)")
    y = np.arange(len(a_))[::-1]
    shown = int(np.ceil(reveal(p, 0, 0.7) * len(a_)))
    cols_ = [BLUE if m.startswith("net") else GREEN if m.startswith("peak") else ORANGE for m in a_["method"]]
    for i in range(shown):
        ax.barh(y[i], a_["mape"][i], color=cols_[i], alpha=0.85)
        ax.errorbar(a_["mape"][i], y[i], xerr=[[a_["mape"][i] - a_["ci_lo"][i]], [a_["ci_hi"][i] - a_["mape"][i]]],
                    fmt="none", ecolor=TEXT, capsize=5)
        ax.text(a_["ci_hi"][i] + 0.5, y[i], f"{a_['mape'][i]:.1f}%", color=TEXT, va="center", fontsize=12)
    ax.set_yticks(y, [m.replace(" (all channels)", "\n(all channels)") for m in a_["method"]], fontsize=11, color=TEXT)
    ax.set_xlim(0, float(a_["ci_hi"].max()) * 1.25)
    a = reveal(p, 0.75, 1.0)
    nets = a_[a_["method"].str.startswith("net")]
    best = nets.loc[nets["mape"].idxmin()]
    txt(fig, 0.04, 0.165, f"Best network input: {best['method'].replace('net: ', '')}  ({best['mape']:.1f}% error, "
        f"95% CI {best['ci_lo']:.1f}% to {best['ci_hi']:.1f}%).  Few recordings means wide intervals.", YELLOW, 13, a)
    allc = a_[a_["method"] == "net: ankle+phone (all channels)"]
    anko = a_[a_["method"] == "net: ankle only"]
    pk = a_[a_["method"] == "peak detector"]
    if len(allc) and len(anko) and len(pk):
        txt(fig, 0.04, 0.115, f"More sensors made it worse: every channel {allc['mape'].iloc[0]:.0f}%, ankle only "
            f"{anko['mape'].iloc[0]:.0f}%. A plain peak detector: {pk['mape'].iloc[0]:.1f}%. Too little data (5 recordings).",
            TEXT, 13, a)
    return fig


SCENES = [("title", 9, scene_title), ("data", 15, scene_data), ("input", 14, scene_input),
          ("network", 15, scene_network), ("loss", 13, scene_loss), ("backprop", 17, scene_backprop),
          ("training", 18, scene_training), ("testing", 14, scene_testing), ("results", 12, scene_results)]


# ---------------------------------------------------------------- rendering

def fig_to_rgba(fig):
    fig.canvas.draw()
    return np.asarray(fig.canvas.buffer_rgba())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stills", action="store_true")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--scenes", nargs="+", help="render only these scenes (names as in SCENES)")
    ap.add_argument("--out", default=os.path.join(OUT_DIR, "step_counter_explained.mp4"))
    args = ap.parse_args()

    print("Preparing real data and the illustration training run...", flush=True)
    D = build(args.quick)

    if args.stills:
        d = os.path.join(OUT_DIR, "video_stills")
        os.makedirs(d, exist_ok=True)
        for name, _, fn in SCENES:
            fig = fn(D, 0.92)
            fig.savefig(os.path.join(d, f"{name}.png"), dpi=100, facecolor=BG)
            plt.close(fig)
            print("  still:", name, flush=True)
        print("Stills in", os.path.abspath(d))
        return

    if args.scenes:
        SCENES[:] = [sc for sc in SCENES if sc[0] in args.scenes]
    ff = imageio_ffmpeg.get_ffmpeg_exe()
    proc = subprocess.Popen([ff, "-y", "-f", "rawvideo", "-pix_fmt", "rgba", "-s", f"{W}x{H}", "-r", str(FPS),
                             "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", "-preset", "medium",
                             "-movflags", "+faststart", args.out], stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)
    total = sum(int(s * FPS) for _, s, _ in SCENES)
    done = 0
    for name, secs, fn in SCENES:
        n = int(secs * FPS)
        for f in range(n):
            fig = fn(D, f / max(n - 1, 1))
            proc.stdin.write(fig_to_rgba(fig).tobytes())
            plt.close(fig)
            done += 1
            if done % 120 == 0:
                print(f"  {done}/{total} frames ({name})", flush=True)
    proc.stdin.close()
    proc.wait()
    print(f"Wrote {os.path.abspath(args.out)}  ({total / FPS:.0f} s, {os.path.getsize(args.out) / 1e6:.1f} MB)")


if __name__ == "__main__":
    sys.exit(main())
