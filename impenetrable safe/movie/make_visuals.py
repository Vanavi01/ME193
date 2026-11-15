"""Generate the movie visuals from the REAL trained model and a real picture of a hand.
    python movie/make_visuals.py
Writes movie/visuals/*.png (print them for the stop-motion set, or film them on a screen) and
movie/visuals/conv_slide/*.png (frames of a filter sliding over the hand)."""
import sys
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt                       # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.patches import FancyArrowPatch, Rectangle  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src import config                                 # noqa: E402
from src.model import load_checkpoint                  # noqa: E402
from src.preprocess import pad_square, preprocess_roi, trim_tape_edges  # noqa: E402
from src.quality import red_fraction                  # noqa: E402

OUT = ROOT / "movie" / "visuals"
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#d9d8d4"
BLUE = "#2a78d6"
# sequential (activation strength): one hue, surface -> dark blue
SEQ = LinearSegmentedColormap.from_list("seq", [SURFACE, "#cde2fb", "#86b6ef", "#3987e5",
                                                "#256abf", "#184f95", "#0d366b"])
# diverging (weights): red = negative, gray = 0, blue = positive
DIV = LinearSegmentedColormap.from_list("div", ["#8f2120", "#e34948", "#f3b1b0", "#f0efec",
                                                "#9ec5f4", "#2a78d6", "#0d366b"])
plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "text.color": INK,
                     "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.edgecolor": GRID, "font.size": 11, "savefig.facecolor": SURFACE})


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  wrote", name)


def off(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def show_rgb(x):
    """(3,H,W) in [-1,1] -> HxWx3 RGB in [0,1]."""
    return np.clip(x.transpose(1, 2, 0) * 0.5 + 0.5, 0, 1)


def run(net, x):
    """Every intermediate value of one forward pass."""
    t = torch.from_numpy(x)[None]
    with torch.no_grad():
        c1 = torch.relu(net.b1[0](t))          # conv1 + ReLU, 64x64x8
        f1 = net.b1(t)                         # after pooling, 32x32x8
        f2 = net.b2(f1)                        # 16x16x16
        f3 = net.b3(f2)                        # 8x8x32
        sq = [net.squash(f) for f in (f1, f2, f3)]                 # each 4x4
        feats = torch.cat([s.flatten(1) for s in sq], 1)[0]        # 896
        logits = net.fc(feats[None])[0]
        probs = torch.softmax(logits, 0)
    return dict(c1=c1[0].numpy(), f1=f1[0].numpy(), f2=f2[0].numpy(), f3=f3[0].numpy(),
                sq=[s[0].numpy() for s in sq], feats=feats.numpy(), logits=logits.numpy(),
                probs=probs.numpy())


def pick_examples(net):
    """One confidently-correct picture per finger count (from the user's own data), preferring
    pictures with no leftover strip of tape at the edge."""
    ex = {}
    for c in config.CLASSES:
        best = None
        for p in sorted((config.DATA_DIR / "victor" / str(c)).glob("*.png"))[::3]:
            img = cv2.imread(str(p))
            x = preprocess_roi(img)
            pr = run(net, x)["probs"]
            if pr.argmax() != c:
                continue
            tape = red_fraction(pad_square(trim_tape_edges(img)))
            score = pr[c] - 0.5 * min(tape * 50, 1.0)         # tape costs up to 0.5
            if best is None or score > best[0]:
                best = (score, p)
        ex[c] = best[1]
    return ex


# --------------------------------------------------------------------------- figures
def fig_pipeline(raw):
    trimmed = trim_tape_edges(raw)
    square = pad_square(trimmed)
    small = cv2.resize(square, (64, 64), interpolation=cv2.INTER_AREA)
    x = preprocess_roi(raw)
    fig, ax = plt.subplots(1, 6, figsize=(17, 3.6), gridspec_kw={"width_ratios": [1.5, 1, 1, 1, 1, 1]})
    ax[0].imshow(cv2.cvtColor(raw, cv2.COLOR_BGR2RGB))
    ax[0].set_title(f"1. crop inside the tape\n{raw.shape[1]}x{raw.shape[0]} pixels")
    ax[1].imshow(cv2.cvtColor(square, cv2.COLOR_BGR2RGB))
    ax[1].set_title(f"2. padded to a square\n{square.shape[1]}x{square.shape[0]}")
    ax[2].imshow(cv2.cvtColor(small, cv2.COLOR_BGR2RGB), interpolation="nearest")
    ax[2].set_title("3. shrunk to 64x64\n= what the network sees")
    for i, (name, tint) in enumerate([("red", "Reds"), ("green", "Greens"), ("blue", "Blues")]):
        ax[3 + i].imshow(x[i], cmap="gray", vmin=-1, vmax=1, interpolation="nearest")
        ax[3 + i].set_title(f"4. {name} channel\n64x64 numbers, -1...1")
    for a in ax:
        off(a)
    fig.suptitle("From camera to numbers: 64 x 64 x 3 = 12,288 numbers go into the network",
                 fontsize=15, y=1.06)
    save(fig, "01_camera_to_numbers.png")


def fig_filters(net):
    w = net.b1[0].weight.detach().numpy()             # (8, 3, 3, 3)
    b = net.b1[0].bias.detach().numpy()
    lim = np.abs(w).max()
    fig, ax = plt.subplots(8, 3, figsize=(6.4, 16))
    for k in range(8):
        for c, name in enumerate(("red", "green", "blue")):
            a = ax[k, c]
            a.imshow(w[k, c], cmap=DIV, vmin=-lim, vmax=lim)
            for (i, j), v in np.ndenumerate(w[k, c]):
                a.text(j, i, f"{v:+.2f}", ha="center", va="center", fontsize=8,
                       color="white" if abs(v) > 0.55 * lim else INK)
            off(a)
            if k == 0:
                a.set_title(name, fontsize=12)
        ax[k, 0].set_ylabel(f"filter {k + 1}\nbias {b[k]:+.2f}", rotation=0, ha="right",
                            va="center", fontsize=11)
        ax[k, 0].yaxis.set_label_coords(-0.08, 0.5)
    fig.suptitle("Layer 1: the 8 learned 3x3 filters\n(blue = positive weight, red = negative)\n"
                 "8 filters x 3 colors x 9 weights + 8 biases = 224 numbers", fontsize=13, y=0.995)
    fig.tight_layout(rect=(0.05, 0, 1, 0.95))
    save(fig, "02_layer1_filters.png")


def best_filter_and_spot(net, x, v):
    k = int(np.argmax(v["c1"].reshape(8, -1).max(1)))
    i, j = np.unravel_index(np.argmax(v["c1"][k]), v["c1"][k].shape)
    i, j = int(np.clip(i, 1, 62)), int(np.clip(j, 1, 62))
    return k, i, j


def fig_one_multiply(net, x, v):
    k, i, j = best_filter_and_spot(net, x, v)
    w = net.b1[0].weight.detach().numpy()[k]
    b = float(net.b1[0].bias.detach()[k])
    patch = np.pad(x, ((0, 0), (1, 1), (1, 1)))[:, i:i + 3, j:j + 3]
    prods = patch * w
    total = prods.sum() + b
    fig = plt.figure(figsize=(16, 5.2))
    g = fig.add_gridspec(2, 6, width_ratios=[2.2, 1, 1, 1, 0.4, 2.2], height_ratios=[1, 1])
    a = fig.add_subplot(g[:, 0])
    a.imshow(show_rgb(x), interpolation="nearest")
    a.add_patch(Rectangle((j - 1.5, i - 1.5), 3, 3, fill=False, ec="#e34948", lw=2.5))
    a.set_title("the filter looks at a 3x3 patch\n(red square)")
    off(a)
    for c, name in enumerate(("red", "green", "blue")):
        a = fig.add_subplot(g[0, 1 + c])
        a.imshow(patch[c], cmap="gray", vmin=-1, vmax=1)
        for (r, s), val in np.ndenumerate(patch[c]):
            a.text(s, r, f"{val:+.2f}", ha="center", va="center", fontsize=9,
                   color="white" if val < 0 else INK)
        a.set_title(f"pixels ({name})", fontsize=10)
        off(a)
        a = fig.add_subplot(g[1, 1 + c])
        lim = np.abs(w).max()
        a.imshow(w[c], cmap=DIV, vmin=-lim, vmax=lim)
        for (r, s), val in np.ndenumerate(w[c]):
            a.text(s, r, f"{val:+.2f}", ha="center", va="center", fontsize=9, color=INK)
        a.set_title(f"x filter {k + 1} weights ({name})", fontsize=10)
        off(a)
    a = fig.add_subplot(g[:, 5])
    a.imshow(v["c1"][k], cmap=SEQ, interpolation="nearest")
    a.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, ec="#e34948", lw=2.5))
    a.set_title(f"filter {k + 1}'s output map (64x64)\nthis spot = one number")
    off(a)
    fig.text(0.585, 0.5, "=", fontsize=40, ha="center", va="center", color=INK2)
    fig.suptitle(f"One step of a convolution: 27 multiplications, add them up, add the bias:  "
                 f"{prods.sum():+.2f} {b:+.2f} = {total:+.2f}  ->  ReLU (negatives become 0)  ->  "
                 f"{max(total, 0):.2f}", fontsize=13, y=1.02)
    save(fig, "03_one_convolution_step.png")
    return k, i, j


def grid_maps(maps, cols, title, name, sub):
    n = len(maps)
    rows = int(np.ceil(n / cols))
    fig, ax = plt.subplots(rows, cols, figsize=(cols * 1.9, rows * 2.05 + 0.9))
    ax = np.atleast_2d(ax)
    vmax = max(float(maps.max()), 1e-6)
    for idx, a in enumerate(ax.flat):
        off(a)
        if idx < n:
            a.imshow(maps[idx], cmap=SEQ, vmin=0, vmax=vmax, interpolation="nearest")
            a.set_title(f"{sub} {idx + 1}", fontsize=10)
            for s in a.spines.values():
                s.set_visible(True)
                s.set_color(GRID)
    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    save(fig, name)


def fig_squash(v):
    sq = v["sq"]
    fig = plt.figure(figsize=(16, 7.6))
    g = fig.add_gridspec(4, 32, height_ratios=[1, 1, 1, 1.35], hspace=0.9)
    labels = ["layer 1: 8 maps -> 8 grids of 4x4 (128 numbers)",
              "layer 2: 16 maps -> 16 grids of 4x4 (256 numbers)",
              "layer 3: 32 maps -> 32 grids of 4x4 (512 numbers)"]
    vmax = max(float(s.max()) for s in sq)
    for r, s in enumerate(sq):
        n = s.shape[0]
        for k in range(n):
            a = fig.add_subplot(g[r, k])
            a.imshow(s[k], cmap=SEQ, vmin=0, vmax=vmax, interpolation="nearest")
            off(a)
        fig.text(0.125, 0.905 - r * 0.19, labels[r], fontsize=11, color=INK2)
    a = fig.add_subplot(g[3, :])
    a.imshow(v["feats"][None, :], cmap=SEQ, vmin=0, vmax=vmax, aspect="auto",
             interpolation="nearest")
    a.set_yticks([])
    a.set_xticks([0, 128, 384, 895])
    a.set_xticklabels(["0", "128 (layer 2 starts)", "384 (layer 3 starts)", "895"])
    a.set_title("all of them in one row: 896 numbers that describe the hand", fontsize=12)
    fig.suptitle("Squash: every map is shrunk to 4x4 by keeping the strongest value in each area",
                 fontsize=14, y=0.99)
    save(fig, "07_squash_to_896.png")


def fig_decision(net, v, label):
    W = net.fc.weight.detach().numpy()          # (6, 896)
    b = net.fc.bias.detach().numpy()
    feats = v["feats"]
    win = int(v["probs"].argmax())
    contrib = W[win] * feats
    fig = plt.figure(figsize=(16, 9.5))
    g = fig.add_gridspec(3, 2, height_ratios=[1, 1, 2.2], hspace=0.55, wspace=0.18)
    lim = np.abs(W).max()
    a = fig.add_subplot(g[0, :])
    a.imshow(W, cmap=DIV, vmin=-lim, vmax=lim, aspect="auto", interpolation="nearest")
    a.set_yticks(range(6))
    a.set_yticklabels([f"{c} fingers" if c else "fist (0)" for c in range(6)])
    a.set_xticks([0, 128, 384, 895])
    a.set_title("the last layer's weights: one row of 896 weights per answer "
                "(6 x 896 + 6 biases = 5,382 numbers)  -  blue = 'this feature votes FOR', "
                "red = 'votes AGAINST'", fontsize=11)
    a = fig.add_subplot(g[1, :])
    cl = np.abs(contrib).max()
    a.imshow(contrib[None, :], cmap=DIV, vmin=-cl, vmax=cl, aspect="auto", interpolation="nearest")
    a.set_yticks([])
    a.set_xticks([0, 128, 384, 895])
    a.set_title(f"votes for '{win}': each of the 896 features x its weight in the '{win}' row "
                f"(sum {contrib.sum():+.1f}, bias {b[win]:+.1f} -> score {v['logits'][win]:+.1f})",
                fontsize=11)
    names = ["0", "1", "2", "3", "4", "5"]
    a = fig.add_subplot(g[2, 0])
    lg = v["logits"]
    a.bar(names, lg, color=BLUE, width=0.6)
    a.axhline(0, color=INK2, lw=1)
    a.set_title("scores (sum of all votes + bias)")
    for i, val in enumerate(lg):
        a.text(i, val + (0.4 if val >= 0 else -0.4), f"{val:+.1f}", ha="center",
               va="bottom" if val >= 0 else "top", fontsize=10, color=INK)
    a.set_xlabel("number of fingers")
    a.grid(axis="y", color=GRID, lw=0.6)
    a.set_axisbelow(True)
    for s in ("top", "right"):
        a.spines[s].set_visible(False)
    a = fig.add_subplot(g[2, 1])
    pr = v["probs"] * 100
    a.bar(names, pr, color=BLUE, width=0.6)
    a.set_ylim(0, 110)
    a.set_title("softmax: scores -> probabilities (they add up to 100%)")
    a.text(win, pr[win] + 2, f"{pr[win]:.1f}%", ha="center", va="bottom", fontsize=12,
           color=INK, fontweight="bold")
    a.set_xlabel("number of fingers")
    a.set_ylabel("%")
    a.grid(axis="y", color=GRID, lw=0.6)
    a.set_axisbelow(True)
    for s in ("top", "right"):
        a.spines[s].set_visible(False)
    fig.suptitle(f"The decision for this picture (really {label} fingers): "
                 f"the network answers {win} with {pr[win]:.1f}% confidence", fontsize=15, y=0.98)
    save(fig, "08_decision.png")


def fig_poster(x, v):
    """The whole network on one page: every stage, how many numbers live there, and how many
    learned weights connect it - 'where all the points of the neural net are'."""
    fig = plt.figure(figsize=(22, 9))
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, 22)
    ax.set_ylim(0, 9)
    off(ax)

    def stack(x0, y0, maps, size, rgb=False, count=None):
        """Up to 6 cards (back ones offset up-right); `count` = how many maps really exist."""
        n_show = 1 if rgb else min(len(maps), 6)
        if not rgb:                            # most active maps in front
            maps = maps[np.argsort(-maps.reshape(len(maps), -1).sum(1))]
        step = 0.18
        for k in reversed(range(n_show)):
            ox, oy = x0 + k * step, y0 + k * step
            z = 2 + n_show - k
            if rgb:
                ax.imshow(maps, extent=(ox, ox + size, oy, oy + size), interpolation="nearest", zorder=z)
            else:
                ax.imshow(maps[k], cmap=SEQ, vmin=0, vmax=max(float(maps.max()), 1e-6),
                          extent=(ox, ox + size, oy, oy + size), interpolation="nearest", zorder=z)
            ax.add_patch(Rectangle((ox, oy), size, size, fill=False, ec=GRID, lw=0.8, zorder=z))
        if count:
            ax.text(x0 + (n_show - 1) * step + size + 0.08, y0 + (n_show - 1) * step + size,
                    f"x{count}", fontsize=11, color=INK2, va="top")

    def arrow(x0, x1, y=4.6, text=""):
        ax.add_patch(FancyArrowPatch((x0, y), (x1, y), arrowstyle="-|>", mutation_scale=18,
                                     color=INK2, lw=1.5))
        if text:
            ax.text((x0 + x1) / 2, y + 0.25, text, ha="center", va="bottom", fontsize=9, color=INK2)

    cols = [  # x, title, numbers line, params line
        (0.4, "INPUT", "64 x 64 x 3\n= 12,288 numbers", ""),
        (3.6, "LAYER 1", "8 maps of 32 x 32\n= 8,192 numbers", "224 weights\n(8 filters, 3x3x3)"),
        (7.3, "LAYER 2", "16 maps of 16 x 16\n= 4,096 numbers", "1,168 weights\n(16 filters, 3x3x8)"),
        (10.9, "LAYER 3", "32 maps of 8 x 8\n= 2,048 numbers", "4,640 weights\n(32 filters, 3x3x16)"),
        (14.6, "SQUASH", "every map -> 4 x 4\n8+16+32 maps = 896 numbers", "no weights"),
        (18.6, "DECISION", "6 scores -> softmax\n= 6 probabilities", "5,382 weights\n(6 x 896 + 6)"),
    ]
    stack(0.4, 3.4, show_rgb(x), 2.4, rgb=True)
    stack(3.6, 3.2, v["f1"], 2.0, count=8)
    stack(7.3, 3.45, v["f2"], 1.5, count=16)
    stack(10.9, 3.7, v["f3"], 1.05, count=32)
    # squash: three small stacks (from layer 1, 2, 3)
    stack(14.6, 5.25, v["sq"][0], 0.6, count=8)
    stack(14.6, 3.85, v["sq"][1], 0.6, count=16)
    stack(14.6, 2.45, v["sq"][2], 0.6, count=32)
    # decision bars
    pr = v["probs"]
    for i, p in enumerate(pr):
        y = 5.6 - i * 0.48
        ax.add_patch(Rectangle((19.2, y), 2.2, 0.32, color="#eceae6", zorder=2))
        ax.add_patch(Rectangle((19.2, y), 2.2 * p, 0.32, color=BLUE, zorder=3))
        ax.text(19.05, y + 0.16, str(i), ha="right", va="center", fontsize=11)
        ax.text(21.5, y + 0.16, f"{p * 100:.0f}%", ha="left", va="center", fontsize=9, color=INK2)
    for x0, title, nums, params in cols:
        ax.text(x0, 8.3, title, fontsize=15, fontweight="bold", va="top")
        ax.text(x0, 7.75, nums, fontsize=10.5, va="top", color=INK)
        if params:
            ax.text(x0, 1.7, params, fontsize=10.5, va="top", color=BLUE)
    arrow(2.95, 3.5, text="conv\n+ReLU\n+pool")
    arrow(6.75, 7.2, text="conv\n+ReLU\n+pool")
    arrow(10.2, 10.8, text="conv\n+ReLU\n+pool")
    arrow(13.4, 14.5, text="max\n4x4")
    arrow(17.8, 18.5, text="weighted\nvotes")
    ax.text(0.4, 0.75, "Learned weights in total: 224 + 1,168 + 4,640 + 5,382 = 11,414.   "
            "Work per camera frame: about 3.2 million multiply-adds, 13.6 ms on the UnoQ.",
            fontsize=12, color=INK)
    ax.text(0.4, 0.3, "Every map shown is the REAL output of the trained network for this picture of "
            "a hand. Darker blue = stronger signal.", fontsize=10.5, color=INK2)
    save(fig, "00_whole_network_poster.png")


def fig_all_counts(net, examples):
    fig, ax = plt.subplots(2, 6, figsize=(18, 6.2), gridspec_kw={"height_ratios": [1.15, 1]})
    for c, p in examples.items():
        x = preprocess_roi(cv2.imread(str(p)))
        pr = run(net, x)["probs"] * 100
        ax[0, c].imshow(show_rgb(x), interpolation="nearest")
        off(ax[0, c])
        ax[0, c].set_title(f"shown: {c}", fontsize=13)
        a = ax[1, c]
        a.bar(range(6), pr, color=BLUE, width=0.65)
        a.set_ylim(0, 115)
        a.set_xticks(range(6))
        a.text(int(pr.argmax()), pr.max() + 3, f"{pr.max():.0f}%", ha="center", fontsize=10)
        a.set_title(f"answer: {int(pr.argmax())}", fontsize=11)
        a.set_yticks([0, 50, 100] if c == 0 else [])
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
    fig.suptitle("All six answers: the 64x64 picture the network sees, and its probabilities",
                 fontsize=15)
    fig.tight_layout()
    save(fig, "09_all_six_answers.png")


def conv_slide_frames(net, x, k, n=24):
    """Stop-motion helper: the 3x3 window sweeping across the image while the output map fills."""
    out_dir = OUT / "conv_slide"
    out_dir.mkdir(parents=True, exist_ok=True)
    full = torch.relu(net.b1[0](torch.from_numpy(x)[None]))[0, k].detach().numpy()
    total = 64 * 64
    for f in range(n):
        upto = int((f + 1) / n * total)
        i, j = divmod(max(upto - 1, 0), 64)
        shown = np.full_like(full, np.nan)
        shown.flat[:upto] = full.flat[:upto]
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        ax[0].imshow(show_rgb(x), interpolation="nearest")
        ax[0].add_patch(Rectangle((j - 1.5, i - 1.5), 3, 3, fill=False, ec="#e34948", lw=2.5))
        ax[0].set_title("input (64x64)")
        cmap = SEQ.copy()
        cmap.set_bad(SURFACE)
        ax[1].imshow(shown, cmap=cmap, vmin=0, vmax=full.max(), interpolation="nearest")
        ax[1].add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, ec="#e34948", lw=2))
        ax[1].set_title(f"filter {k + 1} output, {upto:,} of {total:,} values")
        for a in ax:
            off(a)
            for s in a.spines.values():
                s.set_visible(True)
                s.set_color(GRID)
        fig.savefig(out_dir / f"frame_{f:02d}.png", dpi=120, bbox_inches="tight")
        plt.close(fig)
    print(f"  wrote conv_slide/frame_00..{n - 1:02d}.png")


def main():
    net = load_checkpoint(config.MODEL_PATH)
    print("picking example pictures...")
    examples = pick_examples(net)
    label = 3
    raw = cv2.imread(str(examples[label]))
    x = preprocess_roi(raw)
    v = run(net, x)
    fig_poster(x, v)
    fig_pipeline(raw)
    fig_filters(net)
    k, _, _ = fig_one_multiply(net, x, v)
    grid_maps(v["f1"], 8, "Layer 1 output: 8 maps of 32x32 (after ReLU + pooling)\n"
              "each map = where one filter found its pattern", "04_layer1_maps.png", "map")
    grid_maps(v["f2"], 8, "Layer 2 output: 16 maps of 16x16\nfilters that combine layer-1 "
              "patterns into bigger shapes", "05_layer2_maps.png", "map")
    grid_maps(v["f3"], 8, "Layer 3 output: 32 maps of 8x8\ncoarse 'is there a finger here?' "
              "detectors", "06_layer3_maps.png", "map")
    fig_squash(v)
    fig_decision(net, v, label)
    fig_all_counts(net, examples)
    conv_slide_frames(net, x, k)
    print(f"example picture used: {examples[label].relative_to(ROOT)}")


if __name__ == "__main__":
    main()
