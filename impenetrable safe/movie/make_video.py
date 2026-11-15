"""Render the stop-motion explainer video (silent - the voice-over is recorded over it).
    python movie/make_visuals.py      # first (makes movie/visuals/)
    python movie/make_video.py        # -> movie/impenetrable_safe_explainer.mp4 + movie/TIMING.md

Stop-motion look: 12 frames/s, every pose held for 2 frames ("shooting on twos"), steady camera
(no shake). Every map, weight and percentage is from the REAL trained network.
The safe-screen scenes use the real screen code with a made-up DEMO code (never the real one)."""
import subprocess
import sys
from pathlib import Path

import cv2
import imageio_ffmpeg
import matplotlib
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "movie"))
from make_visuals import SEQ, pick_examples                       # noqa: E402
from src import config                                            # noqa: E402
from src import safe_sim as S                                     # noqa: E402
from src.lock_policy import LockPolicy                            # noqa: E402
from src.model import load_checkpoint                             # noqa: E402
from src.preprocess import preprocess_roi                         # noqa: E402

VIS = ROOT / "movie" / "visuals"
OUT = ROOT / "movie" / "impenetrable_safe_explainer.mp4"
W, H, FPS = 1920, 1080, 12
BG = (251, 252, 252)                       # BGR of #fcfcfb
INK, INK2, BLUE, RED = (11, 11, 11), (78, 81, 82), (214, 120, 42), (72, 73, 227)
DEMO_CODE = (1, 4, 2)                      # NOT the real code
FONT_DIR = Path(matplotlib.get_data_path()) / "fonts" / "ttf"


# ------------------------------------------------------------------ drawing helpers
_font_cache, _text_cache = {}, {}


def font(size, bold):
    key = (size, bold)
    if key not in _font_cache:
        _font_cache[key] = ImageFont.truetype(
            str(FONT_DIR / ("DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf")), size)
    return _font_cache[key]


def text_tile(s, size, color, bold):
    key = (s, size, color, bold)
    if key not in _text_cache:
        f = font(size, bold)
        l, t, r, b = f.getbbox(s)
        im = Image.new("RGBA", (r - l + 4, b - t + 4), (0, 0, 0, 0))
        ImageDraw.Draw(im).text((2 - l, 2 - t), s, font=f, fill=color[::-1] + (255,))
        a = np.array(im)
        _text_cache[key] = (a[..., 2::-1].astype(np.float32), a[..., 3:4].astype(np.float32) / 255)
    return _text_cache[key]


def put(img, s, x, y, size=32, color=INK, bold=False, anchor="l"):
    """Text at (x, y) = top-left ('l'), top-center ('c') or top-right ('r')."""
    for i, line in enumerate(s.split("\n")):
        rgb, a = text_tile(line, size, color, bold)
        h, w = a.shape[:2]
        xx = int(x - (w / 2 if anchor == "c" else w if anchor == "r" else 0))
        yy = int(y + i * size * 1.3)
        x0, y0, x1, y1 = max(xx, 0), max(yy, 0), min(xx + w, W), min(yy + h, H)
        if x1 <= x0 or y1 <= y0:
            continue
        sub = img[y0:y1, x0:x1].astype(np.float32)
        aa, rr = a[y0 - yy:y1 - yy, x0 - xx:x1 - xx], rgb[y0 - yy:y1 - yy, x0 - xx:x1 - xx]
        img[y0:y1, x0:x1] = (sub * (1 - aa) + rr * aa).astype(np.uint8)


def fit(img, mw, mh, up=True):
    s = min(mw / img.shape[1], mh / img.shape[0])
    if not up:
        s = min(s, 1.0)
    return cv2.resize(img, (max(1, int(img.shape[1] * s)), max(1, int(img.shape[0] * s))),
                      interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)


def paste(dst, img, x, y):
    h, w = img.shape[:2]
    x0, y0, x1, y1 = max(x, 0), max(y, 0), min(x + w, W), min(y + h, H)
    if x1 > x0 and y1 > y0:
        dst[y0:y1, x0:x1] = img[y0 - y:y1 - y, x0 - x:x1 - x]


def blank():
    return np.full((H, W, 3), BG, np.uint8)


def title(img, n, s):
    put(img, f"{n}", 48, 28, 30, BLUE, True)
    put(img, s, 100, 28, 30, INK, True)
    cv2.line(img, (48, 80), (W - 48, 80), (217, 216, 212), 2)


def caption(img, s):
    put(img, s, W // 2, H - 78, 30, INK2, False, "c")


def on_twos(k):
    return k - k % 2


def seq_color(v):
    r, g, b, _ = SEQ(float(np.clip(v, 0, 1)))
    return int(b * 255), int(g * 255), int(r * 255)


def load(name):
    img = cv2.imread(str(VIS / name))
    if img is None:
        raise SystemExit(f"missing {name}: run python movie/make_visuals.py first")
    return img


def cover_right(img, frac):
    """Hide everything right of `frac` (a reveal, left to right)."""
    out = img.copy()
    out[:, int(img.shape[1] * frac):] = BG
    return out


def cover_below(img, frac):
    out = img.copy()
    out[int(img.shape[0] * frac):] = BG
    return out


# ------------------------------------------------------------------ the safe screen
class ScreenSim:
    """The real safe screen (src/safe_sim.py drawing code) driven by the real lock policy."""

    def __init__(self, pictures, empty):
        ph = S.H - S.TOP - S.BOTTOM - S.BANNER
        self.guards = [S.fit(S.load_img("guard.jpg"), S.LEFT_W, ph),
                       S.fit(S.load_img("guard_open.jpg"), S.LEFT_W, ph)]
        self.policy = LockPolicy(lambda d: tuple(d) == DEMO_CODE)
        self.t = 0.0
        self.pictures, self.empty = pictures, empty

    def frame(self, show=None, button=False):
        """show = None (empty paper) or a digit 0-5 (that picture, with the model's real answer)."""
        self.t += 1.0 / FPS
        if show is None:
            cam, reading = self.empty, None
        else:
            cam, reading = self.pictures[show]
        pred = reading if self.policy.state.name == "LISTEN" else None
        out = self.policy.tick(self.t, pred, button, False)
        out.lockout_left = 0.0
        c = np.full((S.H, S.W, 3), S.BG, np.uint8)
        S.draw_leds(c, out, self.t)
        S.draw_guard(c, out, self.guards, self.t)
        S.draw_right(c, out, self.policy, cam, reading, self.t, False)
        cv2.putText(c, "demo code - the real code is never shown", (14, S.H - 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, S.DIM, 1, cv2.LINE_AA)
        img = blank()
        sc = fit(c, W - 120, H - 240)
        paste(img, sc, (W - sc.shape[1]) // 2, 100)
        return img, out


# ------------------------------------------------------------------ scenes
def scene_intro(ctx, frames):
    guard = fit(cv2.imread(str(ROOT / "assets" / "guard.jpg")), 560, 820)
    for k in range(frames):
        img = blank()
        kk = on_twos(k)
        x_end = 1300
        x = int(W + 50 - min(kk / 12, 1) * (W + 50 - x_end))    # slides in over 1 s
        paste(img, guard, x, 160)
        if kk >= 12:
            put(img, "The Impenetrable Safe", 120, 330, 76, INK, True)
        if kk >= 24:
            put(img, "No keypad. No key.", 124, 470, 42, INK2)
        if kk >= 40:
            put(img, "Show a 3-digit code with your fingers.", 124, 545, 42, INK2)
        if kk >= 60:
            put(img, "A neural network I built\ndecides what it sees.", 124, 630, 42, BLUE, True)
        if kk >= 84:
            put(img, "Camera + network + lock: all on an Arduino UnoQ", 124, 800, 34, INK2)
        yield img


def scene_button(ctx, frames):
    sim = ScreenSim(ctx["screen_pics"], ctx["empty"])
    for k in range(frames):
        img, out = sim.frame(show=None if k < 48 else 0, button=(k == 24))
        title(img, 2, "Press the button: the safe starts listening")
        if 24 <= k < 48:
            caption(img, "all three LEDs blink once")
        elif k >= 48:
            caption(img, "every camera picture now goes through the network")
        yield img


def scene_numbers(ctx, frames):
    fig = fit(load("01_camera_to_numbers.png"), W - 120, 640)
    for k in range(frames):
        img = blank()
        title(img, 3, "From camera to numbers")
        kk = on_twos(k)
        frac = min(1.0, (kk + 2) / 120)                         # reveal over 10 s
        paste(img, cover_right(fig, frac), 60, 170)
        if kk >= 132:
            cv2.rectangle(img, (W // 2 - 330, 860), (W // 2 + 330, 960), (242, 225, 205), -1)
            put(img, "64 x 64 x 3 = 12,288 numbers", W // 2, 882, 46, INK, True, "c")
        yield img


def scene_poster(ctx, frames):
    raw = load("00_whole_network_poster.png")
    fig = fit(raw, W - 60, 820)
    fx, fy = 30, 150
    fw, fh = fig.shape[1], fig.shape[0]
    cols = [0.4, 3.6, 7.3, 10.9, 14.6, 18.6, 22.0]
    stops = len(cols) - 1
    per = (frames - 48) // stops
    for k in range(frames):
        img = blank()
        title(img, 4, "The whole network: where all the numbers live")
        paste(img, fig, fx, fy)
        kk = on_twos(k)
        stop = min(kk // per, stops - 1)
        local = kk - stop * per
        x_from = cols[max(stop - 1, 0)] + 0.6
        x_to = cols[stop] + 0.6
        p = min(1.0, local / 12)
        xd = x_from + (x_to - x_from) * p
        # highlight the current column
        x0 = fx + int((cols[stop] - 0.25) / 22 * fw)
        x1 = fx + int((cols[stop + 1] - 0.35) / 22 * fw)
        ov = img.copy()
        cv2.rectangle(ov, (x0, fy + 10), (x1, fy + fh - 60), (242, 225, 205), -1)
        img = cv2.addWeighted(ov, 0.35, img, 0.65, 0)
        # the courier token (one number travelling through)
        cx, cy = fx + int(xd / 22 * fw), fy + int(0.49 * fh)
        cv2.circle(img, (cx, cy), 26, RED, -1, cv2.LINE_AA)
        cv2.circle(img, (cx - 8, cy - 8), 7, (255, 255, 255), -1, cv2.LINE_AA)
        if kk >= frames - 48:
            cv2.rectangle(img, (W // 2 - 520, H - 130), (W // 2 + 520, H - 40), (242, 225, 205), -1)
            put(img, "224 + 1,168 + 4,640 + 5,382 = 11,414 weights", W // 2, H - 112, 44, INK,
                True, "c")
        yield img


def scene_conv(ctx, frames):
    filt = fit(load("02_layer1_filters.png"), 700, 940)
    step = fit(load("03_one_convolution_step.png"), W - 120, 760)
    slides = [fit(cv2.imread(str(p)), W - 300, 820)
              for p in sorted((VIS / "conv_slide").glob("frame_*.png"))]
    t1, t2 = 60, 60 + 108                                   # 5 s filters, 9 s one step
    for k in range(frames):
        img = blank()
        title(img, 5, "Convolution: a 3x3 filter slides over the picture")
        if k < t1:
            paste(img, filt, 120, 120)
            put(img, "Layer 1 has 8 filters.", 900, 300, 48, INK, True)
            put(img, "Each one: 3 x 3 weights for each color\n= 27 weights + 1 bias", 900, 390, 38,
                INK2)
            put(img, "blue = positive weight, red = negative", 900, 560, 32, INK2)
        elif k < t2:
            frac = min(1.0, (on_twos(k - t1) + 2) / 60)
            paste(img, cover_right(step, frac), 60, 150)
            if k - t1 > 48:
                caption(img, "multiply 27 pixels by 27 weights, add them up, add the bias, "
                             "then ReLU: negatives become 0")
        else:
            i = min(on_twos(k - t2) * len(slides) // (frames - t2), len(slides) - 1)   # rest of scene
            paste(img, slides[i], (W - slides[i].shape[1]) // 2, 130)
            caption(img, "it does this at all 4,096 positions: the result is a feature map")
        yield img


def scene_pool(ctx, frames):
    c1 = ctx["c1"]
    # an 8x8 patch of the real filter map where the finger edge is (largest variation)
    best, pos = -1, (0, 0)
    for i in range(0, 56, 2):
        for j in range(0, 56, 2):
            s = c1[i:i + 8, j:j + 8].std()
            if s > best:
                best, pos = s, (i, j)
    i0, j0 = pos
    patch = c1[i0:i0 + 8, j0:j0 + 8]
    pooled = patch.reshape(4, 2, 4, 2).max(axis=(1, 3))
    vmax = max(float(patch.max()), 1e-6)
    cell, gx, gy = 88, 140, 200
    ox, oy, ocell = 1130, 330, 96
    hand = fit(ctx["hand_rgb"], 250, 250)
    sc = 250 / 64
    per = max(6, (frames - 36) // 16)
    for k in range(frames):
        img = blank()
        title(img, 6, "Pooling: keep the strongest signal in every 2x2 square")
        n_done = min(16, on_twos(k) // per + 1)
        cur = n_done - 1
        ci, cj = divmod(cur, 4)
        for i in range(8):
            for j in range(8):
                x, y = gx + j * cell, gy + i * cell
                cv2.rectangle(img, (x, y), (x + cell - 3, y + cell - 3), seq_color(patch[i, j] / vmax), -1)
                v = patch[i, j]
                put(img, f"{v:.1f}", x + cell // 2, y + 28, 26,
                    (255, 255, 255) if v / vmax > 0.55 else INK, False, "c")
        x, y = gx + cj * 2 * cell, gy + ci * 2 * cell
        cv2.rectangle(img, (x - 4, y - 4), (x + 2 * cell + 1, y + 2 * cell + 1), RED, 6)
        cv2.arrowedLine(img, (gx + 8 * cell + 40, gy + 4 * cell), (ox - 50, gy + 4 * cell), INK2, 4,
                        tipLength=0.04)
        put(img, "max of 4", (gx + 8 * cell + ox) // 2 - 10, gy + 4 * cell - 60, 30, INK2, True, "c")
        for n in range(16):
            i, j = divmod(n, 4)
            x, y = ox + j * ocell, oy + i * ocell
            if n < n_done:
                cv2.rectangle(img, (x, y), (x + ocell - 3, y + ocell - 3), seq_color(pooled[i, j] / vmax), -1)
                v = pooled[i, j]
                put(img, f"{v:.1f}", x + ocell // 2, y + 32, 28,
                    (255, 255, 255) if v / vmax > 0.55 else INK, True, "c")
            else:
                cv2.rectangle(img, (x, y), (x + ocell - 3, y + ocell - 3), (236, 234, 230), 2)
        cv2.rectangle(img, (ox + cj * ocell - 4, oy + ci * ocell - 4),
                      (ox + (cj + 1) * ocell + 1, oy + (ci + 1) * ocell + 1), RED, 6)
        put(img, "8 x 8 piece of a real feature map", gx, gy - 60, 30, INK2)
        put(img, "after pooling: 4 x 4", ox, oy - 60, 30, INK2)
        paste(img, hand, W - 300, 300)
        cv2.rectangle(img, (W - 300 + int(j0 * sc), 300 + int(i0 * sc)),
                      (W - 300 + int((j0 + 8) * sc), 300 + int((i0 + 8) * sc)), RED, 3)
        put(img, "where on the hand", W - 300, 565, 24, INK2)
        caption(img, "4x fewer numbers, the strongest signals survive (and a small shift of the "
                     "hand barely changes the result)")
        yield img


def scene_layer1(ctx, frames):
    fig = fit(load("04_layer1_maps.png"), W - 100, 500)
    fw = fig.shape[1]
    tags = {0: ("edges", INK), 1: ("skin", RED), 2: ("skin", RED), 3: ("paper", BLUE),
            4: ("paper", BLUE), 7: ("skin", RED)}
    for k in range(frames):
        img = blank()
        title(img, 7, "What layer 1 learned by itself")
        shown = min(8, on_twos(k) // 12 + 1)                     # one map per second
        frac = shown / 8 if shown < 8 else 1.0
        x0 = (W - fw) // 2
        paste(img, cover_right(fig, frac), x0, 220)
        if on_twos(k) >= 8 * 12:
            for i, (t, col) in tags.items():
                cx = x0 + int((i + 0.5) / 8 * fw)
                put(img, t, cx, 220 + fig.shape[0] + 30, 40, col, True, "c")
            caption(img, "nobody told it to look for skin, paper or edges: training found them")
        yield img


def scene_deeper(ctx, frames):
    a = fit(load("05_layer2_maps.png"), W - 200, 860)
    b = fit(load("06_layer3_maps.png"), W - 200, 860)
    half = frames * 12 // 25
    for k in range(frames):
        img = blank()
        title(img, 8, "Going deeper: smaller pictures, more kinds of information")
        fig, kk = (a, k) if k < half else (b, k - half)
        frac = min(1.0, (on_twos(kk) + 2) / 48)
        paste(img, cover_below(fig, frac), (W - fig.shape[1]) // 2, 120)
        yield img


def scene_vote(ctx, frames):
    sq = fit(load("07_squash_to_896.png"), W - 160, 900)
    dec = fit(load("08_decision.png"), W - 300, 930)
    t1 = frames * 18 // 45
    for k in range(frames):
        img = blank()
        if k < t1:
            title(img, 9, "Squash: every map becomes 4 x 4 -> 896 numbers")
            frac = min(1.0, (on_twos(k) + 2) / 84)
            paste(img, cover_below(sq, frac), (W - sq.shape[1]) // 2, 110)
        else:
            title(img, 10, "The vote: 896 numbers x weights -> 6 scores -> softmax")
            kk = k - t1
            frac = min(1.0, 0.36 + on_twos(kk) / 150)
            paste(img, cover_below(dec, frac), (W - dec.shape[1]) // 2, 105)
        yield img


def scene_six(ctx, frames):
    fig = fit(load("09_all_six_answers.png"), W - 100, 780)
    for k in range(frames):
        img = blank()
        title(img, 11, "All six answers")
        shown = min(6, on_twos(k) // 12 + 1)
        paste(img, cover_right(fig, shown / 6), (W - fig.shape[1]) // 2, 160)
        yield img


def scene_training(ctx, frames):
    sheets = [fit(cv2.imread(str(p)), 300, 300)
              for p in sorted(config.AUDIT_DIR.glob("contact_*.png"))[:6]]
    lines = [("720 pictures, 3 people, 120 of each count", INK, True),
             ("start: 11,414 random weights", INK2, False),
             ("each batch of 32: measure how wrong (the loss)", INK2, False),
             ("backpropagation: nudge every weight\n  to be a little less wrong", INK2, False),
             ("60 passes over all the pictures", INK2, False),
             ("random turns, shifts, zooms, brightness", INK2, False),
             ("tested on sessions/people it never saw:", INK, True),
             ("about 86% on me, 66-81% on new people", BLUE, True)]
    for k in range(frames):
        img = blank()
        title(img, 12, "How it learned (on the PC)")
        kk = on_twos(k)
        for i, s in enumerate(sheets):
            t0 = i * 8
            if kk < t0:
                continue
            drop = max(0, 3 - (kk - t0) // 2) * 60             # falls in, 3 steps
            x, y = 80 + (i % 3) * 320, 150 + (i // 3) * 330 - drop
            paste(img, s, x, y)
            put(img, f"{i} fingers" if i else "fist", x + 150, y + 305, 24, INK2, False, "c")
        for i, (s, col, bold) in enumerate(lines):
            if kk >= 48 + i * 20:
                put(img, s, 1060, 160 + i * 98 + (40 if i > 3 else 0), 31, col, bold)
        yield img


def scene_unlock(ctx, frames):
    sim = ScreenSim(ctx["screen_pics"], ctx["empty"])
    plan = [None] * 12 + ["BUTTON"] + [0] * 15
    for d in DEMO_CODE:
        plan += [d] * 18 + [0] * 14
    plan += [0] * 30
    relock_at = frames - 66
    for k in range(frames):
        step = plan[k] if k < len(plan) else 0
        button = step == "BUTTON" or k == relock_at
        show = None if step is None else (0 if step == "BUTTON" else step)
        if k >= relock_at:
            show = None
        img, out = sim.frame(show, button)
        title(img, 13, "From digits to an open door")
        if out.state.name == "UNLOCKED":
            caption(img, "correct code: green flashes, the servo turns the latch open")
        elif out.state.name == "LISTEN":
            caption(img, "same digit, >= 90% sure, 10 pictures in a row -> yellow flash. "
                         "Fist between digits.")
        elif k >= relock_at:
            caption(img, "press the button: locked again")
        yield img


def scene_end(ctx, frames):
    guard = fit(cv2.imread(str(ROOT / "assets" / "guard_open.jpg")), 520, 700)
    for k in range(frames):
        img = blank()
        paste(img, guard, 1250, 200)
        put(img, "The Impenetrable Safe", 140, 330, 80, INK, True)
        put(img, "11,414 numbers guarding the door.", 144, 460, 48, BLUE, True)
        put(img, "PyTorch CNN, built and trained from scratch.\nRunning on an Arduino UnoQ.", 144,
            560, 38, INK2)
        yield img


SCENES = [  # (name, seconds, function, script scene)
    ("The safe", 12, scene_intro, "1"),
    ("Press the button", 10, scene_button, "2"),
    ("From camera to numbers", 18, scene_numbers, "3"),
    ("The whole network", 18, scene_poster, "4"),
    ("Convolution", 24, scene_conv, "5"),
    ("Pooling", 15, scene_pool, "5b"),
    ("What layer 1 learned", 18, scene_layer1, "6"),
    ("Going deeper", 16, scene_deeper, "7"),
    ("Squash and vote", 28, scene_vote, "8"),
    ("All six answers", 10, scene_six, "9"),
    ("How it learned", 20, scene_training, "10"),
    ("From digits to an open door", 20, scene_unlock, "11"),
    ("End card", 5, scene_end, "end"),
]


def context():
    net = load_checkpoint(config.MODEL_PATH)
    print("picking example pictures...")
    ex = pick_examples(net)
    pics = {}
    for c, p in ex.items():
        raw = cv2.imread(str(p))
        x = preprocess_roi(raw)
        with torch.no_grad():
            pr = torch.softmax(net(torch.from_numpy(x)[None]), 1)[0].numpy()
        pics[c] = (raw, (int(pr.argmax()), float(pr.max())))
    x3 = preprocess_roi(cv2.imread(str(ex[3])))
    with torch.no_grad():
        c1 = torch.relu(net.b1[0](torch.from_numpy(x3)[None]))[0].numpy()
    k = int(np.argmax(c1.reshape(8, -1).max(1)))
    hand = cv2.resize(cv2.cvtColor(((x3.transpose(1, 2, 0) * 0.5 + 0.5) * 255).clip(0, 255)
                                   .astype(np.uint8), cv2.COLOR_RGB2BGR), (64, 64))
    empty = np.full_like(pics[0][0], 225)
    bg = config.BACKGROUND_PATH
    if bg.exists():
        empty = cv2.imread(str(bg))
    return {"screen_pics": pics, "c1": c1[k], "hand_rgb": hand, "empty": empty}


def preview(ctx, out_dir):
    """Save 3 frames per scene (start / middle / end) instead of encoding the video."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for idx, (name, secs, fn, ref) in enumerate(SCENES):
        n = secs * FPS
        keep = {n // 6, n // 2, n - 1}
        for k, f in enumerate(fn(ctx, n)):
            if k in keep:
                cv2.imwrite(str(out_dir / f"{idx:02d}_{k:03d}.jpg"), cv2.resize(f, (960, 540)))
    print("preview frames in", out_dir)


def main():
    ctx = context()
    if len(sys.argv) > 1 and sys.argv[1] == "--preview":
        return preview(ctx, Path(sys.argv[2]))
    cmd = [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-f", "rawvideo",
           "-pix_fmt", "bgr24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", "-preset", "medium",
           "-movflags", "+faststart", str(OUT)]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    timing, t = [], 0
    for name, secs, fn, ref in SCENES:
        n = secs * FPS
        timing.append((t, name, ref))
        print(f"  {t // 60}:{t % 60:02d}  {name}")
        for f in fn(ctx, n):
            proc.stdin.write(f.tobytes())
        t += secs
    proc.stdin.close()
    proc.wait()
    lines = ["# Video timing (record your voice-over to these)", "",
             f"`movie/{OUT.name}`, {t // 60}:{t % 60:02d} long, 12 frames/s, silent.", "",
             "| Starts at | Scene | Script section |", "|---|---|---|"]
    lines += [f"| {s // 60}:{s % 60:02d} | {n} | Scene {r} |" for s, n, r in timing]
    (ROOT / "movie" / "TIMING.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {OUT} ({OUT.stat().st_size / 1e6:.1f} MB, {t} s)")


if __name__ == "__main__":
    main()
