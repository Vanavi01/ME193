"""Desktop step counter: replays an ankle recording and counts steps as they happen.

    python step_counter_app.py [recording.csv]

Same look and pictograms as mqtt_viewer.py. The pictogram switches between "stand" and "walk" with
every step, the big number is the running step count, and the strip below shows the ankle gyro with
a marker at each counted step.

Method (menu at the bottom): "neural network" (default; the PyTorch model in step_model.pt) or the
less error-prone "peak detector". Tick "no end stomps" for a recording that ended without them
(for example data/S19_walk.csv).

It replays a recorded file; the board only publishes its activity label over MQTT, not raw samples,
so a live feed would need a change to model2_app/python/main.py.
"""
import os
import sys
import time
import tkinter as tk
from tkinter import filedialog, ttk

import numpy as np

import count_steps as cs
import train_steps as ts

here = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(here, "..", "data")

# Same pictograms as the sketch and mqtt_viewer.py, 8 rows x 13 cols ('#' = lit).
PICTO = {
    "stand": ["......##.....",
              "......##.....",
              "....######...",
              "....#.##.#...",
              "....#.##.#...",
              "......##.....",
              ".....#..#....",
              ".....#..#...."],
    "walk": ["......##.....",
             "......##.....",
             ".....####....",
             ".....#.##.#..",
             ".....#.##.#..",
             "......##.....",
             ".....##.##...",
             "....##...##.."],
}
BG, FG, DIM, OFF = "#111418", "#f2f4f7", "#5b6470", "#1d2229"
BLUE, GREEN = "#4ea1ff", "#3ecf8e"
CELL, GAP = 30, 4
STEP_FLASH_S = 0.30   # pictogram shows "walk" this long after each step
TRACE_S = 8.0         # seconds of gyro shown in the strip


class App:
    def __init__(self, root, path=None):
        self.root = root
        root.title("Step counter")
        root.configure(bg=BG)
        self.t = self.gyro = self.cum = self.step_t = None
        self.pos = 0.0          # playback position, seconds
        self.playing = False
        self.last_tick = 0.0
        self.path = None
        self.error = ""

        w = 13 * CELL + 12 * GAP
        h = 8 * CELL + 7 * GAP
        self.canvas = tk.Canvas(root, width=w, height=h, bg=BG, highlightthickness=0)
        self.canvas.pack(padx=32, pady=(28, 8))
        self.cells = [[self.canvas.create_rectangle(
            c * (CELL + GAP), r * (CELL + GAP), c * (CELL + GAP) + CELL, r * (CELL + GAP) + CELL,
            fill=OFF, width=0) for c in range(13)] for r in range(8)]

        self.count_lbl = tk.Label(root, text="0", font=("Segoe UI", 72, "bold"), bg=BG, fg=FG)
        self.count_lbl.pack()
        tk.Label(root, text="STEPS", font=("Segoe UI", 14), bg=BG, fg=DIM).pack()
        self.info = tk.Label(root, text="open a recording to start", font=("Segoe UI", 11),
                             bg=BG, fg=DIM, wraplength=w)
        self.info.pack(pady=(4, 8))

        self.trace = tk.Canvas(root, width=w, height=90, bg=OFF, highlightthickness=0)
        self.trace.pack(padx=32)
        self.seek = ttk.Scale(root, from_=0, to=1, command=self.on_seek)
        self.seek.pack(fill="x", padx=32, pady=(8, 0))

        bar = tk.Frame(root, bg=BG)
        bar.pack(pady=12)
        self.btn_open = tk.Button(bar, text="Open…", command=self.choose, width=8)
        self.btn_open.grid(row=0, column=0, padx=4)
        self.btn_play = tk.Button(bar, text="Play", command=self.toggle, width=8)
        self.btn_play.grid(row=0, column=1, padx=4)
        tk.Button(bar, text="Restart", command=self.restart, width=8).grid(row=0, column=2, padx=4)
        self.speed = tk.StringVar(value="4x")
        ttk.Combobox(bar, textvariable=self.speed, values=["1x", "2x", "4x", "8x"], width=4,
                     state="readonly").grid(row=0, column=3, padx=4)

        opt = tk.Frame(root, bg=BG)
        opt.pack(pady=(0, 16))
        self.method = tk.StringVar(value="neural network")
        m = ttk.Combobox(opt, textvariable=self.method, state="readonly", width=16,
                         values=["peak detector", "neural network"])
        m.grid(row=0, column=0, padx=6)
        m.bind("<<ComboboxSelected>>", lambda e: self.reload())
        self.no_end = tk.BooleanVar(value=False)
        tk.Checkbutton(opt, text="no end stomps", variable=self.no_end, command=self.reload,
                       bg=BG, fg=FG, selectcolor=OFF, activebackground=BG,
                       activeforeground=FG).grid(row=0, column=1, padx=6)

        self.draw_picto("stand", DIM)
        if path:
            self.load(path)
        self.tick()

    # ---------------------------------------------------------------- loading
    def choose(self):
        p = filedialog.askopenfilename(initialdir=DATA, title="Ankle recording",
                                       filetypes=[("CSV", "*.csv"), ("All", "*.*")])
        if p:
            self.load(p)

    def reload(self):
        if self.path:
            self.load(self.path, keep_pos=True)

    def load(self, path, keep_pos=False):
        self.info.configure(text="counting…", fg=DIM)
        self.root.update_idletasks()
        try:
            t, x = cs.load_ankle_csv(path)
            if self.method.get() == "peak detector":
                pk, w0, w1 = cs.peak_steps(t, x, end_stomps=not self.no_end.get())
                marks = np.zeros(len(t))
                marks[pk] = 2                    # each leg swing = this step + the other leg's
                cum = np.cumsum(marks)
                self.step_t = t[pk]
                note = f"peak detector, walk {w0:.0f}-{w1:.0f} s"
            else:
                tt, rates, _ = cs.count(path)
                cum = np.cumsum(rates.mean(0)) / ts.FS
                sd = (rates.sum(1) / ts.FS).std(ddof=1)
                # a step "happens" each time the running count crosses an integer
                self.step_t = tt[np.searchsorted(cum, np.arange(1, int(cum[-1]) + 1))]
                note = f"neural network, +-{sd:.1f} between 5 nets"
                if cs.trained_on(path):
                    note += "; trained on this recording, so not a fair test"
        except (SystemExit, Exception) as e:   # SystemExit: wrong model file
            self.error = f"could not count {os.path.basename(path)}: {e}"
            self.info.configure(text=self.error, fg="#ff6b6b")
            return
        self.error = ""
        self.path, self.t, self.gyro, self.cum = path, t, x[:, 4], cum
        self.total = cum[-1]
        self.note = note
        known = cs.known_counts(path)
        if "hand" in known:
            self.note += f"; hand count {known['hand']:g}"
        self.seek.configure(to=t[-1])
        if not keep_pos:
            self.pos = 0.0
            self.playing = True
            self.btn_play.configure(text="Pause")
        self.last_tick = time.time()
        self.root.title(f"Step counter - {os.path.basename(path)}")

    # ---------------------------------------------------------------- controls
    def toggle(self):
        if self.t is None:
            return self.choose()
        if not self.playing and self.pos >= self.t[-1]:
            self.pos = 0.0
        self.playing = not self.playing
        self.last_tick = time.time()
        self.btn_play.configure(text="Pause" if self.playing else "Play")

    def restart(self):
        self.pos = 0.0
        self.last_tick = time.time()

    def on_seek(self, v):
        if self.t is not None and abs(float(v) - self.pos) > 0.3:   # user drag, not our own update
            self.pos = float(v)

    # ---------------------------------------------------------------- drawing
    def draw_picto(self, name, color):
        grid = PICTO[name]
        for r in range(8):
            for c in range(13):
                self.canvas.itemconfigure(self.cells[r][c], fill=color if grid[r][c] == "#" else OFF)

    def draw_trace(self):
        cv = self.trace
        cv.delete("all")
        w, h = int(cv["width"]), int(cv["height"])
        lo = max(0.0, self.pos - TRACE_S)
        i0, i1 = int(lo * ts.FS), int(self.pos * ts.FS) + 1
        seg = self.gyro[i0:i1]
        if len(seg) < 2:
            return
        span = max(np.abs(self.gyro).max(), 1.0)
        xs = (np.arange(i0, i0 + len(seg)) / ts.FS - (self.pos - TRACE_S)) / TRACE_S * w
        ys = h / 2 - seg / span * (h / 2 - 6)
        step = max(1, len(seg) // w)
        pts = np.column_stack([xs[::step], ys[::step]]).ravel().tolist()
        cv.create_line(*pts, fill=BLUE, width=1)
        for st in self.step_t[(self.step_t >= lo) & (self.step_t <= self.pos)]:
            x = (st - (self.pos - TRACE_S)) / TRACE_S * w
            cv.create_line(x, 4, x, h - 4, fill=GREEN, width=2)

    def tick(self):
        now = time.time()
        if self.t is not None:
            if self.playing:
                self.pos = min(self.t[-1], self.pos + (now - self.last_tick) * float(self.speed.get()[:-1]))
                if self.pos >= self.t[-1]:
                    self.playing = False
                    self.btn_play.configure(text="Play")
            self.last_tick = now
            i = min(int(self.pos * ts.FS), len(self.cum) - 1)
            n = int(round(self.cum[i]))
            recent = self.step_t[(self.step_t <= self.pos) & (self.step_t > self.pos - STEP_FLASH_S)]
            walking = len(recent) > 0
            self.count_lbl.configure(text=str(n), fg=GREEN if walking else FG)
            self.draw_picto("walk" if walking else "stand", GREEN if walking else BLUE)
            self.draw_trace()
            self.seek.set(self.pos)
            if not self.error:
                self.info.configure(text=f"{os.path.basename(self.path)}   {self.pos:.0f} / {self.t[-1]:.0f} s"
                                         f"   ({self.note})", fg=DIM)
        self.root.after(33, self.tick)


def main():
    root = tk.Tk()
    path = sys.argv[1] if len(sys.argv) > 1 else None
    App(root, path)
    root.mainloop()


if __name__ == "__main__":
    main()
