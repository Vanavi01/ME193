"""Live step counter: the Uno Q streams raw ankle samples over MQTT, the PyTorch network counts steps.

    pip install paho-mqtt torch numpy pandas
    python live_step_counter.py [broker]            # live from the board (model2_app MODE = "stream")
    python live_step_counter.py --replay ../data/S21_walk.csv   # fake the board with a recorded file

Only the ankle strap is read. The main number counts both feet from the left-ankle swings (each left
swing = a left step, and a right step follows half a stride later unless you stop); the PyTorch network
runs alongside and is shown in small print (use --net to make it the main number). Same look and pictograms as mqtt_viewer.py. The count lags the leg by
about 0.65 s because the network looks that far ahead. Accuracy on unseen recordings was about 14%
average error (see ../results/uncertainty_report.md), so treat the number as an estimate.
"""
import argparse
import json
import os
import queue
import threading
import time
import tkinter as tk
from tkinter import ttk

import numpy as np
import pandas as pd

import live_steps as ls
from step_counter_app import PICTO, BG, FG, DIM, OFF, BLUE, GREEN, CELL, GAP, TRACE_S

BROKER = "broker.hivemq.com"      # match MQTT_BROKER in model2_app/python/main.py
TOPIC = "tufts/gait/steps/imu"    # match STREAM_TOPIC there
PORT = 1883
STALE_S = 3.0                     # no samples for this long -> grey out
FLASH_S = 0.30
UPDATE_MS = 250
here = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(here, "..", "data")

inbox = queue.Queue()             # ("status", text) | ("samples", seq, rows)


KEEP_LAST_RIGHT = False   # --keep-last-right: when you stop, still count the right step after the last left swing
USE_NET = False   # --net: show the neural network's count as the main number instead of the swing count


def start_mqtt(broker):
    import paho.mqtt.client as mqtt
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2) if hasattr(mqtt, "CallbackAPIVersion") else mqtt.Client()

    def on_connect(c, *a, **k):
        c.subscribe(TOPIC, qos=1)
        inbox.put(("status", f"connected to {broker}  -  waiting for the board on {TOPIC}"))

    def on_disconnect(c, *a, **k):
        inbox.put(("status", f"disconnected from {broker}, retrying..."))

    def on_message(c, u, msg):
        try:
            d = json.loads(msg.payload)
            inbox.put(("samples", d.get("seq"), d["s"]))
        except (ValueError, KeyError):
            pass

    client.on_connect, client.on_disconnect, client.on_message = on_connect, on_disconnect, on_message
    client.connect_async(broker, PORT, keepalive=30)
    client.loop_start()
    return client


def start_replay(path, speed):
    """Pretend to be the board: feed a recorded CSV in 0.2 s batches at `speed` x real time."""
    df = pd.read_csv(path)
    rows = df[["t_ms"] + ls.ts.ANKLE].to_numpy(float)

    def run():
        inbox.put(("status", f"replaying {os.path.basename(path)} (as if streamed from the board)"))
        t_start, i, seq = time.time(), 0, 0
        while i < len(rows):
            due = (rows[i, 0] - rows[0, 0]) / 1000.0 / speed
            if time.time() - t_start < due:
                time.sleep(0.02)
                continue
            j = i
            while j < len(rows) and (rows[j, 0] - rows[0, 0]) / 1000.0 / speed <= time.time() - t_start:
                j += 1
            inbox.put(("samples", seq, rows[i:j].tolist()))
            seq += 1
            i = j
        inbox.put(("status", "replay finished"))
    threading.Thread(target=run, daemon=True).start()


class App:
    def __init__(self, root):
        self.root = root
        root.title("Step counter - live")
        root.configure(bg=BG)
        self.lc = ls.LiveCounter()
        self.sw = ls.SwingTracker(retract_at_stop=not KEEP_LAST_RIGHT)
        self.use_net = USE_NET
        self.last_rx = 0.0
        self.flash_until = 0.0
        self.last_seq = None
        self.missed = 0
        self.rx_samples = 0
        self.rx_t0 = None

        w = 13 * CELL + 12 * GAP
        h = 8 * CELL + 7 * GAP
        self.canvas = tk.Canvas(root, width=w, height=h, bg=BG, highlightthickness=0)
        self.canvas.pack(padx=32, pady=(28, 8))
        self.cells = [[self.canvas.create_rectangle(
            c * (CELL + GAP), r * (CELL + GAP), c * (CELL + GAP) + CELL, r * (CELL + GAP) + CELL,
            fill=OFF, width=0) for c in range(13)] for r in range(8)]
        self.count_lbl = tk.Label(root, text="0", font=("Segoe UI", 72, "bold"), bg=BG, fg=DIM)
        self.count_lbl.pack()
        tk.Label(root, text="STEPS", font=("Segoe UI", 14), bg=BG, fg=DIM).pack()
        self.sub = tk.Label(root, text="", font=("Segoe UI", 12), bg=BG, fg=DIM)
        self.sub.pack(pady=(2, 6))
        self.trace = tk.Canvas(root, width=w, height=90, bg=OFF, highlightthickness=0)
        self.trace.pack(padx=32)
        bar = tk.Frame(root, bg=BG)
        bar.pack(pady=10)
        tk.Button(bar, text="Reset count", command=self.reset, width=12).grid(row=0, column=0, padx=4)
        tk.Button(bar, text="Save samples...", command=self.save, width=14).grid(row=0, column=1, padx=4)
        self.status = tk.Label(root, text="", font=("Segoe UI", 10), bg=BG, fg=DIM, wraplength=w)
        self.status.pack(pady=(0, 14))
        self.draw_picto("stand", DIM)
        self.tick()

    # ------------------------------------------------------------------ actions
    def reset(self):
        self.lc.reset()
        self.sw.reset()
        self.missed = self.rx_samples = 0
        self.last_seq = self.rx_t0 = None

    def save(self):
        if len(self.lc.t) == 0:
            return
        os.makedirs(DATA, exist_ok=True)
        path = os.path.join(DATA, time.strftime("live_%Y%m%d_%H%M%S.csv"))
        df = pd.DataFrame(self.lc.x, columns=ls.ts.ANKLE)
        df.insert(0, "t_ms", np.round(self.lc.t * 1000 + self.lc.t0).astype(int))
        df.to_csv(path, index=False)
        self.status.configure(text=f"saved {len(df)} samples to {os.path.abspath(path)}")

    # ------------------------------------------------------------------ drawing
    def draw_picto(self, name, color):
        grid = PICTO[name]
        for r in range(8):
            for c in range(13):
                self.canvas.itemconfigure(self.cells[r][c], fill=color if grid[r][c] == "#" else OFF)

    def draw_trace(self, color):
        cv = self.trace
        cv.delete("all")
        w, h = int(cv["width"]), int(cv["height"])
        n = len(self.lc.t)
        if n < 2:
            return
        i0 = max(0, n - int(TRACE_S * 100))
        seg = self.lc.x[i0:, 4]
        tt = self.lc.t[i0:]
        span = max(np.abs(self.lc.x[max(0, n - 3000):, 4]).max(), 20.0)
        xs = (tt - (self.lc.now - TRACE_S)) / TRACE_S * w
        ys = h / 2 - seg / span * (h / 2 - 6)
        step = max(1, len(seg) // w)
        cv.create_line(*np.column_stack([xs[::step], ys[::step]]).ravel().tolist(), fill=color, width=1)
        # lag marker: counts to the left of this line are final
        x = (self.lc.done - (self.lc.now - TRACE_S)) / TRACE_S * w
        cv.create_line(x, 4, x, h - 4, fill=GREEN, width=2)

    def tick(self):
        got = False
        while True:
            try:
                msg = inbox.get_nowait()
            except queue.Empty:
                break
            if msg[0] == "status":
                self.status.configure(text=msg[1])
                continue
            _, seq, rows = msg
            if seq is not None and self.last_seq is not None and seq > self.last_seq + 1:
                self.missed += seq - self.last_seq - 1
            if seq is not None:
                self.last_seq = seq
            a = np.asarray(rows, float)
            if a.size:
                self.lc.add(a[:, 0], a[:, 1:7])
                self.rx_samples += len(a)
                got = True
            self.last_rx = time.time()
            if self.rx_t0 is None:
                self.rx_t0 = self.last_rx
        now = time.time()
        if got:
            self.lc.update()
        if self.lc.t0 is not None and len(self.lc.t) > 50:
            # extrapolate the clock between batches so a right step is credited on time and a stop is noticed
            t_now = self.lc.now + (min(now - self.last_rx, STALE_S) if self.last_rx else 0.0)
            if self.sw.update(self.lc.t, self.lc.x, t_now) and not self.use_net:
                self.flash_until = now + FLASH_S
        stale = self.last_rx == 0.0 or now - self.last_rx > STALE_S
        walking = now < self.flash_until
        color = DIM if stale else (GREEN if walking else BLUE)
        self.draw_picto("walk" if walking and not stale else "stand", color)
        shown = self.lc.cum if self.use_net else self.sw.count
        self.count_lbl.configure(text=str(int(shown + 1e-6)), fg=DIM if stale else (GREEN if walking else FG))
        if self.last_rx == 0.0:
            self.sub.configure(text="waiting for samples...")
        else:
            rate = self.rx_samples / max(now - self.rx_t0, 1e-6) if self.rx_t0 else 0
            txt = (f"left {self.sw.left_steps} + right {self.sw.right_steps}   "
                   f"network: {self.lc.cum:.0f} (+-{self.lc.spread:.1f})   "
                   f"{self.lc.now:.0f} s   {rate:.0f} samples/s")
            if self.use_net:
                txt = (f"network {self.lc.cum:.1f} (+-{self.lc.spread:.1f} between nets)   "
                       f"swing count: {self.sw.count}   {self.lc.now:.0f} s   {rate:.0f} samples/s")
            if self.missed or self.lc.gaps:
                txt += f"   lost batches: {self.missed}, gaps: {self.lc.gaps}"
            if stale:
                txt = f"no samples for {now - self.last_rx:.0f} s.  last count {shown:.0f}"
            self.sub.configure(text=txt)
        self.draw_trace(DIM if stale else BLUE)
        self.root.after(UPDATE_MS if not got else 60, self.tick)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("broker", nargs="?", default=BROKER)
    ap.add_argument("--replay", help="feed this ankle CSV instead of listening to the board")
    ap.add_argument("--speed", type=float, default=1.0, help="replay speed (1 = real time)")
    ap.add_argument("--net", action="store_true", help="count with the neural network instead of the swing counter")
    ap.add_argument("--keep-last-right", action="store_true",
                    help="when you stop, keep the right step that follows your last left step (default: take it back)")
    args = ap.parse_args()
    global USE_NET, KEEP_LAST_RIGHT
    USE_NET = args.net
    KEEP_LAST_RIGHT = args.keep_last_right
    root = tk.Tk()
    app = App(root)
    client = None
    if args.replay:
        start_replay(args.replay, args.speed)
    else:
        try:
            client = start_mqtt(args.broker)
        except ImportError:
            app.status.configure(text="paho-mqtt is not installed:  pip install paho-mqtt")
    try:
        root.mainloop()
    finally:
        if client:
            client.loop_stop()
            client.disconnect()


if __name__ == "__main__":
    main()
