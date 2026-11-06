"""Streaming version of the neural-network step counter (ankle sensor only).

LiveCounter takes raw ankle samples as they arrive and keeps a running step count from the saved
PyTorch ensemble (pc/step_model.pt). The network looks about 0.6 s ahead of every output, so a step
rate is only final once it is 0.6 s old: the count lags the leg by about 0.6 s.

    python live_steps.py --check            # stream every recording in data/ in 0.2 s chunks and
                                            # compare with the offline count (no board needed)
    python live_steps.py --check data/S21_walk.csv

How it works: every `update` the last WINDOW_S seconds of samples are resampled to 100 Hz, run
through all 5 networks, and the rates that are now older than LAG_S are added to the count once.
"""
import argparse
import os
import time

import numpy as np
import pandas as pd
import torch
from scipy.signal import butter, find_peaks, sosfilt

import train_steps as ts

WINDOW_S = 15.0    # seconds of history given to the network each update
LAG_S = 0.65       # output is final this long after it is produced (receptive field is 0.6 s)
GAP_WARN_S = 0.3   # a hole in the sample stream longer than this is counted as a gap

here = os.path.dirname(os.path.abspath(__file__))


class LiveCounter:
    def __init__(self, model_path=ts.MODEL_PATH):
        ck = torch.load(model_path, weights_only=False)
        if ck["channels"] != ts.ANKLE:
            raise SystemExit(f"{model_path} is not the ankle-only model; run train_steps.py --configs ankle --main ankle")
        self.mean, self.std = ck["mean"], ck["std"]
        self.nets = []
        for st in ck["states"]:
            m = ts.StepNet(len(ts.ANKLE), ck["width"])
            m.load_state_dict(st)
            m.eval()
            self.nets.append(m)
        self.reset()

    def reset(self):
        self.t = np.empty(0)             # sample times, s
        self.x = np.empty((0, 6))        # ax ay az gx gy gz
        self.t0 = None                   # time of first sample, ms
        self.done = 0.0                  # everything before this time (s) is already counted
        self.cum = 0.0                   # steps counted so far (final, lags by LAG_S)
        self.net_cum = np.zeros(len(self.nets))
        self.gaps = 0
        self.step_times = []             # times at which the count crossed a whole number

    def add(self, t_ms, x):
        """Append samples: t_ms (n,), x (n, 6) in g and deg/s. Out-of-order samples are sorted."""
        t_ms = np.asarray(t_ms, float)
        x = np.asarray(x, float).reshape(-1, 6)
        if len(t_ms) == 0:
            return
        if self.t0 is None:
            self.t0 = t_ms.min()
        t = (t_ms - self.t0) / 1000.0
        if len(self.t) and t.min() < self.t[-1]:
            keep = t > self.t[-1]                      # drop duplicates / late repeats
            t, x = t[keep], x[keep]
        if len(self.t) and len(t) and t[0] - self.t[-1] > GAP_WARN_S:
            self.gaps += 1
        order = np.argsort(t)
        self.t = np.r_[self.t, t[order]]
        self.x = np.vstack([self.x, x[order]])

    @property
    def now(self):
        return self.t[-1] if len(self.t) else 0.0

    def update(self):
        """Run the networks on recent history; return the number of new whole steps counted."""
        if len(self.t) < 2 or self.now - self.done < 0.25 + LAG_S:
            return 0
        lo = max(self.t[0], self.now - WINDOW_S)
        grid = np.arange(lo, self.now, 1 / ts.FS)
        raw = np.column_stack([np.interp(grid, self.t, self.x[:, i]) for i in range(6)])
        feats = ((raw - self.mean) / self.std).T.astype(np.float32)
        rates = np.array([ts.predict(m, feats) for m in self.nets])      # (n_nets, T)
        # final part: older than LAG_S and, unless the window starts at the recording start,
        # also past the zero-padded left edge (0.6 s)
        edge = lo + (0.0 if lo <= self.t[0] + 1e-9 else LAG_S)
        sel = (grid >= max(self.done, edge)) & (grid < self.now - LAG_S)
        if not sel.any():
            return 0
        before = self.cum
        self.net_cum += rates[:, sel].sum(1) / ts.FS
        self.cum = float(self.net_cum.mean())
        self.done = grid[sel][-1] + 1 / ts.FS
        n_before, n_after = int(before), int(self.cum)
        new = max(0, n_after - n_before)
        for k in range(new):
            # spread the new steps across the newly final stretch
            self.step_times.append(grid[sel][0] + (k + 1) / (new + 1) * (grid[sel][-1] - grid[sel][0]))
        return new

    def flush(self):
        """End of recording: count the last LAG_S too (it is then also right-edge padded, as in training)."""
        if len(self.t) < 2:
            return
        lo = max(self.t[0], self.now - WINDOW_S)
        grid = np.arange(lo, self.now, 1 / ts.FS)
        raw = np.column_stack([np.interp(grid, self.t, self.x[:, i]) for i in range(6)])
        feats = ((raw - self.mean) / self.std).T.astype(np.float32)
        rates = np.array([ts.predict(m, feats) for m in self.nets])
        sel = grid >= max(self.done, lo + (0.0 if lo <= self.t[0] + 1e-9 else LAG_S))
        self.net_cum += rates[:, sel].sum(1) / ts.FS
        self.cum = float(self.net_cum.mean())
        self.done = self.now

    @property
    def spread(self):
        """Standard deviation of the count between the 5 networks."""
        return float(self.net_cum.std(ddof=1)) if len(self.net_cum) > 1 else 0.0


class SwingTracker:
    """Counts BOTH feet from the left-ankle gyro, live, without the neural network.

    The strap only sees the left leg. Each left swing (a peak in the gyro y axis) is a left step. A
    right step follows every left step, half a stride later, unless the walker stops: if no further
    left swing comes within STOP_FACTOR strides, the right step that was credited after the last
    swing is taken back (retract_at_stop). Stride time is the median of the last few swing intervals.
    Latency is about 0.2 s, much less than the network's 0.65 s.
    """
    AXIS = 1               # gyro y: the leg-swing axis for this strap (see README)
    MIN_HEIGHT = 90.0      # deg/s: a swing is at least this fast (walking swings are 130-430)
    MIN_PROM = 60.0
    MIN_PERIOD_S = 0.55    # the same leg cannot swing faster than this
    WINDOW_S = 8.0
    DEFAULT_STRIDE_S = 1.2
    STOP_FACTOR = 1.7      # no swing for this many strides = stopped
    ISOLATED_S = 3.0       # a lone swing with no other swing within this long is not a step (foot shuffle,
                           # picking the board up); it is counted at first, then taken back

    def __init__(self, retract_at_stop=True):
        self.retract_at_stop = retract_at_stop
        self.sos = butter(4, 4.0 / (ts.FS / 2), output="sos")
        self.reset()

    def reset(self):
        self.swings = []          # times of left swings, s
        self.heights = []
        self.right_done = []      # right step credited for swing k?
        self.right_live = []      # ...and not taken back since
        self.dropped = []         # swing k judged a lone shuffle
        self.left_steps = 0
        self.right_steps = 0
        self.events = []          # (time, "L" | "R" | "-R") in order
        self.stopped_after = -1   # index of the swing whose right step was retracted

    @property
    def count(self):
        return self.left_steps + self.right_steps

    def stride(self):
        d = np.diff(self.swings[-5:])
        d = d[d < 2.5]
        return float(np.clip(np.median(d), 0.8, 1.8)) if len(d) else self.DEFAULT_STRIDE_S

    def update(self, t, x, now):
        """t (s), x (n, 6) are all samples so far; now is the newest sample time. -> new events"""
        n0 = len(self.events)
        if len(t) > 50:
            grid = np.arange(max(t[0], now - self.WINDOW_S), now, 1 / ts.FS)
            gy = np.interp(grid, t, x[:, 3 + self.AXIS])
            f = sosfilt(self.sos, gy)
            thr = self.MIN_HEIGHT
            if self.heights:
                thr = max(thr, 0.4 * float(np.median(self.heights[-6:])))
            pk, _ = find_peaks(f, height=thr, prominence=self.MIN_PROM, distance=int(self.MIN_PERIOD_S * ts.FS))
            for i in pk:
                tp = grid[i]
                if tp < grid[0] + 0.5:                    # filter start-up
                    continue
                if self.swings and tp - self.swings[-1] < self.MIN_PERIOD_S:
                    continue
                if self.swings and tp <= self.swings[-1]:
                    continue
                self.swings.append(float(tp))
                self.heights.append(float(f[i]))
                self.right_done.append(False)
                self.right_live.append(False)
                self.dropped.append(False)
                self.left_steps += 1
                self.events.append((float(tp), "L"))
        if self.swings:
            T = self.stride()
            for k, ts_ in enumerate(self.swings):
                if not self.right_done[k] and now >= ts_ + T / 2:
                    self.right_done[k] = self.right_live[k] = True
                    self.right_steps += 1
                    self.events.append((ts_ + T / 2, "R"))
            # a lone swing: no neighbour within ISOLATED_S on either side -> not a step
            for k, ts_ in enumerate(self.swings):
                if self.dropped[k] or now - ts_ <= self.ISOLATED_S:
                    continue
                prev_gap = ts_ - self.swings[k - 1] if k > 0 else np.inf
                next_gap = self.swings[k + 1] - ts_ if k + 1 < len(self.swings) else np.inf
                if prev_gap > self.ISOLATED_S and next_gap > self.ISOLATED_S:
                    self.dropped[k] = True
                    self.left_steps -= 1
                    if self.right_live[k]:
                        self.right_live[k] = False
                        self.right_steps -= 1
                    self.events.append((now, "-L"))
            # a pause (or the end of the walk): the right step after the last left step is taken back
            k = len(self.swings) - 1
            if (self.retract_at_stop and self.right_live[k] and not self.dropped[k]
                    and now - self.swings[k] > max(self.STOP_FACTOR * T, 1.6)):
                self.right_live[k] = False
                self.right_steps -= 1
                self.events.append((now, "-R"))
        return self.events[n0:]


def check_swings(paths=None):
    """Stream recordings through SwingTracker (walk part only) for both stop rules."""
    import prepare_steps as ps
    import count_steps as cs
    truth = {"S18": 47, "S19": 51.5, "S21": 76, "S22": 79, "S23": 57}
    print("walk part only (between start and end stomps), 0.2 s chunks; 'x2' = twice the left swings")
    print(f"{'rec':5s} {'truth':>6s} {'x2':>5s} {'live, keep last R':>18s} {'live, retract at stop':>22s}")
    err = {"keep": [], "retract": []}
    for sname, tr in truth.items():
        t, x = cs.load_ankle_csv(os.path.join(here, "..", "data", f"{sname}_walk.csv"))
        st, en = ps.find_stomps(t, x[:, :3])
        w0 = st[-1] + 0.5
        w1 = t[-1] if sname == "S19" else en[0] - 0.5
        m = (t >= w0) & (t <= w1)
        tt, xx = t[m], x[m]
        res = {}
        for name, retract in (("keep", False), ("retract", True)):
            sw = SwingTracker(retract_at_stop=retract)
            for i in range(0, len(tt), 20):
                sw.update(tt[:i + 20], xx[:i + 20], tt[min(i + 19, len(tt) - 1)])
            # let it run out as if the stream went on (stopped) for 3 s
            sw.update(tt, xx, tt[-1] + 3.0)
            res[name] = sw.count
            err[name].append(abs(sw.count - tr) / tr * 100)
        print(f"{sname:5s} {tr:6g} {2 * len(sw.swings):5d} {res['keep']:18d} {res['retract']:22d}")
    print(f"mean abs error: keep {np.mean(err['keep']):.1f}%  retract {np.mean(err['retract']):.1f}%")


def check(paths):
    import count_steps as cs
    lc = LiveCounter()
    print(f"{'recording':14s} {'offline':>8s} {'streamed':>9s} {'diff':>6s}  chunks of 0.2 s, update after each")
    worst = 0.0
    for p in paths:
        df = pd.read_csv(p)
        t_ms = df["t_ms"].to_numpy(float)
        x = df[ts.ANKLE].to_numpy(float)
        _, rates, _ = cs.count(p)
        off = rates.sum(1).mean() / ts.FS
        lc.reset()
        t0 = time.time()
        step = 20
        for i in range(0, len(t_ms), step):
            lc.add(t_ms[i:i + step], x[i:i + step])
            lc.update()
        lc.flush()
        worst = max(worst, abs(lc.cum - off))
        print(f"{os.path.basename(p):14s} {off:8.1f} {lc.cum:9.1f} {lc.cum - off:+6.1f}   "
              f"({time.time() - t0:.1f} s compute for {t_ms[-1] / 1000 - t_ms[0] / 1000:.0f} s of data)")
    print(f"largest difference: {worst:.2f} steps")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--check-swings", action="store_true", help="test the swing counter on data/")
    ap.add_argument("--check", nargs="*", metavar="CSV", help="stream recordings and compare with the offline count")
    a = ap.parse_args()
    if a.check_swings:
        check_swings()
    elif a.check is not None:
        data = os.path.join(here, "..", "data")
        paths = a.check or [os.path.join(data, f"{s}_walk.csv") for s in ("S18", "S19", "S21", "S22", "S23")]
        check(paths)
    else:
        ap.print_help()
