"""Turn raw step-counter recordings into training data: sync ankle + phone, find the walk, label every step.

Usage: python prepare_steps.py [session ...]      (default: every session in data/steps_log.csv)

Each recording is one ankle CSV logged on the board (data/S17_walk.csv) plus the phone files from
phone_receiver.py started with the SAME session name (python phone_receiver.py S17 ->
data/phone/S17_*.csv). The protocol for every recording:

    3 hard stomps, stand still ~2 s, walk (30 s), stand still ~2 s, 3 hard stomps

data/steps_log.csv has one row per recording (a template is written if it does not exist):

    session,activity,true_steps,offset_s,walk_start_s,walk_end_s,stomp_times_s,notes
    S17,walk_slow,42,,,,,tally counter

true_steps is your hand count of the walking steps (both feet, NOT the stomps: subtract them
from a count that included them). stomp_times_s lists stomps you did in the middle of the
recording (ankle clock, seconds, separated by ;) so the leg lifts around them are not counted as
steps. The override columns are normally left blank; fill them only if the plot shows the automatic
sync (offset_s = ankle time minus phone time) or the walk boundaries are wrong.

What it does per session:
  1. Puts the ankle (board clock) and phone (epoch clock) streams on one 100 Hz timeline.
     The offset comes from cross-correlating acceleration spikes; the stomps at both ends
     make the match unique, and matching the start and end separately also corrects drift.
  2. Finds the three start stomps and three end stomps; the walk is everything between them.
  3. Labels steps: each swing of the ankle leg (ankle gyro) is one step of that foot, each
     swing of the phone leg (phone gyro, front pocket) is one step of the other foot. Without
     phone data the phone-leg steps are put halfway between ankle-leg steps.
  4. Checks the label count against true_steps and against the phone's own pedometer.

Writes data/steps/<session>.csv (100 Hz: all sensor channels, in_walk, step, foot),
data/steps/<session>.png (check this!) and data/steps/summary.csv.
"""
import glob
import os
import sys
from itertools import combinations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.ndimage import median_filter  # noqa: E402
from scipy.signal import butter, filtfilt, find_peaks  # noqa: E402

FS = 100
G = 9.80665
ANKLE_COLS = ["ax_g", "ay_g", "az_g", "gx_dps", "gy_dps", "gz_dps"]
# phone sensor -> (value columns to keep, scale to the ankle's units)
PHONE_SENSORS = {
    "totalacceleration": (["x", "y", "z"], 1 / G),   # m/s^2 -> g, includes gravity like the ankle
    "accelerometer": (["x", "y", "z"], 1 / G),       # user acceleration (gravity removed)
    "gravity": (["x", "y", "z"], 1 / G),
    "gyroscope": (["x", "y", "z"], 180 / np.pi),     # rad/s -> dps
    "orientation": (["qw", "qx", "qy", "qz", "pitch", "roll"], 1.0),  # yaw left out: wraps at +-pi
    "magnetometer": (["x", "y", "z"], 1.0),          # uT
    "barometer": (["pressure", "relativeAltitude"], 1.0),
}
STOMPS = 3
STOMP_SEARCH_S = 12        # stomps must be within this many seconds of each end
MAX_STOMP_GAP_S = 2.5      # consecutive stomps in a triple are at most this far apart
STOMP_LIFT_S = (0.8, 0.4)   # a stomp's leg lift comes this long before / after its impact spike
ISOLATED_S = 3.0           # a lone swing with no same-foot swing this close is not a step
MIN_PHONE_COVERAGE = 0.8   # use the phone gyro if it covers at least this much of the walk
MAX_DRIFT_S = 0.03         # start-vs-end offset change allowed (real clocks differ by ~ms over a minute)
MAX_SYNC_LAG_S = 600       # coarse search range for the ankle/phone offset
MIN_STEP_PERIOD_S = 0.6    # same foot can't swing faster than this (fast walking ~1.0 s)
ANKLE_FOOT, PHONE_FOOT = "L", "R"   # strap on the left ankle, phone in the right pocket
LOG_COLS = ["session", "activity", "true_steps", "offset_s", "walk_start_s", "walk_end_s",
            "stomp_times_s", "notes"]

here = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(here, "..", "data")
OUT = os.path.join(DATA, "steps")
LOG = os.path.join(DATA, "steps_log.csv")


def lowpass(x, hz, order=4):
    b, a = butter(order, hz / (FS / 2))
    return filtfilt(b, a, x, axis=0)


def resample(t, values, grid):
    """Linear interpolation of each column onto grid; NaN outside the recorded range."""
    out = np.column_stack([np.interp(grid, t, v) for v in np.atleast_2d(values.T)])
    out[(grid < t[0]) | (grid > t[-1])] = np.nan
    return out


def spikes(acc_g):
    """Acceleration-magnitude spikes above the local baseline, squared so stomps dominate walking."""
    mag = np.linalg.norm(acc_g, axis=1)
    hp = np.clip(mag - median_filter(mag, size=FS + 1, mode="nearest"), 0, None)
    return hp ** 2


def xcorr_lag(a, b, max_lag):
    """Lag (samples) that best aligns b to a: a[i] ~ b[i - lag]. Uses FFT cross-correlation."""
    a = np.nan_to_num(a - np.nanmean(a))
    b = np.nan_to_num(b - np.nanmean(b))
    n = 1 << int(np.ceil(np.log2(len(a) + len(b))))
    c = np.fft.irfft(np.fft.rfft(a, n) * np.conj(np.fft.rfft(b, n)), n)
    lags = np.r_[np.arange(0, len(a)), np.arange(-len(b) + 1, 0)]
    c = np.r_[c[:len(a)], c[n - len(b) + 1:]]
    ok = np.abs(lags) <= max_lag
    i = np.argmax(np.where(ok, c, -np.inf))
    return int(lags[i]), float(c[i] / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


# ---------------------------------------------------------------- loading

def load_ankle(session):
    paths = [p for p in glob.glob(os.path.join(DATA, "*.csv"))
             if os.path.basename(p).upper().startswith(session.upper() + "_")]
    if not paths:
        raise FileNotFoundError(f"no ankle file data/{session}_*.csv")
    df = pd.read_csv(paths[0])
    t = (df["t_ms"].to_numpy() - df["t_ms"].iloc[0]) / 1000.0
    grid = np.arange(0, t[-1], 1 / FS)
    x = resample(t, df[ANKLE_COLS].to_numpy(float), grid)
    gaps = np.diff(t)
    if gaps.max() > 0.1:
        print(f"  WARNING ankle file has a {gaps.max():.2f} s gap (dropped samples)")
    return os.path.basename(paths[0]), grid, x


def load_phone(session):
    """{sensor: (t_s, values)} with t_s in seconds since the first phone sample; plus t0_ns."""
    raw = {}
    for name in list(PHONE_SENSORS) + ["pedometer"]:
        p = os.path.join(DATA, "phone", f"{session}_{name}.csv")
        if os.path.exists(p) and os.path.getsize(p) > 0:
            # Sensor Logger sometimes delivers a ~2 s batch late; sorting by time restores it
            # exactly (verified: 100 samples/s, no duplicates or gaps afterwards).
            df = pd.read_csv(p).dropna(subset=["t_ns"]).drop_duplicates("t_ns").sort_values("t_ns")
            if len(df) > 1:
                raw[name] = df
    if not raw:
        return None, None
    t0_ns = min(int(df["t_ns"].iloc[0]) for n, df in raw.items() if n != "pedometer")
    phone = {}
    for name, df in raw.items():
        t = (df["t_ns"].to_numpy(np.int64) - t0_ns) / 1e9
        if name == "pedometer":
            phone[name] = (t, df["steps"].to_numpy(float))
            continue
        cols, scale = PHONE_SENSORS[name]
        cols = [c for c in cols if c in df]
        phone[name] = (t, df[cols].to_numpy(float) * scale, cols)
    return phone, t0_ns


# ---------------------------------------------------------------- sync

def sync_signal(acc_g, gyro_dps):
    """What the two sensors have in common: stomp spikes (compressed so one hard stomp can't
    dominate and mimic the whole triple) plus how much the leg is rotating."""
    z = lambda x: (x - np.nanmean(x)) / (np.nanstd(x) + 1e-9)
    motion = lowpass(np.nan_to_num(np.linalg.norm(gyro_dps, axis=1)), 4)
    return z(spikes(acc_g) ** 0.25) + z(motion)


def sync(t_ank, ank_acc, ank_gyr, phone, manual_offset):
    """Return (scale, offset) with t_ankle = scale * t_phone + offset, and a note for the summary."""
    if not np.isnan(manual_offset):
        return 1.0, manual_offset, "manual"
    key = "totalacceleration" if "totalacceleration" in phone else "accelerometer"
    tp, vp, _ = phone[key]
    gp = np.arange(0, tp[-1], 1 / FS)
    ph_acc = resample(tp, vp, gp)
    if key == "accelerometer":  # user acceleration has no gravity; magnitude spikes still match
        ph_acc[:, 2] += 1.0
    if "gyroscope" in phone:
        tg, vg, _ = phone["gyroscope"]
        a, b = sync_signal(ank_acc, ank_gyr), sync_signal(ph_acc, resample(tg, vg, gp))
    else:
        a, b = spikes(ank_acc) ** 0.25, spikes(ph_acc) ** 0.25

    # Coarse: whole recording. The stomp pairs at both ends make this peak unique.
    lag, score = xcorr_lag(a, b, MAX_SYNC_LAG_S * FS)
    # Fine: snap to the stomp spikes alone, refitting start and end separately (+-0.75 s)
    # to measure clock drift.
    a, b = spikes(ank_acc) ** 0.25, spikes(ph_acc) ** 0.25
    fits = []
    w = STOMP_SEARCH_S * FS
    for seg in (slice(0, w), slice(max(len(a) - w, 0), len(a))):
        idx = np.arange(len(a))[seg]
        src = idx - lag
        keep = (src >= 0) & (src < len(b))
        if keep.sum() < FS:
            continue
        bb = np.zeros(len(idx))
        bb[keep] = b[src[keep]]
        pad = 3 * FS // 4
        best = max(range(-pad, pad + 1),
                   key=lambda d: np.dot(a[idx], np.roll(bb, d)))
        fits.append((idx.mean() / FS, (lag + best) / FS))   # (ankle time, offset there)
    if len(fits) == 2 and abs(fits[1][1] - fits[0][1]) < MAX_DRIFT_S:
        (ta, oa), (tb, ob) = fits
        drift = (ob - oa) / (tb - ta)          # offset change per second
        scale = 1 + drift
        offset = oa - drift * ta
        note = f"xcorr {score:.2f}, drift {drift * 1e6:+.0f} ppm"
    else:
        scale, offset = 1.0, (fits[0][1] if fits else lag / FS)
        note = f"xcorr {score:.2f}, start only (end fit disagreed)"
    if score < 0.2:
        print(f"  WARNING weak sync match ({score:.2f}); check the plot, set offset_s if needed")
    return scale, offset, note


# ---------------------------------------------------------------- stomps and steps

def find_stomps(t, acc):
    """Times of the stomp triple near each end: the 3 evenly spaced spikes whose weakest is strongest."""
    s = spikes(acc)
    out = []
    for lo, hi in ((0, STOMP_SEARCH_S), (t[-1] - STOMP_SEARCH_S, t[-1])):
        m = (t >= lo) & (t <= hi)
        pk, _ = find_peaks(np.where(m, s, 0), distance=int(0.3 * FS))
        top = np.sort(pk[np.argsort(s[pk])[::-1][:8]])   # the 8 biggest spikes, in time order
        best, best_h = [], 0.0
        for trio in combinations(top, STOMPS):
            gaps = np.diff(trio)
            if gaps.max() > MAX_STOMP_GAP_S * FS or gaps.min() < 0.4 * FS:
                continue
            h = s[list(trio)].min()
            if h > best_h:
                best, best_h = list(trio), h
        out.append(t[np.array(best, int)])
    return out


def swings(t, gyro, walk):
    """One peak per swing of the leg the gyro is on. Picks the axis/sign with the strongest swing."""
    g = lowpass(np.nan_to_num(gyro), 4)
    gw = g[walk]
    best = None
    for ax in range(3):
        for sign in (1, -1):
            hi = np.percentile(sign * gw[:, ax], 98)
            if best is None or hi > best[0]:
                best = (hi, ax, sign)
    hi, ax, sign = best
    sig = sign * g[:, ax]
    rest = np.percentile(np.abs(gw[:, ax]), 20)
    height = rest + 0.35 * (hi - rest)
    pk, _ = find_peaks(np.where(walk, sig, -np.inf), height=height,
                       distance=int(MIN_STEP_PERIOD_S * FS), prominence=0.3 * (hi - rest))
    return pk, sig, ax


def drop_isolated(idx, foot):
    """Remove swings with no other swing of the same foot within ISOLATED_S on either side
    (a foot shuffle while standing, not a step)."""
    if len(idx) < 3:
        return idx
    gap_before = np.r_[np.inf, np.diff(idx)]
    gap_after = np.r_[np.diff(idx), np.inf]
    keep = (gap_before <= ISOLATED_S * FS) | (gap_after <= ISOLATED_S * FS)
    if (~keep).any():
        print(f"  dropped {(~keep).sum()} isolated {foot} swing(s) at {np.round(idx[~keep] / FS, 1)} s")
    return idx[keep]


def label_steps(t, ank, phone_gyro, walk):
    ank_steps, a_sig, a_ax = swings(t, ank[:, 3:6], walk)
    ank_steps = drop_isolated(ank_steps, ANKLE_FOOT)
    fin = np.isfinite(phone_gyro).all(axis=1) if phone_gyro is not None else None
    if fin is not None and fin[walk].mean() >= MIN_PHONE_COVERAGE:
        ph_steps, p_sig, p_ax = swings(t, phone_gyro, walk & fin)
        ph_steps = drop_isolated(ph_steps, PHONE_FOOT)
        how = "phone gyro"
        tail = walk & ~fin       # the phone stopped streaming before the walk ended
        if tail.any():
            gaps = np.diff(ank_steps)
            mid = (ank_steps[:-1] + gaps // 2)[gaps < 2 * FS]
            ph_steps = np.sort(np.r_[ph_steps, mid[tail[mid]]])
            how = f"phone gyro, midpoints for the last {tail.sum() / FS:.1f} s"
    else:
        # Fallback: each phone-leg step falls halfway between two ankle-leg steps (skip across stops).
        gaps = np.diff(ank_steps)
        ph_steps = (ank_steps[:-1] + gaps // 2)[gaps < 2 * FS]
        p_sig, p_ax, how = None, None, "midpoints (no phone)"
    return ank_steps, ph_steps, (a_sig, a_ax, p_sig, p_ax), how


# ---------------------------------------------------------------- main

def process(row):
    session = row["session"]
    print(f"\n{session} ({row['activity']})")
    ank_name, t, ank = load_ankle(session)
    phone, t0_ns = load_phone(session)

    channels = {c: ank[:, i] for i, c in enumerate(ANKLE_COLS)}
    phone_gyro, ped, sync_note = None, None, "no phone data"
    if phone is not None and ("totalacceleration" in phone or "accelerometer" in phone):
        scale, offset, sync_note = sync(t, ank[:, :3], ank[:, 3:6], phone, row["offset_s"])
        print(f"  sync: ankle t = {scale:.6f} * phone t {offset:+.3f} s  ({sync_note})")
        for name, (tp, vp, cols) in ((n, v) for n, v in phone.items() if n != "pedometer"):
            vals = resample(scale * tp + offset, vp, t)
            for i, c in enumerate(cols):
                channels[f"ph_{name}_{c}"] = vals[:, i]
        if "gyroscope" in phone:
            phone_gyro = np.column_stack([channels[f"ph_gyroscope_{c}"] for c in "xyz"])
        if "pedometer" in phone:
            tp, steps = phone["pedometer"]
            ped = (scale * tp + offset, steps)
    else:
        print("  no phone files found; ankle only")

    starts, ends = find_stomps(t, ank[:, :3])
    w0 = row["walk_start_s"] if not np.isnan(row["walk_start_s"]) else \
        (starts[-1] + 0.5 if len(starts) else t[0])
    w1 = row["walk_end_s"] if not np.isnan(row["walk_end_s"]) else \
        (ends[0] - 0.5 if len(ends) else t[-1])
    walk = (t >= w0) & (t <= w1)
    print(f"  stomps at {np.round(starts, 2)} and {np.round(ends, 2)} s -> walk {w0:.1f}-{w1:.1f} s "
          f"({w1 - w0:.1f} s)")

    ank_steps, ph_steps, sigs, how = label_steps(t, ank, phone_gyro, walk)
    stomps = [float(x) for x in str(row["stomp_times_s"]).replace(",", ";").split(";")
              if x.strip() and x.strip() != "nan"]
    if stomps:
        def not_stomp(idx):
            keep = np.ones(len(idx), bool)
            for st in stomps:
                keep &= ~((t[idx] >= st - STOMP_LIFT_S[0]) & (t[idx] <= st + STOMP_LIFT_S[1]))
            return idx[keep]
        n0 = len(ank_steps) + len(ph_steps)
        ank_steps, ph_steps = not_stomp(ank_steps), not_stomp(ph_steps)
        print(f"  removed {n0 - len(ank_steps) - len(ph_steps)} swings that are leg lifts for the "
              f"{len(stomps)} marked stomps")
    labeled = len(ank_steps) + len(ph_steps)
    step = np.zeros(len(t), int)
    foot = np.full(len(t), "", object)
    step[ank_steps], foot[ank_steps] = 1, ANKLE_FOOT
    step[ph_steps], foot[ph_steps] = 1, PHONE_FOOT

    after = ~walk & (t > w1)
    if after.any():
        more = len(swings(t, ank[:, 3:6], after)[0])
        if more >= 6:
            print(f"  WARNING {more} more leg swings after the detected walk end ({w1:.1f} s): the end "
                  "stomps are probably missing from the ankle file (board stopped early?); "
                  "set walk_end_s in steps_log.csv")

    ped_steps = np.nan
    if ped is not None:
        tp, steps = ped
        before = steps[tp <= w0]
        upto = steps[tp <= w1 + 2.0]   # pedometer reports late; allow 2 s
        if len(upto):
            ped_steps = upto[-1] - (before[-1] if len(before) else steps[0])

    true = row["true_steps"]
    diff = labeled - true if not np.isnan(true) else np.nan
    print(f"  labels: {len(ank_steps)} {ANKLE_FOOT} (ankle) + {len(ph_steps)} {PHONE_FOOT} "
          f"(phone, {how}) = {labeled}   "
          f"true {true:g}   diff {diff:+g}   phone pedometer {ped_steps:g}")
    if not np.isnan(diff) and abs(diff) > max(2, 0.05 * true):
        print("  WARNING labels disagree with your count; look at the plot before training on this")

    os.makedirs(OUT, exist_ok=True)
    df = pd.DataFrame({"t_s": np.round(t, 3), **channels,
                       "in_walk": walk.astype(int), "step": step, "foot": foot})
    df.insert(1, "session", session)
    df.insert(2, "activity", row["activity"])
    df.to_csv(os.path.join(OUT, f"{session}.csv"), index=False, float_format="%.5g")
    plot(session, row["activity"], t, ank, channels, walk, starts, ends, ank_steps, ph_steps, sigs,
         ped, true, w0)

    return {"session": session, "activity": row["activity"], "ankle_file": ank_name,
            "walk_s": round(w1 - w0, 1), "true_steps": true, "labeled_steps": labeled,
            "ankle_steps": len(ank_steps), "phone_steps": len(ph_steps), "diff": diff,
            "cadence_spm": round(labeled / (w1 - w0) * 60, 1), "phone_pedometer": ped_steps,
            "phone_steps_from": how, "sync": sync_note}


def plot(session, activity, t, ank, ch, walk, starts, ends, ank_steps, ph_steps, sigs, ped, true, w0):
    a_sig, a_ax, p_sig, p_ax = sigs
    fig, axs = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
    a = axs[0]
    a.plot(t, np.linalg.norm(ank[:, :3], axis=1), lw=0.7, label="ankle |a| (g)")
    if "ph_totalacceleration_x" in ch:
        pm = np.linalg.norm(np.column_stack([ch[f"ph_totalacceleration_{c}"] for c in "xyz"]), axis=1)
        a.plot(t, pm, lw=0.7, label="phone |a| (g), synced")
    for s in np.r_[starts, ends]:
        a.axvline(s, color="k", ls=":", lw=0.8)
    a.set_title(f"{session} {activity}: sync check (stomp spikes should line up; dotted = stomps)")
    a.legend(loc="upper right")

    axs[1].plot(t, a_sig, lw=0.7)
    axs[1].plot(t[ank_steps], a_sig[ank_steps], "rv", label=f"{ANKLE_FOOT} steps ({len(ank_steps)})")
    axs[1].set_title(f"ankle gyro axis {'xyz'[a_ax]} ({ANKLE_FOOT} leg swing)")
    axs[1].legend(loc="upper right")

    if p_sig is not None:
        axs[2].plot(t, p_sig, lw=0.7, color="tab:orange")
        axs[2].plot(t[ph_steps], p_sig[ph_steps], "bv", label=f"{PHONE_FOOT} steps ({len(ph_steps)})")
        axs[2].set_title(f"phone gyro axis {'xyz'[p_ax]} ({PHONE_FOOT} leg swing)")
    else:
        axs[2].plot(t[ph_steps], np.zeros(len(ph_steps)), "bv",
                    label=f"{PHONE_FOOT} steps ({len(ph_steps)})")
        axs[2].set_title(f"{PHONE_FOOT} steps estimated as midpoints (no phone gyro)")
    axs[2].legend(loc="upper right")

    events = np.sort(np.r_[ank_steps, ph_steps])
    cum = np.zeros(len(t))
    cum[events] = 1
    axs[3].plot(t, np.cumsum(cum), label="labeled steps")
    if ped is not None:
        tp, steps = ped
        b = steps[tp <= w0]
        axs[3].step(tp, steps - (b[-1] if len(b) else steps[0]), where="post",
                    label="phone pedometer")
    if not np.isnan(true):
        axs[3].axhline(true, color="k", ls="--", lw=0.8, label=f"your count ({true:g})")
    axs[3].set_ylim(bottom=0)
    axs[3].set_xlim(t[0] - 1, t[-1] + 1)
    axs[3].set_xlabel("time (s, ankle clock)")
    axs[3].set_title("cumulative steps")
    axs[3].legend(loc="upper left")
    for a in axs:
        a.axvspan(t[walk][0], t[walk][-1], color="g", alpha=0.06)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, f"{session}.png"), dpi=110)
    plt.close(fig)


def main():
    if not os.path.exists(LOG):
        pd.DataFrame(columns=LOG_COLS).to_csv(LOG, index=False)
        print(f"Wrote an empty {LOG}; add one row per recording and run again.")
        return
    log = pd.read_csv(LOG, dtype={"session": str, "activity": str, "notes": str,
                                  "stomp_times_s": str})
    for c in LOG_COLS:
        if c not in log:
            log[c] = np.nan
    for c in ["true_steps", "offset_s", "walk_start_s", "walk_end_s"]:
        log[c] = pd.to_numeric(log[c], errors="coerce")
    wanted = {s.upper() for s in sys.argv[1:]}
    rows = [r for _, r in log.iterrows() if not wanted or r["session"].upper() in wanted]
    if not rows:
        print("No matching sessions in", LOG)
        return

    results = []
    for r in rows:
        try:
            results.append(process(r))
        except FileNotFoundError as e:
            print(f"\n{r['session']}: SKIPPED, {e}")
    if not results:
        return
    summary = pd.DataFrame(results)
    summary.to_csv(os.path.join(OUT, "summary.csv"), index=False)
    print("\n" + summary[["session", "activity", "walk_s", "true_steps", "labeled_steps", "diff",
                          "cadence_spm", "phone_pedometer"]].to_string(index=False))
    print(f"\nWrote {os.path.abspath(OUT)}: <session>.csv, <session>.png, summary.csv")


if __name__ == "__main__":
    main()
