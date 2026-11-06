"""Per-window features from all six IMU channels. Pure Python (no numpy) so the exact same
code runs in training on the PC and live on the Uno Q.

Window input: a list of (ax, ay, az, gx, gy, gz) tuples, accel in g, gyro in deg/s.

Raw channel means are deliberately NOT used: gravity's sign/axis on the ankle changes when the
strap is rotated (it flipped between recording sessions), so means would teach the model the
strap orientation instead of the activity. Spread, peaks and magnitudes are orientation-robust.
"""
import math

FS = 100
CHANNELS = ["ax", "ay", "az", "gx", "gy", "gz"]
LAG_MIN, LAG_MAX = 25, 120  # 0.25-1.2 s: covers running through slow walking/pedalling

FEATURE_NAMES = (
    [f"{c}_{s}" for c in CHANNELS for s in ("std", "peak", "mad")]
    + ["amag_mean", "amag_std", "amag_max", "gmag_mean", "gmag_std", "gmag_max",
       "acorr_peak", "acorr_lag_s"]
)


def _spread(x):
    n = len(x)
    m = sum(x) / n
    dev = [v - m for v in x]
    std = math.sqrt(sum(d * d for d in dev) / n)
    peak = max(abs(d) for d in dev)    # largest excursion from the window mean
    mad = sum(abs(d) for d in dev) / n  # mean absolute deviation
    return std, peak, mad


def _mean_std_max(x):
    n = len(x)
    m = sum(x) / n
    return m, math.sqrt(sum((v - m) ** 2 for v in x) / n), max(x)


def _autocorr_peak(x):
    """Strength (0..1) and lag (s) of the strongest repeat in the window = step/pedal rhythm."""
    n = len(x)
    m = sum(x) / n
    d = [v - m for v in x]
    denom = sum(v * v for v in d)
    if denom < 1e-9:
        return 0.0, 0.0
    best, best_lag = -1.0, 0
    for lag in range(LAG_MIN, min(LAG_MAX, n - 1) + 1):
        r = sum(d[i] * d[i + lag] for i in range(n - lag)) / denom
        if r > best:
            best, best_lag = r, lag
    return best, best_lag / FS


def window_features(win):
    cols = list(zip(*win))
    feats = []
    for c in cols:
        feats.extend(_spread(c))
    amag = [math.sqrt(a * a + b * b + c * c) for a, b, c, _, _, _ in win]
    gmag = [math.sqrt(a * a + b * b + c * c) for _, _, _, a, b, c in win]
    feats.extend(_mean_std_max(amag))
    feats.extend(_mean_std_max(gmag))
    feats.extend(_autocorr_peak(amag))
    return feats
