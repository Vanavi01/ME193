"""Gait logger / live Model 1 classifier for the Uno Q.

MODE = "log"  : record raw IMU samples for LABEL into a CSV (used for training on the PC)
MODE = "live" : classify each window with model1.json (written by pc/train_model1.py)
"""
import csv
import json
import os
import re
import subprocess
import threading
import time
from collections import deque

from arduino.app_utils import App, Bridge

# ---- edit these per run -------------------------------------------------
MODE = "log"            # "log" or "live"
LABEL = "walk"          # stand | walk | run | bike | air_bike
SESSION = "auto"        # "auto" = next free S01, S02, ... ; or set your own id
DURATION_S = 60         # recording length
COUNTDOWN_S = 10        # time to get the strap on and start moving; shown on the LED matrix
TRIM_S = 2              # seconds discarded after the countdown (settling)
SHUTDOWN_WHEN_DONE = True   # try a clean Linux shutdown after saving (see run_log)
# --------------------------------------------------------------------------

FS = 100
WINDOW = 2 * FS
HOP = WINDOW // 2
ACC_LSB_PER_G = 2048.0
GYR_LSB_PER_DPS = 32.8

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "..", "data")
MODEL_PATH = os.path.join(HERE, "model1.json")

lock = threading.Lock()
samples = deque()  # (t_ms, ax, ay, az, gx, gy, gz) raw counts
first_t = None


def imu_sample(t_ms, ax, ay, az, gx, gy, gz):
    global first_t
    with lock:
        if first_t is None:
            first_t = t_ms
        samples.append((t_ms, ax, ay, az, gx, gy, gz))


Bridge.provide("imu_sample", imu_sample)


def status(code):
    """Tell the sketch what to show on the LED matrix."""
    try:
        Bridge.notify("set_status", code)
    except Exception as e:  # LED feedback must never break a recording
        print("status notify failed:", repr(e), "- trying call")
        try:
            Bridge.call("set_status", code)
        except Exception as e2:
            print("status call failed:", repr(e2))


def next_session():
    nums = [int(m.group(1)) for fn in os.listdir(DATA_DIR)
            if (m := re.match(r"[sS](\d+)_", fn))]
    return f"S{max(nums, default=0) + 1:02d}"


def clean_shutdown():
    os.sync()
    try:
        r = subprocess.run(["sudo", "-n", "shutdown", "-h", "now"], timeout=10)
        return r.returncode == 0
    except Exception as e:
        print("shutdown not available from the app:", e)
        return False


def run_log():
    os.makedirs(DATA_DIR, exist_ok=True)
    session = next_session() if SESSION == "auto" else SESSION
    path = os.path.join(DATA_DIR, f"{session}_{LABEL}.csv")
    print(f"Will log {LABEL} to {path}: {COUNTDOWN_S}s countdown, then {DURATION_S}s")

    for k in range(COUNTDOWN_S, 0, -1):
        status(k)
        time.sleep(1)
    with lock:
        samples.clear()  # drop everything streamed during the countdown
    status(100)

    start = time.time()
    t0 = None
    n = 0
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t_ms", "ax_g", "ay_g", "az_g", "gx_dps", "gy_dps", "gz_dps",
                    "label", "session"])
        while time.time() - start < DURATION_S:
            time.sleep(0.2)
            with lock:
                batch = list(samples)
                samples.clear()
            for t, ax, ay, az, gx, gy, gz in batch:
                if t0 is None:
                    t0 = t
                if (t - t0) / 1000.0 < TRIM_S:
                    continue
                w.writerow([t,
                            ax / ACC_LSB_PER_G, ay / ACC_LSB_PER_G, az / ACC_LSB_PER_G,
                            gx / GYR_LSB_PER_DPS, gy / GYR_LSB_PER_DPS, gz / GYR_LSB_PER_DPS,
                            LABEL, session])
                n += 1
            f.flush()
            os.fsync(f.fileno())  # rows reach flash every batch
    print(f"Done: {n} rows (expect about {(DURATION_S - TRIM_S) * FS}) -> {path}")

    status(101)  # check mark: file is saved
    if SHUTDOWN_WHEN_DONE:
        time.sleep(2)
        status(102)
        if not clean_shutdown():
            status(101)
            print("Clean shutdown failed; the file is already synced, check mark = safe to unplug.")


def classify(peak, model):
    for thr, label in zip(model["thresholds"], model["labels"]):
        if peak <= thr:
            return label
    return model["labels"][-1]


def run_live():
    with open(MODEL_PATH) as f:
        model = json.load(f)
    axis = model["axis"]  # 1=ax 2=ay 3=az index into the sample tuple
    buf = deque(maxlen=WINDOW)
    since = 0
    print(f"Live Model 1: axis={model['axis_name']}, thresholds={model['thresholds']}")
    while True:
        time.sleep(0.05)
        with lock:
            batch = list(samples)
            samples.clear()
        for s in batch:
            buf.append(abs(s[axis] / ACC_LSB_PER_G))
            since += 1
            if len(buf) == WINDOW and since >= HOP:
                since = 0
                peak = max(buf)
                print(f"peak={peak:5.2f} g -> {classify(peak, model)}")


def user_loop():
    # Wait for the first sample so timing starts when the sketch is streaming.
    while first_t is None:
        time.sleep(0.1)
    if MODE == "log":
        run_log()
        # Idle instead of returning: App.run may call user_loop again, which would
        # start a second recording and overwrite this file.
        while True:
            time.sleep(1)
    run_live()


App.run(user_loop=user_loop)
