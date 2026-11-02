"""Gait logger / live Model 2 (all six IMU channels) classifier for the Uno Q.

MODE = "log"  : record raw IMU samples for LABEL into a CSV (used for training on the PC)
MODE = "live" : classify each 2 s window with model2.json (written by pc/train_model2.py)
                and publish each prediction as JSON to MQTT_TOPIC (view it with pc/mqtt_viewer.py)
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

import features
import model_runtime

# ---- edit these per run -------------------------------------------------
MODE = "log"            # "log" or "live"
LABEL = "walk"          # stand | walk | run | bike | air_bike
SESSION = "auto"        # "auto" = next free S01, S02, ... ; or set your own id
DURATION_S = 60         # recording length
COUNTDOWN_S = 10        # time to get the strap on and start moving; shown on the LED matrix
TRIM_S = 2              # seconds discarded after the countdown (settling)
SHUTDOWN_WHEN_DONE = True   # try a clean Linux shutdown after saving (see run_log)
MQTT_BROKER = "broker.hivemq.com"   # public test broker; or your PC's IP if it runs mosquitto
MQTT_PORT = 1883
MQTT_TOPIC = "tufts/gait/model2/state"   # must match pc/mqtt_viewer.py
# --------------------------------------------------------------------------

FS = 100
WINDOW = 2 * FS
HOP = WINDOW // 2
ACC_LSB_PER_G = 2048.0
GYR_LSB_PER_DPS = 32.8

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "..", "data")
MODEL_PATH = os.path.join(HERE, "model2.json")

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


def mqtt_connect():
    """Background MQTT client that reconnects on its own; None if paho is missing."""
    try:
        import paho.mqtt.client as mqtt
    except ImportError:
        print("paho-mqtt not installed (python/requirements.txt); not publishing")
        return None
    if hasattr(mqtt, "CallbackAPIVersion"):  # paho 2.x
        client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    else:
        client = mqtt.Client()
    client.connect_async(MQTT_BROKER, MQTT_PORT, keepalive=30)
    client.loop_start()
    print(f"Publishing predictions to mqtt://{MQTT_BROKER}:{MQTT_PORT}/{MQTT_TOPIC}")
    return client


def mqtt_publish(client, label, classes, probs):
    if client is None:
        return
    msg = {"label": label, "confidence": round(max(probs), 4),
           "probs": {c: round(p, 4) for c, p in zip(classes, probs)},
           "t": time.time()}
    try:  # retained, so a viewer that starts later sees the current state at once
        client.publish(MQTT_TOPIC, json.dumps(msg), qos=0, retain=True)
    except Exception as e:  # a network hiccup must never stop classification
        print("mqtt publish failed:", repr(e))


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


def run_live():
    with open(MODEL_PATH) as f:
        model = json.load(f)
    # LED matrix shows a stick-figure pictogram per class (codes defined in the sketch);
    # a class without a pictogram falls back to its digit.
    pictogram = {"stand": 110, "walk": 111, "run": 112, "bike": 113, "air_bike": 113}
    digit = {c: pictogram.get(c, i + 1) for i, c in enumerate(model["classes"])}
    print("Live Model 2 (all IMU channels). LED code legend:", digit)
    mqtt = mqtt_connect()
    buf = deque(maxlen=WINDOW)
    since = 0
    while True:
        time.sleep(0.05)
        with lock:
            batch = list(samples)
            samples.clear()
        for _, ax, ay, az, gx, gy, gz in batch:
            buf.append((ax / ACC_LSB_PER_G, ay / ACC_LSB_PER_G, az / ACC_LSB_PER_G,
                        gx / GYR_LSB_PER_DPS, gy / GYR_LSB_PER_DPS, gz / GYR_LSB_PER_DPS))
            since += 1
            if len(buf) == WINDOW and since >= HOP:
                since = 0
                label, probs = model_runtime.predict(model, features.window_features(list(buf)))
                status(digit[label])
                mqtt_publish(mqtt, label, model["classes"], probs)
                conf = max(probs)
                print(f"-> {label:6s} (confidence {conf:.2f})")


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
