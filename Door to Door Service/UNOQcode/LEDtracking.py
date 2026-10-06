"""
UNO Q Linux side (App Lab app). Subscribes to the minifig topic, shows the dot
on the LED matrix, and decides motor speed.

Set MODE = "display" for Part 1 (LED only) or "drive" for Part 2 (LED + motor).
"""
import json, time
import paho.mqtt.client as mqtt
from arduino.app_utils import App, Bridge

BROKER = "broker.hivemq.com"        # must match the laptop
PORT = 1883
TOPIC = "me35/victor/minifig"     # must match the laptop

MODE = "drive"        # "display" or "drive"
COLS, ROWS = 13, 8

# --- Part 2 tuning ---
DEADBAND = 0.05       # stop when |x - 0.5| < this (5% of frame width)
KP = 600              # extra speed per unit of error
MIN_SPEED = 90        # lowest PWM that actually turns your motor
MAX_SPEED = 200
DIRECTION = 1         # flip to -1 if the cart runs away from center
TIMEOUT = 1.0         # stop motor if no message for this long (s)

latest = None
last_rx = 0.0

def on_connect(client, userdata, flags, rc, props=None):
    print("MQTT connected:", rc)
    client.subscribe(TOPIC)

def on_message(client, userdata, m):
    global latest, last_rx
    try:
        latest = json.loads(m.payload)
        last_rx = time.time()
    except ValueError:
        print("bad message:", m.payload)

def speed_from_x(x):
    err = x - 0.5
    if abs(err) < DEADBAND:
        return 0
    mag = min(MAX_SPEED, MIN_SPEED + abs(err) * KP)
    return int(DIRECTION * mag * (1 if err > 0 else -1))

def loop():
    global latest
    msg, latest = latest, None

    if time.time() - last_rx > TIMEOUT:      # safety watchdog
        Bridge.call("drive", 0)
        time.sleep(0.05)
        return

    if msg is not None:
        if msg.get("found"):
            x, y = float(msg["x"]), float(msg["y"])
            y = 1 - y   # flip vertically to match the screen
            col = max(0, min(COLS - 1, round(x * (COLS - 1))))
            row = max(0, min(ROWS - 1, round(y * (ROWS - 1))))
            Bridge.call("show", col, row)
            if MODE == "drive":
                spd = speed_from_x(x)
                Bridge.call("drive", spd)
                print(f"x={x:.2f} -> col {col}, row {row}, speed {spd}")
        else:
            Bridge.call("clear")
            Bridge.call("drive", 0)
    time.sleep(0.05)

client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
client.on_connect = on_connect
client.on_message = on_message
client.connect(BROKER, PORT)
client.loop_start()

App.run(user_loop=loop)