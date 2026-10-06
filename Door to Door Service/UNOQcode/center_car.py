"""
Runs on the Arduino UNO Q (Linux side, via Arduino App Lab). Subscribes to the
minifig position published by detect_publish.py on the laptop and drives the
car forward/back until the minifig sits in the center of the laptop camera view.

The motor pins (M1: 10/11, M2: 6/5) live on the UNO Q's microcontroller, so both
motors are driven together by calling "set_motor" in motor_test.ino over the
Bridge (same sketch as motor_test.py).

  App Lab app layout:
    python/main.py           <- this file
    python/requirements.txt  <- paho-mqtt
    sketch/sketch.ino        <- motor_test.ino

  Laptop:  python detect_publish.py
"""
import json, threading, time
import paho.mqtt.client as mqtt
from arduino.app_utils import App, Bridge

BROKER = "broker.hivemq.com"   # must match detect_publish.py
PORT = 1883
TOPIC = "me35/victor/minifig"

DIRECTION = 1      # flip to -1 if the car drives away from center instead of toward it
DEADBAND = 0.05    # |x - 0.5| below this counts as centered -> stop
KP = 400           # PWM per unit of error (error ranges -0.5..0.5)
MIN_SPEED = 90     # smallest PWM that actually gets the car moving
MAX_SPEED = 200    # used until the laptop's "max speed" slider value arrives
TIMEOUT = 0.5      # seconds without a fresh detection -> stop

latest = {"found": False}
last_seen = 0.0
lock = threading.Lock()


def on_message(client, userdata, msg):
    global latest, last_seen
    try:
        data = json.loads(msg.payload)
    except ValueError:
        return
    with lock:
        latest = data
        if data.get("found"):
            last_seen = time.time()


def on_connect(client, userdata, flags, reason_code, properties):
    print("connected to broker, subscribing to", TOPIC)
    client.subscribe(TOPIC)


client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
client.on_connect = on_connect
client.on_message = on_message
client.connect(BROKER, PORT)
client.loop_start()


def speed_for(x, max_speed):
    error = x - 0.5                       # negative = minifig left of center
    if abs(error) < DEADBAND:
        return 0
    speed = MIN_SPEED + KP * (abs(error) - DEADBAND)
    speed = min(int(speed), max_speed)    # slider below MIN_SPEED still caps it
    return DIRECTION * speed * (1 if error > 0 else -1)


def loop():
    with lock:
        msg, seen = latest, last_seen

    if msg.get("found") and time.time() - seen < TIMEOUT:
        speed = speed_for(msg["x"], msg.get("max_speed", MAX_SPEED))
    else:
        speed = 0                         # lost the minifig -> don't drive blind

    Bridge.call("set_motor", speed)
    time.sleep(0.05)                      # ~20 Hz control loop


App.run(user_loop=loop)
