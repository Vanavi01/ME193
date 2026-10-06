"""
Runs on the Arduino UNO Q (Linux side, via Arduino App Lab). Drives both motors
on the Cytron Maker Drive (M1 + M2): forward 3 s, stop 1 s, back 3 s, stop 1 s, repeat.

Pins 10/11 live on the UNO Q's microcontroller, so Python can't drive them
directly -- it calls the "set_motor" function provided by motor_test.ino
over the Bridge.

  App Lab app layout:
    python/main.py     <- this file
    sketch/sketch.ino  <- motor_test.ino
"""
import time
from arduino.app_utils import App, Bridge

SPEED = 200   # 0-255 PWM duty; positive = forward, negative = backward


def loop():
    print("clockwise")
    Bridge.call("set_motor", SPEED)
    time.sleep(3)

    print("pause")
    Bridge.call("set_motor", 0)
    time.sleep(1)

    print("counter-clockwise")
    Bridge.call("set_motor", -SPEED)
    time.sleep(3)

    print("pause")
    Bridge.call("set_motor", 0)
    time.sleep(1)


App.run(user_loop=loop)
