import time
from lelib import doubleMotor

CARD_SERIAL = "0999"   # <- change to the number on YOUR card

dm = doubleMotor()

print(f"Connecting to double motor {CARD_SERIAL}...")
dm.connect(card_serial=CARD_SERIAL)
print("Connected.")

try:
    print("Both wheels forward, 2 seconds...")
    dm.run_time(2000)
    time.sleep(1)

    print("Left wheel only...")
    dm.set_speed_left(50)
    dm.set_speed_right(0)
    dm.run_left()
    dm.run_right()
    time.sleep(2)
    dm.stop()

    print("Right wheel only...")
    dm.set_speed_left(0)
    dm.set_speed_right(50)
    dm.run_left()
    dm.run_right()
    time.sleep(2)
finally:
    dm.stop()
    dm.disconnect()
    print("Done.")
