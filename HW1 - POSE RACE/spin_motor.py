import time
from lelib import singleMotor

sm = singleMotor()

print("Connecting to single motor 1142...")
sm.connect(card_serial="1142")
print("Connected.")

try:
    print("Spinning 2 rotations...")
    sm.spin(2)
    time.sleep(1)

    print("Running at speed 50 for 3 seconds...")
    sm.run(50)
    time.sleep(3)
finally:
    sm.stop()
    sm.disconnect()
    print("Done.")