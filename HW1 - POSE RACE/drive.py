import time
import cv2

import gestures
import motor
from crash_guard import CrashGuard
from motor import CarMotor

CARD_SERIAL = "0999"
COLOR_CARD_SERIAL = None  # first advertising color sensor - fine with only one powered on
MODEL = "gesture_recognizer.task"

speed = 60
SPEED_REPEAT = 0.4   # seconds between bumps while holding the gesture
HOLD_FRAMES = 3
LOST_TIMEOUT = 1.5


cams = []
for i in range(2):
    c = cv2.VideoCapture(i)
    if c.isOpened():
        cams.append(i)
        c.release()
print("Available cameras:", cams)
cap = cv2.VideoCapture(int(input("Which camera index? ")))

recognizer = gestures.create_recognizer(MODEL)
print(f"Loading gesture classifier {gestures.GESTURE_MODEL_PATH}...")
gestures.load_classifier()

car = CarMotor(CARD_SERIAL)
car.connect()

crash_guard = CrashGuard(COLOR_CARD_SERIAL)
crash_guard.connect()

candidate, streak, command = "STOP", 0, "STOP"
last_seen = time.time()
last_speed_change = 0.0

try:
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        now = time.time()
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = gestures.recognize(recognizer, rgb)
        hands = gestures.hands_from_result(result)

        detected = None
        if hands:
            detected = gestures.classify(hands)
            last_seen = now

        if detected is not None:
            if detected == candidate:
                streak += 1
            else:
                candidate, streak = detected, 1

            if streak >= HOLD_FRAMES:
                if candidate in ("FASTER", "SLOWER"):
                    if now - last_speed_change > SPEED_REPEAT:
                        if candidate == "FASTER":
                            speed = min(motor.MAX_SPEED, speed + motor.SPEED_STEP)
                        else:
                            speed = max(motor.MIN_SPEED, speed - motor.SPEED_STEP)
                        last_speed_change = now
                        car.reset_send_cache()
                else:
                    command = candidate

        if now - last_seen > LOST_TIMEOUT:
            command = "STOP"

        crashed = crash_guard.update()
        if crashed and command in ("FORWARD", "LEFT", "RIGHT"):
            command = "STOP"

        cv2.putText(frame, command, (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 255, 0), 3)
        if crashed:
            cv2.putText(frame, "WALL - BACK UP", (20, 225),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        cv2.putText(frame, f"SPEED {speed}", (20, 105),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
        cv2.putText(frame, f"L: {hands.get('Left', '-')}", (20, 145),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        cv2.putText(frame, f"R: {hands.get('Right', '-')}", (20, 180),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        car.send(command, speed, now)

        cv2.imshow("Gesture drive - press q to quit", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key in (ord("+"), ord("=")):
            speed = min(motor.MAX_SPEED, speed + motor.SPEED_STEP)
            car.reset_send_cache()
        elif key in (ord("-"), ord("_")):
            speed = max(motor.MIN_SPEED, speed - motor.SPEED_STEP)
            car.reset_send_cache()
finally:
    car.stop()
    car.disconnect()
    cap.release()
    cv2.destroyAllWindows()
    print("Stopped.")
