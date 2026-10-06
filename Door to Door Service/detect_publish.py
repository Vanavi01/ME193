"""
Run on your laptop. Detects the green minifig with YOLO and publishes its
normalized position over MQTT.

  pip install ultralytics paho-mqtt opencv-python
  python detect_publish.py --topic me35/yourname/minifig
"""
import argparse, json, time
import cv2
import paho.mqtt.client as mqtt
from ultralytics import YOLO

p = argparse.ArgumentParser()
p.add_argument("--model", default="runs/detect/train/weights/best.pt")  # your trained weights
p.add_argument("--broker", default="broker.hivemq.com")  # use your class broker
p.add_argument("--port", type=int, default=1883)
p.add_argument("--topic", default="me35/victor/minifig")
p.add_argument("--cam", type=int, default=0)
p.add_argument("--conf", type=float, default=0.5)
p.add_argument("--rate", type=float, default=10.0)    # messages per second
args = p.parse_args()

model = YOLO(args.model)
client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
client.connect(args.broker, args.port)
client.loop_start()

cap = cv2.VideoCapture(args.cam)
last_pub = 0.0

# Speed slider: max motor PWM sent to the car (0 = stop, 255 = full speed)
cv2.namedWindow("minifig")
cv2.createTrackbar("max speed", "minifig", 150, 255, lambda v: None)

while True:
    ok, frame = cap.read()
    if not ok:
        break
    h, w = frame.shape[:2]
    result = model(frame, conf=args.conf, verbose=False)[0]

    msg = {"found": False}
    if len(result.boxes):
        best = result.boxes[int(result.boxes.conf.argmax())]   # most confident box
        x1, y1, x2, y2 = best.xyxy[0].tolist()
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        msg = {"found": True, "x": round(cx / w, 3), "y": round(cy / h, 3),
               "conf": round(float(best.conf), 2)}
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.circle(frame, (int(cx), int(cy)), 7, (255, 0, 0), -1)  # blue (BGR)

    msg["max_speed"] = cv2.getTrackbarPos("max speed", "minifig")

    cv2.line(frame, (w // 2, 0), (w // 2, h), (200, 200, 200), 1)  # center line
    cv2.putText(frame, json.dumps(msg), (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (255, 255, 255), 2)

    now = time.time()
    if now - last_pub >= 1.0 / args.rate:
        client.publish(args.topic, json.dumps(msg))
        last_pub = now

    cv2.imshow("minifig", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
client.loop_stop()
