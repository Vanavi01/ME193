import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision

MODEL = "pose_landmarker_lite.task"
LEFT_WRIST, RIGHT_WRIST = 15, 16

cams = []
for i in range(2):
    c = cv2.VideoCapture(i)
    if c.isOpened():
        cams.append(i)
        c.release()
print("Available cameras:", cams)
choice = int(input("Which camera index? "))
cap = cv2.VideoCapture(choice)

landmarker = vision.PoseLandmarker.create_from_model_path(MODEL)

while True:
    ok, frame = cap.read()
    if not ok:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = landmarker.detect(mp_image)

    cv2.line(frame, (0, h // 2), (w, h // 2), (0, 255, 255), 2)

    if result.pose_landmarks:
        lm = result.pose_landmarks[0]
        lw, rw = lm[LEFT_WRIST], lm[RIGHT_WRIST]

        cv2.circle(frame, (int(lw.x * w), int(lw.y * h)), 12, (255, 0, 0), -1)
        cv2.circle(frame, (int(rw.x * w), int(rw.y * h)), 12, (0, 0, 255), -1)

        left_speed = max(-100, min(100, int((0.5 - lw.y) * 200)))
        right_speed = max(-100, min(100, int((0.5 - rw.y) * 200)))

        cv2.putText(frame, f"L {left_speed}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.putText(frame, f"R {right_speed}", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        print(f"left={left_speed:4d}  right={right_speed:4d}")

    cv2.imshow("Wrist tracker - press q to quit", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
