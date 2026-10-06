"""Press SPACE to save a frame, q to quit. Saves to ./dataset_raw/"""
import os, time, cv2

os.makedirs("dataset_raw", exist_ok=True)
cap = cv2.VideoCapture(0)
n = 0
while True:
    ok, frame = cap.read()
    if not ok:
        break
    cv2.imshow("capture (SPACE=save, q=quit)", frame)
    k = cv2.waitKey(1) & 0xFF
    if k == ord(" "):
        cv2.imwrite(f"dataset_raw/img_{int(time.time()*1000)}.jpg", frame)
        n += 1
        print("saved", n)
    elif k == ord("q"):
        break
cap.release()
cv2.destroyAllWindows()
