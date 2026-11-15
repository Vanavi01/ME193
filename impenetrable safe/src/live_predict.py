"""Live finger-count test: shows what the model thinks and how confident it is.
  python -m src.live_predict
Keys: s = save a snapshot (to data/_snapshots), q = quit.

The big number is the prediction for this frame; the bars show the probability of every count
0-5; "lock would accept" turns green when the lock policy would count it (confidence >= the
threshold held for STABLE_FRAMES frames in a row)."""
import time

import cv2
import numpy as np

from . import config
from .camera import open_from_calibration, read_frame
from .infer import Predictor
from .preprocess import crop_roi, model_view

FONT = cv2.FONT_HERSHEY_SIMPLEX
GREEN, YELLOW, RED, WHITE, GRAY = (0, 200, 0), (0, 220, 255), (0, 0, 255), (240, 240, 240), (120, 120, 120)


def panel(probs, digit, conf, stable, fps, width):
    p = np.zeros((230, width, 3), np.uint8)
    color = GREEN if conf >= config.CONF_THRESHOLD else (YELLOW if conf >= 0.6 else RED)
    cv2.putText(p, str(digit), (15, 120), FONT, 4.0, color, 8)
    cv2.putText(p, f"{conf:.0%} sure", (10, 160), FONT, 0.8, color, 2)
    ok = stable >= config.STABLE_FRAMES
    cv2.putText(p, "lock would accept" if ok else f"steady {stable}/{config.STABLE_FRAMES}",
                (10, 195), FONT, 0.55, GREEN if ok else GRAY, 2)
    cv2.putText(p, f"{fps:.0f} fps   s = snapshot   q = quit", (10, 222), FONT, 0.45, GRAY, 1)

    x0, bar_w = 200, width - 260                       # probability bars for 0-5
    for c, pr in enumerate(probs):
        y = 18 + c * 34
        cv2.putText(p, str(c), (x0, y + 18), FONT, 0.7, WHITE, 2)
        cv2.rectangle(p, (x0 + 25, y), (x0 + 25 + bar_w, y + 22), GRAY, 1)
        cv2.rectangle(p, (x0 + 25, y), (x0 + 25 + int(bar_w * pr), y + 22),
                      color if c == digit else (90, 90, 90), -1)
        cv2.putText(p, f"{pr:.0%}", (x0 + 32 + bar_w, y + 18), FONT, 0.55, WHITE, 1)
    return p


def main():
    calib = config.load_calibration()
    if calib is None:
        raise SystemExit("No calibration. Run: python -m src.calibrate")
    if not config.MODEL_PATH.exists():
        raise SystemExit("No trained model. Run: python -m src.train")
    predictor = Predictor()
    cap, _ = open_from_calibration(calib)
    bx, by, bw, bh = calib.get("border_roi", calib["roi"])
    last, stable, fps, n, t0 = None, 0, 0.0, 0, time.time()
    snap_dir = config.DATA_DIR / "_snapshots"
    try:
        while True:
            frame = read_frame(cap)
            probs = predictor.probabilities(frame)
            digit, conf = int(probs.argmax()), float(probs.max())
            # same rule as the lock: confident AND the same answer for several frames in a row
            stable = stable + 1 if (digit == last and conf >= config.CONF_THRESHOLD) else \
                (1 if conf >= config.CONF_THRESHOLD else 0)
            last = digit
            n += 1
            if time.time() - t0 >= 1:
                fps, n, t0 = n / (time.time() - t0), 0, time.time()

            view = frame.copy()
            cv2.rectangle(view, (bx, by), (bx + bw, by + bh), (0, 0, 255), 2)
            mv = model_view(crop_roi(frame, calib["roi"]), size=140)
            view[10:150, view.shape[1] - 150:view.shape[1] - 10] = mv
            cv2.putText(view, "model sees", (view.shape[1] - 150, 165), FONT, 0.45, (255, 0, 0), 1)
            cv2.imshow("finger count test", np.vstack([view, panel(probs, digit, conf, stable,
                                                                    fps, view.shape[1])]))
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("s"):
                snap_dir.mkdir(parents=True, exist_ok=True)
                path = snap_dir / f"pred{digit}_{int(conf * 100)}_{int(time.time() * 1000)}.png"
                cv2.imwrite(str(path), frame)
                print("saved", path)
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
