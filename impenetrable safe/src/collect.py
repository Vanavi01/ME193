"""Guided, automatic data collection. Anyone can add their hand:
    python -m src.collect                    (asks for your name; training data)
    python -m src.collect --person alex      (resumes where alex left off)
    python -m src.collect --person alex --test   (separate TEST session, never trained on)

The screen asks for finger counts in RANDOM order, a few pictures each, so every count is
recorded at many different moments. Placement rule: wrist on the BOTTOM blue tape, fingers
pointing up, fingers spread apart, hand inside the border (the model only sees the inside). Hold still: a picture is
taken automatically when the hand is steady, sharp and has changed since the last picture. When it beeps and says SWITCH, show the new count.

Keys: SPACE pause/resume | u undo last picture | n skip this prompt | q quit"""
import argparse
import csv
import re
import time

import cv2
import numpy as np

from . import config
from .autocapture import AutoCapture
from .camera import open_from_calibration, read_frame
from .preprocess import crop_roi, model_view
from .quality import calibration_drift, detect_red_square, iou

try:
    import winsound
except ImportError:                     # not Windows (e.g. the UnoQ)
    winsound = None

PROMPTS = {0: "make a FIST (0 fingers)", 1: "show 1 FINGER", 2: "show 2 FINGERS",
           3: "show 3 FINGERS", 4: "show 4 FINGERS", 5: "show 5 FINGERS (open hand)"}
# Placement rule (same for everyone): wrist at the BOTTOM tape, fingers pointing up, hand
# inside the border. Variety comes from these, not from where the hand is.
HINTS = ["slide a little left / right", "tilt your hand a little", "use your OTHER hand",
         "SPREAD your fingers (gaps between them)", "use DIFFERENT fingers",
         "keep the wrist on the bottom tape", "palm up or down"]
GREEN, RED, WHITE, YELLOW, ORANGE, GRAY = ((0, 200, 0), (0, 0, 255), (230, 230, 230),
                                           (0, 255, 255), (0, 140, 255), (150, 150, 150))
FONT = cv2.FONT_HERSHEY_SIMPLEX


def beep(kind):
    if winsound is None:
        print("\a", end="", flush=True)
        return
    try:
        if kind == "switch":
            winsound.Beep(1100, 90)
        elif kind == "done":
            for f in (660, 880, 1100, 1320):
                winsound.Beep(f, 120)
    except RuntimeError:
        pass


def clean_name(name):
    return re.sub(r"[^a-z0-9_-]", "", name.strip().lower())


def draw(view, ac, res, moved, flash, person, test):
    H, W = view.shape[:2]
    if flash:
        cv2.rectangle(view, (0, 0), (W - 1, H - 1), GREEN, 12)
    mv = model_view(res["crop"], size=150)
    view[10:160, W - 160:W - 10] = mv
    cv2.putText(view, "model sees", (W - 160, 175), FONT, 0.45, (255, 0, 0), 1)
    if test:
        cv2.putText(view, "TEST SESSION", (10, 30), FONT, 0.8, ORANGE, 2)

    panel = np.zeros((175, W, 3), np.uint8)
    now = time.time()
    if ac.done:
        cv2.putText(panel, f"ALL DONE - thank you, {person}!", (10, 38), FONT, 1.0, GREEN, 2)
    elif now < ac.switch_until:
        cv2.putText(panel, f"SWITCH! {PROMPTS[ac.label]}", (10, 38), FONT, 0.95, ORANGE, 2)
    else:
        cv2.putText(panel, PROMPTS[ac.label], (10, 38), FONT, 1.1, YELLOW, 2)

    done, total = ac.total()                              # overall progress bar
    cv2.rectangle(panel, (10, 52), (W - 10, 70), GRAY, 1)
    cv2.rectangle(panel, (10, 52), (10 + int((W - 20) * done / max(1, total)), 70), GREEN, -1)
    cv2.putText(panel, f"{done}/{total} pictures", (W - 170, 67), FONT, 0.5, WHITE, 1)

    for i, c in enumerate(config.CLASSES):
        finished = ac.counts[c] >= ac.targets[c]
        color = GREEN if finished else (YELLOW if c == ac.label else WHITE)
        cv2.putText(panel, f"{c}:{ac.counts[c]}/{ac.targets[c]}", (10 + 105 * i, 95), FONT, 0.55,
                    color, 2)

    reason = res.get("reason")
    if res.get("event") == "capture":
        cv2.putText(panel, "picture taken", (10, 122), FONT, 0.6, GREEN, 2)
    elif reason and reason not in ("wait", "switching"):
        color = YELLOW if reason in ("hold still", "move your hand more") else RED
        cv2.putText(panel, reason, (10, 122), FONT, 0.6, color, 2)
    if not ac.done:
        tips = res.get("warnings") or []
        if moved:
            tips = ["camera/border moved (ok, but recalibrate soon)"] + tips
        hint = tips[0] if tips else HINTS[(done // 3) % len(HINTS)]
        cv2.putText(panel, f"tip: {hint}", (10, 147), FONT, 0.5, ORANGE if tips else WHITE, 1)
    cv2.putText(panel, "SPACE pause | u undo | n skip prompt | q quit", (10, 168), FONT, 0.45, GRAY, 1)
    return np.vstack([view, panel])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--person")
    ap.add_argument("--test", action="store_true",
                    help="record a separate TEST session (data/_test), never used for training")
    ap.add_argument("--per-class", type=int)
    ap.add_argument("--more", action="store_true",
                    help="another round on top of what this person already has")
    args = ap.parse_args()

    calib = config.load_calibration()
    if calib is None or not config.BACKGROUND_PATH.exists():
        raise SystemExit("No calibration. Run: python -m src.calibrate")
    person = clean_name(args.person or input("Your first name (your pictures are saved under it): "))
    if not person:
        raise SystemExit("need a name")
    session = time.strftime("%Y%m%d-%H%M")
    if args.test:
        person_dir = config.TEST_DIR_PATH / f"{person}_{session}"
        per_class = args.per_class or config.AUTO_TEST_PER_CLASS
    else:
        person_dir = config.DATA_DIR / person
        per_class = args.per_class or config.AUTO_PER_CLASS

    bg = cv2.imread(str(config.BACKGROUND_PATH))
    roi_box, outer = tuple(calib["roi"]), tuple(calib["outer"])
    cap, _ = open_from_calibration(calib)
    for issue in calibration_drift(read_frame(cap), calib):
        print("CALIBRATION CHECK:", issue)

    person_dir.mkdir(parents=True, exist_ok=True)
    counts = {c: len(list((person_dir / str(c)).glob("*.png"))) for c in config.CLASSES}
    targets = {c: per_class + (config.AUTO_EXTRA_THREE if c == 3 else 0)
               + (counts[c] if args.more else 0) for c in config.CLASSES}
    ac = AutoCapture(targets, counts, calib, bg)
    meta_path = person_dir / "meta.csv"
    new_meta = not meta_path.exists()
    meta_f = open(meta_path, "a", newline="")
    meta = csv.writer(meta_f)
    if new_meta:
        meta.writerow(["file", "label", "time", "session", "hand_frac", "sharpness", "empty"])

    print(f"\nHi {person}! {'TEST session' if args.test else 'Training session'} {session}. "
          "Follow the prompt under the video; pictures are taken automatically.\n"
          "RULE: wrist on the BOTTOM blue tape, fingers pointing up, hand inside the border, "
          "fingers SPREAD apart. No blue sleeves.\n")
    ac.switch_until = time.time() + config.SWITCH_PAUSE_S       # time to get ready
    history, moved, frame_i, flash_until, done_at = [], False, 0, 0.0, None
    try:
        while True:
            frame = read_frame(cap)
            crop = crop_roi(frame, roi_box).copy()
            if frame_i % 15 == 0:
                det = detect_red_square(frame)
                if det is not None:
                    moved = iou(det["outer"], outer) < config.MOVE_IOU_MIN
            frame_i += 1
            now = time.time()
            res = ac.step(crop, now)                      # moved = tip only (shown in draw)
            res["crop"] = crop

            if res["event"] == "capture":
                label = res["label"]
                d = person_dir / str(label)
                d.mkdir(exist_ok=True)
                path = d / f"{int(now * 1000)}.png"
                cv2.imwrite(str(path), crop)
                m = res["metrics"]
                meta.writerow([path.name, label, f"{now:.3f}", session, m["hand_frac"],
                               f"{m['sharpness']:.1f}", int(res["empty"])])
                meta_f.flush()
                history.append((label, path, res["empty"]))
                flash_until = now + 0.15
                if "next" in res:
                    if ac.done:
                        beep("done")
                        done_at = now
                        print(f"ALL DONE: {ac.counts}")
                    else:
                        beep("switch")

            view = draw(frame.copy(), ac, res, moved, now < flash_until, person, args.test)
            cv2.imshow(f"collect: {person}{' (TEST)' if args.test else ''}", view)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or (done_at and now - done_at > 3):
                break
            if key == ord(" "):
                ac.paused = not ac.paused
            elif key == ord("u") and history:
                label, path, was_empty = history.pop()
                path.unlink(missing_ok=True)
                ac.undo(label, was_empty)
                done_at = None
            elif key == ord("n"):
                ac.skip(now)
    finally:
        meta_f.close()
        cap.release()
        cv2.destroyAllWindows()
    print("pictures per count:", ac.counts)
    print("next: python -m src.train" if not args.test else "next: python -m src.evaluate")


if __name__ == "__main__":
    main()
