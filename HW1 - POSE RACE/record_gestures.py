'''
Records normalized hand-landmark samples labeled with a ground-truth gesture
name, for training a custom classifier (see train_gestures.py). Also stores
MediaPipe's own gesture_recognizer.task prediction on each sample so the two
models can be compared later on identical rows.

Usage:
    python record_gestures.py <label> <count> [--csv gesture_landmarks.csv]
'''

import argparse
import csv
import os

import cv2

from gestures import LANDMARK_COLUMNS, create_recognizer, normalize_landmarks, recognize

MODEL = "gesture_recognizer.task"
FIELDNAMES = ["session", "true_label", "handedness", "google_label"] + LANDMARK_COLUMNS


def next_session_id(csv_path):
    """One recording run = one session, so frames within a run stay
    correlated but train_gestures.py can hold out whole sessions."""
    if not os.path.exists(csv_path):
        return 0
    with open(csv_path, newline="") as f:
        sessions = [int(row["session"]) for row in csv.DictReader(f) if row.get("session")]
    return max(sessions, default=-1) + 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("label", help="Ground-truth gesture label, e.g. Closed_Fist")
    parser.add_argument("count", type=int, help="Number of samples to record")
    parser.add_argument("--csv", default="gesture_landmarks.csv", help="CSV file to append to")
    args = parser.parse_args()

    cams = []
    for i in range(2):
        c = cv2.VideoCapture(i)
        if c.isOpened():
            cams.append(i)
            c.release()
    print("Available cameras:", cams)
    cap = cv2.VideoCapture(int(input("Which camera index? ")))

    recognizer = create_recognizer(MODEL)

    session = next_session_id(args.csv)
    write_header = not os.path.exists(args.csv)
    csv_file = open(args.csv, "a", newline="")
    writer = csv.DictWriter(csv_file, fieldnames=FIELDNAMES)
    if write_header:
        writer.writeheader()

    recorded = 0
    print(f"Recording session {session}: {args.count} samples for '{args.label}'. Hold the gesture up. Press q to stop early.")

    try:
        while recorded < args.count:
            ok, frame = cap.read()
            if not ok:
                break

            # Not mirrored for display on purpose: normalize_landmarks() already
            # canonicalizes Left/Right via MediaPipe's own handedness label, so
            # flipping here would only make that label mean the opposite hand.
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = recognize(recognizer, rgb)

            for hand_landmarks, handed, gesture in zip(result.hand_landmarks, result.handedness, result.gestures):
                if recorded >= args.count:
                    break
                side = handed[0].category_name
                google_label = gesture[0].category_name
                normalized = normalize_landmarks(hand_landmarks, side)

                row = {"session": session, "true_label": args.label, "handedness": side, "google_label": google_label}
                row.update(zip(LANDMARK_COLUMNS, normalized.tolist()))
                writer.writerow(row)
                recorded += 1

            cv2.putText(frame, f"{args.label}: {recorded}/{args.count}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            cv2.imshow("Recording gestures - press q to stop", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        csv_file.close()
        cap.release()
        cv2.destroyAllWindows()
        print(f"Recorded {recorded} samples to {args.csv}")


if __name__ == "__main__":
    main()
