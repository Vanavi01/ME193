'''
Turns a MediaPipe hand-landmark detection into a drive command. Each hand's
shape is classified by our own model (trained via record_gestures.py +
train_gestures.py), not MediaPipe's built-in gesture_recognizer.task labels
- we still use that task file for hand detection/landmarks/handedness, just
not its gesture classification output.
'''

import joblib
import mediapipe as mp
import numpy as np
import pandas as pd
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python import vision

SWAP_HANDS = False

GESTURE_MODEL_PATH = "gesture_model.joblib"
NUM_LANDMARKS = 21
LANDMARK_COLUMNS = [f"{axis}{i}" for i in range(NUM_LANDMARKS) for axis in ("x", "y", "z")]

_classifier = None


def create_recognizer(model_path, num_hands=2):
    return vision.GestureRecognizer.create_from_options(
        vision.GestureRecognizerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            num_hands=num_hands,
        )
    )


def recognize(recognizer, frame_rgb):
    return recognizer.recognize(mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb))


def normalize_landmarks(hand_landmarks, handedness_label):
    """Wrist-centered, scale-normalized landmark vector, with Left hands
    mirrored onto Right so both hands share one feature space. Must match
    record_gestures.py exactly - it's what the classifier was trained on."""
    pts = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks], dtype=np.float64)
    pts -= pts[0]
    max_dist = np.linalg.norm(pts, axis=1).max()
    if max_dist > 0:
        pts /= max_dist
    if handedness_label == "Left":
        pts[:, 0] *= -1
    return pts.flatten()


def load_classifier(path=GESTURE_MODEL_PATH):
    """Load and cache the trained gesture classifier. hands_from_result()
    calls this lazily on first use; call it explicitly beforehand (e.g. at
    startup, before connecting to hardware) to fail fast on a missing or
    stale model file instead of mid-frame-loop."""
    global _classifier
    if _classifier is None:
        _classifier = joblib.load(path)
    return _classifier


def hands_from_result(result, swap_hands=SWAP_HANDS):
    clf = load_classifier()
    hands = {}
    for hand_landmarks, handed in zip(result.hand_landmarks, result.handedness):
        raw_side = handed[0].category_name
        normalized = normalize_landmarks(hand_landmarks, raw_side)
        features = pd.DataFrame([normalized], columns=LANDMARK_COLUMNS)
        label = clf.predict(features)[0]

        side = raw_side
        if swap_hands:
            side = "Left" if raw_side == "Right" else "Right"
        hands[side] = label
    return hands


def classify(hands):
    left = hands.get("Left")
    right = hands.get("Right")

    # speed modifiers
    if left == "Pointing_Up" and right == "Pointing_Up":
        return "FASTER"
    if left == "Victory" and right == "Victory":
        return "SLOWER"

    # drive commands
    if left == "Closed_Fist" and right == "Closed_Fist":
        return "STOP"
    if left == "Thumb_Down" and right == "Thumb_Down":
        return "FORWARD"
    if left == "Open_Palm" and right == "Open_Palm":
        return "BACKWARD"
    if right == "Open_Palm" and left != "Open_Palm":
        return "RIGHT"
    if left == "Open_Palm" and right != "Open_Palm":
        return "LEFT"
    return None
