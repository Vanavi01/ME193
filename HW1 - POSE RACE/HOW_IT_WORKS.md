# How the Gesture-Controlled LEGO Car Works

A walkthrough of every component in this repo and how they chain together to
turn a hand gesture in front of a webcam into a LEGO Education Double Motor
spinning over Bluetooth.

## The pipeline, end to end

```
Webcam frame
     │
     ▼
MediaPipe hand landmarks + handedness   (gesture_recognizer.task, via gestures.py)
     │
     ▼
Our own Random Forest classifier        (gesture_model.joblib, via gestures.py)
     │  → per-hand label, e.g. Left: "Closed_Fist", Right: "Closed_Fist"
     ▼
classify()                              (gestures.py)
     │  → one drive command, e.g. "FORWARD" / "STOP" / "LEFT" / "FASTER"
     ▼
Debounce (streak-of-frames) + lost-hands failsafe   (drive.py)
     │
     ▼
Crash guard override                    (crash_guard.py)
     │  → forces FORWARD/LEFT/RIGHT to STOP if a wall/object was just detected
     ▼
speeds_for() → (left_speed, right_speed)            (motor.py)
     │
     ▼
Send-throttled BLE write                (motor.py → lelib.py → legoeducation)
     │
     ▼
LEGO Education Double Motor spins the wheels
```

Every stage below is one file in the repo.

## 1. Seeing the hand: MediaPipe (`gestures.py`)

`drive.py` grabs a frame from the webcam, flips it (so the on-screen preview
acts like a mirror), and hands it to a MediaPipe `GestureRecognizer` built
from the `gesture_recognizer.task` model file. We only use MediaPipe for what
it's genuinely good at — finding hands in the frame and returning 21 3D hand
landmarks per hand plus a Left/Right handedness label. We do **not** use
MediaPipe's own built-in gesture classification (see §2 for why).

## 2. Classifying the shape: our own model, not Google's

MediaPipe ships a built-in gesture classifier, but it's tuned for full-body
photos, not a hand held up close to a webcam for driving a robot, and its
accuracy for our five gesture shapes wasn't good enough. So this project
trains its own classifier:

- **`record_gestures.py`** — hold up a gesture, run
  `python record_gestures.py Closed_Fist 100`, and it saves 100 rows of
  *normalized* landmark coordinates to `gesture_landmarks.csv`, each labeled
  with the ground-truth gesture name. It also records what MediaPipe's own
  model guessed for the same frame, purely so the two can be compared later.
  Recordings are grouped into "sessions" (one per run) so that near-duplicate
  frames from the same recording don't leak between train and test sets.
- **`gestures.normalize_landmarks()`** — the shared normalization both
  recording and live inference use: landmarks are re-centered on the wrist,
  scaled so the farthest point is distance 1 (makes the model distance- and
  hand-size-invariant), and Left-hand landmarks are mirrored onto the Right
  hand's coordinate space so one model can classify either hand.
- **`train_gestures.py`** — trains a `RandomForestClassifier` on the recorded
  landmarks and saves it to `gesture_model.joblib`. It evaluates two ways: a
  naive random train/test split (optimistic, because it leaks near-duplicate
  frames from the same session into both sides), and **leave-one-session-out
  cross-validation**, which always tests on a recording session the model
  never trained on — a fairer estimate of real accuracy. It also reports how
  our model's accuracy compares to Google's built-in classifier on the exact
  same frames.
- **`gestures.hands_from_result()`** — at drive time, loads the cached
  `gesture_model.joblib`, normalizes each detected hand's landmarks the same
  way, and predicts a label per hand: `{"Left": "Closed_Fist", "Right":
  "Open_Palm"}`, etc.

## 3. Turning two hand shapes into one command: `classify()`

`gestures.classify(hands)` is a lookup table over the pair of hand labels:

| Left hand | Right hand | Command |
|---|---|---|
| Pointing_Up | Pointing_Up | `FASTER` |
| Victory | Victory | `SLOWER` |
| Closed_Fist | Closed_Fist | `STOP` |
| Thumb_Down | Thumb_Down | `FORWARD` |
| Open_Palm | Open_Palm | `BACKWARD` |
| anything else | Open_Palm | `RIGHT` |
| Open_Palm | anything else | `LEFT` |
| *(none of the above)* | | `None` (no command this frame) |

Straight-line driving and speed changes deliberately require **both hands**
to agree — this cuts down on false positives from a single misread hand.
Turning only needs **one** palm open, because requiring two hands to agree on
a turn would mean inventing a whole extra gesture per direction.

## 4. Making it steady: debounce and the lost-hands failsafe (`drive.py`)

A single frame's classification is noisy — hands are mid-transition between
shapes plenty of the time. `drive.py` fixes this with a streak counter: a
candidate command has to repeat for `HOLD_FRAMES` (3) consecutive frames
before it's trusted and becomes the active `command`. Critically, a frame
where no gesture matched (`detected is None`) does *not* reset this — it just
fails to add to the streak — so the car keeps executing its last committed
command through the brief gaps between hand shapes, instead of stuttering.

The one thing that unconditionally overrides all of this is `LOST_TIMEOUT`
(1.5s): if no hands have been visible at all for that long, `command` is
forced to `STOP` regardless of streak state. That's the actual failsafe for
"the operator walked away or the camera lost the hands."

`FASTER`/`SLOWER` bypass the drive-command streak entirely and use their own
independent rate limiter, `SPEED_REPEAT` (0.4s), so holding the "both
pointing up" gesture ramps speed smoothly in `SPEED_STEP` (10) increments,
clamped to `[MIN_SPEED, MAX_SPEED]` = `[10, 100]`.

## 5. Crash protection: the Color Sensor (`crash_guard.py`)

There's no distance sensor on this hub, so `CrashGuard` improvises one out of
the LEGO Education Color Sensor, mounted facing forward/down. It averages the
first 15 frames into a baseline color (the floor), then watches for later
readings that differ from that baseline — a wall, box, or table edge under/in
front of the car reads as a different color (or a sudden reflectivity spike
from being much closer). That color change is debounced the same way gesture
detection is (3 consecutive frames before it's trusted) so one noisy reading
doesn't trip it.

Every frame, right after the lost-hands failsafe check, `drive.py` asks the
guard whether it's active and, if so, forces `FORWARD`/`LEFT`/`RIGHT` to
`STOP` — but leaves `BACKWARD` (and `STOP`) completely alone. That's the key
design point: **the crash guard can only stop forward motion, never take
control away from the operator**, so you can always gesture `BACKWARD` to
pull off whatever tripped it. Once the sensor sees the baseline floor color
again, the guard clears itself automatically and forward driving works again.

## 6. Turning a command into wheel speeds: `motor.py`

`speeds_for(cmd, spd)` is a pure lookup from command + current speed to a
`(left, right)` speed pair — e.g. `"FORWARD"` → `(spd, spd)`,
`"LEFT"` → `(inner, spd)` where `inner = spd * TURN_RATIO`.
`TURN_RATIO = 0.0` means an arc turn (inner wheel just stops); it's kept
gentle on purpose since a gesture-driven car can easily hold a turn command
longer than intended.

`CarMotor.send()` then does two things before it's allowed to actually write
to the motors:
1. **Throttles BLE writes** — only sends if `SEND_INTERVAL` (0.1s) has
   passed since the last write, so a ~30fps camera loop doesn't flood the
   BLE link.
2. **Dedupes** — only sends if the `(left, right)` tuple actually changed
   since the last send, so holding a steady command doesn't re-send
   identical speeds every throttle window either.

Because the two motors are mounted **mirrored** on the chassis, the same
signed speed means opposite physical rotation on each side. `RIGHT_FLIP = -1`
/ `LEFT_FLIP = 1` correct for that right before the BLE write.

## 7. Talking to the hardware: `lelib.py`

`lelib.py` is a thin shared wrapper around the `legoeducation` package (BLE
RPC to the SPIKE-family hub) used by every script in this repo, not just
`drive.py`. The one quirk worth knowing: every `connect()` retries up to 5
times with a 1s sleep whenever the hub reports "not ready" — a real,
reproducible BLE state where the hub starts advertising slightly before its
RPC service can actually accept a connection. Both the Double Motor and the
Color Sensor connections drive.py makes go through this retry path.

## Putting it together: what happens when you make a fist with both hands

1. Webcam captures a frame; MediaPipe finds both hands' 21 landmarks each
   plus Left/Right handedness.
2. Each hand's landmarks are normalized and fed to `gesture_model.joblib`,
   which predicts `"Closed_Fist"` for both.
3. `classify()` sees `left == right == "Closed_Fist"` → returns `"STOP"`.
4. If `"STOP"` repeats for 3 frames running, it becomes the committed
   `command`.
5. The crash guard is checked — `STOP` isn't in the set it can override
   anyway, so it passes through unchanged.
6. `speeds_for("STOP", speed)` → `(0, 0)`.
7. `CarMotor.send()` writes `(0, 0)` to both motors over BLE (throttle
   permitting), and the wheels stop.

The same seven steps, with `Thumb_Down`/`Thumb_Down` → `FORWARD` → `(speed,
speed)`, are how the car drives forward — and if the color sensor trips
mid-drive, step 5 is where that forward command gets overridden to `STOP`
before it ever reaches the motors.
