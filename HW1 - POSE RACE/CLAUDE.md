# ME193 Gesture-Controlled LEGO Car

Webcam gestures (MediaPipe) → drive commands → LEGO Education Double Motor over BLE
(via the `legoeducation` / SimpleLE package, wrapped by `lelib.py`).

## Hardware

- **LEGO Education Double Motor** (SPIKE-family BLE hub with two motor outputs +
  onboard IMU). Identified by **Connection Card serial `0999`** (`CARD_SERIAL` in
  `drive.py`, `spin_double.py`). Other scripts (`test_devices.py`) connect with
  `card_serial=None`, which just grabs the first advertising device of that type —
  fine when only one is powered on, ambiguous if several are.
- **LEGO Education Single Motor**, Connection Card `1142` (`spin_motor.py`) — used
  for standalone motor tests, not part of the car itself.
- **LEGO Education Color Sensor** — now wired into the car as crash protection
  (`crash_guard.py`, connected with `card_serial=None` like `test_devices.py`).
  Mount it facing forward/down. Also supported by `lelib.py` but not currently
  wired into the car: **Controller** (twin joysticks).
- Any Mac/PC webcam (`drive.py` and `track_hands.py` both prompt for a camera index
  at startup after probing indices 0–1 with `cv2.VideoCapture`).
- The two motors are mounted **mirrored** on the chassis — see "Signed speed = the
  only direction control" below for why `drive.py` negates one side.

## Environment

- Needs **Python 3.11+**; macOS system Python (3.9) won't run `legoeducation`. The
  committed venv (`my_env/`, gitignored) is 3.14.
- `mediapipe` is pinned to `==0.10.35` in the README install command — don't bump
  it casually, the `vision.GestureRecognizer` / `BaseOptions` task API has broken
  across versions before.
- Two MediaPipe task models are checked into the repo (large binaries, not code):
  `gesture_recognizer.task` (used by `drive.py`) and `pose_landmarker_lite.task`
  (used by `track_hands.py`).

## Files

| File | Role |
|---|---|
| `drive.py` | **The car.** Gesture → command → BLE motor speeds. |
| `crash_guard.py` | Color-sensor crash protection, imported by `drive.py`. See below. |
| `lelib.py` | Shared thin wrapper around `legoeducation`, imported by every other script. |
| `track_hands.py` | Earlier/alternate prototype: pose landmarker tracks wrist height directly to tank-drive speeds (no gesture vocabulary, no debounce). Not used by `drive.py`; kept as reference, not wired to the motor. |
| `scan.py` | `bleak` BLE scan to list advertising LEGO devices by name/address — use when a device won't connect and you need to confirm it's advertising at all. |
| `test_devices.py` | Smoke test: connects to one of each device type, exercises sensors/motors, disconnects. Good first thing to run after any `lelib.py` change. |
| `spin_motor.py`, `spin_double.py` | Minimal single-purpose hardware scripts for manual motor checks. |

## BLE quirks (all handled inside `lelib.py`)

- **"not ready" on connect is transient, not fatal.** Every `connect()` override in
  `lelib.py` (`singleMotor`, `doubleMotor`, `controller`, `colorSensor`) retries up
  to 5 times with a 1s sleep when the underlying exception message contains
  "not ready" (case-insensitive), and re-raises anything else or the final failure.
  This is a real, reproducible BLE hub state (the hub advertises before its RPC
  service is actually ready to accept a connection) — don't remove the retry loop
  or treat a single connect failure as proof the hardware is broken.
- **Signed speed is the only direction control that matters for continuous drive.**
  `motor_run()` / the BLE RPC layer takes both a `direction` enum
  (`MOTOR_MOVE_DIRECTION_CLOCKWISE`/`COUNTERCLOCKWISE`) and a signed speed
  (-100..100, sign = clockwise/counter-clockwise). `lelib.py`'s `run_left()` /
  `run_right()` **hardcode `direction=COUNTERCLOCKWISE`** and rely entirely on the
  sign of whatever speed was last pushed via `set_speed_left()`/`set_speed_right()`
  to determine actual rotation. Because the two motors are mounted mirrored on the
  chassis, one wheel's "forward" is the other's "backward" for the same sign —
  `drive.py` compensates with `RIGHT_FLIP = -1`, `LEFT_FLIP = 1` applied right
  before sending. If you ever see the car drive backward on one side only, check
  those flip constants before suspecting the motor wiring.
- **BLE writes are throttled, deliberately.** `drive.py` only calls
  `set_speed_*`/`run_left`/`run_right` when `now - last_send > SEND_INTERVAL` (0.1s)
  **and** the `(left, right)` speed tuple actually changed since `last_cmd`. This
  isn't a UI framerate limiter — it's there to avoid flooding the BLE link with
  redundant identical commands every camera frame (~30/s) when nothing changed.
- **No async/notification-driven state** — `lelib.py` is a synchronous polling
  wrapper (`self.scanned_card.serial`, `self.imu_device.yaw`, etc. are read
  synchronously); there's no event loop to manage in `drive.py` beyond OpenCV's.

## Design decisions in `drive.py`

- **Both hands must agree for drive commands, not turns.** `classify()` requires
  both hands doing the same gesture for `STOP`/`FORWARD`/`BACKWARD`/`FASTER`/`SLOWER`
  (reduces false positives from a single misread hand), but a turn is triggered by
  *one* palm open while the other is anything else (`RIGHT` = right palm open,
  left not open). This is intentional asymmetry, not a missed case — turning with
  two hands would need a third gesture per direction.
- **Debounce via streak-of-frames, separate from lost-hands failsafe.** A candidate
  gesture must repeat for `HOLD_FRAMES` (3) consecutive frames before it becomes the
  active `command`. Critically, a frame with *no* matching gesture (`detected is
  None`, e.g. hands present but between shapes) does **not** reset `command` — it
  just fails to advance the streak. The car keeps executing the last committed
  command through such gaps. Only `LOST_TIMEOUT` (1.5s of literally no hands
  visible) forces `STOP`. This is the failsafe described in the README ("holds the
  last command... if no hands are visible for 1.5 seconds, the motors cut").
- **Speed changes (`FASTER`/`SLOWER`) use their own rate limiter**, `SPEED_REPEAT`
  (0.4s), independent of `HOLD_FRAMES`/`SEND_INTERVAL`. This lets you hold the
  "both pointing up" gesture to ramp speed in `SPEED_STEP` (10) increments, clamped
  to `[MIN_SPEED, MAX_SPEED]` = `[10, 100]`, without it also being gated by the
  drive-command streak logic. `+`/`-` keys do the same adjustment directly, bypassing
  gesture recognition entirely (handy for testing without a working camera feed).
- **`TURN_RATIO` controls turn geometry, not turn detection.** `0.0` = arc turn
  (inner wheel simply stops); `-1.0` would spin the inner wheel backward for a
  pivot/spin-in-place turn. Currently set to arc (gentler, more predictable for a
  gesture-driven car where turn commands may be held inadvertently).
- **`SWAP_HANDS` exists because MediaPipe's Left/Right handedness labeling and the
  mirrored (`cv2.flip`) display can disagree** depending on camera/mount — flip this
  bool rather than rewriting `classify()` if left/right ever come out backward for
  a given setup.
- **`speeds_for()` is a pure lookup, called every frame regardless of whether
  anything changed** — the actual BLE throttling happens after, at the
  `last_send`/`last_cmd` check (see BLE quirks above). Keeping the mapping pure and
  side-effect-free is what makes that later dedup check possible.

## Crash protection (`crash_guard.py`)

- **Color change is the proximity cue, not a distance reading.** There's no
  distance sensor on this hub, so `CrashGuard` uses the Color Sensor's
  `detect_color()` instead: it averages the first `BASELINE_FRAMES` (15) readings
  into a baseline (the floor color), then treats any later reading that differs
  from that baseline as "too close to a wall/object." Debounced the same way as
  gestures — `CRASH_HOLD_FRAMES` (3) consecutive frames of the new state before
  `active` flips — so one noisy reading at a seam or shadow doesn't trip it.
- **Blocks forward motion only, never backward.** `drive.py` calls
  `crash_guard.update()` every frame, right after the `LOST_TIMEOUT` check, and
  forces `command` to `STOP` only when it's `FORWARD`/`LEFT`/`RIGHT` and the guard
  is `active`. `BACKWARD` and `STOP` are left untouched, so the driver can always
  back off whatever tripped it. Once the sensor sees the baseline color again
  (the car has backed clear), `active` drops back to `False` on its own and
  forward driving is re-enabled — there's no separate "reset" step.
- **Baseline is captured once, at startup, not re-averaged later.** If the car
  starts already facing a wall, that wall color becomes the baseline and the
  guard won't catch it. Point the sensor at open floor before/while the first
  ~15 frames come in.
