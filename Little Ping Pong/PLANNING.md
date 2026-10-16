# Little Ping Pong: Project Planning Notes

*Brainstorm and design log, started 2026-10-08. No code has been written yet.*

---

## 0. Project Goal (original brief)

Play **virtual ping-pong** using a **LEGO Education double motor** mounted on a real paddle. The system must include:

- **Pose information / recognition**: know whether the paddle is in the right location.
- **IMU information**: know whether the person is swinging.
- **AprilTag info**: set the player level (ball speed) and start the game.
- **Real-time score**: post to `ME193/Rogers/Victor` as a **floating-point number** equal to the **record number of continuous hits** (best streak).

### Ideas added during brainstorming
- A **red paddle** so the camera can track it easily.
- The double motor's **IMU** provides the swing data.
- **Spin mechanics** plus **hit strength/speed**.
- Possibly an **AprilTag on the paddle itself** for the paddle angle.

---

## 1. Paddle Tracking: Train a Model or Not?

Because the paddle is bright red, you most likely don't need a trained model.

| Approach | Effort | Speed | What you get | Weakness |
|---|---|---|---|---|
| HSV colour threshold + contour (OpenCV) | Low | 60+ fps | Centroid (x, y), blob area (rough depth), ellipse angle | Confused by red shirts or other red objects |
| MediaPipe Pose/Hands | Low | ~30 fps | Wrist, elbow, shoulder keypoints | Doesn't see the paddle itself |
| Custom YOLO model | High (hundreds of labelled images) | 20–60 fps | Robust paddle bounding box | Lots of work for a small gain |

**Decision: colour tracking and MediaPipe Pose together.**
- The red blob gives the paddle position.
- MediaPipe gives the wrist position (this covers the "pose recognition" requirement).
- Only accept a red blob within some distance of the detected wrist. This rejects background clutter without training.
- Fallback: train a small YOLO model only if lighting or clutter breaks colour tracking.

**Mapping to the virtual paddle:** normalise the blob centroid to screen coordinates and use blob area for depth. Add a short **calibration step** where the player holds the paddle in the corners of the play area.

---

## 2. Can an AprilTag on the Paddle Give the Angle and Spin?

**Partly. It gives a good angle when the paddle moves slowly, and fails during the swing, which is when spin matters.**

**Strengths**
- With a calibrated camera and a known tag size, `solvePnP` returns the full **6-DoF pose** (3D position plus orientation). That's an accurate face angle when the paddle is still or moving slowly.
- The tag ID also identifies the paddle or player.

**Weaknesses**
- **Motion blur:** a real swing at 30 fps smears the tag and detection drops out. This is the main problem.
- **Steep tilt:** detection becomes unreliable past about 60–70° away from the camera.
- **Distance:** the tag needs to be about 5 cm or larger to be detected a couple of metres from the camera.

**Decision: fuse the AprilTag with the IMU.**
- **AprilTag:** the absolute orientation reference whenever it's visible (ready position, between shots).
- **IMU gyro:** high-rate orientation and angular velocity during the swing, integrated from the last good AprilTag fix.
- Gyro drift over a roughly 300 ms swing is small, so a **complementary filter** is enough. A Kalman filter isn't needed at first.

---

## 3. Spin and Hit Strength

Physically, a hit depends on three things at the moment of contact:

1. **Paddle face normal** (orientation) → ball **direction**
2. **Paddle velocity along the normal** → ball **speed** (hit strength)
3. **Paddle velocity across the face** (brushing) → **spin**

| Quantity | Best source | Backup |
|---|---|---|
| Face angle | IMU orientation, corrected by the AprilTag | Fitted ellipse angle of the red blob |
| Swing speed / strength | Peak gyro angular velocity (rad/s) | Frame-to-frame blob velocity |
| Swing direction (up, down or sideways) | Gyro rotation axis plus blob motion | — |
| Spin type | Face tilt combined with swing direction | — |

**Warning:** don't double-integrate the accelerometer to get velocity, because it drifts within a fraction of a second. Use **peak angular velocity × estimated arm length** as the hit speed instead.

### Spin implementation, in two phases
- **Phase A (first): classify the swing.**
  - Upward brush with a closed face → **topspin**
  - Downward chop with an open face → **backspin**
  - Sideways swing → **sidespin**
  - Fast and flat → **smash**
  - Each class applies a preset spin to the virtual ball. This is robust and easy to tune.
- **Phase B (later): continuous physics.** Compute spin from the actual tangential velocity, and add **Magnus-force** curvature and a spin-dependent bounce.

### Hit detection rule
A hit counts when **all** of these are true:
1. The virtual ball reaches the paddle plane.
2. The paddle is within radius *r* of the ball (camera / location check).
3. The gyro magnitude goes over a threshold within about ±100 ms (IMU / swing check).

---

## 4. AprilTags for Game Control

Use separate printed cards, not the paddle tag:

| Tag ID | Meaning |
|---|---|
| 0 | Start / reset game |
| 1 | Level: Beginner (slow ball) |
| 2 | Level: Intermediate |
| 3 | Level: Pro (fast ball; later maybe more spin or AI aggressiveness) |
| (separate ID) | Paddle tag, never mistaken for a control card |

---

## 5. Bonus Idea: Haptic Feedback

The double motor is an **actuator** as well as a sensor:
- A short, sharp motor pulse on contact gives a "thock" you can feel.
- A stronger pulse for a smash, and a buzz on a miss.
- Mount it on the **handle** so the paddle's balance isn't too far off.

---

## 6. Proposed Architecture

```
 Webcam ──► OpenCV: red blob + AprilTags ─┐
        └─► MediaPipe Pose (wrist gate) ──┤
                                          ├─► Sensor fusion ─► Game engine ─► Renderer (pygame / browser)
 Double motor ── BLE ──► IMU stream ──────┘   (paddle pose,   (ball physics,      │
        ▲                                     speed, spin)    hit detection,      │
        └────────── haptic pulse ◄─────────────────────────── streak score) ──────┴─► Publish best streak (float)
                                                                                      to ME193/Rogers/Victor
```

**Latency budget:** camera about 33 ms per frame at 30 fps, plus BLE about 20–50 ms. Keep the hit-detection window tolerant of this.

---

## 7. Phased Plan

1. **Hardware check:** stream IMU data from the double motor and measure sample rate and BLE latency. *Everything else depends on this.*
2. **Vision:** red-blob tracking, MediaPipe wrist gate, camera calibration.
3. **AprilTags:** start and level cards, plus the paddle tag pose.
4. **Game loop:** 2D/2.5D table, ball physics, hit detection, streak counter.
5. **Score publishing:** send the best streak (float) to `ME193/Rogers/Victor` in real time.
6. **Swing classification:** strength plus spin types (Phase A), and haptic feedback.
7. **Polish:** continuous spin physics (Phase B), sound, difficulty tuning.

---

## 8. Open Questions (answered 2026-10-08)

1. **`ME193/Rogers/Victor`** → an **MQTT topic** to publish to. The broker is probably `test.mosquitto.org`, the same one `apriltags/mqttlib.py` uses (still to confirm).
2. **Double motor** → read from **Python** with the `legoeducation` package, like the AprilTags project. The package source confirms it exposes **raw accelerometer X/Y/Z, raw gyro X/Y/Z, and yaw/pitch/roll** (int16, units undocumented). The default stream rate is 10 Hz and the fastest is 15 ms (~66 Hz).
3. **Visual setup** → one webcam facing the player. The game screen with the virtual paddle is on the monitor in front of the player, with the **live webcam feed in the top-right corner**.
4. **Opponent** → a **CPU opponent**.
5. **Platform** → **Python, OpenCV, MediaPipe**.

**Working rule:** build incrementally, so every piece can be tested on its own before moving on.

---

## 9. Decision Log

| Date | Decision | Reason |
|---|---|---|
| 2026-10-08 | Brainstorm and plan before writing any code | User request |
| 2026-10-08 | Use a red paddle with colour tracking (no custom model at first) | Simple, fast, no training data needed |
| 2026-10-08 | Use MediaPipe Pose as a wrist gate for the red blob | Covers the pose requirement and rejects clutter |
| 2026-10-08 | Fuse the paddle AprilTag (absolute) with the IMU gyro (fast) for orientation | AprilTag fails under motion blur; gyro drifts over long periods |
| 2026-10-08 | Do discrete spin classes first, continuous Magnus physics later | Robust and tunable before going for realism |
| 2026-10-08 | Use separate AprilTag cards for start and levels | Avoids confusion with the paddle tag |
| 2026-10-08 | Confirmed: spin starts as classification; haptic feedback is in | User agreed |
| 2026-10-08 | CPU opponent; game on the monitor with the webcam feed in the top-right corner | User's visual setup |
| 2026-10-08 | Score goes over MQTT to `ME193/Rogers/Victor` | User answer |
| 2026-10-08 | Stream the IMU at 15 ms (fastest allowed) using a callback, not polling | The 100 ms default is too slow for swings |
| 2026-10-08 | Build the game screen and swing detection before camera tracking | Swing-to-hit timing is the biggest unknown; vision is well understood |
| 2026-10-08 | The mouse stands in for the paddle position until step 3 | Lets the game be played and tested without any vision code |
| 2026-10-08 | Every step gets unit tests (`python -m unittest`) plus a hardware/visual test | User request |
| 2026-10-08 | 2.5D look inspired by Wii Sports Resort table tennis | User request |
| 2026-10-08 | A swing starts above 600 deg/s; it is reported at its peak | From recordings: slow turns stay under 400, swings reach 900-1900 |
| 2026-10-08 | Levels set the game speed (slow motion); shots keep real physics | Slow flights under real gravity became 1.7 m lobs |
| 2026-10-08 | Hit = paddle within reach when the ball arrives AND a swing within +-0.2 s (real time) | Location + swing rule; the window absorbs Bluetooth delay |
| 2026-10-08 | Camera raised (0.85 m above the table, 19 deg tilt) | User wanted a slightly higher point of view |
| 2026-10-08 | Hybrid test mode `play.py --imu`: mouse = paddle position, IMU = swing timing + strength | Tests real swings in the game before camera tracking exists |
| 2026-10-11 | Paddle face colour (red/black) facing the camera decides forehand vs backhand; red = forehand by default | The rubber facing the camera is the one facing the ball; configurable with `--forehand` |
| 2026-10-11 | Search for the paddle only beyond each wrist (MediaPipe Pose wrist + elbow) | Rejects red/black clutter elsewhere; covers the pose requirement |
| 2026-10-11 | Red wins whenever present; black needs a larger, compact blob | Red is distinctive; black is easily confused with shadows, hair and clothes |
| 2026-10-11 | Vision runs in a background thread; the game reads the latest state | MediaPipe takes ~30 ms per frame; the game must stay at 60 fps |
| 2026-10-11 | AprilTags deferred | User request |
| 2026-10-12 | Colour thresholds come from calibration on the real paddle, saved to `vision_config.json` | Hard-coded HSV guesses missed the real red/black under real lighting |
| 2026-10-12 | Hold the last paddle-to-wrist offset when no face is seen; also search around last position | Removes position snapping and losses on a bent wrist |
| 2026-10-12 | One Euro filter instead of fixed EMA | One setting can't be both steady when still and responsive when swinging |
| 2026-10-12 | Track only the configured racket arm (default right) | Free-hand controller/motor must not be detected as the paddle |
| 2026-10-12 | Run pose on the raw (un-mirrored) frame and mirror only the coordinates | On a mirrored frame MediaPipe's left/right labels swap |
| 2026-10-12 | Cut the hand and gripped objects out of the colour search | Track the paddle head, not the hand/handle/motor |
| 2026-10-12 | Game tracking: red-only, x-only, no MediaPipe (`red_tracker.py`); forehand/backhand dropped from the game | User wanted faster tracking and a proper rally first; pose cost ~12 ms/frame and stole CPU from the game |
| 2026-10-12 | Faster pace: levels 0.75/0.95/1.15x, +3% ball speed per hit (max +30%), 0.9 s pause | User wanted a faster-paced game |

---

## 10. Progress Log

How to run all unit tests:  `python -m unittest -v`

### Step 1: Double motor connection and IMU recording ✅ (tested on hardware)
- `paddle_imu.py`: the `PaddleIMU` class. It connects at the fastest stream rate (15 ms), uses the library's notification callback, and has `drain()` so a game loop can pick up new samples each frame.
- `record_imu.py`: records N seconds to `recordings/imu_<time>.csv` and prints a summary.
- `plot_imu.py`: plots a recording, with detected swings marked in red.
- The first run failed with `VersionMismatchError`. Fixed by updating the motor's firmware at code.legoeducation.com.
- **Measured:** ~64 Hz with gaps up to ~90 ms. Accel ~1024 raw = 1 g. **Gyro in deg/s** (a 90 deg turn integrates to ~90), saturating near 2000. **Yaw/pitch/roll in tenths of a degree** (the "degrees" comment in lelib.py is wrong). Still: |gyro| < 10. Slow turns: 100-400. Swings: 900-1900 deg/s and 3-10 g.

### Step 2a: Swing detection ✅ (user confirmed: detects a good range of strengths)
- `swing.py`: `SwingDetector` with start 600 deg/s, end 300 deg/s, and a 0.25 s refractory period. It reports at the peak with a strength from 0 to 1 and the rotation axis (the axis will be used later for spin classes).
- `tests/test_swing.py`: 11 tests covering stillness, slow rotation, single and double swings, wobble, Bluetooth gaps, early reporting, and a regression test on a real recording (`tests/data/swings_4.csv`).
- **Hardware test:** `python live_swing.py`. Each swing should print exactly one SWING line; still or slow movement should print nothing.
- Known gap: the paddle swinging *back* after a swing also counts as a swing (opposite rotation axis). It's harmless for now; the axis can filter it out later.

### Step 2b: 2.5D game screen ✅ (user confirmed the game works, including `--imu` hybrid mode)
- `physics.py`: real table dimensions, gravity, bounces, `aim()` (always clears the net), and `predict_crossing()`.
- `camera.py`: perspective camera behind and above the player, and the reverse projection so the paddle sits under the mouse.
- `game.py`: rally rules. The CPU serves and always returns; the player hits by location + swing timing; streak and best are tracked; events are emitted for haptics and MQTT later.
- `play.py`: the pygame renderer (Wii Sports Resort style), with a webcam placeholder box in the top right.
- `tests/test_physics.py` and `tests/test_game.py`: 24 tests, including a scripted perfect player keeping a 10-hit rally at every level, each miss type, the edges of the timing window, and level speed.
- **Play test:** `python play.py`, then Enter, mouse, and Space/click. Then `python play.py --imu`.

### Step 3: Vision tracking + forehand/backhand (unit tested; live test pending), 2026-10-11
The paddle has a **red side and a black side**. The side facing the camera is the rubber facing the (virtual) ball, so it tells forehand from backhand. The default is red = forehand; change it with `--forehand black`.
- `paddle_vision.py`: pure image functions.
  - **Wrist gate:** search only a circle beyond each wrist, along the forearm (elbow to wrist), sized from forearm length. Red shirts or dark hair elsewhere in the frame are ignored.
  - **Colours:** HSV red mask (hue wraps around 0) and a black mask (dark and unsaturated), cleaned with a morphological open.
  - **Side decision:** red wins whenever there's enough of it (≥4% of the search circle), even in front of a dark jumper. Black needs ≥8% *and* a compact blob (solidity ≥0.8), because shadows and hair are also black. Edge-on gives no side.
  - `SideFilter` debounce: the side switches only when the new side is seen in 3 of the last 5 frames, and clears after 5 frames with no face. `Smoother` smooths position.
  - `image_to_game()`: maps the mirrored image (central 70% of width and height) to the game's hit plane (x −0.9..0.9 m, y 0.8..−0.1 m).
- `tracker.py`: `VisionTracker` runs the webcam (640×480) and MediaPipe Pose (VIDEO mode, `models/pose_landmarker_lite.task`, copied from HW1) in a background thread, so the game stays at 60 fps. The per-frame logic is in `TrackerCore.step()` so it can be tested without a camera. The image is mirrored, so moving right moves the paddle right. If the player leaves, the paddle holds its last position, and after 0.5 s the tracker reports "lost". Measured on this laptop: **~20 fps** with an empty room.
- `game.py`: stores the current `stroke`. Hits say "Nice forehand!" / "Nice backhand!", and the hit event carries the stroke (for spin classes later).
- `play.py --vision` (works with or without `--imu`):
  - The webcam feed with overlay is in the top-right box (now 4:3).
  - A FOREHAND (red) / BACKHAND (black) badge sits under the feed.
  - The on-screen paddle shows the rubber colour facing the camera, and fades when edge-on.
- `live_vision.py`: standalone camera test with overlay (forearm, yellow search circle, dot on the detected face, FOREHAND/BACKHAND). Press `m` to show the colour masks for checking lighting.
- Tests: `tests/test_paddle_vision.py` (17) and `tests/test_tracker.py` (6) use synthetic frames covering red, black, edge-on, no paddle, red shirt away from the hand, a dark streak (hair/shadow), red in front of a dark jumper, picking the paddle hand, mapping, debounce, smoothing, and dropout. `test_game.py` gained a stroke test. **59 tests total.**
- A bug the tests caught: at first, black could outvote red when a dark jumper filled the search circle. Fixed by letting red win whenever it's present.
- **Known limitations:** a red shirt directly behind the paddle reads as red, so avoid red clothing. Black detection is the most lighting-sensitive part; check with `live_vision.py` and the `m` key. The HSV thresholds are constants at the top of `paddle_vision.py`.
- **Live tests:** `python live_vision.py` (check both sides, edge-on, and walking out of view), then `python play.py --vision --imu --serial 0994`.

### Step 3b: Tracking fine-tuning (unit tested; calibration on real paddle pending), 2026-10-12
User feedback: tracking was "slightly glitchy" and didn't pick up the red and black faces well.

**Causes found in the code:**
1. The colour thresholds were fixed guesses, not measured from the real paddle and lighting.
2. Whenever a frame missed the face, the position snapped to the guessed point beyond the wrist, then snapped back. This was the main source of jumpiness.
3. Search only happened beyond the wrist along the forearm, so a bent wrist lost the paddle.
4. Fixed smoothing (EMA 0.5): jittery when still and laggy when swinging.

**Fixes:**
- **`VisionConfig` + `vision_config.json`:** every tunable number is in one dataclass, saved and loaded as JSON. `tracker.py`, `live_vision.py` and `play.py --vision` load it automatically; the defaults are used if the file is missing.
- **`calibrate_vision.py`:**
  - *Part 1, colour sampling:* hold the red side, then the black side, over a box; then hide the paddle for a background check.
    - Red: the hue range is computed with a +90 shift so a range that wraps around 0 works. Saturation and value floors are set at the 5th percentile minus 25.
    - Black: value and saturation ceilings are set at the 95th percentile plus a margin.
    - `tighten_for_background()` then raises the red saturation floor and lowers the black value ceiling until less than 1% of the room is marked, while keeping at least 70% of the paddle's own pixels.
  - *Part 2:* live tuning with 15 sliders (colours, minimum sizes, search area, noise filter, smoothing, exposure) next to the masks and live red/black %. `s` saves, `r` resamples.
- **Search around last frame's paddle** (`track_radius`), as well as beyond the wrist.
- **No more snapping:** when no face is visible, the position holds the last paddle-to-wrist offset.
- **Hand stickiness:** the tracker prefers the hand that held the paddle last frame.
- **One Euro filter** replaces the EMA: steady when still, low lag when swinging. It filters in pixels; `min_cutoff` and `beta` are sliders.
- **Manual exposure option** (`exposure`, 0 = auto): shorter exposure means less motion blur on swings. It depends on the webcam.
- **`live_vision.py`:** shows live red/black %, and `p` saves a snapshot (camera + masks) to `snapshots/` to share when asking for help.
- **Tests:** 12 new tests (71 total). They cover config save/load, the wrapping red hue, calibration of a dull red the defaults miss, the background check tightening black while keeping the paddle, a bent wrist found near the last position, no snapping, hand stickiness, and One Euro (jitter reduction, convergence, low lag on a 1500 px/s sweep). Mutation checks confirm the continuity tests fail when those features are removed.

**How to tune (order matters):**
1. Light the player from the front; avoid a bright window behind you.
2. Avoid red or black clothing on your paddle arm.
3. Run `python calibrate_vision.py`, do the three samples, then play-test with the sliders and press `s`.
4. Check with `python live_vision.py` (red/black % readout; `m` for masks).

User result: red/black distinction "looks good" after calibration ✅

### Step 3c: Racket arm only, track the paddle head not the hand (unit tested; live test pending), 2026-10-12
User request: limit tracking to the racket arm, and track the paddle rather than the controller or hand.
- **Racket arm only:** `racket_hand` in the config (`'right'` default, `'left'`, or `'auto'` for the old behaviour), also set by `--hand left|right` on `play.py`, `live_vision.py` and `calibrate_vision.py`, or by a slider. The other arm is never searched, so a motor or controller held in the free hand (hybrid `--imu` mode) can't be mistaken for the paddle.
- **Left/right fix:** pose used to run on the *mirrored* frame. MediaPipe can't tell a mirrored person, so it most likely swapped left and right. Pose now runs on the raw camera frame and only the coordinates are mirrored (`hands_from_landmarks(..., mirror=True)`), so "right" is the player's real right arm. If a webcam driver already mirrors its output the names would swap; the overlay labels the racket arm so this is easy to spot, and `--hand` fixes it.
- **Hand cut-out:** a circle covering the hand and whatever it grips (handle, Double Motor) is removed from the colour search (`hand_circle()`, size slider "hand cut-out %"). It spans from the wrist to the knuckles, which are now read from Pose landmarks 17-20 (index/pinky). Without knuckles it uses the first 20% of the forearm direction. Only the paddle head counts, and the reported position is the head's centre.
- **Overlay:** the racket arm is white and labelled "racket"; the other arm is grey; the hand cut-out is a grey circle.
- **Calibration hardening:** the red hue range is now capped at the paddle's median hue ±14 (+4 margin). The saved `vision_config.json` had red = hue 108..7, which includes purple and blue, probably from stray pixels in the sampling box. **Recalibrate** to get the tighter range.
- **Tests:** 81 total (+10).
  - Ignoring a red object in the free hand; left-handed players; racket arm hidden gives no observation.
  - A dark controller in the fist is not the black side (and *would* be without the cut-out); the position is the paddle head, not the hand, for red and for black with a dark controller in the fist.
  - Knuckle-based cut-out; mirrored coordinates keep the arm names; stray pixels don't widen red into purple or blue.
  - The synthetic paddle in the tests was moved to a realistic 0.8 forearm lengths beyond the wrist (it used to overlap the hand).
- **Note:** during testing the raw webcam delivered only 1 frame/s because another app held the camera (another Python process and Teams were running). Close other camera users before playing.

### Step 4: Fast red-only tracking + faster rally (unit tested + measured with the real webcam), 2026-10-12
User feedback: tracking in game was slow and clunky. Wanted faster tracking, a faster-paced game and a proper rally first; red side only; x-axis only; forehand/backhand can go.

**Measured causes:**
- The webcam is capped at 30 fps (MJPG/60 fps requests make no difference).
- MediaPipe Pose took ~12 ms per frame and competed with the game for the CPU.
- The game itself spent **10.6 ms** drawing each frame (of a 16.7 ms budget at 60 fps). Rounded panels and the paddle each allocated full-screen transparent surfaces, and fonts and outlined text were rebuilt every frame.

**Changes:**
- `red_tracker.py` (new; used by `play.py --vision`):
  - No pose detection. A red mask on a 320×240 frame, using the calibrated red thresholds from `vision_config.json`.
  - Searches near the last position first, then the whole frame; the biggest red blob wins and specks under 60 px are ignored.
  - One Euro filter on x only. Camera buffer set to 1 frame, so it always uses the newest frame.
  - **1.4–1.6 ms per frame; 30.7 fps (the webcam's max).**
  - A look-ahead option was tested and left at 0, because the filter already has ~no lag and look-ahead overshoots when the paddle stops.
- **X-only paddle:** the game takes `(x, None)`. Only left/right distance decides reach, and the paddle is drawn at the incoming ball's predicted height (`expected_y`, computed once per CPU shot).
- **Faster pace:**
  - Levels run at 0.75 / 0.95 / 1.15× speed (was 0.5 / 0.7 / 0.9).
  - CPU flight 0.55 s; player returns 0.40–0.65 s.
  - Pause after a miss 0.9 s (was 1.5 s).
  - **The rally speeds up +3% per consecutive hit (max +30%)**, and resets on a miss.
- **Rendering:** fonts and outlined text are cached, and the shadow and paddle surfaces are sized to their content. **Frame draw: 10.6 ms → 1.9 ms**; the game measures **59.9 fps** with real tracking.
- **Forehand/backhand removed from the game screen.** The camera box shows the tracking view; a chip asks to "show the RED side" only when tracking is lost. The pose-based modules (`paddle_vision.py`, `tracker.py`, `live_vision.py`, `calibrate_vision.py`) stay; calibration is still how the red thresholds are set.
- **Tests: 96 total.**
  - `test_red_tracker.py` (8): mapping, play-area edges, a fast sweep with <0.04 lag, re-finding after a big jump, specks ignored, holding then lost, using the red calibration, and <5 ms per frame.
  - `test_render.py` (3, headless): draw time under a third of a frame; x-only paddle lifted to ball height; mouse mode.
  - `test_game.py` (+4): rally speed-up, cap and reset; x-only hits ignore height but not left/right reach; paddle drawn at ball height.
- **Live tests:** `python live_red.py`, then `python play.py --vision --imu --serial 0994`.

### Step 5: MediaPipe Pose, without slowing the game (plan agreed 2026-10-12)
Assignment requirement: the game must use MediaPipe Pose information.
**Rule:** red tracking stays the fast path for paddle x. Pose runs beside it at a lower rate (MediaPipe `LIVE_STREAM` async mode on a small frame) and only feeds things that tolerate 10–15 Hz. The game never waits for pose.
Options considered: A) async `LIVE_STREAM` (chosen), B) separate process (fallback if fps drops), C) every Nth frame (adds 12 ms hiccups), D) smaller input (combined with A).
- **5a:** async pose + **skeleton overlay** in the camera box (feature 6). Guard: game stays ~60 fps and red tracking ~30 fps with pose on.
- **5b:** **racket-arm gate** (feature 1): only accept red near the racket wrist, which brings back protection against red shirts. **Player presence** (feature 3): auto-pause when the player leaves, resume when back.
- **5c:** **start gesture** (feature 4): raise both hands above your head to start.
- Later: forehand/backhand from body side (5), auto-fit the play area to the shoulders (2), paddle height from the wrist (7).

**Step 5 built (2026-10-12): features 6, 1, 3 and 4, built and tested in stages 5a → 5b → 5c.**
- `pose_async.py`: `AsyncPose` runs MediaPipe Pose in `LIVE_STREAM` mode on the 320×240 *un-mirrored* frame. It returns immediately; results arrive by callback, each stamped with its frame's capture time. The tracker uses the latest pose only if it's less than 0.5 s old.
- `pose_logic.py` (pure, testable):
  - `pose_from_landmarks` mirrors x and keeps the player's real body-side names.
  - `skeleton_lines` gives the line segments to draw.
  - `racket_gate`: a box around wrist + 0.8 × forearm; half-size is 1.6 × forearm length, with a minimum of 0.10.
  - `Presence`: both shoulders must be visible; present after 0.5 s, gone after 1.0 s.
  - `StartGesture`: both wrists above the nose for 0.6 s, firing once; it re-arms when the hands drop below the shoulders.
- `red_tracker.py`: when pose is fresh and the racket arm is visible, red is only searched inside the gate; otherwise it falls back to the old search. The overlay draws the skeleton (green) and the gate (yellow). New `RedState` fields: `pose`, `pose_hz`, `present`, `gate`, `start_count`.
- `game.py`: `pause()` / `resume()` freeze time, so the swing window and timers stay valid. Swings during a pause are ignored.
- `play.py --vision`: pauses when the player leaves and resumes when they're back, showing "Paused / step back into view". "Raise both hands" starts the game from the start screen.
- **Measured with the real webcam, all features on: game 60.5 fps, red tracker 29.8 fps (1.2 ms/frame), pose 29.8 Hz.** No slowdown; pose keeps up with the camera because of the small input.
- **Tests: 119 total** (+23 in `test_pose_logic.py`):
  - Mirroring/visibility and the skeleton.
  - The gate: a red shirt is ignored with pose, while without pose the shirt wins; red in the free hand is ignored; left-handed players; a stale pose falls back.
  - Presence: appearing, brief dropouts, leaving and returning, needing both shoulders.
  - The gesture: hold, quick wave, one hand, re-arm, no pose, the tracker's counter.
  - Game pause freezing the ball and ignoring swings.
- **Live test:** `python play.py --vision --imu --serial 0994`. Stand back so your shoulders and racket arm are in view; you should see the skeleton and the yellow gate box in the camera panel.

### Step 5d: Hand detected as red, paddle jumping between hand and paddle (fixed + unit tested), 2026-10-12
**Cause:** the saved red range (hue 165..11, saturation ≥70) included skin. Skin is orange-red: hue ~9-11, saturation 70-130. The red rubber sits at hue ~177-178 with saturation >150. When the paddle was tilted (little red visible), the hand was the bigger "red" blob, so the tracker jumped to it.
**Fixes (each works on its own):**
1. **Calibration learns skin.** `calibrate_vision.py` has a new step 2: hold your open hand over the box. `tighten_red_against_skin()` then repeatedly takes the better of two moves (trim the orange end of the hue range by 1, or raise the saturation floor by 5) until <1% of skin counts as red, keeping ≥70% of the paddle's red. With the user's settings and 4 skin tones: hue 165..11 → 165..5; skin counted as red 71% → 0%; paddle stays at 100%.
2. **Pose cuts out the racket hand.** `hand_cutout()` makes a circle from wrist to knuckles (new landmarks: index 19/20, pinky 17/18), or uses the forearm direction if knuckles aren't visible. It's removed from the red mask, so the hand can never be the paddle. Shown as a grey circle in the camera box.
**Tests:** `tests/test_skin.py` (9). It reproduces the bug (with the user's settings and no pose, the tracker locks onto the hand), then shows each fix keeps every frame on the paddle. Also: the skin step excludes 4 skin tones and keeps the paddle; good settings are left untouched; the cut-out covers the hand but doesn't reach the paddle. **128 tests total.**
**User action:** rerun `python calibrate_vision.py` (the hand step is new) and press `s` to save.

### Step 6: Paddle height fixed, backhand back (unit tested + measured), 2026-10-12
User: "plays well". Asked to fix the paddle randomly jumping up, keep it x-axis only, and bring back the backhand: show "Backhand" and turn the paddle black, with tracking unchanged.
- **Jump bug cause:** in x-only mode the paddle was drawn at each incoming shot's predicted crossing height (`expected_y`), recomputed on every CPU hit. It jumped by up to 0.12 m (~65 px) per shot, plus 0.25 → ~0.40 m on the first shot. **Fix:** a fixed height `X_ONLY_PADDLE_Y = 0.38` m (the typical crossing height); `expected_y` was removed. Gameplay is unchanged, since reach already used x only.
- **Backhand detection (`red_tracker.py`):** only with fresh pose, and only inside the racket-arm gate. If no red face is found there, it looks for a compact black blob (area ≥150 px at 320×240, solidity ≥ `min_black_solidity`, calibrated black thresholds). The hand cut-out and a new **forearm cut-out** (thick elbow→wrist line, for dark sleeves) are removed first. Red always wins over black. A `SideFilter` (3 of the last 5 frames) debounces the side. The black face's position keeps driving `paddle_x`, so tracking continues through a backhand. Without pose, black is never trusted.
- **Display (`play.py`):** the paddle is drawn black while `backhand` is on, with a "Backhand" label above it (a small pop when it first appears). Hits during a backhand say "Nice backhand!" (`game.stroke`). Forehand hits still say "Nice!"/"Smash!".
- **Tests: 138 total.**
  - `tests/test_backhand.py` (10): black = backhand and still tracked; follows a moving black face; red isn't backhand; switching needs a few frames and one stray frame doesn't flip it; a dark sleeve isn't backhand (fails without the forearm cut-out); a thin dark streak isn't backhand; black is ignored without pose; the calibrated black threshold is used; the paddle is drawn dark on backhand; the hit message.
  - `test_game.py`: the paddle height stays the same through a 6-hit rally (replaces the "ball height" test).
- **Real webcam:** tracker 30.7 fps, 0.7 ms/frame, pose 30.7 Hz.
- Fixed along the way: the cached fonts (step 4 speed-up) went stale after `pygame.quit()` (seen when test files re-initialise pygame). The `Renderer` now clears the font/text caches when it is created.
- **User feedback (2026-10-12):** forehand works well; backhand is "still a little temperamental". **Deferred: backhand reliability** (ideas: per-session black calibration under playing light, pose-based backhand from body side as a second signal, a lower debounce).

### Step 2c: Haptic feedback (unit tested + checked end to end with a fake motor; feel test pending), 2026-10-12
User chose haptics next; spin is on hold (low confidence it would feel right).
- `haptics.py`: patterns are lists of (direction, speed %, ms) sent as non-blocking `motor_run_for_time(..., motor=MOTOR_BOTH, blocking=False)` from a worker thread, so the game loop never waits. A forward-then-back kick feels sharper. Newest feedback wins: a waiting pattern is replaced, never queued up.
  - **hit:** 45 ms forward + 45 ms back; speed 50–100% from swing strength.
  - **smash** (strength >0.75): 3 × 60 ms at 100%.
  - **miss:** 110 ms buzz, 90 ms gap, 110 ms buzz.
  - Motor errors are printed, never raised.
- **The motor shakes its own IMU.** Without handling, a pulse reads as a fake swing. `SwingDetector.suppress_until(t)` plus `Haptics.quiet_until()` (pattern length + 0.12 s settle), converted to the IMU clock with `PaddleIMU.sample_time()`, make `play.py` ignore swings while the motor runs.
- **Game:** `play.py --imu` plays a pattern on each hit/smash/miss event (`pattern_for_event`). `--no-haptics` turns it off.
- **Tests (149 total):** `tests/test_haptics.py` (11): strength→pulse, patterns under 0.4 s, event mapping, `play()` returns in <10 ms, non-blocking BLE commands on both motors, step spacing, the quiet window, disabled mode, motor errors swallowed, and a motor shake after a hit is not a swing (2 swings without suppression, 1 with).
- **End-to-end check with a fake motor and IMU:** a scripted swing gives a hit → hit pulse sent; a missed ball → miss buzz sent; a simulated 900 deg/s motor shake never reached the game, and with suppression switched off it did (a fake swing).
- **Hardware test:** `python live_haptics.py` plays each pattern 3 times and prints the peak gyro during the pulse and how many fake swings it would have caused. For more rumble, fix an off-centre LEGO piece on one motor output.

**Question: would AprilTags on the paddle give orientation precise enough for spin?** Mostly no; see the answer in chat. Summary: the tag blurs out exactly during the swing (30 fps webcam, the paddle moves several m/s at contact) and would cover the red face we track. The IMU on the paddle already measures face tilt (from gravity) and the swing's rotation axis at 64 Hz. If spin is attempted later, use IMU-only classification checked against labelled recordings first.

### Haptics simplified (2026-10-12)
User: no need for different feedback per event. Every hit/smash/miss now plays one pattern, `buzz` (the former miss pattern: 110 ms at 45%, 90 ms gap, 110 ms). Swing suppression is unchanged. Tests updated (148 at that point).

### Spin experiment: hit-type detection from the IMU (separate tool, forehand/red only; unit tested, real data pending), 2026-10-12
User wants to *test* whether spin is detectable before touching the game. Types: **topspin** (closed face, brush up), **backspin** (open face, chop down), **flat** (drive).
- `strokes.py` (pure):
  - **Window:** 0.35 s before to 0.15 s after each swing peak.
  - **17 features:** rotation axis at the peak; face tilt from gravity before the swing; axis·gravity; acceleration direction at the peak; pitch/roll change (wrap-safe); peak speed; duration; total rotation.
  - `StrokeCollector` cuts windows from a live stream and skips swings within 0.6 s of the previous one (the paddle swinging back).
  - **Nearest-centroid classifier** on standardised features (simple on purpose: ~10 swings per type). Confidence = margin between the closest and second-closest type.
  - **Leave-one-out evaluation**, so the accuracy isn't flattered, plus a confusion table and per-feature separation scores.
- `record_strokes.py`: guided recording (Enter, then N swings per type; keep or redo) → `recordings/strokes/strokes_<time>.csv`. Train uses every file, so data can be added over sessions.
- `train_strokes.py`: prints the honest accuracy vs chance, the confusion table, the 5 most useful measurements, and a verdict (≥85% reliable / ≥65% promising / below that, not usable). Saves `models/stroke_model.json`.
- `live_strokes.py`: prints the detected type and confidence for each swing. Not connected to the game.
- **Tests: 159 total** (`tests/test_strokes.py`, 11): window centring/rejection, angle wrap, features reflect tilt and axis, one window per stroke with return swings ignored (fails without the cool-down), honest LOO ≥90% on separable synthetic data, **shuffled labels score near chance (no leakage)**, confidence and model save/load, useful features ranked, CSV round trip across files, `train_strokes` end to end.
- Checked on the real recording `tests/data/swings_4.csv`: 3 stroke windows (the swing-back skipped), ~30 samples each, all features finite.
- **Next:** the user records ~10 of each type, then runs `train_strokes.py`. Only if it's reliable should spin go into the game.

### Step 7: Spin in the game (unit tested + checked end to end; play test pending), 2026-10-12
**Experiment result (user's data):** 60 forehand swings (20 topspin, 20 backspin, 20 flat), **leave-one-out accuracy 100%** vs 33% chance. The most separating features were `axis_z`, `axis_y` and `d_pitch`. Caveat: each type was recorded as one block, so `live_strokes.py` with mixed strokes is the real confirmation.
**Shorter window:** accuracy on the same data with the window ending 150 / 100 / 50 / 0 ms after the peak was 100 / 100 / 97 / 95%. `FEATURE_POST_S = 0.10` keeps 100% and knows the spin 50 ms sooner. The model was retrained on that window.
**User's spec:** topspin takes a lower arc and gets to the CPU faster; a chop has a similar arc but goes less far and slower.
- **Physics:** each `Ball` has its own effective gravity (spin as a vertical force) and a bounce factor on forward speed. `aim()` takes `g`, and skips the net check for a ball already past the net.
- **`SPINS` table in `game.py`** (all tunable):

  | Spin | Flight time | Gravity | Depth | Bounce | Message |
  |---|---|---|---|---|---|
  | topspin | ×0.75 | ×1.4 | +0.15 m | ×1.15 (kicks on) | "Topspin!" |
  | backspin | ×1.25 | ×0.65 | −0.45 m | ×0.6 (checks up) | "Chop!" |
  | flat | unchanged | unchanged | unchanged | unchanged | unchanged |
- **Measured in game (Beginner, 10 rallies):**

  | | Time to CPU | Arc top | Lands at |
  |---|---|---|---|
  | flat | 0.89 s | 0.56 m | 2.25 m |
  | topspin | 0.62 s | 0.49 m | 2.43 m |
  | chop | 1.68 s | 0.56 m | 1.79 m |
- **Timing:** the hit launches flat right away. `LiveStrokeClassifier` classifies the swing 0.10 s after its peak, then `Game.apply_spin()` re-aims the shot from its current position with the remaining flight time. A classification arriving before the hit (early swing, ≤0.4 s) is used by the hit; one later than 0.35 s after the hit is ignored. Only one spin per shot. **Backhand shots get no spin** (the model is forehand-only).
- **CPU:** returns everything flat and resets the ball's spin. It also takes a chop that bounced on its side and drops below 0.12 m, so a short dying chop can't stall the rally.
- **Display:** "Topspin!" (orange) / "Chop!" (light blue) pop-ups, and a coloured trail behind a spinning ball.
- **`play.py --imu`** turns spin on automatically when `models/stroke_model.json` exists; `--no-spin` turns it off.
- **Tests: 178 total** (`tests/test_spin.py`, 19):
  - Physics: aim lands under any gravity; topspin is lower and faster; chop has a similar arc and is shorter and slower; bounce factors; every spin clears the net.
  - Game rules: spin after the hit reshapes the shot; messages and events; flat unchanged; early classification used; stale or late ignored; one spin per shot; backhand gets none; CPU resets spin; CPU takes a dying chop (fails without the rule); a 15-hit rally with random spins at every level.
  - Live classifier: fires once, at peak + 0.10 s; makes no guess without history.
  - Drawing: the trail renders.
- **End-to-end trace (fake IMU streaming a synthetic topspin swing into `play.py`):** "HIT (launched flat)" at 1.24 s → "classified topspin → ball spin topspin, 'Topspin!', 0.12 s after the hit".

### Webcam selection (tested), 2026-10-12
User asked to find and use another webcam.
- **Windows devices:** "Full HD webcam" (USB VID_1BCF), "Integrated Camera" and "Integrated IR Camera" (laptop), plus virtual cameras "Camera (NVIDIA Broadcast)" and "OBS Virtual Camera". The NVIDIA virtual camera prints pages of `[DSH]` log lines whenever it's opened.
- **OpenCV indices after plugging in the USB webcam** (the numbering shifted):

  | Index | Resolution | fps | Brightness | Verdict |
  |---|---|---|---|---|
  | 0 | 640×480 | 10 | ~9 | dark |
  | 1 | 720p max | 30 | ~2 | black (built-in camera, cover likely closed) |
  | **2** | **640×480 (up to 1080p)** | **30** | **127–142** | **the only live camera: the USB webcam** |
  | 3 | 640×480 | 1 | 0 | virtual |
- `find_camera.py`: probes indices 0–5 with driver log output suppressed. A camera is **live** if brightness ≥15 and ≥15 fps; `pick_camera` prefers the higher frame rate, then resolution. It saves the choice as `camera` in `vision_config.json` (`--save`; `--preview` shows each live camera).
- **The camera is now a setting** (`VisionConfig.camera`). `play.py`, `live_red.py`, `calibrate_vision.py` and `live_vision.py` use it unless `--camera N` is given. **Saved: camera 2.**
- **Real check on camera 2:** game 60.4 fps, tracker 30.4 fps (1.6 ms), pose 30.4 Hz, person detected.
- **Tests: 184 total** (`tests/test_find_camera.py`, 6): picks camera 2 from this laptop's real readings; dark or slow cameras aren't live; faster beats sharper; none when nothing is live; setting round trip; the tracker uses the saved camera and treats 0 as a real choice.
- **To do (user):** recalibrate red, skin and black on the new camera with `python calibrate_vision.py`, since colours differ between cameras.

### Step 8: Forehand Drill, a separate smash / chop practice mini-game (unit tested + end to end; play test pending), 2026-10-12
User idea: forehands only, camera skewed to the right, a smash/chop practice game instead of a full match. **Separate game**; `play.py` is unchanged.
- **Camera:** `camera.py` gained `yaw_deg` (and `center_shift`). It builds full camera axes from yaw then pitch; with yaw 0 it's identical to before (all old tests pass). The drill camera is **over the forehand shoulder**: `pos=(0.25, 0.9, -1.7)`, pitch 18°, yaw +14°. The table runs diagonally to the upper-left and the hitting zone sits on the right of the screen. It's mirrored for `racket_hand: "left"`. Several angles were tried and rendered: a camera right of the table looking left put the action on the *left*; turning right left half the screen as empty floor.
- **`drill.py` (pure):** a ball machine at the far end, forehand side, feeds one ball at a time and calls **SMASH** or **CHOP**.
  - **Smash balls:** lofted, land mid-table, arrive at the paddle **0.68–0.75 m** high.
  - **Chop balls:** low, short, backspin, arrive **0.31–0.38 m** high. Every feed reaches the player at all levels; the time to the player shrinks with level.
  - **Hit rule:** same as the game (x reach plus a swing within ±0.2 s).
  - **Judging:** the stroke type comes from `set_stroke()` (IMU classifier, or keys). Smash = flat/topspin with strength ≥0.5; chop = backspin. No type within 0.35 s → judged flat. A type up to 0.4 s *before* the hit counts.
  - **Scoring:** right stroke = 100 + 20 × streak; wrong stroke = 0 and the streak resets ("That was a topspin!" / "Too soft!"); miss = streak resets ("No swing!" / "Too far!").
  - The return flies with that stroke's spin physics.
  - **Rounds:** 20 balls (`--balls`), 1 s gap between balls, results with score, accuracy, smash x/y, chop x/y and best streak. Pause and resume are supported.
- **`drill_play.py`:** `DrillRenderer` extends the game's `Renderer` (which now takes a `camera`, plus a `cam_box` position).
  - Ball machine drawn at the far end.
  - Big orange **SMASH!** / blue **CHOP!** call banner while the ball comes in.
  - Score, streak, ball count and level top-left; webcam feed bottom-left (only with `--vision`); green/red result pop-ups; ready and results screens.
  - **Inputs:** mouse plus Space (smash) / C (chop) / T (topspin) without the IMU; `--imu` gives swings, the classifier and haptics (with swing suppression); `--vision` gives red-tracker x, presence pause, and a hands-up start/replay.
- **Tests: 206 total.**
  - `tests/test_drill.py` (18): correctness rules; feeds on the forehand side, with smash balls clearly higher than chop balls; left-handed feeds; scoring and the streak bonus; correct chop flies with backspin; wrong stroke; soft smash; no type → flat; early type counts; late type ignored; no swing / too far; gap then next ball; the return leaves over the net; round ends with results; per-call accuracy; pause; a perfect player gets 100% at every level. Mutation checks: "every stroke correct" and "no judge timeout" are both caught.
  - `tests/test_drill_render.py` (4): every state draws; draw time under half a frame with the webcam feed; the forehand zone is on the right of the screen (mirrored for lefties); the mouse maps correctly through the skewed camera.
- **End-to-end runs of `drill_play.main()`:**
  - Keyboard mode: 6 balls, all correct, round over, 60 fps, score 900 (= 6 × 100 + 20 × 15).
  - **Fake IMU streaming synthetic swings + stroke classifier:** smash → flat ✓, chop → backspin ✓ (6/6), 12 haptic pulses.
- **Follow-up (2026-10-12): "tracking seems worse in the drill".** Both games use the **same** data (`models/stroke_model.json`, `vision_config.json`, the same camera) and the same `RedTracker`. The cause was the mapping: the webcam play area was still mapped onto the *whole table* (x −0.9..0.9), but drill balls only arrive at x +0.16..+0.58. So the paddle used just **17%** of the image width, each webcam pixel was 0.8 cm of paddle, and the closer drill camera magnified the jitter. **Fix:** `RedTracker(x_range=...)`; the drill maps the play area onto `FOREHAND_X_RANGE = (0.0, 0.75)` (mirrored for left-handers); the main game is unchanged. **Measured:** image use 17% → **40%**, 1 webcam px = 0.80 → **0.33 cm**, on-screen shake from ±2 px jitter 0.80 → **0.35 px** (2.4× less). Tests: `tests/test_drill_tracking.py` (5); 211 total. My first zone (−0.1..0.8) only gave 2.0×; the tests caught the over-claim before the zone was tightened.
- **Follow-up (2026-10-12): normal hits mixed in.** A third call, **HIT!** (green, "normal ball: a steady forehand"). Feed: target z 0.45–0.85, 0.6 s flight, no spin, like the CPU's rally balls. Correct = flat or topspin at **any** strength; a chop is wrong ("That was a chop!"); success message "Good hit!". The calls are mixed evenly (smash / hit / chop). Heights at the paddle: smash 0.68–0.75 m, **hit 0.40–0.50 m**, chop 0.31–0.38 m; all arrive at every level. Results add a Hit x/y row; keyboard **H** = normal hit. The paddle is drawn behind the results panel. Tests: three distinct height bands, an even mix (300 feeds), correctness for hit, hit judged right/wrong; **213 total**.

### Step 9: Drill results published over MQTT (unit tested + verified through the real broker), 2026-10-12
User: publish the final score and the number of hits/misses to the topic after a completed round.
- **Messages** (on `round_over`), via `mqttlib.py` (copied from the AprilTags project; paho-mqtt, test.mosquitto.org):

  | Topic | Payload |
  |---|---|
  | `ME193/Rogers/Victor` | final score as a **float**, e.g. `900.0` (keeps the original brief's format) |
  | `ME193/Rogers/Victor/hits` | balls returned, with any stroke |
  | `ME193/Rogers/Victor/misses` | balls missed |
- `score_publisher.py`: `ScorePublisher` connects lazily and publishes (qos 1) on a **background thread**, so the game never waits on the network. It reuses the connection and retries after a failure. Failures are printed, never raised. A broker that never answers counts as failed, not "sent" (mqttlib's connect doesn't raise, so the connected flag is checked). `close()` lets a just-finished round go out when quitting.
- `drill_play.py`: publishes on `round_over`. The results screen shows "Sending results..." / "Results sent to ME193/Rogers/Victor" / "Couldn't send results (offline?)". Options `--no-mqtt` and `--topic T`.
- **Tests (223 total):** `tests/test_score_publisher.py` (10): hit/miss counts; exact topics and payloads; float score; publishes in the background (a 0.2 s slow connect doesn't block, all qos 1); reuses the connection; offline is reported; a broker that never answers is a failure; retry after failure; disabled sends nothing; close flushes the pending round.
- **Verified through the real broker** (to a throwaway topic `littlepingpong-selftest/<random>`, so no test data hit the class topic): a full 6-ball `drill_play.main()` round (4 hits, 2 misses, score 440) → the subscriber received `440.0`, `4`, `2` on the three topics.
- `requirements.txt`: added paho-mqtt.

### Camera choice by kind: laptop vs USB (tested), 2026-10-13
User: switch back to the laptop webcam, or add an option to pick the integrated vs a connected camera.
- **The numbers shift.** With the USB webcam unplugged today: 0 = Integrated Camera, 1 = Pixel 10 Pro (Windows Virtual Camera, from the phone), 2 = NVIDIA Broadcast, 3 = OBS. Yesterday's "black 720p camera 1" was the phone's virtual camera, not the laptop camera.
- `camera_select.py`: names come from **cv2-enumerate-cameras** (installed; added to requirements). The `camera` setting can be **`'laptop'`** (built-in, never the IR camera), **`'usb'`** (a physical non-built-in webcam; **falls back to the laptop camera** with a note if none is plugged in), a number, or part of a name. Virtual cameras (no USB id, or "virtual/broadcast/obs" in the name) are never picked for `laptop`/`usb`. `pick()` resolves against what's plugged in when the game starts and prints the device chosen.
- `VisionConfig.camera` now defaults to `'laptop'`. **The user's setting is now `'laptop'`.** `--camera laptop|usb|N` works on `play.py`, `drill_play.py`, `live_red.py`, `calibrate_vision.py` and `live_vision.py`.
- `find_camera.py`: lists cameras by name (skips IR); saves the suggestion as `'laptop'`/`'usb'` rather than a number; **`--use laptop|usb|N`** switches instantly.
- **Checked live:** the drill on the saved `laptop` → Integrated Camera #0, game 60.3 fps, tracker 30.4 fps, person seen. `--camera usb` with the USB webcam unplugged → falls back to the laptop camera with a note, 60.0 / 29.6 fps.
- **Tests: 229 total** (`tests/test_camera_select.py`, 6, using today's and yesterday's real device lists): laptop found whatever its number; usb found; usb unplugged → laptop with a note; never the IR or a virtual camera; numbers and name parts; device kinds. The camera default test is updated to `'laptop'`.
- **Reminder:** colours differ per camera. Recalibrate (`calibrate_vision.py`) when switching cameras, or keep one calibration per camera in mind.

### Double Motor connection: always OUR motor (tested on hardware), 2026-10-13
User: "the double motor is not connecting properly".
- **Diagnosis:** the Bluetooth adapter (RZ616) and service were fine. A scan found **4 Double Motors advertising** (later a different set: 0994, 3664 and two without a card). Their names look like "🟩 0994 Double Motor". `connect()` without `--serial` takes **whichever motor answers first**, often someone else's or a cardless one. Connecting by serial to 0994 worked first time: 2.7 s, 62 Hz IMU.
- `motor_select.py`: parses advertised names (serial + card colour); `choose_serial(requested, saved, motors)`: `--serial` > saved (`motor.json`) > the only motor in range; with several motors and no serial it **refuses to guess** and lists them. `explain()` turns failures into advice (firmware → code.legoeducation.com; otherwise "switched on? close other programs using it, wait ~5 s").
- `PaddleIMU.connect()`: uses that choice, retries "not found / not ready" up to 3× with 1.5 s waits, never retries a firmware mismatch, and records `self.serial`. Every script (`play.py`, `drill_play.py`, `record_*`, `live_*`) benefits without changes.
- `find_motor.py`: lists motors in range (marks yours); `--use 0994` saves it; `--test` connects and checks the IMU rate. **Saved: 0994.** Real test: connected in 1.6 s, 61 Hz, OK.
- **Tests: 242 total** (`tests/test_motor_select.py`, 13, using the real advertised names): parsing; requested > saved; saved used; refuses to guess in a classroom (and never calls connect); a single motor is fine; none found; save/load; advice; retry then success; 3 failed attempts → advice; firmware not retried; unknown colour symbol.

### AprilTag difficulty: live test (unit tested; waiting on the live camera test), 2026-10-15
User: use AprilTags to set the difficulty. Build a live test that labels the tags "Normal", "Hard" and "Pro". How do you "train" it: compare against a JPG, or show the tag to the camera?
- **No training and no image matching.** Each tag's pattern encodes an ID number, which OpenCV's built-in detector reads straight from the frame (`cv2.aruco`, family **tag36h11**, the same family as `../apriltags/autopark.py`). The only setup is an **ID -> level table** in `level_tags.json`. The default is tag 0 = Normal (level 1), 1 = Hard (2), 2 = Pro (3). These map onto game.py's levels 1-3 (named Beginner/Intermediate/Pro there; the names aren't unified yet).
- "Teaching" it = hold a tag up in `live_level_tags.py` and press 1/2/3. That assigns the tag in view and saves it; each level keeps exactly one tag. `r` resets to the defaults.
- `level_tags.py`: `LevelTagDetector.detect(frame)` returns the tags in view, biggest (closest) first. **`LevelChooser`** only switches the level after the same level's tag has been seen for **5 frames in a row** (~0.2 s), so a glimpse or a misread can't change it. The chosen level stays after the tag is put away.
- `python live_level_tags.py --make-tags` writes printable tags `level_tags/tag{0,1,2}_{normal,hard,pro}.png`. Print them 5-10 cm wide and keep the white margin.
- **Tests: 253 total** (`tests/test_level_tags.py`, 11): each default tag is labelled correctly on synthetic frames; no tag; an unassigned tag; the biggest tag first; a rotated tag; table save/load and defaults; one tag per level; a level needs steady frames; flicker never switches it; the level stays after the tag is removed, with each change reported once; unassigned tags are ignored. The generated PNGs are also detected correctly.
- **Next (after the live test):** hook `LevelChooser` into play.py / drill_play.py (show a tag on the start screen to set the level).

### AprilTag level choice in the drill (unit tested; waiting on the real-camera test), 2026-10-15
User: the tags are printed; check for AprilTags before the drill starts.
- **Levels renamed** in game.py to **Normal / Hard / Pro** to match the printed tags (they were Beginner / Intermediate / Pro). `level_tags.LEVEL_NAMES` now comes from `game.LEVELS`.
- `RedTracker`: new `tag_detector` and `watch_tags` attributes. While `watch_tags` is on, the tracker thread also reads tags from the **full 640x480 frame** (~3 ms, so they can be read from further away than the 320x240 tracking frame allows). It puts them in `RedState.tags` and outlines them, mirrored, in the webcam preview: green with the level name for a level tag, grey for others. With `watch_tags` off, `tags` is None and costs nothing.
- `drill_play.py --vision`: `watch_tags` is on only on the **start and results screens** (never during a round). Each new webcam frame's tags go through `choose_level(drill, chooser, tags)`. A tag held steady for 5 frames sets the level. The screen shows "Level: Normal (show a tag: Normal / Hard / Pro)", an orange bar while a tag is held, and a green "Level: HARD" flash when the level changes. The HUD level badge updates too. Enter or both hands up still starts the round; showing a tag first is optional. The 1/2/3 keys still work, and a tag works again after a key change (the chooser syncs to the drill's level).
- **Tests: 260 total.** `tests/test_level_tags.py` now has 19: progress only counts towards a change; a steady tag sets the drill level; a tag for the current level does nothing; a tag works again after a keyboard change; a real tag image end to end (detect -> Hard); preview outlines scaled and mirrored. `tests/test_drill_render.py` adds 1: the tag prompt draws on the start and results screens. A screenshot of both screens was checked. A smoke test of the tracker thread with a fake camera showing tag 2: no tags while off, ['Pro'] while on, none after turning off again.

### Drill difficulty = ball speed + time between serves (unit tested), 2026-10-15
User: the tags work great. Now the difficulty should set how fast all the incoming balls are and how much time there is between serves.
- `drill.py`: a new **`DRILL_PACE`** table, used by the drill only. It replaces the shared `game.LEVELS` time_scale and the fixed `FEED_GAP_S = 1.0`:

  | Level | Ball speed (every feed and return) | Gap between serves | Feed -> paddle (measured) |
  |---|---|---|---|
  | Normal | 75% | 1.8 s | hit/chop 1.03-1.22 s, smash 1.43-1.65 s |
  | Hard | 100% | 1.0 s | 0.78-0.92 s, smash 1.07-1.23 s |
  | Pro | 130% | 0.5 s | 0.60-0.70 s, smash 0.83-0.95 s |

  `Drill.pace` gives the current level's Pace. `speed` comes from it, and the gap is read when a ball ends, so a tag or key change during a round's gap applies from the next ball. Reach (`reach_m`) still comes from game.LEVELS (30 / 25 / 20 cm). The main game (play.py) is unchanged.
- `drill_play.py`: under the level line, the start and results screens show "ball speed 130%   0.5 s between serves". This also shows in keyboard mode ("1 / 2 / 3 to change").
- **Tests: 265 total.** test_drill: the gap is checked at each level (1.8 / 1.0 / 0.5 s); speeds rise and gaps shrink with level; the same ball (same seed) arrives at least 0.15 s / 0.12 s sooner each level up; reaction time stays between 0.55 s and 1.8 s; returns also fly 1.4x+ faster on Pro; a level change in the gap applies to the next ball. "A perfect player gets 100% at every level" still passes at Pro speed. Screenshots of the start and results screens were checked.
