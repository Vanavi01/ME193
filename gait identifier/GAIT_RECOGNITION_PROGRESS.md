# Ankle-Worn Activity Recognition — Project Framework

## Project goal

Build a wearable using the Uno Q and 5060 IMU to classify four activities:

1. Standing still
2. Walking
3. Running
4. Biking

The first placement to investigate is an ankle strap. It is convenient for collecting leg-motion patterns, though biking can be harder to distinguish from walking or running than the other classes. If the intended biking activity is air-biking while lying on the back, record and evaluate it explicitly; do not assume it has the same sensor pattern as riding a bicycle.

The assignment calls for two non-neural models:

- **Model 1 — single-axis peak:** uses only the peak acceleration from one selected accelerometer axis in each analysis window.
- **Model 2 — full IMU:** uses accelerometer and gyroscope outputs from all three axes.

No neural networks will be used.

## Activity definitions and working hypotheses

These are hypotheses to test with data, not guaranteed rules. Speed, individual movement style, strap orientation, terrain, and sensor placement can change the signals.

| Label | Operational definition for data collection | Expected ankle-IMU pattern |
|---|---|---|
| Standing still | Participant stands comfortably, feet mostly planted, for a marked interval. | Low movement overall; accelerometer mainly reflects gravity, with small noise or postural sway and little gyroscope activity. |
| Walking | Participant walks at a comfortable, steady pace on a level surface. | Repeating steps with moderate, fairly regular impacts and ankle rotation; usually a lower step cadence and impact intensity than running. |
| Running | Participant runs at a comfortable, steady pace on a level surface. | Repeating steps, often at a higher cadence and with stronger or sharper impact peaks and larger short-term changes than walking. |
| Biking | Participant pedals a bicycle at a steady, comfortable cadence. Record whether seated or standing, and whether riding indoors or outdoors. | Repeating pedal cycles and ankle rotation. The motion may be smoother and more continuously cyclic than foot strikes, but depends on technique and bicycle setup. |
| Air-biking (optional biking variant) | Participant lies on their back and performs a continuous bicycle-leg motion. | Cyclic leg motion without ground-contact impacts. It may be distinguishable from standing and foot-strike activities, but could overlap with bicycle riding or other leg exercises. Keep it as a separate sublabel in the raw data. |

**Labeling decision:** Keep `bike` and `air_bike` distinct in the collected data. If the assignment requires only four output classes, decide and document whether air-biking is included in the `biking` class or excluded from the final test. Report that choice rather than silently merging different protocols.

## Wearable and sensor setup

- Secure the board and IMU firmly to one ankle. Choose a consistent side and position, and record which one.
- Mark the sensor's axis directions relative to the leg/foot so the single-axis experiment is reproducible. Do not rotate or reposition the strap between train and test sessions without recording the change.
- Record timestamped accelerometer and gyroscope readings on all three axes. Preserve the original units and sensor configuration in the dataset metadata.
- Select and record a sampling rate the hardware can sustain reliably; around 50–100 Hz is a reasonable starting range for gait motion. Verify actual timestamps and dropped samples rather than assuming the configured rate was achieved.
- Record sensor range, units, sampling rate, axis mapping, strap location, participant/session ID, activity label, and protocol notes alongside each recording.
- Check for clipping/saturation, missing timestamps, and loose-strap artifacts before training.

## Data collection plan

1. Write down a repeatable protocol for each activity, including pace, duration, surface, rest periods, and biking posture.
2. Collect several separate recordings per activity, preferably across multiple sessions. Include more than one participant if feasible.
3. Capture a short still period at the beginning or end of each session to check the sensor and labels.
4. Use explicit start/stop markers or a synchronized event log. Trim transitions and setup periods so the training windows represent the intended activity.
5. Keep a balanced number of usable windows per class where possible. Record real-world variation (for example, different comfortable walking/running paces) and note it rather than mixing conditions unknowingly.
6. Save raw data unchanged. Create processed windows as a separate derived dataset so feature extraction can be repeated.

### Suggested recording log

| Session | Participant | Activity/subtype | Duration | Pace/setup | Strap side/location | Notes |
|---|---|---|---:|---|---|---|
| To fill in | To fill in | To fill in | To fill in | To fill in | To fill in | To fill in |

## Windowing and preprocessing

- Start with fixed windows of about 2 seconds, with 50% overlap. Treat this as an initial setting to compare, not a final requirement.
- Use the same window boundaries and preprocessing for both models. Do not include label transitions in windows.
- Check timestamp regularity. If resampling is necessary, document the method and apply it consistently.
- Avoid orientation-dependent assumptions unless orientation is calibrated and consistent. For Model 2, retaining all three axes makes the classifier less dependent on any one axis, but does not make it immune to strap rotation.
- Fit any scaling or preprocessing parameters using only the training split.

## Model 1 — peak acceleration on one axis

**Input:** exactly one value per window: the maximum acceleration on one preselected accelerometer axis. Select and document the axis before comparing results. Keep all other IMU channels and features out of this model.

**Classifier:** begin with a small decision tree using this single feature, or a documented set of threshold rules. A one-feature classifier is a deliberately simple baseline and may not separate all four activities well. Do not add cadence, gyroscope values, other axes, or additional statistics to this model.

**Questions to test:**

- Which axis is stable and interpretable with the chosen strap orientation?
- Does peak acceleration separate running from walking?
- Do standing and biking have overlapping peak values?
- How much does the result change across sessions and participants?

## Model 2 — all IMU outputs

**Input:** all six channels for each window: accelerometer X/Y/Z and gyroscope X/Y/Z.

Start with interpretable per-channel features such as mean, standard deviation, RMS, minimum/maximum, peak-to-peak range, and signal energy. Consider periodicity or cadence features only if they are computed consistently from the full IMU window and documented. Alternatively, use windowed samples directly with a classical classifier if the sample count and evaluation design support it.

**Classifiers to compare:** a decision tree or random forest is a practical first choice; a support vector machine is another non-neural option. Keep the comparison small and tune parameters using training/validation data only. Do not use neural networks.

## Evaluation plan

- Split by complete recording session (and preferably participant), not by randomly assigning overlapping windows. Random window splits can put near-identical samples in both training and test sets and inflate performance.
- Keep a held-out test set untouched until model choices are complete.
- Report a confusion matrix and per-class precision, recall, and F1, plus macro-F1 and overall accuracy. Macro-F1 helps show when a model performs poorly on a less frequent activity.
- Compare both models on the same held-out sessions. Include the class counts, window size/overlap, chosen axis for Model 1, feature list/classifier for Model 2, and split method.
- Inspect mistakes, especially walking vs. running and biking vs. air-biking. Do not rely on overall accuracy alone.
- Record a simple baseline (for example, majority-class prediction) so model results have context.

## Progress tracker

- [ ] Confirm the exact Uno Q / 5060 sensor interface, available IMU channels, units, and achievable sampling rate.
- [ ] Choose and document ankle side, strap position, and axis orientation.
- [ ] Implement timestamped raw IMU logging and an activity/session metadata format.
- [ ] Run a short pilot for all four required classes; separately pilot air-biking if it is in scope.
- [ ] Inspect signal quality, axis mapping, timestamps, and clipping; revise the collection protocol if needed.
- [ ] Collect labeled sessions with repeated trials and representative variation.
- [ ] Build consistent windowing and preprocessing.
- [ ] Train and evaluate the single-axis peak baseline.
- [ ] Train and evaluate the all-IMU classical model.
- [ ] Compare models using session/participant-separated evaluation and document errors.
- [ ] Record final limitations, results, and a short wearable demonstration.

## Current status and open decisions

- **Placement:** ankle strap (proposed).
- **Required labels:** standing still, walking, running, biking.
- **Air-biking:** retain as a separate raw-data sublabel; decide whether it is included in the required biking class.
- **Model restriction:** two non-neural models as specified above.
- **Scoring rubric:** not included in the project description provided so far; map the final report and demo to the rubric once available.
- **Progress:** framework drafted; hardware verification, data collection, modeling, and evaluation remain to be done.

## Model 1 build log

- **Hardware:** Cytron SN-MPU6050 (I2C 0x68), read from the sketch with raw Wire register access. Ranges: accel +/-16 g, gyro +/-1000 dps, 100 Hz, DLPF ~44 Hz.
- **Code:** `unoq_app/sketch/sketch.ino` (MCU, streams samples via Bridge), `unoq_app/python/main.py` (`MODE = "log"` records CSV, `MODE = "live"` classifies with `model1.json`), `pc/train_model1.py` (peak feature, axis comparison, leave-one-session-out eval, exports `model1.json`).
- **Status:** code written, not yet run on hardware.
- [x] IMU responds on `Wire` (header SDA/SCL, 0x68), confirmed with imu_test.
- [ ] Verify logger timestamps are about 10 ms apart.
- [ ] Record at least 3 sessions per class (60 s each, unique SESSION ids).
- [ ] Run `train_model1.py`, record the chosen axis, confusion matrix and macro-F1 here.
- [ ] Copy `model1.json` to `unoq_app/python/`, switch to `MODE = "live"`, and test on the board.

### Model 1 results (data as of 2026-10-28)

- Label and session now come from the CSV **file name** (`S15_stand.csv`), not the label column inside; the training script prints a note when they disagree.
- Data: stand 3 sessions / 162 windows, walk 3 / 162, bike (air-bike) 4 sessions but only 2 full / 113, run 3 sessions but only 2 usable / 28 (short, 15 s each).
- Chosen axis: **ay** (macro-F1 0.86 leave-one-session-out; ax 0.61, az 0.50). Rule: peak |ay| <= 0.375 g stand, <= 0.991 g bike, <= 3.016 g walk, above run.
- Held-out result: accuracy 0.86, macro-F1 0.86. Stand 162/162, bike 95/113, run 23/28, walk 122/162. Main error: 38 walking windows called bike (single-axis overlap), the motivation for Model 2.
- Caveats: run data is thin; strap orientation changed between sessions (sign of ax gravity flips); several recordings (S04, S10-S12, S16 run) were cut off early; run peaks reach ~14.6 g against the +/-16 g sensor limit.
- `model1.json` copied to `unoq_app/python/` for live mode.
- [ ] Live test on the board (`MODE = "live"`), including untethered.
- [ ] More run data (45-60 s sessions), one more air-bike session, consistent strap mounting.

## Model 2 build log (all six IMU channels)

- **App:** separate app in `model2_app/` (same streaming sketch; `python/main.py` has log + live modes, `features.py`, `model_runtime.py`, `model2.json`). Pure Python on the board, no extra libraries. Live mode shows the class as a digit on the LED matrix (legend printed in the console).
- **Features (26 per 2 s window, hop 1 s):** per channel std / peak excursion / mean absolute deviation, acceleration and gyro magnitude mean/std/max, and autocorrelation strength + lag (step/pedal rhythm). Raw channel means are excluded on purpose: gravity's sign flipped between strap mountings, so means would encode orientation, not activity.
- **Training:** `pc/train_model2.py` compares decision tree, random forest and SVM with leave-one-session-out, scores Model 1 on the same folds, and exports the better of tree/forest (SVM is compared but not deployed).
- **Results (460 windows, 10 sessions; recordings under 3 windows skipped):** Model 1 macro-F1 0.857 / acc 0.865; tree 0.742 / 0.787; forest 0.902 / 0.885; SVM 0.939 / 0.978. Deployed: random forest (30 trees, depth 5). Exported runtime matches sklearn on 100% of windows.
- **Errors are concentrated in whole sessions:** the forest calls 43 of 55 windows of walk session S03 "bike" (S03 was strapped on differently / walked faster); SVM misses 9 of 26 run windows. The model is limited by session variety more than by feature choice.
- [ ] Live test of Model 2 on the board.
- [ ] More data with varied strap mounting and pace (especially run and walk) and re-run both training scripts.
