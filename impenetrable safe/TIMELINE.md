# One-Day Prototype Timeline (~11.5 h)

Scope: the tasks listed in the assignment paste (PyTorch finger-counting lock, movie, fun/cool feature, 4 report sections, template questions, Notion/GitHub page). Task 1 (V-data fit) is not part of this plan, and there are no augmentation plots.

**Fun/cool feature = Hall-effect door sensor** (forced-entry detection + auto re-lock). The logic is already written and unit-tested in `src/lock_policy.py`; the day's work is wiring it up and demoing it.

**Already done:** model, train/collect scripts, live PC demo with simulated button/door, lock + breach state machine (11 passing tests). Training a ~11k-parameter model on a few hundred small images takes minutes, so the real time goes to data collection, wiring and the UnoQ.

Times are clock-hours from start. **SW** = Claude + me; **HW** = me, in parallel.

| Time | SW (Claude + me) | HW (me, in parallel) |
|---|---|---|
| 0:00–0:30 | Mount webcam at 9 in, draw red square, calibrate `ROI`, run `collect.py` to check it. **Also test whether PyTorch installs on the UnoQ (5 min check); this is the biggest risk.** | Parts check |
| 0:30–1:30 | Collect my images: ~100 per class (0–5), burst-saving, extra 3s | Wire 3 LEDs + button on the UnoQ |
| 1:30–2:15 | First train; check the confusion matrix; fix obvious label/crop problems | Wire servo + Hall sensor |
| 2:15–3:00 | Collect 1–2 other people's hands (15 min each); retrain with one person held out | Mount magnet + Hall sensor on the box; check pole and gap |
| 3:00–3:45 | Tune the 3s (more data, epochs, widths); live test with `run_lock` (keys `b` and `d` simulate button and door); set the code; tune confidence and frames | Write/test the MCU sketch: button, LEDs, servo, Hall sensor |
| 3:45–4:30 | Write `UnoQIO` (button, Hall sensor, LEDs, servo) | Finish the sketch |

**Milestone A (~4:30): model counts fingers live on the PC; the real hardware responds from the sketch.**

| Time | SW | HW |
|---|---|---|
| 4:30–6:00 | PC webcam + model + policy driving the real servo/LEDs/Hall sensor via the UnoQ. Test: correct code unlocks; wrong code stays locked; forced open → alarm; valid unlock → no alarm; auto re-lock; disarm | Mechanical fixes (latch binding, sensor placement) |
| 6:00–7:30 | Move to the UnoQ: install PyTorch or export to ONNX; same webcam via hub; same-image check (PC vs UnoQ predictions match) | |
| 7:30–8:15 | Run the full lock standalone on the UnoQ; measure PC vs UnoQ: inference time, FPS, accuracy, setup difficulty | |

**Milestone B (~8:15): box opens only for my code, on the UnoQ; forced entry triggers the alarm.**

| Time | Task |
|---|---|
| 8:15–9:45 | **Stop-action movie** (~90 min): storyboard (width/depth, conv, pooling, ReLU, softmax/cross-entropy, backprop, gradient descent, then the Hall sensor tripwire); shoot with props or paper cards; I record the voice-over; quick edit. Claude can generate the diagrams/feature maps from the real model to cut down on filming. |
| 9:45–11:30 | **Notion + GitHub (~105 min):** Claude drafts the intro, the 4 sections (lock policy, ≥4 limitations, how the net works → the movie, UnoQ vs computer using the measurements above) and the template questions; I edit and add my own voice; embed the demo + movie; push the repo with a short README. |

---

## Hour budget

| Block | Hours |
|---|---|
| Camera, data, training, PC live test | ~3.75 |
| UnoQ I/O + integration with real hardware | ~2.25 |
| Moving to the UnoQ + comparison | ~2.25 |
| Movie | ~1.5 |
| Notion + GitHub + report | ~1.75 |
| **Total** | **~11.5** (the HW track runs in parallel, inside the first 6 h) |

## Where this can slip (and what to cut)

- **PyTorch won't install on the UnoQ:** switch to ONNX Runtime (adds up to ~1 h). Test this in the first 30 minutes so there's no surprise at hour 6.
- **The 3s stay confused:** collect extra 3s rather than tuning; more data beats more tuning.
- **Hall sensor/magnet misaligned:** the pole orientation is the usual culprit; flip the magnet before debugging software.
- **If behind schedule, cut:** the buzzer, tuning of the re-lock/unlock timers, and the second other person's data. Don't cut the UnoQ run or the voice-over.
- **Writing time is short** because Claude drafts it; reserve time for you to read it and add your own words.
