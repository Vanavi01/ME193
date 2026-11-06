# Neural-Net Step Counter — Plan

## Assignment

Extend the ankle walker to count steps with a neural net built in PyTorch, using every sensor available. The writeup needs:

- GitHub link, reflection questions, clean layout
- A working step counter (video demo) **with an uncertainty analysis** (how good is it?)
- A Claude-made video/animation of how the whole system works: sensors → neural net → PyTorch training → back-propagation
- At most three paragraphs explaining how the neural net works

## Sensors

| Source | Placement | Channels | Rate | Path to PC |
|---|---|---|---|---|
| Uno Q + MPU6050 | Ankle strap, **left** leg (keep the existing mounting, and mark it) | accel xyz (±16 g), gyro xyz (±1000 dps) | 100 Hz | CSV logged on the board (`model2_app` log mode) |
| Phone (Sensor Logger app) | Front pocket, **right** leg, screen facing out, top up | accelerometer, gyroscope, gravity, orientation, magnetometer, barometer, pedometer | 100 Hz requested | HTTP push over Wi-Fi to `pc/phone_receiver.py` |

Why both: one ankle only sees its own foot, so each stride it sees is two steps. The phone on the other leg sees the opposite foot's steps, so together they cover every step. The phone's built-in pedometer is **not** ground truth. It is a commercial baseline we compare against in the uncertainty analysis.

Magnetometer, orientation, and barometer are logged because the assignment says "every sensor". Whether they help gets tested with an ablation (train with and without them) rather than assumed.

## Ground truth (the step labels)

The existing `data/*.csv` files have activity labels but no step counts, so they cannot train or score a step counter by themselves.

For every new recording:

1. **True step count:** count steps with a hand tally counter (or a counting app on a second device), or film the feet from a laptop/friend's phone and count later. Film at least the sessions used for the demo and the test set.
2. **Step times (optional, better):** if filmed, mark each heel strike in the video and save the times. Per-step labels let the net learn *when* steps happen, not just how many.
3. Write the count in `data/steps_log.csv`: `session,activity,true_steps,counted_by,notes`.

## Synchronizing the two sensors

The board's clock is milliseconds since boot and the phone's is wall-clock time, so they don't line up on their own. At the start **and** end of every recording, do **3 hard stomps** with both feet. The stomps show up as sharp spikes in both streams. Cross-correlating acceleration magnitude around the spikes gives the time offset, and doing it at both ends also corrects clock drift. Then resample both streams onto one 100 Hz timeline.

## Data to collect

The data set is four recordings (changed 2026-11-03 from the original 25–30 min plan):

| Session | Activity (`steps_log.csv`) | Walking time |
|---|---|---|
| 1 | `walk_slow` | 30 s |
| 2 | `walk_normal` | 30 s |
| 3 | `walk_fast` | 30 s |
| 4 | `stop_go` (walk, stand 5–10 s, walk, ...) | 30 s or more |

Each recording follows this protocol: 3 hard stomps, stand still for about 2 s, walk, stand still for about 2 s, 3 hard stomps. The board's LED matrix guides the timeline (`START_PHASE_S`, `WALK_S`, `END_PHASE_S` in `model2_app/python/main.py`, 52 s total): a **border** means stomp (start, then end), and a **number** is the seconds left until the end stomps, so walk until it reaches 1, then stand still and stomp. Start the phone (`python phone_receiver.py S17`, using the board's session number) **before** the board's countdown ends, and stop it after the board finishes. The tally count covers walking steps only, not stomps.

Zero-step negatives: the existing stand and air-bike CSVs (ankle only).

With only 4 recordings (about 2 min and 200–250 steps), testing is leave-one-recording-out, and the confidence intervals will be wide. Say this in the uncertainty analysis.

## Interpreting the recordings (`pc/prepare_steps.py`)

`python pc/prepare_steps.py` reads `data/steps_log.csv` (`session,activity,true_steps,offset_s,walk_start_s,walk_end_s,notes`). For each session it:
1. Syncs the phone to the ankle by cross-correlating acceleration spikes, with a separate fit at the start and the end to correct drift.
2. Finds the walk between the stomps.
3. Labels each left step as a swing peak in the ankle gyro, and each right step as a swing peak in the phone gyro (or the midpoint between left steps when there is no phone data). The sides are set by `ANKLE_FOOT`/`PHONE_FOOT` at the top of the script.
4. Compares the labeled count with the tally count and with the phone pedometer.

Output goes to `data/steps/<session>.csv` (100 Hz: all channels, `in_walk`, `step`, `foot`), `<session>.png` (check this file), and `summary.csv`. If the plot shows a bad sync or bad walk boundaries, fill in the override columns. Tested on synthetic recordings: exact counts and correct offsets.

## Model

- **Input:** 2 s windows (200 samples) of all aligned channels (6 ankle + phone channels), normalized using training-set statistics only.
- **Network:** small 1-D CNN (a few conv layers → pooling → linear). Train two versions and compare them:
  - **A — count regression:** output = number of steps in the window; sum non-overlapping windows for the total.
  - **B — step-event detection (preferred if step times exist):** output = per-sample probability that a heel strike happens here; count peaks above a threshold.
- **Training:** PyTorch, Adam, MSE (A) or binary cross-entropy (B). Leave-one-recording-out or a fixed held-out test set.
- **Baselines:** a classic peak detector on the ankle accelerometer, and the phone's own pedometer.

## Uncertainty analysis

- Per test recording: predicted vs. true steps, signed error, and % error.
- Mean absolute % error with a **95% confidence interval** (bootstrap over recordings).
- Breakdown by activity and speed (where does it fail?).
- Model uncertainty: train 5 models with different seeds (an ensemble) or use MC dropout, and report the spread of the predicted count.
- Ablation: ankle-only vs. phone-only vs. both vs. both + magnetometer/barometer.
- Same metrics for the peak-detector and phone-pedometer baselines.

## Live demo

The board streams to the PC (or logs while the phone pushes live), and the PC runs the trained net and shows a running step count next to the true count. Film it with a tally counter in frame.

## Deliverables checklist

- [ ] Put the project in a git repo and push it to GitHub (it is not a repo yet)
- [x] Phone streaming working (`pc/phone_receiver.py`)
- [x] Sync + merge + step-labeling script (`pc/prepare_steps.py`)
- [x] Data collection with true step counts (5 recordings: S18, S19, S21, S22, S23)
- [x] PyTorch training script + saved model (`pc/train_steps.py`, `pc/step_model.pt`)
- [x] Uncertainty analysis figures/tables (`results/uncertainty_report.md`)
- [~] Step tracker program (`pc/count_steps.py`, replays a recording; no live feed from the board); demo video still to film
- [x] Animation: sensors → windows → CNN → loss → backprop → weight update
- [~] Three-paragraph explanation done (`NEURAL_NET_EXPLANATION.md`); reflection questions still to write
- [~] README.md written; writeup layout still to finish

## Progress log

- **2026-11-03:** Plan written. PC has torch 2.14.1, numpy, pandas, scipy, matplotlib. Starting with phone streaming.
- **2026-11-03:** Data set reduced to 3 walks at different speeds (30 s each) plus 1 stop-and-go. Wrote `pc/prepare_steps.py` to sync, find the walk, and label steps.
- **2026-11-04:** Network trained and evaluated. Final result: ankle-only network 13.9% mean error on held-out recordings (95% CI 8-23%), worse than the peak detector (2.2%) and the phone pedometer (4.1%); adding phone channels made it worse. Finalized as a negative result.
