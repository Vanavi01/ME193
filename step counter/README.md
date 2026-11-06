# Neural-net step counter (ankle IMU + phone)

A PyTorch 1-D CNN that counts steps from an ankle-worn accelerometer/gyroscope (Arduino Uno Q + MPU6050), with a phone in the opposite pocket used for step labels and tested as extra input.

**Most accurate counter: the peak detector (`count_steps.py --method peak`; the live window uses a streaming version of it).** It counts leg swings in the ankle gyroscope (each swing = 2 steps) between the start and end stomps. Counts vs hand counts: S18 46 vs 47, S19 52 vs 51.5 (run with `--no-end-stomps`; that recording has none), S21 76 vs 76, S22 86 vs 79, S23 60 vs 57: **3.5% mean error, worst 9%**. Its settings were chosen while looking at these same recordings, and S18's truth is the detector's own labels, so a new walk is the real test.

**The neural network (`count_steps.py`'s default, `--method net`; the PyTorch part of the assignment) works only partly.** On recordings it had never seen, the network's step count was off by **13.9% on average (95% CI 8-23%)**. A plain peak detector was off by 2.2% and the phone's pedometer by 4.1%, so the network did not beat the simple methods. Adding phone sensors made it worse. Five recordings is too little data. Full numbers: [results/uncertainty_report.md](results/uncertainty_report.md).

| Held-out recording | Truth | Network |
|---|---|---|
| S18 walk | 47 (labels) | 52 |
| S19 fast walk | 51.5 | 48 |
| S21 walk | 76 | 70.5 |
| S22 stop-and-go | 79 | 89 |
| S23 stop-and-go | 57 | 39 |

## Live tests on new walks (never used for training)

Recorded with the board streaming to the PC (`RUN_LIVE_COUNTER.md`). Hand counts are walks out and back with a pause and turn between.

| Recording | Hand count | Swing counter (live window default) | Neural network |
|---|---|---|---|
| S24, walk 1 | 12 (unsure; the data looks like 14) | 14 | 10 |
| S24, walk 2 | 14 | 14 | 10 |
| S25 | 14 (7 out, 7 back) | 14 | 13.7 |
| S26 | 38 (19 out, 19 back) | 36 | 35.5 |

The swing counter was within 2 steps in every case except the uncertain S24 walk 1. It credits a right step after each left swing and takes one back at each pause, so which foot starts and ends a leg of the walk can shift the total by 1 or 2: the left-ankle sensor cannot see the right foot. The network undercounted on S24 (10 of 14) and was close on S25 and S26. These are a handful of walks, not a proper evaluation.

## Run the tracker

**Live (board streaming, PyTorch network):** see [RUN_LIVE_COUNTER.md](RUN_LIVE_COUNTER.md). `python pc/live_step_counter.py`

Window app (same pictograms and dark theme as the activity viewer in `../gait identifier/pc/mqtt_viewer.py`): `python pc/step_counter_app.py` (or give it a CSV). Open a recording, press Play: the pictogram steps and the big number counts up as the walk is replayed. Menu at the bottom switches between the peak detector and the neural network; tick "no end stomps" for S19.

Command line:

```
cd pc
python count_steps.py ../data/S21_walk.csv              # neural-network count, replayed in a window
python count_steps.py ../data/S21_walk.csv --method peak # peak-detector count instead
python count_steps.py ../data/S19_walk.csv --method peak --no-end-stomps   # recording without end stomps
python count_steps.py <your ankle CSV> --no-plot        # just print the count
```

The saved model is `pc/step_model.pt` (ankle-only, 5 networks). The five recordings in `data/` were all used to train it, so counting one of them is not a test (the program warns you); record a new walk and count it by hand for a fair check. It replays a recorded file; it is not a live feed from the board.

## What is where

| Path | What it is |
|---|---|
| `NEURAL_NET_EXPLANATION.md` | three-paragraph explanation of the network, PyTorch and back-propagation |
| `results/step_counter_explained.mp4` | explainer video (sensors, network, loss, back-propagation, results) |
| `results/` | accuracy tables and figures; `run1_all_channels/` and `run2_motion_channels/` keep the earlier input-set experiments |
| `pc/live_step_counter.py`, `pc/live_steps.py` | live window and the streaming network counter (ankle only) |
| `pc/step_counter_app.py` | window app: replays a recording and counts steps |
| `pc/count_steps.py` | the tracker (command line; peak detector or network) |
| `pc/train_steps.py` | trains the network; leave-one-recording-out evaluation |
| `pc/analyze_uncertainty.py` | uncertainty analysis and report |
| `pc/prepare_steps.py` | syncs ankle + phone, finds the walk, labels every step |
| `pc/make_video.py` | renders the explainer video |
| `pc/phone_receiver.py` | receives the phone's sensor stream (Sensor Logger app) |
| `model2_app/` | Uno Q app that logs ankle data (LED matrix guides the stomps and countdown) |
| `data/` | ankle CSVs (`S*_walk.csv`), `phone/` raw phone streams, `steps/` aligned and labeled data, `steps_log.csv` (hand counts) |
| `STEP_COUNTER_PLAN.md` | original plan and recording protocol |

## Pipeline

1. Record: `model2_app` logs the ankle at 100 Hz while `pc/phone_receiver.py` saves the phone. 3 stomps at the start and end of each recording line the two clocks up.
2. `python pc/prepare_steps.py` aligns the sensors and labels each step (leg-swing peaks), then checks against hand counts in `data/steps_log.csv`.
3. `python pc/train_steps.py --configs ankle --main ankle` trains and evaluates (about 20 minutes).
4. `python pc/analyze_uncertainty.py` writes the report.

## Limits

- 5 recordings, about 5 minutes, one walker; wide confidence intervals.
- The training labels come from the same peak detector that is the baseline, so the network learns to imitate it.
- The input set (ankle only) was chosen after seeing results on these same recordings, so the 13.9% is a little optimistic.
- Stop-and-go with long pauses is the worst case (S23: 39 counted against 57).

The earlier activity classifier (stand/walk/run/bike, no neural network) and its recordings S01-S16 live in the sibling folder `../gait identifier`. This folder holds only the step counter: the board app `model2_app` (`MODE = "stream"` for the live counter, `"log"` to record), the PC programs in `pc/`, the step-counter recordings S18-S26 in `data/`, and the results.
