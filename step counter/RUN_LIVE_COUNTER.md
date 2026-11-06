# Run the live step counter

Ankle strap only. The Uno Q streams raw samples over Wi-Fi (MQTT); the PyTorch network counts on your PC.

## One-time setup
1. PC: `pip install paho-mqtt torch numpy pandas matplotlib scipy`
2. Board app `model2_app/python/main.py`: set `MODE = "stream"` (and `LABEL = "walk"`). Check that
   `MQTT_BROKER` is the same as in `pc/live_step_counter.py` (default `broker.hivemq.com`, a public
   broker: anyone can read your sensor data; use your PC's IP with mosquitto to keep it private).
3. Copy/sync `model2_app` to the Uno Q as before.

## Each run
1. Start the PC window first: `cd pc` then `python live_step_counter.py`
2. Power the board with the strap on your LEFT ankle and start the app. The LED shows the walking figure
   when it is streaming; the window shows "waiting for samples..." until data arrives.
3. Walk. The count follows your steps about 0.65 s late. "Reset count" restarts at 0.
   "Save samples..." writes the received stream to `data/live_<time>.csv`.
4. The board also saves a backup CSV (`model2_app/data/S##_walk.csv`). Use it to check by hand count:
   count your steps and compare with the number on screen.

## No board? Test the window
`python live_step_counter.py --replay ../data/S21_walk.csv --speed 4`

## How the count works now
- Each left-ankle swing is a left step; a right step follows half a stride later (the strap cannot see the
  right leg). If you stop (no left swing for ~1.7 strides) the right step after your last left step is taken
  back, so the count can drop by 1 when you stop. If your right foot really does take a last step when you
  stop, run `python live_step_counter.py --keep-last-right`.
- The PyTorch network still runs and its count is shown in small print; `--net` makes it the main number.
- Don't do start/end stomps with this live counter: a stomp is a leg lift and counts as steps.

## Notes
- Swing counter: 2.1% mean error on the 5 recordings (walk parts, settings chosen on them); network alone about 14% (undercounts), see `results/uncertainty_report.md`.
- Grey window = no samples for 3 s (Wi-Fi or board stopped). "lost batches" counts dropped messages.
- The window and board code were tested in replay only; the MQTT path to the real board is untested.
