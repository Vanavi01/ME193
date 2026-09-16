# Gesture-Controlled LEGO Car

Drive a LEGO Education Double Motor with hand gestures read from a webcam.
Python sees your hands with MediaPipe and sends drive commands to the motor
over Bluetooth Low Energy.

## Hardware

- LEGO Education Double Motor (Connection Card serial `0999`)
- LEGO Education Color Sensor (crash protection - connects to the first
  advertising one, no Connection Card needed if only one is powered on)
- Any Mac or PC webcam

## Setup

Requires Python 3.11 or newer. The system Python on macOS is 3.9 and will not work.

```bash
python3 -m venv my_env
source my_env/bin/activate          # Windows: my_env\Scripts\activate
pip install --upgrade pip
pip install legoeducation opencv-python "mediapipe==0.10.35"
```

`lelib.py` and both `.task` model files are included in this repo.

## Running

```bash
python drive.py
```

Pick a camera index when prompted. Press `q` in the video window to quit.

## Gestures

| Gesture | Action |
|---|---|
| Both thumbs down | Forward |
| Both open palms | Backward |
| Right palm only | Turn right |
| Left palm only | Turn left |
| Both fists | Stop |
| Both pointing up | Faster |
| Both peace signs | Slower |

An unrecognized gesture holds the last command rather than stopping, so the
car keeps driving through the moment your hands change shape. If no hands are
visible for 1.5 seconds, the motors cut as a failsafe.

## Crash protection

A color sensor mounted facing forward/down watches for a color change (floor
color vs. wall/object color) as a stand-in for a proximity sensor. When it
trips, `FORWARD`/`LEFT`/`RIGHT` are forced to `STOP`, but `BACKWARD` still
works normally so you can back away from whatever tripped it - once the
sensor sees the floor color again, forward driving is re-enabled.