"""All tunable settings in one place. No heavy imports here (the policy tests use this)."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"          # data/<person>/<label>/*.png ; folders starting with "_" are not data
CALIB_DIR = DATA_DIR / "_calibration"
CALIB_PATH = CALIB_DIR / "calibration.json"
BACKGROUND_PATH = CALIB_DIR / "background.png"
AUDIT_DIR = DATA_DIR / "_audit"
TEST_DIR_PATH = DATA_DIR / "_test"   # separate recording sessions, never used for training
MODELS_DIR = ROOT / "models"
MODEL_PATH = MODELS_DIR / "finger_net.pt"
CODE_PATH = MODELS_DIR / "code.json"


def load_calibration():
    """Calibration written by `python -m src.calibrate`, or None."""
    if CALIB_PATH.exists():
        return json.loads(CALIB_PATH.read_text())
    return None


# ---- camera / image ---------------------------------------------------------
FRAME_W, FRAME_H = 640, 480       # same on PC and UnoQ
# "border": the model sees only the inside of the tape border (hand bigger, background fixed).
# "frame": the model sees the whole camera frame (old setup; the border was only a guide).
CROP_MODE = "border"
# Non-square crops are padded to a square (edge pixels repeated) instead of stretched,
# so finger shapes keep their true proportions.
PAD_TO_SQUARE = True
# Strips of border tape along the crop edges (paper nudged after calibration) are cut off.
TRIM_TAPE_EDGES = True
EXPOSURE_TARGET = 205             # locked-exposure brightness of the empty white paper
IMG_SIZE = 64
IMG_CHANNELS = 3                  # 3 = color (RGB); 1 = grayscale (old models)
CLASSES = [0, 1, 2, 3, 4, 5]      # 0 = fist / empty square

# ---- border tape detection --------------------------------------------------
# Blue painter's tape, measured on this camera (exposure -4): HSV median (109, 178, 195).
# Skin is hue ~5-8 and the white paper has sat ~34, so neither can be mistaken for the tape.
# (The function names in quality.py still say "red" - historic; they detect this tape.)
TAPE_H_MIN, TAPE_H_MAX = 98, 125  # OpenCV hue 0-179
TAPE_S_MIN, TAPE_V_MIN = 100, 70
MIN_SQUARE_AREA_FRAC = 0.05       # square must cover >= 5% of the frame
RED_CLEAN_FRAC = 0.001            # crop is "clean" when < 0.1% of it is tape
ROI_EXTRA_MARGIN = 4              # extra px inside the border
MAX_TILT_DEG = 5.0
MOVE_IOU_MIN = 0.90               # square moved if IoU with the calibrated one drops below this

# ---- calibration acceptance -------------------------------------------------
MIN_ROI_SIDE = 100
BG_MEAN_RANGE = (120, 250)
BG_STD_MAX = 25.0                 # shadows / glare
BG_DRIFT_MAX = 6.0                # brightness change over ~1 s with nothing moving
OPEN_HAND_FRAC_RANGE = (0.10, 0.70)
EXPOSURE_SHIFT_MAX = 15.0         # background brightness change when a hand enters

# ---- per-image quality gates ------------------------------------------------
HAND_DIFF_THRESH = 30             # gray-level difference from the empty background
HAND_SAT_THRESH = 35              # saturation difference (catches light skin on white)
HAND_MAX_FRAC = 0.95              # big hands fill most of this crop; only reject a fully covered view
EDGE_CONTACT_MAX = 0.03           # hand touching a non-arm edge over 3% of its length = cut off
REJECT_CUT_OFF_FINGERS = False    # lenient collection: fingers leaving the view only give a tip
# hand present = largest connected hand region >= this fraction of the crop. Measured on real
# data: real hands >= 0.14, lighting-noise "hands" on empty paper <= 0.023.
DEFAULT_HAND_FRAC_MIN = 0.06      # replaced by the calibrated value
DEFAULT_SHARPNESS_MIN = 10.0      # replaced by the calibrated value (lenient: only really blurry)
DARK_MEAN_MIN = 70
SATURATED_MAX = 0.50              # fraction of pixels at 253+
LIGHT_DRIFT_MAX = 40.0            # visible paper next to a big hand is partly shadow
BAND_FRAC = 0.06                  # outer band of the crop checked for red
BAND_RED_MAX = 0.02
DUP_DIFF = 0.8                    # mean abs diff of 16x16 thumbnails (small moves = new picture)
MIN_SAVE_INTERVAL_S = 0.2

# ---- automatic guided collection --------------------------------------------
AUTO_PER_CLASS = 30               # pictures per finger count per person
AUTO_EXTRA_THREE = 0              # balanced classes (extra 3s made the model guess "3")
AUTO_TEST_PER_CLASS = 10          # pictures per count in a separate TEST session
AUTO_BURST = 4                    # pictures per prompt before switching to another count
AUTO_MIN_CHANGE = 2.5             # thumbnail difference needed between pictures (move more)
AUTO_EMPTY_MAX = 5                # empty-area pictures allowed in class 0 (the rest are fists)
MOTION_MAX = 5.0                  # mean gray change between frames to count as "still" (lenient)
STEADY_FRAMES = 2                 # still this many frames in a row before a picture
AUTO_INTERVAL_S = 0.2             # min time between pictures
SWITCH_PAUSE_S = 2.0              # pause after "switch!" so the change itself isn't captured
# only these block a picture; anything else (out of view, lighting, camera moved) is a tip
AUTO_BLOCKING = ("no hand", "blurry", "too dark", "FIST", "covered")

# ---- audit ------------------------------------------------------------------
MIN_TOTAL_PER_CLASS = 30
IMBALANCE_MAX = 1.6
DUP_WARN_FRAC = 0.10
FLAG_WARN_FRAC = 0.10

# ---- lock policy ------------------------------------------------------------
CODE_LEN = 3
CONF_THRESHOLD = 0.90             # min softmax confidence for a reading to count
STABLE_FRAMES = 10                # same digit this many frames in a row to accept
INPUT_TIMEOUT_S = 10.0            # no hand activity -> back to sleep
MAX_ATTEMPTS = 3
LOCKOUT_S = 30.0
READY_FLASH_S = 0.5               # all LEDs on after the button: "show your code"
DIGIT_FLASH_S = 0.5               # yellow flash per registered digit
RESULT_FLASH_S = 2.0              # red (wrong) / green (correct) flashing

# ---- UnoQ hardware (servo, 3 LEDs, button) ----------------------------------
UNOQ_PORT = 5005                  # TCP port of the hardware server running on the UnoQ
# Continuous-rotation servo (M5Stack 360): the pulse sets SPEED + DIRECTION, not an angle.
# 1500 us = stop; further from 1500 = faster. A quarter turn = spinning for the right TIME.
# Calibrate with:  python -m src.hw_test --host <IP> --servo   (saved to SERVO_CALIB_PATH and
# sent to the board every time the PC connects).
SERVO_STOP_US = 1500
SERVO_CALIB_PATH = CALIB_DIR / "servo.json"
# Physical stops at OPEN and CLOSED: every move spins a bit longer than needed (500 ms at
# speed 200) and ends against a stop, so drift cannot build up.
SERVO_DEFAULT = {"lock_us": 1300, "unlock_us": 1700, "lock_ms": 500, "unlock_ms": 500}
SERVO_NUDGE_MS = 60               # one small nudge during calibration


def load_servo_calibration():
    cal = dict(SERVO_DEFAULT)
    if SERVO_CALIB_PATH.exists():
        cal.update(json.loads(SERVO_CALIB_PATH.read_text()))
    return cal
HALL_SENSOR = False               # not installed yet: the door always reads "closed"
POLL_INTERVAL_S = 0.05            # how often the PC asks the board for button presses

# ---- Hall-effect door sensor ------------------------------------------------
DOOR_DEBOUNCE_S = 0.20            # raw door reading must hold this long to count
UNLOCK_WINDOW_S = None            # None = stays open until the button; a number = auto re-lock (s)
RELOCK_DELAY_S = 2.0              # door closed this long after opening -> re-lock

# ---- training ---------------------------------------------------------------
AUG_ROTATE = 15                   # random rotation (deg) during training
AUG_TRANSLATE = 0.10              # random shift (fraction of the image)
AUG_SCALE = (0.9, 1.1)            # random zoom
EPOCHS = 30
BATCH_SIZE = 32
LR = 1e-3
