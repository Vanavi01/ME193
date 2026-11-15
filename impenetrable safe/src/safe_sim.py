"""The safe on screen: the real lock logic + the trained model.

  python -m src.safe_sim                     # webcam + model, simulated hardware
  python -m src.safe_sim --keys              # no camera: type 1-5 for digits (test the lock flow)
  python -m src.safe_sim --host <UnoQ IP>    # webcam + model + the REAL servo/LEDs/button
                                             # (the screen mirrors the hardware)

Top: red / yellow / green LEDs (yellow flash = digit registered, red flashing = wrong code,
green flashing = correct code). Left: the guard (closed door = locked, opening door = unlocked).
Keys: SPACE or b = button | d = open/close door (Hall sensor stand-in) | q / ESC = quit
      --keys mode: 1-5 = show that many fingers, 0 = fist
"""
import argparse
import time

import cv2
import numpy as np

from . import code_store, config
from .lock_policy import LockPolicy, State

W, H = 1040, 720
TOP, BOTTOM = 120, 44
LEFT_W = 440
BANNER = 62                       # state text under the guard photo
BG, PANEL, TEXT, DIM = (32, 26, 22), (48, 40, 34), (235, 235, 235), (140, 140, 140)
LED_COLORS = {"red": (60, 60, 235), "yellow": (40, 210, 245), "green": (90, 210, 60)}
STATE_TEXT = {State.SLEEP: ("LOCKED", (60, 60, 235)), State.LISTEN: ("SHOW YOUR CODE", (40, 210, 245)),
              State.UNLOCKED: ("UNLOCKED", (90, 210, 60)), State.LOCKOUT: ("LOCKED OUT", (60, 60, 235)),
              State.BREACH: ("BREACH!", (60, 60, 235))}
FONT, BOLD = cv2.FONT_HERSHEY_SIMPLEX, cv2.FONT_HERSHEY_DUPLEX
ASSETS = config.ROOT / "assets"


def load_img(name):
    img = cv2.imread(str(ASSETS / name))
    if img is None:
        raise SystemExit(f"missing image {ASSETS / name}")
    return img


def fit(img, w, h, bg=PANEL):
    """Scale to fit inside w x h, centered on a background."""
    s = min(w / img.shape[1], h / img.shape[0])
    r = cv2.resize(img, (int(img.shape[1] * s), int(img.shape[0] * s)), interpolation=cv2.INTER_CUBIC)
    out = np.full((h, w, 3), bg, np.uint8)
    y, x = (h - r.shape[0]) // 2, (w - r.shape[1]) // 2
    out[y:y + r.shape[0], x:x + r.shape[1]] = r
    return out


def text_c(img, s, cx, y, scale, color, thick=2, font=BOLD):
    (tw, _), _ = cv2.getTextSize(s, font, scale, thick)
    cv2.putText(img, s, (int(cx - tw / 2), y), font, scale, color, thick, cv2.LINE_AA)


def led_is_lit(mode, now):
    return mode == "on" or (mode == "fast" and int(now * 8) % 2 == 0)     # 4 Hz blink


def draw_leds(canvas, out, now):
    cv2.rectangle(canvas, (0, 0), (W, TOP), (20, 16, 14), -1)
    glow = np.zeros((TOP, W, 3), np.uint8)              # LED strip only (cheap on the UnoQ)
    names = [("red", out.red if out else "off"), ("yellow", out.yellow if out else "off"),
             ("green", out.green if out else "off")]
    for i, (name, mode) in enumerate(names):
        cx, cy = W // 2 + (i - 1) * 170, 52
        c = LED_COLORS[name]
        lit = led_is_lit(mode, now)
        cv2.circle(canvas, (cx, cy), 34, (70, 70, 70), -1, cv2.LINE_AA)               # bezel
        if lit:
            cv2.circle(glow, (cx, cy), 46, c, -1, cv2.LINE_AA)
            cv2.circle(canvas, (cx, cy), 28, c, -1, cv2.LINE_AA)
            cv2.circle(canvas, (cx - 9, cy - 9), 8, (255, 255, 255), -1, cv2.LINE_AA)  # shine
        else:
            cv2.circle(canvas, (cx, cy), 28, tuple(int(v * 0.25) for v in c), -1, cv2.LINE_AA)
        text_c(canvas, name.upper(), cx, 108, 0.5, TEXT if lit else DIM, 1, FONT)
    glow = cv2.GaussianBlur(glow, (0, 0), 14)
    canvas[:TOP] = cv2.add(canvas[:TOP], (glow * 0.8).astype(np.uint8))


def draw_guard(canvas, out, guards, now):
    x0, y0, w, h = 0, TOP, LEFT_W, H - TOP - BOTTOM
    locked = out is None or out.servo_locked
    canvas[y0:y0 + h - BANNER, x0:x0 + w] = guards[0 if locked else 1]
    label, color = STATE_TEXT[out.state if out else State.SLEEP]
    if out and out.state is State.LOCKOUT:
        label += f" {out.lockout_left:.0f}s"
    cv2.rectangle(canvas, (x0, y0 + h - BANNER), (x0 + w, y0 + h), (20, 16, 14), -1)
    text_c(canvas, label, x0 + w // 2, y0 + h - 20, 1.1, color, 2)
    icon = "LOCK CLOSED" if locked else "LOCK OPEN"
    cv2.rectangle(canvas, (x0 + 10, y0 + 10), (x0 + 190, y0 + 44), (20, 16, 14), -1)
    cv2.putText(canvas, icon, (x0 + 20, y0 + 35), FONT, 0.65,
                (60, 60, 235) if locked else (90, 210, 60), 2, cv2.LINE_AA)


def draw_right(canvas, out, policy, cam_view, reading, now, keys_mode):
    x0, y0 = LEFT_W, TOP
    w, h = W - LEFT_W, H - TOP - BOTTOM
    cv2.rectangle(canvas, (x0, y0), (x0 + w, y0 + h), BG, -1)
    cx = x0 + w // 2

    # camera / model view
    vw, vh = 360, 240
    vx, vy = cx - vw // 2, y0 + 20
    if cam_view is not None:
        canvas[vy:vy + vh, vx:vx + vw] = fit(cam_view, vw, vh, BG)
    else:
        cv2.rectangle(canvas, (vx, vy), (vx + vw, vy + vh), PANEL, -1)
        text_c(canvas, "keyboard mode", cx, vy + vh // 2 - 10, 0.8, DIM, 2)
        text_c(canvas, "press 1-5 = fingers, 0 = fist", cx, vy + vh // 2 + 25, 0.55, DIM, 1, FONT)
    cv2.rectangle(canvas, (vx, vy), (vx + vw, vy + vh), (90, 90, 90), 1)

    # current reading + confidence + hold progress
    ry = vy + vh + 40
    if reading is not None:
        d, conf = reading
        ok = conf >= config.CONF_THRESHOLD
        cv2.putText(canvas, f"reading: {d}", (vx, ry), BOLD, 0.8, TEXT, 2, cv2.LINE_AA)
        bx, bw = vx + 175, vw - 175
        cv2.rectangle(canvas, (bx, ry - 18), (bx + bw, ry - 2), PANEL, -1)
        cv2.rectangle(canvas, (bx, ry - 18), (bx + int(bw * conf), ry - 2),
                      (90, 210, 60) if ok else (40, 140, 245), -1)
        cv2.putText(canvas, f"{conf:.0%}", (bx + bw - 48, ry - 4), FONT, 0.45, TEXT, 1, cv2.LINE_AA)
    if out and out.state is State.LISTEN and policy.cand:
        frac = min(1.0, policy.cand_n / config.STABLE_FRAMES)
        cv2.putText(canvas, "hold", (vx, ry + 30), FONT, 0.55, DIM, 1, cv2.LINE_AA)
        cv2.rectangle(canvas, (vx + 60, ry + 18), (vx + vw, ry + 32), PANEL, -1)
        cv2.rectangle(canvas, (vx + 60, ry + 18), (vx + 60 + int((vw - 60) * frac), ry + 32),
                      LED_COLORS["yellow"], -1)

    # code slots
    sy = ry + 100
    n = out.digits_entered if out else 0
    for i in range(config.CODE_LEN):
        sx = cx + (i - 1) * 95
        filled = i < n
        cv2.rectangle(canvas, (sx - 36, sy - 46), (sx + 36, sy + 26),
                      LED_COLORS["yellow"] if filled else (90, 90, 90), 2, cv2.LINE_AA)
        if filled:
            cv2.circle(canvas, (sx, sy - 10), 14, TEXT, -1, cv2.LINE_AA)      # hidden like a PIN
    # instructions
    tips = {State.SLEEP: "press SPACE (the button) to start",
            State.LISTEN: "show a digit, then a fist between digits",
            State.UNLOCKED: "open! press the button to lock",
            State.LOCKOUT: "too many wrong codes - wait",
            State.BREACH: "forced entry! SPACE + correct code to disarm"}
    text_c(canvas, tips[out.state if out else State.SLEEP], cx, sy + 75, 0.6, TEXT, 1, FONT)
    if out and out.alarm and led_is_lit("fast", now):
        cv2.rectangle(canvas, (x0 + 4, y0 + 4), (x0 + w - 4, y0 + h - 4), (60, 60, 235), 6)


class SafeApp:
    """Camera + model + lock logic + screen, one frame per step(). No window: safe_sim shows the
    frames in a window on the PC, safe_server streams them to a browser from the UnoQ."""

    def __init__(self, io=None, cap=None, predictor=None, calib=None, keys=False):
        self.io, self.cap, self.predictor, self.keys = io, cap, predictor, keys
        self.roi = tuple(calib["roi"]) if calib else None
        panel_h = H - TOP - BOTTOM - BANNER
        self.guards = [fit(load_img("guard.jpg"), LEFT_W, panel_h),
                       fit(load_img("guard_open.jpg"), LEFT_W, panel_h)]
        self.policy = LockPolicy(code_store.verify)
        self.door_open = False
        self._button = False
        self._fake, self._fake_until = None, 0.0
        self.out = None
        self.help = "SPACE/b = button   d = door open/close   q = quit"

    # inputs (keyboard, web page)
    def press_button(self):
        self._button = True

    def toggle_door(self):
        self.door_open = not self.door_open

    def fake_digit(self, d):
        self._fake, self._fake_until = d, time.monotonic() + 0.6

    def step(self):
        from .camera import read_frame
        now = time.monotonic()
        cam_view = reading = None
        if self.cap is not None:
            frame = read_frame(self.cap)
            reading = self.predictor.predict(frame)
            cam_view = frame.copy()
            bx, by, bw, bh = self.roi
            cv2.rectangle(cam_view, (bx, by), (bx + bw, by + bh), (0, 255, 255), 2)
        elif self._fake is not None:
            reading = (self._fake, 1.0) if now < self._fake_until else (0, 1.0)
            if now > self._fake_until + 0.4:
                self._fake = None
        policy = self.policy
        pred = reading if policy.state is State.LISTEN else None
        button, self._button = self._button, False
        if self.io is not None:
            button = self.io.read_button() or button          # real button
            if config.HALL_SENSOR:
                self.door_open = self.io.read_door_open()
        out = policy.tick(now, pred, button, self.door_open)
        out.lockout_left = max(0.0, policy.lockout_until - now)
        if self.io is not None:
            self.io.apply(out)                                 # servo + LEDs follow the screen
        if out.event:
            print(f"{time.strftime('%H:%M:%S')}  {out.event:12s} -> {out.state.name}", flush=True)
        self.out = out

        canvas = np.full((H, W, 3), BG, np.uint8)
        draw_leds(canvas, out, now)
        draw_guard(canvas, out, self.guards, now)
        draw_right(canvas, out, policy, cam_view, reading, now, self.keys)
        help_ = self.help + ("   1-5 = fingers   0 = fist" if self.keys else "")
        cv2.putText(canvas, help_, (14, H - 16), FONT, 0.5, DIM, 1, cv2.LINE_AA)
        if self.io is not None:
            ok = getattr(self.io, "connected", True)
            cv2.putText(canvas, "UnoQ: connected" if ok else "UnoQ: RECONNECTING...",
                        (W - 330, H - 16), FONT, 0.5, (90, 210, 60) if ok else (60, 60, 235),
                        1 if ok else 2, cv2.LINE_AA)
        cv2.putText(canvas, "door: " + ("OPEN" if self.door_open else "closed"), (W - 150, H - 16),
                    FONT, 0.5, TEXT, 1, cv2.LINE_AA)
        return canvas

    def close(self):
        if self.io is not None:
            self.io.close()                                    # leaves the real safe LOCKED
        if self.cap is not None:
            self.cap.release()


def connect_board(host):
    """Hardware server on the UnoQ (App Lab app): lock against the stop, then background I/O."""
    from .io_backend import ThreadedIO, UnoQNetIO
    first = UnoQNetIO(host)                                    # sends the servo calibration
    print(f"connected to the board at {host}; locking the latch (known position)...", flush=True)
    first.home_locked()                                        # full spin against the CLOSED stop
    time.sleep(config.load_servo_calibration()["lock_ms"] / 1000 + 0.3)
    return ThreadedIO(lambda: UnoQNetIO(host), first)


def open_vision(camera=None):
    from .camera import open_from_calibration
    from .infer import Predictor
    calib = config.load_calibration()
    if calib is None:
        raise SystemExit("No calibration. Run: python -m src.calibrate --quick")
    cap, _ = open_from_calibration(calib, camera)
    return cap, Predictor(), calib


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--keys", action="store_true", help="no camera: type digits on the keyboard")
    ap.add_argument("--camera", type=int)
    ap.add_argument("--host", help="UnoQ IP address: drive the real servo/LEDs, read the real button")
    args = ap.parse_args()
    if not config.CODE_PATH.exists():
        raise SystemExit("No code set yet. Run e.g.:  python -m src.code_store 2 5 3")
    io = connect_board(args.host) if args.host else None
    cap = predictor = calib = None
    if not args.keys:
        cap, predictor, calib = open_vision(args.camera)
    app = SafeApp(io, cap, predictor, calib, args.keys)
    cv2.namedWindow("impenetrable safe", cv2.WINDOW_AUTOSIZE)
    try:
        while True:
            cv2.imshow("impenetrable safe", app.step())
            key = cv2.waitKey(15 if cap is None else 1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key in (ord(" "), ord("b")):
                app.press_button()
            elif key == ord("d"):
                app.toggle_door()
            elif args.keys and key in [ord(str(i)) for i in range(6)]:
                app.fake_digit(key - ord("0"))
            if cv2.getWindowProperty("impenetrable safe", cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        app.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
