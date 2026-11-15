"""Check the UnoQ wiring from the PC, step by step (no camera/model needed).
  python -m src.hw_test --host <UnoQ IP>            # LEDs, servo lock/unlock, button
  python -m src.hw_test --host <UnoQ IP> --servo    # (re)calibrate the 360 servo first

The servo is CONTINUOUS rotation: it turns for a TIME, not to an angle. Calibration:
  1. nudge the latch to the OPEN position
  2. try closing with a spin time; not right -> it goes back to open, change the time, try again
  3. same for opening
The result is saved to data/_calibration/servo.json and sent to the board on every connect."""
import argparse
import json
import time

from . import config
from .io_backend import SAFE_STATE, UnoQNetIO

LED_STEPS = [
    ("RED LED (D5) on", dict(SAFE_STATE, r=1)),
    ("YELLOW LED (D6) on", dict(SAFE_STATE, y=1)),
    ("GREEN LED (D7) on", dict(SAFE_STATE, g=1)),
    ("all three LEDs blinking fast", dict(SAFE_STATE, y=2, g=2, r=2)),
]


def ask(prompt):
    return input(f"  {prompt} > ").strip()


def spin(io, pulse_us, ms):
    io.servo_run(pulse_us, ms)
    time.sleep(ms / 1000 + 0.3)                         # wait until it has stopped


def nudge_to(io, cal, where):
    """Small nudges until the latch is where it should be. a = toward OPEN, d = toward CLOSED."""
    print(f"\n  Nudge the latch to the {where} position:")
    print("    a = nudge toward OPEN   d = nudge toward CLOSED   (aaa / ddd = bigger)   "
          "Enter = it's there")
    while True:
        cmd = ask(f"nudge ({where})")
        if not cmd:
            return
        n = len(cmd)
        if set(cmd) == {"a"}:
            spin(io, cal["unlock_us"], config.SERVO_NUDGE_MS * n)
        elif set(cmd) == {"d"}:
            spin(io, cal["lock_us"], config.SERVO_NUDGE_MS * n)
        else:
            print("    type only a's or d's (or just Enter)")


def set_speed(cal, speed):
    """Same speed both ways, keeping the current directions."""
    speed = max(50, min(1000, speed))
    stop = config.SERVO_STOP_US
    cal["lock_us"] = stop + speed if cal["lock_us"] > stop else stop - speed
    cal["unlock_us"] = stop + speed if cal["unlock_us"] > stop else stop - speed


def tune(io, cal, which):
    """Try lock/unlock with the current time; adjust until the user is happy."""
    us_key, ms_key = f"{which}_us", f"{which}_ms"
    goal = "CLOSED" if which == "lock" else "OPEN"
    other = "unlock" if which == "lock" else "lock"
    print(f"\n  Tuning {which.upper()} (open -> closed)" if which == "lock" else
          "\n  Tuning UNLOCK (closed -> open)")
    print("    Enter = try   a number = new spin time in ms   + / - = 10% more / less")
    print("    f = flip direction   s N = speed N (50-1000, distance from stop)   y = done")
    while True:
        speed = abs(cal[us_key] - config.SERVO_STOP_US)
        cmd = ask(f"{which}: {cal[ms_key]} ms at speed {speed}")
        if cmd == "y":
            return
        if cmd == "":
            spin(io, cal[us_key], cal[ms_key])
            ok = ask(f"did it end exactly {goal}? [y/n]").lower().startswith("y")
            if ok:
                return
            # go back to the starting position for the next try
            print(f"    going back (reverse spin)...")
            spin(io, cal[f"{other}_us"], cal[ms_key])
            nudge_to(io, cal, "OPEN" if which == "lock" else "CLOSED")
        elif cmd.isdigit():
            cal[ms_key] = max(20, int(cmd))
        elif cmd in ("+", "-"):
            cal[ms_key] = max(20, round(cal[ms_key] * (1.1 if cmd == "+" else 0.9)))
        elif cmd == "f":
            cal["lock_us"], cal["unlock_us"] = cal["unlock_us"], cal["lock_us"]
            print("    direction flipped (a/d nudges flip too)")
        elif cmd.startswith("s") and cmd[1:].strip().isdigit():
            set_speed(cal, int(cmd[1:]))
        else:
            print("    ?")


def calibrate_servo(io):
    cal = config.load_servo_calibration()
    print("\nSERVO CALIBRATION (continuous servo: it turns for a TIME, not to an angle)")
    print("  Tip: if nudges go the wrong way, use f in the tuning step to flip the direction.")
    nudge_to(io, cal, "OPEN")
    tune(io, cal, "lock")                   # ends CLOSED
    tune(io, cal, "unlock")                 # ends OPEN
    config.SERVO_CALIB_PATH.parent.mkdir(parents=True, exist_ok=True)
    config.SERVO_CALIB_PATH.write_text(json.dumps(cal, indent=2))
    print(f"\n  saved {config.SERVO_CALIB_PATH}: {cal}")
    io.send_servo_calibration(cal)
    io.servo_state(locked=False)            # the latch is OPEN now
    return cal


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", required=True)
    ap.add_argument("--servo", action="store_true", help="(re)calibrate the 360 servo")
    args = ap.parse_args()
    io = UnoQNetIO(args.host)                           # also sends the saved servo calibration
    print("connected to the UnoQ")
    results = {}
    try:
        for name, state in LED_STEPS:
            io.request({"cmd": "out", **state})
            results[name] = ask(f"{name}? [y/n]").lower().startswith("y")
        io.request({"cmd": "out", **SAFE_STATE})

        if args.servo or not config.SERVO_CALIB_PATH.exists() or \
                ask("calibrate the servo now? [y/N]").lower().startswith("y"):
            calibrate_servo(io)
            locked_now = False
        else:
            locked_now = ask("is the latch CLOSED right now? [Y/n]").lower() != "n"
            io.servo_state(locked=locked_now)

        # lock/unlock through the normal command (what the real safe does)
        order = [(0, "UNLOCK: did it turn back to OPEN"), (1, "LOCK: did it turn to CLOSED")]
        if not locked_now:
            order.reverse()
        for servo, name in order + [order[0]]:
            io.request({"cmd": "out", **dict(SAFE_STATE, servo=servo, g=0 if servo else 1)})
            time.sleep(1.0)
            results[f"servo {name}"] = ask(f"{name}? [y/n]").lower().startswith("y")
        io.request({"cmd": "out", **SAFE_STATE})            # leave it locked
        time.sleep(1.0)

        io.request({"cmd": "poll"})                         # clear old presses
        print("  press the BUTTON 3 times within 10 s (live level shown below)...")
        presses, t0, seen_down, last = 0, time.time(), False, None
        while presses < 3 and time.time() - t0 < 10:
            presses += io.request({"cmd": "poll"})["button"]
            try:
                down = bool(io.request({"cmd": "button_raw"})["pressed"])
            except RuntimeError:
                down = None                               # older board app: no live level
            if down is not None and down != last:
                print(f"    D2 reads: {'PRESSED (LOW)' if down else 'released (HIGH)'}")
                last = down
            seen_down |= bool(down)
            time.sleep(0.05)
        results["button (3 presses)"] = presses == 3
        print(f"  counted {presses} presses")
        if presses < 3:
            if last is None:
                print("  -> update the board app (sketch + main.py) for the live check")
            elif not seen_down:
                print("  -> D2 never went LOW: button not reaching D2/GND. Use two DIAGONAL legs "
                      "of the button (one to D2, one to GND), across the breadboard's middle gap.")
            elif last:
                print("  -> D2 stuck LOW: the two legs you used are always connected - "
                      "use the diagonal pair instead.")
    finally:
        io.close()
    print("\nHARDWARE TEST")
    for name, ok in results.items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    if any(not ok for k, ok in results.items() if k.startswith("servo")):
        print("  servo off? rerun with --servo to recalibrate (or nudge it back first)")


if __name__ == "__main__":
    main()
