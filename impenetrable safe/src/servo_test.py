"""Lock / unlock the real safe from the terminal (no camera, no model).
  python -m src.servo_test --host <UnoQ IP>

Commands (type, then Enter):
  l        lock   (spin toward CLOSED for lock_ms, ends against the stop; red LED blinks once)
  u        unlock (spin toward OPEN for unlock_ms; green LED stays on while unlocked)
  t        toggle
  c N      cycle lock -> unlock N times (reliability check)
  b        button mode: each press of the real button toggles lock/unlock (Ctrl+C to stop)
  a / d    small nudge toward OPEN / CLOSED (aaa / ddd = bigger)
  q        quit (the board locks the safe when the PC disconnects)

Every lock/unlock is a full spin even if the board thinks it's already there: with the physical
stops that is harmless and pulls the latch back to a known position."""
import argparse
import time

from . import config
from .io_backend import SAFE_STATE, UnoQNetIO


class Safe:
    def __init__(self, io, cal):
        self.io, self.cal = io, cal
        self.locked = None

    def move(self, locked):
        key = "lock" if locked else "unlock"
        t0 = time.time()
        self.io.servo_run(self.cal[f"{key}_us"], self.cal[f"{key}_ms"])
        self.io.servo_state(locked)                      # board now agrees where the latch is
        self.io.request({"cmd": "out", **dict(SAFE_STATE, servo=int(locked),
                                               r=1 if locked else 0, g=0 if locked else 1)})
        time.sleep(self.cal[f"{key}_ms"] / 1000 + 0.3)
        if locked:                                        # red flash only briefly
            self.io.request({"cmd": "out", **SAFE_STATE})
        self.locked = locked
        print(f"  {'LOCKED' if locked else 'UNLOCKED'}  ({time.time() - t0:.1f} s)")

    def nudge(self, toward_open, n):
        us = self.cal["unlock_us"] if toward_open else self.cal["lock_us"]
        ms = config.SERVO_NUDGE_MS * n
        self.io.servo_run(us, ms)
        time.sleep(ms / 1000 + 0.3)


def button_mode(io, safe):
    """Real button toggles the lock. Stops after 60 s without a press or Ctrl+C."""
    print("  button mode: press the button on the safe to lock/unlock "
          "(stops after 60 s without a press, or Ctrl+C)")
    io.request({"cmd": "poll"})                          # forget old presses
    last = time.time()
    try:
        while time.time() - last < 60:
            if io.request({"cmd": "poll"})["button"]:
                safe.move(not safe.locked)
                last = time.time()
                io.request({"cmd": "poll"})              # ignore presses during the move
            time.sleep(0.05)
    except KeyboardInterrupt:
        pass
    print("  button mode off")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", required=True)
    args = ap.parse_args()
    cal = config.load_servo_calibration()
    io = UnoQNetIO(args.host)                            # also sends the servo calibration
    safe = Safe(io, cal)
    print(f"connected. lock {cal['lock_ms']} ms @ {cal['lock_us']} us, "
          f"unlock {cal['unlock_ms']} ms @ {cal['unlock_us']} us")
    print(__doc__.split("Commands (type, then Enter):")[1].split("Every lock")[0].rstrip())
    try:
        safe.move(True)                                  # start from a known position: LOCKED
        while True:
            cmd = input(f"[{'locked' if safe.locked else 'UNLOCKED'}] > ").strip().lower()
            if cmd in ("q", "quit", "exit"):
                break
            if cmd == "l":
                safe.move(True)
            elif cmd == "u":
                safe.move(False)
            elif cmd == "t":
                safe.move(not safe.locked)
            elif cmd.startswith("c"):
                arg = cmd[1:].strip()
                n = int(arg) if arg.isdigit() else 5
                for i in range(n):
                    print(f"  cycle {i + 1}/{n}")
                    safe.move(True)
                    safe.move(False)
                safe.move(True)
            elif cmd == "b":
                button_mode(io, safe)
            elif cmd and set(cmd) == {"a"}:
                safe.nudge(True, len(cmd))
            elif cmd and set(cmd) == {"d"}:
                safe.nudge(False, len(cmd))
            elif cmd:
                print("  ? (l, u, t, c N, b, a/d, q)")
    except (KeyboardInterrupt, EOFError):
        print()
    finally:
        io.close()                                       # board locks if it was open
        print("disconnected (safe locked)")


if __name__ == "__main__":
    main()
