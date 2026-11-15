"""Hardware abstraction so the same lock policy runs in simulation, from the PC with the real
hardware on the UnoQ, and (later) entirely on the UnoQ.

SimIO      keyboard stand-ins for the button + door, LEDs drawn on screen (no hardware)
UnoQNetIO  PC -> Wi-Fi -> hardware server on the UnoQ -> Bridge -> sketch (servo/LEDs/button)
BridgeIO   running ON the UnoQ: talks to the sketch directly through the Bridge
ThreadedIO wraps UnoQNetIO in a background thread: Wi-Fi delays never stall the camera loop,
           and a dropped connection is retried automatically
"""
import json
import socket
import threading
import time
from abc import ABC, abstractmethod

from . import config
from .lock_policy import Output

LED_CODES = {"off": 0, "on": 1, "fast": 2}


def encode(out: Output) -> dict:
    """Policy output -> what the sketch needs (servo 1 = locked / 0 = unlocked; the board turns
    the continuous servo only when this changes. LED codes 0 off / 1 on / 2 blink)."""
    return {"servo": 1 if out.servo_locked else 0,
            "y": LED_CODES[out.yellow], "g": LED_CODES[out.green], "r": LED_CODES[out.red],
            "alarm": int(out.alarm)}


SAFE_STATE = {"servo": 1, "y": 0, "g": 0, "r": 0, "alarm": 0}


class LockIO(ABC):
    last = None

    @abstractmethod
    def read_button(self) -> bool:
        """True once per press."""

    @abstractmethod
    def read_door_open(self) -> bool:
        """True when the door is open (always False without the Hall sensor)."""

    @abstractmethod
    def apply(self, out: Output) -> None:
        """Drive servo, LEDs and alarm from the policy output."""

    def handle_key(self, key):
        pass

    def close(self):
        pass

    def overlay_lines(self):
        o = self.last
        if o is None:
            return []
        return [f"state {o.state.name}  servo {'LOCKED' if o.servo_locked else 'OPEN'}"
                f"  door {'OPEN' if o.door_open else 'closed'}",
                f"Y:{o.yellow} G:{o.green} R:{o.red}  digits {o.digits_entered}"
                f"{'  *** ALARM ***' if o.alarm else ''}",
                "keys: b=button  d=toggle door  q=quit"]


class SimIO(LockIO):
    """Keys:  b = button,  d = toggle door open/closed (simulates the Hall sensor)."""

    def __init__(self):
        self._button = False
        self._door_open = False

    def handle_key(self, key):
        if key == ord("b"):
            self._button = True
        elif key == ord("d"):
            self._door_open = not self._door_open

    def read_button(self):
        pressed, self._button = self._button, False
        return pressed

    def read_door_open(self):
        return self._door_open

    def apply(self, out):
        self.last = out


class _BoardIO(LockIO):
    """Shared logic: send outputs only when they change, poll button/door every POLL_INTERVAL_S.
    Subclasses implement _set_outputs(dict) and _poll() -> (presses, door)."""

    def __init__(self):
        self._sent = None
        self._presses = 0
        self._door = False
        self._last_poll = 0.0
        self._fake_button = False

    def _maybe_poll(self):
        now = time.monotonic()
        if now - self._last_poll >= config.POLL_INTERVAL_S:
            self._last_poll = now
            presses, door = self._poll()
            self._presses += presses
            self._door = bool(door) and config.HALL_SENSOR

    def read_button(self):
        self._maybe_poll()
        if self._fake_button:                 # keyboard 'b' still works for testing
            self._fake_button = False
            return True
        if self._presses:
            self._presses -= 1
            return True
        return False

    def read_door_open(self):
        return self._door

    def handle_key(self, key):
        if key == ord("b"):
            self._fake_button = True

    def apply(self, out):
        self.last = out
        o = encode(out)
        if o != self._sent:
            self._set_outputs(o)
            self._sent = o


class UnoQNetIO(_BoardIO):
    """PC side: JSON lines over TCP to unoq/safe_app/python/main.py on the board."""

    def __init__(self, host, port=None, timeout=2.0):
        super().__init__()
        self.sock = socket.create_connection((host, port or config.UNOQ_PORT), timeout=timeout)
        self.f = self.sock.makefile("rw", encoding="utf-8", newline="\n")
        if not self.request({"cmd": "ping"}).get("ok"):
            raise RuntimeError("UnoQ hardware server did not answer ping")
        self.send_servo_calibration(config.load_servo_calibration())

    def send_servo_calibration(self, cal):
        """How the continuous servo locks/unlocks (pulse width + spin time each way)."""
        self.request({"cmd": "servo_config", **{k: int(cal[k]) for k in config.SERVO_DEFAULT}})

    def servo_run(self, pulse_us, ms):
        """Spin at `pulse_us` for `ms` milliseconds, then stop (calibration)."""
        self.request({"cmd": "servo_run", "pulse": int(pulse_us), "ms": int(ms)})

    def home_locked(self):
        """Full lock spin (ends against the CLOSED stop) so the latch is at a known position."""
        cal = config.load_servo_calibration()
        self.servo_run(cal["lock_us"], cal["lock_ms"])
        self.servo_state(True)

    def servo_state(self, locked):
        """Tell the board where the latch is now, without moving it."""
        self.request({"cmd": "servo_state", "locked": int(bool(locked))})

    def request(self, msg):
        self.f.write(json.dumps(msg) + "\n")
        self.f.flush()
        reply = json.loads(self.f.readline())
        if not reply.get("ok"):
            raise RuntimeError(f"UnoQ error: {reply.get('error')}")
        return reply

    def _set_outputs(self, o):
        self.request({"cmd": "out", **o})

    def _poll(self):
        r = self.request({"cmd": "poll"})
        return r["button"], r["door"]

    def close(self):
        if self.sock is None:
            return
        try:
            self.request({"cmd": "out", **SAFE_STATE})          # leave it locked
        finally:
            self.f.close()
            self.sock.close()
            self.sock = None


class BridgeIO(_BoardIO):
    """On the UnoQ itself (final stage): the same calls, straight to the sketch."""

    def __init__(self, bridge):
        super().__init__()
        self.bridge = bridge

    def _set_outputs(self, o):
        self.bridge.call("set_outputs", o["servo"], o["y"], o["g"], o["r"], o["alarm"])

    def _poll(self):
        return int(self.bridge.call("get_button")), int(self.bridge.call("get_door"))


class ThreadedIO(LockIO):
    """Board I/O on a background thread. The UI thread only reads/writes local variables, so a
    slow Wi-Fi round trip (~30-120 ms on campus) never slows the camera + model loop.
    If the connection drops, it reconnects every 2 s and re-sends the current outputs
    (the board itself locks the safe while the PC is gone)."""

    def __init__(self, connect, first=None):
        self._connect = connect                  # () -> UnoQNetIO
        self._io = first if first is not None else connect()
        self._lock = threading.Lock()
        self._presses = 0
        self._door = False
        self._want = None
        self._sent = None
        self.connected = True
        self.error = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        while not self._stop.is_set():
            try:
                if self._io is None:
                    self._io = self._connect()
                    self._sent = None            # board was reset to locked: re-send everything
                    self.connected, self.error = True, None
                with self._lock:
                    want = self._want
                if want is not None and want != self._sent:
                    self._io._set_outputs(want)
                    self._sent = want
                presses, door = self._io._poll()
                with self._lock:
                    self._presses += presses
                    self._door = bool(door) and config.HALL_SENSOR
            except (OSError, RuntimeError, ValueError) as e:
                self.connected, self.error = False, str(e) or type(e).__name__
                self._drop()
                self._stop.wait(2.0)
                continue
            self._stop.wait(config.POLL_INTERVAL_S)

    def _drop(self):
        io, self._io = self._io, None
        if io is not None:
            try:
                io.f.close()
                io.sock.close()
            except OSError:
                pass
            io.sock = None

    def read_button(self):
        with self._lock:
            if self._presses:
                self._presses -= 1
                return True
        return False

    def read_door_open(self):
        with self._lock:
            return self._door

    def apply(self, out):
        self.last = out
        with self._lock:
            self._want = encode(out)

    def close(self):
        self._stop.set()
        self._thread.join(timeout=3)
        if self._io is not None:
            try:
                self._io.close()                 # leaves the real safe LOCKED
            except (OSError, RuntimeError, ValueError):
                pass
