"""Hardware-independent lock state machine.

Call tick() once per camera frame. It takes the latest model prediction, the
button edge, and the raw Hall-sensor door reading, and returns an Output telling
the I/O layer what the servo / LEDs / alarm should do.

States
  SLEEP     locked, LEDs off, waiting for the button
  LISTEN    reading digits (all LEDs blink once when the button wakes it up)
  UNLOCKED  servo open, green on; the BUTTON locks it again (with the Hall sensor also
            after the door is opened and closed)
  LOCKOUT   too many wrong codes; red on until the timer ends
  BREACH    door opened while the servo was LOCKED (forced entry): red fast + alarm.
            Press the button and enter the correct code to disarm.

LED signals
  yellow flash  a digit was registered
  red flashing  wrong code
  green flashing correct code (then green stays on while unlocked)
"""
from dataclasses import dataclass
from enum import Enum, auto
from typing import Callable, Optional, Sequence, Tuple

from . import config


class State(Enum):
    SLEEP = auto()
    LISTEN = auto()
    UNLOCKED = auto()
    LOCKOUT = auto()
    BREACH = auto()


@dataclass
class Output:
    state: State
    servo_locked: bool
    yellow: str          # "off" | "on" | "fast"
    green: str
    red: str
    alarm: bool
    event: Optional[str]
    digits_entered: int
    door_open: bool


class LockPolicy:
    def __init__(self, check_code: Callable[[Sequence[int]], bool], cfg=config):
        self.check_code = check_code
        self.cfg = cfg
        self.state = State.SLEEP
        self.servo_locked = True
        # door (debounced)
        self.door_open = False
        self._first = True
        self._raw_door = False
        self._raw_since = 0.0
        # digit entry
        self.digits = []
        self.cand: Optional[int] = None
        self.cand_n = 0
        self.need_release = False
        self.attempts = 0
        self.last_input = 0.0
        self.disarming = False
        # timers
        self.lockout_until = 0.0
        self.ready_until = 0.0       # all LEDs on: "button pressed, show your code"
        self.yellow_until = 0.0      # digit registered
        self.green_until = 0.0       # correct code
        self.red_until = 0.0         # wrong code
        # unlocked bookkeeping
        self.unlocked_at = 0.0
        self.door_seen_open = False
        self.closed_since: Optional[float] = None

    # ------------------------------------------------------------------ public
    def tick(self, now: float, pred: Optional[Tuple[int, float]] = None,
             button: bool = False, door_raw: bool = False) -> Output:
        """pred = (digit, confidence) or None. door_raw True = door open."""
        door_event = self._update_door(now, door_raw)
        if door_event == "opened" and self.servo_locked and self.state is not State.BREACH:
            self._to_breach()
            event = "breach"
        else:
            handler = {
                State.SLEEP: self._sleep,
                State.LISTEN: self._listen,
                State.UNLOCKED: self._unlocked,
                State.LOCKOUT: self._lockout,
                State.BREACH: self._breach,
            }[self.state]
            event = handler(now, pred, button)
        return self._output(now, event)

    # -------------------------------------------------------------------- door
    def _update_door(self, now, raw) -> Optional[str]:
        if self._first:                       # trust the first reading (no alarm at boot)
            self._first = False
            self._raw_door, self._raw_since, self.door_open = raw, now, raw
            return None
        if raw != self._raw_door:
            self._raw_door, self._raw_since = raw, now
        if raw != self.door_open and now - self._raw_since >= self.cfg.DOOR_DEBOUNCE_S:
            self.door_open = raw
            return "opened" if raw else "closed"
        return None

    # ---------------------------------------------------------------- handlers
    def _sleep(self, now, pred, button):
        if button:
            self._wake(now, disarming=False)
            return "wake"
        return None

    def _breach(self, now, pred, button):
        if button:
            self._wake(now, disarming=True)
            return "disarm_start"
        return None

    def _lockout(self, now, pred, button):
        if now >= self.lockout_until:
            self.state = State.SLEEP
            return "lockout_end"
        return None

    def _listen(self, now, pred, button):
        if now - self.last_input >= self.cfg.INPUT_TIMEOUT_S:
            self._abort()
            return "timeout"
        digit = pred[0] if (pred is not None and pred[1] >= self.cfg.CONF_THRESHOLD) else None
        if digit is not None:
            self.last_input = now
        if self.need_release:                 # wait for fist / empty between digits
            if digit == 0:
                self.need_release = False
                self.cand, self.cand_n = None, 0
            return None
        if digit is None or digit == 0:
            self.cand, self.cand_n = None, 0
            return None
        if digit == self.cand:
            self.cand_n += 1
        else:
            self.cand, self.cand_n = digit, 1
        if self.cand_n >= self.cfg.STABLE_FRAMES:
            return self._accept(now, digit)
        return None

    def _unlocked(self, now, pred, button):
        """Stays open until the BUTTON is pressed (door closed). With the Hall sensor it also
        re-locks by itself once the door has been opened and closed again; an optional
        UNLOCK_WINDOW_S re-locks if nobody does anything."""
        if self.door_open:
            self.door_seen_open = True
            self.closed_since = None
            return None
        if button:
            return self._relock()
        if self.door_seen_open:               # opened earlier, closed now
            if self.closed_since is None:
                self.closed_since = now
            if now - self.closed_since >= self.cfg.RELOCK_DELAY_S:
                return self._relock()
        elif self.cfg.UNLOCK_WINDOW_S and now - self.unlocked_at >= self.cfg.UNLOCK_WINDOW_S:
            return self._relock()
        return None

    # ----------------------------------------------------------- transitions
    def _wake(self, now, disarming):
        self.state = State.LISTEN
        self.digits = []
        self.cand, self.cand_n = None, 0
        self.need_release = False
        self.last_input = now
        self.disarming = disarming
        self.ready_until = now + self.cfg.READY_FLASH_S

    def _abort(self):
        self.digits = []
        self.state = State.BREACH if self.disarming else State.SLEEP
        self.disarming = False

    def _to_breach(self):
        self.state = State.BREACH
        self.servo_locked = True
        self.digits = []
        self.disarming = False

    def _accept(self, now, digit):
        self.digits.append(digit)
        self.cand, self.cand_n = None, 0
        self.need_release = True
        self.yellow_until = now + self.cfg.DIGIT_FLASH_S
        if len(self.digits) < self.cfg.CODE_LEN:
            return "digit"
        ok = self.check_code(tuple(self.digits))
        self.digits = []
        if ok:
            self.attempts = 0
            self.green_until = now + self.cfg.RESULT_FLASH_S
            self._unlock(now)
            return "unlock"
        self.attempts += 1
        self.red_until = now + self.cfg.RESULT_FLASH_S
        if self.attempts >= self.cfg.MAX_ATTEMPTS:
            self.attempts = 0
            if self.disarming:
                self._to_breach()
            else:
                self.state = State.LOCKOUT
                self.lockout_until = now + self.cfg.LOCKOUT_S
            return "lockout"
        return "wrong_code"

    def _unlock(self, now):
        self.state = State.UNLOCKED
        self.servo_locked = False
        self.disarming = False
        self.unlocked_at = now
        self.door_seen_open = self.door_open
        self.closed_since = None

    def _relock(self):
        self.state = State.SLEEP
        self.servo_locked = True
        return "relock"

    # ------------------------------------------------------------------ output
    def _output(self, now, event) -> Output:
        y = g = r = "off"
        alarm = False
        s = self.state
        if s is State.LISTEN:
            if self.disarming:
                r = "fast"
        elif s is State.UNLOCKED:
            g = "on"
        elif s is State.LOCKOUT:
            r = "on"
        elif s is State.BREACH:
            r, alarm = "fast", True
        if now < self.ready_until:
            y = g = r = "on"
        if now < self.yellow_until:
            y = "on"
        if now < self.green_until:
            g = "fast"
        if now < self.red_until and s is not State.BREACH:
            r = "fast"
        return Output(s, self.servo_locked, y, g, r, alarm, event,
                      len(self.digits), self.door_open)
