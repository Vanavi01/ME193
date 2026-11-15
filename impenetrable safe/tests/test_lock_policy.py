"""Run from the project root:  python -m unittest discover -s tests -t ."""
import unittest

from src import config
from src.lock_policy import LockPolicy, State

CODE = (2, 5, 3)


class Sim:
    def __init__(self):
        self.p = LockPolicy(lambda d: tuple(d) == CODE)
        self.t = 0.0
        self.door = False
        self.out = self.step()

    def step(self, pred=None, button=False, dt=0.1, n=1):
        for _ in range(n):
            self.t += dt
            self.out = self.p.tick(self.t, pred, button, self.door)
        return self.out

    def wake(self):
        return self.step(button=True)

    def enter(self, digits):
        for d in digits:
            self.step((d, 0.99), n=config.STABLE_FRAMES)
            self.step((0, 0.99), n=2)
        return self.out


class LockPolicyTests(unittest.TestCase):
    def test_correct_code_unlocks(self):
        s = Sim()
        s.wake()
        self.assertEqual(s.out.state, State.LISTEN)
        s.enter(CODE)
        self.assertEqual(s.out.state, State.UNLOCKED)
        self.assertFalse(s.out.servo_locked)

    def test_low_confidence_ignored(self):
        s = Sim()
        s.wake()
        s.step((2, 0.5), n=50, dt=0.1)  # also exceeds nothing: below threshold
        self.assertEqual(s.out.digits_entered, 0)

    def test_wrong_code_three_times_locks_out_then_recovers(self):
        s = Sim()
        s.wake()
        for _ in range(config.MAX_ATTEMPTS):
            s.enter((1, 2, 4))
        self.assertEqual(s.out.state, State.LOCKOUT)
        self.assertTrue(s.out.servo_locked)
        s.step(n=int(config.LOCKOUT_S / 0.1) + 5)
        self.assertEqual(s.out.state, State.SLEEP)

    def test_timeout_returns_to_sleep(self):
        s = Sim()
        s.wake()
        s.step(n=int(config.INPUT_TIMEOUT_S / 0.1) + 5)
        self.assertEqual(s.out.state, State.SLEEP)

    def test_forced_door_while_locked_is_breach(self):
        s = Sim()
        s.door = True
        s.step(n=5)
        self.assertEqual(s.out.state, State.BREACH)
        self.assertTrue(s.out.alarm)
        self.assertEqual(s.out.red, "fast")

    def test_door_flicker_shorter_than_debounce_ignored(self):
        s = Sim()
        s.door = True
        s.step(dt=0.05)
        s.door = False
        s.step(n=10, dt=0.05)
        self.assertEqual(s.out.state, State.SLEEP)

    def test_open_after_valid_unlock_is_not_breach_and_relocks(self):
        s = Sim()
        s.wake()
        s.enter(CODE)
        s.door = True
        s.step(n=10)
        self.assertEqual(s.out.state, State.UNLOCKED)
        s.door = False
        s.step(n=int(config.RELOCK_DELAY_S / 0.1) + 10)
        self.assertEqual(s.out.state, State.SLEEP)
        self.assertTrue(s.out.servo_locked)

    def test_stays_open_until_button_then_locks(self):
        s = Sim()
        s.wake()
        s.enter(CODE)
        s.step(n=600)                                     # a minute later: still open
        self.assertEqual(s.out.state, State.UNLOCKED)
        self.assertFalse(s.out.servo_locked)
        s.step(button=True)
        self.assertEqual(s.out.event, "relock")
        self.assertEqual(s.out.state, State.SLEEP)
        self.assertTrue(s.out.servo_locked)

    def test_optional_timed_relock(self):
        from unittest import mock
        with mock.patch.object(config, "UNLOCK_WINDOW_S", 10.0):
            s = Sim()
            s.wake()
            s.enter(CODE)
            s.step(n=105)
            self.assertEqual(s.out.state, State.SLEEP)

    def test_button_relock_ignored_while_door_open(self):
        s = Sim()
        s.wake()
        s.enter(CODE)
        s.door = True
        s.step(n=5)
        s.step(button=True)
        self.assertEqual(s.out.state, State.UNLOCKED)     # don't throw the bolt into an open door

    def test_breach_can_be_disarmed_with_code(self):
        s = Sim()
        s.door = True
        s.step(n=5)
        s.door = False
        s.step(n=5)
        self.assertEqual(s.out.state, State.BREACH)
        s.wake()
        self.assertEqual(s.out.state, State.LISTEN)
        s.enter(CODE)
        self.assertEqual(s.out.state, State.UNLOCKED)

    def test_failed_disarm_returns_to_breach(self):
        s = Sim()
        s.door = True
        s.step(n=5)
        s.wake()
        for _ in range(config.MAX_ATTEMPTS):
            s.enter((1, 2, 4))
        self.assertEqual(s.out.state, State.BREACH)

    def test_yellow_flashes_per_digit(self):
        s = Sim()
        s.wake()
        s.step(n=int(config.READY_FLASH_S / 0.1) + 1)
        self.assertEqual((s.out.red, s.out.yellow, s.out.green), ("off", "off", "off"))
        s.step((2, 0.99), n=config.STABLE_FRAMES)
        self.assertEqual(s.out.event, "digit")
        self.assertEqual(s.out.yellow, "on")
        s.step((0, 0.99), n=int(config.DIGIT_FLASH_S / 0.1) + 1)
        self.assertEqual(s.out.yellow, "off")

    def test_wake_lights_all_leds_briefly(self):
        s = Sim()
        s.wake()
        self.assertEqual((s.out.red, s.out.yellow, s.out.green), ("on", "on", "on"))

    def test_wrong_code_flashes_red(self):
        s = Sim()
        s.wake()
        s.enter((1, 2, 4))
        self.assertEqual(s.out.red, "fast")
        self.assertEqual(s.out.green, "off")
        s.step(n=int(config.RESULT_FLASH_S / 0.1) + 1)
        self.assertEqual(s.out.red, "off")

    def test_correct_code_flashes_green_then_steady(self):
        s = Sim()
        s.wake()
        s.enter(CODE)
        self.assertEqual(s.out.green, "fast")
        self.assertEqual(s.out.red, "off")
        s.step(n=int(config.RESULT_FLASH_S / 0.1) + 1)
        self.assertEqual(s.out.green, "on")

    def test_door_open_at_boot_does_not_alarm(self):
        p = LockPolicy(lambda d: False)
        out = p.tick(0.0, None, False, True)
        out = p.tick(1.0, None, False, True)
        self.assertEqual(out.state, State.SLEEP)


if __name__ == "__main__":
    unittest.main()
