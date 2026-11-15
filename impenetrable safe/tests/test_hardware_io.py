"""PC <-> UnoQ hardware path, with a fake Bridge in place of the microcontroller sketch."""
import importlib.util
import time
import unittest
from pathlib import Path
from unittest import mock

from src import config
from src.io_backend import ThreadedIO, SAFE_STATE, BridgeIO, UnoQNetIO, encode
from src.lock_policy import LockPolicy, Output, State

spec = importlib.util.spec_from_file_location(
    "unoq_main", Path(__file__).resolve().parent.parent / "unoq" / "safe_app" / "python" / "main.py")
unoq_main = importlib.util.module_from_spec(spec)
spec.loader.exec_module(unoq_main)


class FakeBridge:
    """Mimics the sketch's Bridge functions."""

    def __init__(self):
        self.calls, self.presses, self.door = [], 0, 0

    def call(self, name, *args):
        self.calls.append((name, args))
        if name == "get_button":
            n, self.presses = self.presses, 0
            return n
        if name == "get_door":
            return self.door
        if name == "get_button_raw":
            return 1 if self.presses else 0
        return None

    def outputs(self):
        return [a for n, a in self.calls if n == "set_outputs"]


def out(**kw):
    base = dict(state=State.SLEEP, servo_locked=True, yellow="off", green="off", red="off",
                alarm=False, event=None, digits_entered=0, door_open=False)
    base.update(kw)
    return Output(**base)


class EncodeTests(unittest.TestCase):
    def test_locked_and_unlocked(self):
        self.assertEqual(encode(out())["servo"], 1)
        e = encode(out(servo_locked=False, green="on", red="fast"))
        self.assertEqual((e["servo"], e["g"], e["r"], e["y"]), (0, 1, 2, 0))


class BoardTests(unittest.TestCase):
    def setUp(self):
        self.bridge = FakeBridge()
        self.board = unoq_main.Board(self.bridge)

    def test_out_calls_sketch(self):
        r = self.board.handle({"cmd": "out", "servo": 0, "y": 1, "g": 0, "r": 2, "alarm": 0})
        self.assertTrue(r["ok"])
        self.assertEqual(self.bridge.outputs(), [(0, 1, 0, 2, 0)])

    def test_servo_commands_reach_the_sketch(self):
        self.board.handle({"cmd": "servo_config", "lock_us": 1300, "unlock_us": 1700,
                           "lock_ms": 280, "unlock_ms": 310})
        self.board.handle({"cmd": "servo_run", "pulse": 1300, "ms": 60})
        self.board.handle({"cmd": "servo_state", "locked": 0})
        self.assertEqual([c for c in self.bridge.calls if c[0].startswith("servo")],
                         [("servo_config", (1300, 1700, 280, 310)), ("servo_run", (1300, 60)),
                          ("servo_state", (0,))])

    def test_button_raw(self):
        self.bridge.presses = 1
        self.assertEqual(self.board.handle({"cmd": "button_raw"}), {"ok": True, "pressed": 1})

    def test_poll(self):
        self.bridge.presses = 2
        self.assertEqual(self.board.handle({"cmd": "poll"}), {"ok": True, "button": 2, "door": 0})

    def test_bad_messages_do_not_crash(self):
        self.assertFalse(self.board.handle_line("not json")["ok"])
        self.assertFalse(self.board.handle_line('{"cmd": "explode"}')["ok"])


@mock.patch.object(config, "POLL_INTERVAL_S", 0.0)
class NetworkTests(unittest.TestCase):
    def setUp(self):
        self.bridge = FakeBridge()
        self.board = unoq_main.Board(self.bridge)
        self.srv = unoq_main.serve(self.board, "127.0.0.1", 0)
        self.io = UnoQNetIO("127.0.0.1", self.srv.server_address[1])

    def tearDown(self):
        try:
            self.io.close()
        except OSError:
            pass                       # already closed by the test
        self.srv.shutdown()
        self.srv.server_close()

    def test_servo_calibration_sent_on_connect(self):
        cfg = [c for c in self.bridge.calls if c[0] == "servo_config"]
        cal = config.load_servo_calibration()
        self.assertEqual(cfg, [("servo_config", (cal["lock_us"], cal["unlock_us"],
                                                 cal["lock_ms"], cal["unlock_ms"]))])

    def test_outputs_sent_only_on_change(self):
        self.io.apply(out())
        self.io.apply(out())
        self.io.apply(out(yellow="on"))
        self.assertEqual(len(self.bridge.outputs()), 2)

    def test_each_press_is_one_button_event(self):
        self.bridge.presses = 2
        self.assertEqual([self.io.read_button() for _ in range(3)], [True, True, False])

    def test_door_ignored_without_hall_sensor(self):
        self.bridge.door = 1
        self.io.read_button()
        self.assertFalse(self.io.read_door_open())

    def test_close_and_disconnect_leave_it_locked(self):
        self.io.apply(out(servo_locked=False, green="on"))
        self.io.close()
        time.sleep(0.2)
        safe = tuple(SAFE_STATE[k] for k in ("servo", "y", "g", "r", "alarm"))
        self.assertEqual(self.bridge.outputs()[-1], safe)

    def test_button_wakes_the_real_lock(self):
        policy = LockPolicy(lambda d: tuple(d) == (2, 5, 3))
        self.bridge.presses = 1
        o = policy.tick(1.0, None, self.io.read_button(), self.io.read_door_open())
        self.io.apply(o)
        self.assertEqual(o.state, State.LISTEN)
        self.assertEqual(self.bridge.outputs()[-1][1], 1)          # yellow on at the board


def wait_for(cond, timeout=3.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if cond():
            return True
        time.sleep(0.02)
    return False


@mock.patch.object(config, "POLL_INTERVAL_S", 0.01)
class ThreadedIOTests(unittest.TestCase):
    def setUp(self):
        self.bridge = FakeBridge()
        self.board = unoq_main.Board(self.bridge)
        self.srv = unoq_main.serve(self.board, "127.0.0.1", 0)
        self.port = self.srv.server_address[1]
        self.io = ThreadedIO(lambda: UnoQNetIO("127.0.0.1", self.port))

    def tearDown(self):
        self.io.close()
        self.srv.shutdown()
        self.srv.server_close()

    def test_outputs_reach_the_board_without_blocking(self):
        t0 = time.time()
        self.io.apply(out(servo_locked=False, green="on"))
        self.assertLess(time.time() - t0, 0.01)                   # apply() never waits on Wi-Fi
        self.assertTrue(wait_for(lambda: self.bridge.outputs()[-1:] == [(0, 0, 1, 0, 0)]))

    def test_button_presses_arrive(self):
        self.bridge.presses = 2
        self.assertTrue(wait_for(lambda: self.io.read_button()))
        self.assertTrue(wait_for(lambda: self.io.read_button()))
        self.assertFalse(self.io.read_button())

    def test_close_leaves_it_locked(self):
        self.io.apply(out(servo_locked=False, green="on"))
        wait_for(lambda: self.bridge.outputs()[-1:] == [(0, 0, 1, 0, 0)])
        self.io.close()
        time.sleep(0.2)
        self.assertEqual(self.bridge.outputs()[-1][0], 1)

    def test_reconnects_and_resends_outputs(self):
        self.io.apply(out(servo_locked=False, green="on"))
        wait_for(lambda: self.bridge.outputs()[-1:] == [(0, 0, 1, 0, 0)])
        self.srv.shutdown()                                        # board app restarts
        self.srv.server_close()
        self.io._drop()                                            # (socket noticed as gone)
        self.srv = unoq_main.serve(self.board, "127.0.0.1", self.port)
        self.bridge.calls.clear()
        self.assertTrue(wait_for(lambda: self.io.connected and
                                 self.bridge.outputs()[-1:] == [(0, 0, 1, 0, 0)], timeout=6))


class BridgeIOTests(unittest.TestCase):
    @mock.patch.object(config, "POLL_INTERVAL_S", 0.0)
    def test_on_board_path(self):
        b = FakeBridge()
        io = BridgeIO(b)
        b.presses = 1
        self.assertTrue(io.read_button())
        io.apply(out(servo_locked=False, green="on"))
        self.assertEqual(b.outputs()[-1], (0, 0, 1, 0, 0))


if __name__ == "__main__":
    unittest.main()
