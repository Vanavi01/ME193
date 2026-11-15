"""Impenetrable Safe - UnoQ Linux side.

Runs a small TCP server (port 5005) so the PC can drive the servo/LEDs and read the button
through the Bridge while the camera + model still run on the PC. Protocol: one JSON object
per line, e.g.
    {"cmd": "ping"}                                     -> {"ok": true}
    {"cmd": "out", "servo": 1, "y": 1, "g": 0, "r": 2, "alarm": 0}  -> {"ok": true}
        servo: 1 = locked, 0 = unlocked (continuous servo: it turns only when this changes)
    {"cmd": "servo_config", "lock_us": 1300, "unlock_us": 1700, "lock_ms": 300, "unlock_ms": 300}
    {"cmd": "servo_run", "pulse": 1300, "ms": 60}      -> spin, then stop (calibration)
    {"cmd": "servo_state", "locked": 0}                -> where the latch is now (no movement)
    {"cmd": "poll"}                                     -> {"ok": true, "button": 1, "door": 0}
If the PC disconnects, the safe goes back to locked with all LEDs off.

SELF_TEST = True instead makes the button cycle through all outputs (wiring check, no PC).
"""
import json
import socketserver
import threading
import time

try:
    from arduino.app_utils import App, Bridge
except ImportError:          # running the unit tests on the PC
    App = Bridge = None

PORT = 5005
LOCKED, UNLOCKED = 1, 0      # continuous servo: the PC sends the lock STATE, the sketch spins
SAFE = {"servo": LOCKED, "y": 0, "g": 0, "r": 0, "alarm": 0}
SELF_TEST = False


class Board:
    def __init__(self, bridge):
        self.bridge = bridge
        self.lock = threading.Lock()      # one Bridge call at a time

    def handle(self, msg):
        cmd = msg.get("cmd")
        with self.lock:
            if cmd == "ping":
                return {"ok": True}
            if cmd == "out":
                self.bridge.call("set_outputs", int(msg["servo"]), int(msg["y"]), int(msg["g"]),
                                 int(msg["r"]), int(msg.get("alarm", 0)))
                return {"ok": True}
            if cmd == "servo_config":
                self.bridge.call("servo_config", int(msg["lock_us"]), int(msg["unlock_us"]),
                                 int(msg["lock_ms"]), int(msg["unlock_ms"]))
                return {"ok": True}
            if cmd == "servo_run":
                self.bridge.call("servo_run", int(msg["pulse"]), int(msg["ms"]))
                return {"ok": True}
            if cmd == "servo_state":
                self.bridge.call("servo_state", int(msg["locked"]))
                return {"ok": True}
            if cmd == "button_raw":           # live button level (wiring check)
                return {"ok": True, "pressed": int(self.bridge.call("get_button_raw"))}
            if cmd == "poll":
                return {"ok": True, "button": int(self.bridge.call("get_button")),
                        "door": int(self.bridge.call("get_door"))}
        return {"ok": False, "error": f"unknown cmd {cmd!r}"}

    def handle_line(self, line):
        try:
            return self.handle(json.loads(line))
        except Exception as e:            # never crash the server on a bad message
            return {"ok": False, "error": str(e)}

    def safe(self):
        self.handle({"cmd": "out", **SAFE})


class _Server(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


def serve(board, host="0.0.0.0", port=PORT):
    class Handler(socketserver.StreamRequestHandler):
        def handle(self):
            print("PC connected:", self.client_address, flush=True)
            try:
                for raw in self.rfile:
                    reply = board.handle_line(raw.decode("utf-8"))
                    self.wfile.write((json.dumps(reply) + "\n").encode("utf-8"))
            finally:
                board.safe()                  # PC gone -> locked, LEDs off
                print("PC disconnected; safe locked", flush=True)

    srv = _Server((host, port), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def self_test_loop(board):
    """Each button press: next step. Steps: Y on, G on, R on, all blink, servo open, servo locked."""
    steps = [dict(SAFE, y=1), dict(SAFE, g=1), dict(SAFE, r=1), dict(SAFE, y=2, g=2, r=2),
             dict(SAFE, servo=UNLOCKED, g=1), dict(SAFE)]
    i = 0
    board.handle({"cmd": "out", **steps[i]})
    while True:
        if board.handle({"cmd": "poll"})["button"]:
            i = (i + 1) % len(steps)
            board.handle({"cmd": "out", **steps[i]})
            print("self-test step", i, steps[i], flush=True)
        time.sleep(0.05)


if __name__ == "__main__":
    board = Board(Bridge)
    board.safe()
    if SELF_TEST:
        threading.Thread(target=self_test_loop, args=(board,), daemon=True).start()
        print("SELF TEST: press the button to step through LEDs and servo", flush=True)
    else:
        serve(board)
        print(f"safe hardware server listening on port {PORT}", flush=True)
    App.run()
