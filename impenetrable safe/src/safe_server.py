"""Runs ON the UnoQ: camera + model + lock logic + guard screen, all on the board. The screen is
streamed to any browser on the same Wi-Fi (PC or phone): http://<UnoQ IP>:8080

  python3 -m src.safe_server                        # real camera + hardware (App Lab app on :5005)
  python3 -m src.safe_server --keys --board none    # test anywhere: no camera, no hardware

The servo/LEDs/button are driven by the App Lab app (unoq/safe_app) through 127.0.0.1:5005,
so nothing goes over Wi-Fi except the picture you look at.
The web page also has a "button" (same as the real one) and, in --keys mode, digit buttons."""
import argparse
import signal
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2

from . import config
from .safe_sim import SafeApp, connect_board, open_vision

PAGE = """<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Impenetrable Safe</title>
<style>
 :root { color-scheme: dark; }
 body { margin: 0; background: #16191e; color: #e8e8e8; font-family: system-ui, sans-serif;
        display: flex; flex-direction: column; align-items: center; }
 img { width: 100%; max-width: 1040px; display: block; }
 .bar { display: flex; flex-wrap: wrap; gap: 8px; padding: 12px 16px; justify-content: center; }
 button { font-size: 18px; padding: 10px 18px; border-radius: 8px; border: 1px solid #555;
          background: #2a2f37; color: #e8e8e8; cursor: pointer; }
 button:active { background: #3d4450; }
 #st { color: #999; font-size: 14px; padding-bottom: 12px; }
</style></head><body>
<img src="/stream" alt="safe screen">
<div class="bar">
 <button onclick="send('button')">Button</button>
 %DIGITS%
</div>
<div id="st">Impenetrable Safe - streamed from the UnoQ</div>
<script>
function send(what) {
  fetch('/' + what, {method: 'POST'}).catch(() => {
    document.getElementById('st').textContent = 'lost connection to the UnoQ';
  });
}
document.addEventListener('keydown', e => {
  if (e.key === ' ' || e.key === 'b') { e.preventDefault(); send('button'); }
  if (%KEYS% && e.key >= '0' && e.key <= '5') send('digit/' + e.key);
});
</script></body></html>"""


class Stream:
    """Latest JPEG frame, shared between the camera loop and any number of viewers."""

    def __init__(self):
        self.cond = threading.Condition()
        self.jpeg, self.seq = None, 0

    def publish(self, jpeg):
        with self.cond:
            self.jpeg, self.seq = jpeg, self.seq + 1
            self.cond.notify_all()

    def wait_next(self, seq, timeout=2.0):
        with self.cond:
            self.cond.wait_for(lambda: self.seq != seq, timeout=timeout)
            return self.jpeg, self.seq


def make_handler(app, stream, keys):
    digits = "".join(f'<button onclick="send(\'digit/{d}\')">{d if d else "fist"}</button>'
                     for d in range(6)) if keys else ""
    page = PAGE.replace("%DIGITS%", digits).replace("%KEYS%", "true" if keys else "false").encode()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):                   # keep the console for lock events
            pass

        def _send(self, code, body=b"", ctype="text/plain"):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/":
                return self._send(200, page, "text/html; charset=utf-8")
            if self.path == "/snapshot.jpg":
                jpeg, _ = stream.wait_next(-1, 0)
                return self._send(200, jpeg or b"", "image/jpeg")
            if self.path == "/stream":
                self.send_response(200)
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                seq = -1
                try:
                    while True:
                        jpeg, seq = stream.wait_next(seq)
                        if jpeg is None:
                            continue
                        self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                                         + str(len(jpeg)).encode() + b"\r\n\r\n" + jpeg + b"\r\n")
                except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                    return                            # viewer closed the page
            return self._send(404, b"not found")

        def do_POST(self):
            if self.path == "/button":
                app.press_button()
                return self._send(200, b"ok")
            if keys and self.path.startswith("/digit/") and self.path[7:].isdigit():
                app.fake_digit(int(self.path[7:]))
                return self._send(200, b"ok")
            return self._send(404, b"not found")

    return Handler


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8080, help="web page port")
    ap.add_argument("--board", default="127.0.0.1",
                    help="hardware server (App Lab app); 'none' = no hardware")
    ap.add_argument("--keys", action="store_true", help="no camera: digits from the web page")
    ap.add_argument("--camera", type=int)
    ap.add_argument("--fps", type=float, default=12.0, help="max frames/s sent to viewers")
    args = ap.parse_args()
    if not config.CODE_PATH.exists():
        raise SystemExit("No code set yet. Run e.g.:  python3 -m src.code_store 2 5 3")

    io = None if args.board == "none" else connect_board(args.board)
    cap = predictor = calib = None
    if not args.keys:
        cap, predictor, calib = open_vision(args.camera)
    app = SafeApp(io, cap, predictor, calib, args.keys)
    app.help = "button: the real one, or the web page"

    stream = Stream()
    srv = ThreadingHTTPServer(("0.0.0.0", args.port), make_handler(app, stream, args.keys))
    srv.daemon_threads = True
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    print(f"safe running - open http://<this board's IP>:{args.port} in a browser", flush=True)

    stop = threading.Event()
    signal.signal(signal.SIGTERM, lambda *a: stop.set())        # systemd stop -> clean exit
    min_dt, last_pub, frames, t_fps = 1.0 / args.fps, 0.0, 0, time.monotonic()
    try:
        while not stop.is_set():
            canvas = app.step()
            now = time.monotonic()
            if now - last_pub >= min_dt:
                ok, jpeg = cv2.imencode(".jpg", canvas, [cv2.IMWRITE_JPEG_QUALITY, 80])
                if ok:
                    stream.publish(jpeg.tobytes())
                last_pub = now
            frames += 1
            if now - t_fps >= 30:                               # loop speed in the log
                print(f"{frames / (now - t_fps):.1f} frames/s", flush=True)
                frames, t_fps = 0, now
            if cap is None:
                time.sleep(0.02)
    except KeyboardInterrupt:
        pass
    finally:
        srv.shutdown()
        app.close()                                             # leaves the safe LOCKED
        print("stopped (safe locked)", flush=True)


if __name__ == "__main__":
    main()
