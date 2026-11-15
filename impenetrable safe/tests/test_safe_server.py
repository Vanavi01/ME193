"""The UnoQ web view: page, live frames and the web button, without camera or hardware."""
import threading
import time
import unittest
import urllib.request
from http.server import ThreadingHTTPServer
from unittest import mock

from src import code_store, config
from src.lock_policy import State
from src.safe_server import Stream, make_handler
from src.safe_sim import SafeApp


class SafeServerTests(unittest.TestCase):
    def setUp(self):
        p = mock.patch.object(code_store, "verify", lambda d: tuple(d) == (2, 5, 3))
        p.start()
        self.addCleanup(p.stop)
        self.app = SafeApp(keys=True)
        self.stream = Stream()
        self.srv = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(self.app, self.stream, True))
        self.srv.daemon_threads = True
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.srv.server_address[1]}"
        self.stop = threading.Event()

        def loop():
            import cv2
            while not self.stop.is_set():
                ok, jpeg = cv2.imencode(".jpg", self.app.step())
                self.stream.publish(jpeg.tobytes())
                time.sleep(0.02)
        self.loop = threading.Thread(target=loop, daemon=True)
        self.loop.start()

    def tearDown(self):
        self.stop.set()
        self.loop.join(2)
        self.srv.shutdown()
        self.srv.server_close()

    def post(self, path):
        return urllib.request.urlopen(urllib.request.Request(self.url + path, method="POST"),
                                      timeout=3).read()

    def wait_state(self, state, timeout=3):
        t0 = time.time()
        while time.time() - t0 < timeout:
            if self.app.out is not None and self.app.out.state is state:
                return True
            time.sleep(0.02)
        return False

    def test_page_and_snapshot(self):
        html = urllib.request.urlopen(self.url + "/", timeout=3).read().decode()
        self.assertIn('src="/stream"', html)
        self.assertIn("digit/5", html)
        time.sleep(0.2)
        jpeg = urllib.request.urlopen(self.url + "/snapshot.jpg", timeout=3).read()
        self.assertEqual(jpeg[:2], b"\xff\xd8")                  # a JPEG

    def test_stream_sends_frames(self):
        r = urllib.request.urlopen(self.url + "/stream", timeout=3)
        self.assertIn("multipart/x-mixed-replace", r.headers["Content-Type"])
        self.assertTrue(r.read(4000).startswith(b"--frame"))
        r.close()

    def test_web_button_and_digits_unlock(self):
        self.post("/button")
        self.assertTrue(self.wait_state(State.LISTEN))
        for d in (2, 5, 3):
            self.post(f"/digit/{d}")
            time.sleep(1.2)                                      # digit 0.6 s, then fist
        self.assertTrue(self.wait_state(State.UNLOCKED))
        self.post("/button")                                     # button locks again
        self.assertTrue(self.wait_state(State.SLEEP))

    def test_unknown_path(self):
        with self.assertRaises(urllib.error.HTTPError):
            self.post("/explode")


if __name__ == "__main__":
    unittest.main()
