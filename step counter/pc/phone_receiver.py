"""Receive live phone IMU data from the Sensor Logger app (HTTP push) and save it as CSV.

Phone setup (Sensor Logger, iOS or Android):
  Settings -> Data Streaming -> HTTP Push: on, URL = http://<this PC's IP>:8000/data
  Turn on Accelerometer, Gyroscope, Gravity, Orientation, Magnetometer, Barometer, Pedometer
  and set the sampling rate to 100 Hz (or the highest offered). Then press Start Recording.

    python phone_receiver.py [session_name] [port]

Each sensor goes to its own file, data/phone/<session>_<sensor>.csv, with columns
t_ns (phone clock, epoch nanoseconds), rx_time (PC clock, s), then that sensor's values.
A one-line rate summary per sensor prints every 2 s so you can see the stream is alive.
"""
import csv
import json
import os
import socket
import sys
import threading
import time
from collections import Counter
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

SESSION = sys.argv[1] if len(sys.argv) > 1 else time.strftime("P%Y%m%d_%H%M%S")
PORT = int(sys.argv[2]) if len(sys.argv) > 2 else 8000

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "..", "data", "phone")

lock = threading.Lock()
writers = {}     # sensor name -> (file, csv.writer, value keys)
counts = Counter()
seen = set()


def write_reading(r, rx):
    name = r.get("name", "unknown").lower()
    values = r.get("values", {})
    if name not in writers:
        keys = sorted(values)
        f = open(os.path.join(OUT_DIR, f"{SESSION}_{name}.csv"), "w", newline="")
        w = csv.writer(f)
        w.writerow(["t_ns", "rx_time"] + keys)
        writers[name] = (f, w, keys)
        print(f"  new sensor: {name} {keys}")
    f, w, keys = writers[name]
    w.writerow([r.get("time"), f"{rx:.3f}"] + [values.get(k) for k in keys])
    counts[name] += 1


class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        rx = time.time()
        body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        try:
            msg = json.loads(body)
        except ValueError:
            self.send_response(400)
            self.end_headers()
            return
        with lock:
            for r in msg.get("payload", []):
                write_reading(r, rx)
        self.send_response(200)
        self.end_headers()

    def do_GET(self):  # lets you check from the phone's browser that the PC is reachable
        body = b"phone_receiver is running. Point Sensor Logger HTTP Push here."
        self.send_response(200)
        self.send_header("Content-Type", "text/plain")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):
        # Only log the first request of each kind (a data stream would flood the console).
        key = (self.command, self.path)
        if key not in seen:
            seen.add(key)
            print(f"  request from {self.client_address[0]}: {self.command} {self.path}")


def local_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))  # no packet is sent; picks the outbound interface
        return s.getsockname()[0]
    finally:
        s.close()


def report():
    last = Counter()
    while True:
        time.sleep(2)
        with lock:
            for f, _, _ in writers.values():
                f.flush()
            now = counts.copy()
        if now:
            rates = "  ".join(f"{k} {(now[k] - last[k]) / 2:.0f}Hz" for k in sorted(now))
            print(f"[{time.strftime('%H:%M:%S')}] {rates}")
        last = now


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"Session {SESSION} -> {os.path.abspath(OUT_DIR)}")
    print(f"Point Sensor Logger HTTP Push at  http://{local_ip()}:{PORT}/data")
    print("Waiting for data (Ctrl+C to stop)...")
    threading.Thread(target=report, daemon=True).start()
    server = ThreadingHTTPServer(("0.0.0.0", PORT), Handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        with lock:
            for f, _, _ in writers.values():
                f.close()
        print("Saved:", dict(counts))


if __name__ == "__main__":
    main()
