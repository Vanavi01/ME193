"""Live viewer for Model 2: subscribes to the Uno Q's MQTT topic and shows the
predicted activity as a pictogram (same 8x13 drawings as the LED matrix) with the
model's confidence and per-class probabilities.

    pip install paho-mqtt
    python mqtt_viewer.py [broker] [topic]
"""
import json
import queue
import sys
import time
import tkinter as tk

import paho.mqtt.client as mqtt

BROKER = sys.argv[1] if len(sys.argv) > 1 else "broker.hivemq.com"   # match MQTT_BROKER in main.py
TOPIC = sys.argv[2] if len(sys.argv) > 2 else "tufts/gait/model2/state"
PORT = 1883
STALE_S = 5   # no message for this long -> grey out (board stopped / disconnected)

# Same pictograms as the sketch, 8 rows x 13 cols ('#' = lit).
PICTO = {
    "stand": ["......##.....",
              "......##.....",
              "....######...",
              "....#.##.#...",
              "....#.##.#...",
              "......##.....",
              ".....#..#....",
              ".....#..#...."],
    "walk": ["......##.....",
             "......##.....",
             ".....####....",
             ".....#.##.#..",
             ".....#.##.#..",
             "......##.....",
             ".....##.##...",
             "....##...##.."],
    "run": ["........##...",
            "........##...",
            "....#..###...",
            ".....####.#..",
            "......###....",
            ".....##.##...",
            "....#.....#..",
            "...##.....##."],
    "bike": [".............",
             "...###....##.",
             "....#..####..",
             "....#.#...#..",
             ".###.#...###.",
             "#.#########.#",
             "#...#...#...#",
             ".###.....###."],
}
PICTO["air_bike"] = PICTO["bike"]

BG, FG, DIM, OFF = "#111418", "#f2f4f7", "#5b6470", "#1d2229"
COLOR = {"stand": "#4ea1ff", "walk": "#3ecf8e", "run": "#ff8a3d", "bike": "#c77dff", "air_bike": "#c77dff"}
CELL, GAP = 34, 4

inbox = queue.Queue()


def on_connect(client, userdata, flags, reason_code, properties=None):
    client.subscribe(TOPIC)
    inbox.put(("status", f"connected to {BROKER}  ·  {TOPIC}"))


def on_disconnect(client, userdata, *args):
    inbox.put(("status", f"disconnected from {BROKER}, retrying…"))


def on_message(client, userdata, msg):
    try:
        inbox.put(("msg", json.loads(msg.payload)))
    except ValueError:
        pass


class Viewer:
    def __init__(self, root):
        self.root = root
        self.last = None        # last message
        self.last_rx = 0.0      # local receive time, so board/PC clock skew doesn't matter
        root.title("Gait — Model 2 live")
        root.configure(bg=BG)

        w = 13 * CELL + 12 * GAP
        h = 8 * CELL + 7 * GAP
        self.canvas = tk.Canvas(root, width=w, height=h, bg=BG, highlightthickness=0)
        self.canvas.pack(padx=32, pady=(32, 12))
        self.cells = [[self.canvas.create_rectangle(
            c * (CELL + GAP), r * (CELL + GAP),
            c * (CELL + GAP) + CELL, r * (CELL + GAP) + CELL, fill=OFF, width=0)
            for c in range(13)] for r in range(8)]

        self.label = tk.Label(root, text="waiting…", font=("Segoe UI", 36, "bold"), bg=BG, fg=FG)
        self.label.pack()
        self.conf = tk.Label(root, text="", font=("Segoe UI", 22), bg=BG, fg=DIM)
        self.conf.pack(pady=(0, 16))

        self.bars = tk.Frame(root, bg=BG)
        self.bars.pack(fill="x", padx=32)
        self.bar_rows = {}

        self.status = tk.Label(root, text=f"connecting to {BROKER}…", font=("Segoe UI", 10),
                               bg=BG, fg=DIM)
        self.status.pack(pady=(16, 12))
        self.poll()

    def bar_row(self, cls):
        if cls not in self.bar_rows:
            row = tk.Frame(self.bars, bg=BG)
            row.pack(fill="x", pady=3)
            tk.Label(row, text=cls, width=9, anchor="w", font=("Segoe UI", 12),
                     bg=BG, fg=FG).pack(side="left")
            track = tk.Canvas(row, height=16, bg=OFF, highlightthickness=0)
            track.pack(side="left", fill="x", expand=True, padx=8)
            pct = tk.Label(row, width=5, anchor="e", font=("Segoe UI", 12), bg=BG, fg=FG)
            pct.pack(side="left")
            self.bar_rows[cls] = (track, pct)
        return self.bar_rows[cls]

    def draw(self, msg, stale):
        label = msg.get("label", "?")
        color = DIM if stale else COLOR.get(label, FG)
        grid = PICTO.get(label)
        for r in range(8):
            for c in range(13):
                on = grid is not None and grid[r][c] == "#"
                self.canvas.itemconfigure(self.cells[r][c], fill=color if on else OFF)
        self.label.configure(text=label.replace("_", " ").upper(), fg=color)
        self.conf.configure(text=f"{msg.get('confidence', 0) * 100:.0f}% confidence")

        for cls, p in sorted(msg.get("probs", {}).items()):
            track, pct = self.bar_row(cls)
            track.update_idletasks()
            track.delete("all")
            fill = DIM if stale or cls != label else COLOR.get(cls, FG)
            track.create_rectangle(0, 0, track.winfo_width() * p, 16, fill=fill, width=0)
            pct.configure(text=f"{p * 100:.0f}%")

    def poll(self):
        while not inbox.empty():
            kind, data = inbox.get()
            if kind == "status":
                self.status.configure(text=data)
            else:
                self.last, self.last_rx = data, time.time()
        if self.last is not None:
            stale = time.time() - self.last_rx > STALE_S
            self.draw(self.last, stale)
            if stale:
                self.conf.configure(text=f"no update for {time.time() - self.last_rx:.0f} s")
        self.root.after(100, self.poll)


def main():
    if hasattr(mqtt, "CallbackAPIVersion"):  # paho 2.x
        client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    else:
        client = mqtt.Client()
    client.on_connect = on_connect
    client.on_disconnect = on_disconnect
    client.on_message = on_message
    client.connect_async(BROKER, PORT, keepalive=30)
    client.loop_start()

    root = tk.Tk()
    Viewer(root)
    try:
        root.mainloop()
    finally:
        client.loop_stop()
        client.disconnect()


if __name__ == "__main__":
    main()
