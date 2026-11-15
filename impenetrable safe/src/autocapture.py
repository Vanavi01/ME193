"""Guided automatic capture: decides, frame by frame, whether to take a picture and which
finger count to ask for next. No camera or window code (testable).

Prompts come in RANDOM order, AUTO_BURST pictures at a time, so no count is recorded in one
continuous burst (otherwise the model can learn "what that 30-second burst looked like" -
lighting, arm angle, hand position - instead of counting fingers)."""
import random

import cv2
import numpy as np

from . import config
from .quality import check_sample, signature


class AutoCapture:
    def __init__(self, targets, counts, calib, bg, burst=None, seed=None, classes=None):
        self.classes = list(classes or config.CLASSES)
        self.targets = dict(targets)
        self.counts = {c: counts.get(c, 0) for c in self.classes}
        self.calib, self.bg = calib, bg
        self.burst = burst or config.AUTO_BURST
        self.rng = random.Random(seed)
        self.prev = None
        self.steady = 0
        self.motion = 0.0
        self.last_save = -1e9
        self.last_sig = None              # last saved picture (any count)
        self.switch_until = 0.0
        self.empties = 0
        self.paused = False
        self.label = None
        self.in_burst = 0
        self._pick(0.0, pause=False)

    # ---------------------------------------------------------------- state
    @property
    def done(self):
        return self.label is None

    def total(self):
        return sum(self.counts.values()), sum(self.targets.values())

    def _pick(self, now, pause=True):
        """Next prompt: a random unfinished count, different from the current one if possible."""
        left = [c for c in self.classes if self.counts[c] < self.targets[c]]
        if not left:
            self.label = None
            return
        choices = [c for c in left if c != self.label] or left
        self.label = self.rng.choice(choices)
        self.in_burst = 0
        if pause:
            self.switch_until = now + config.SWITCH_PAUSE_S

    # ---------------------------------------------------------------- per frame
    def step(self, crop, now, moved=False):
        """Returns {"event": "capture"|None|"done", "reason": why not, "metrics": ...}.
        On capture also "label", "empty", and "next" when the prompt changes."""
        g = cv2.resize(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), (80, 60),
                       interpolation=cv2.INTER_AREA).astype(np.float32)
        self.motion = float(np.abs(g - self.prev).mean()) if self.prev is not None else 99.0
        self.prev = g
        self.steady = self.steady + 1 if self.motion < config.MOTION_MAX else 0
        if self.done:
            return {"event": "done", "reason": "all counts finished", "metrics": None,
                    "warnings": []}

        label = self.label
        issues, m = check_sample(crop, label, self.calib, self.bg)
        if moved:
            issues.append("camera moved: recalibrate")
        hand_min = self.calib.get("hand_frac_min", config.DEFAULT_HAND_FRAC_MIN)
        empty = m["hand_blob"] is not None and m["hand_blob"] < hand_min
        if label == 0 and empty and self.empties >= config.AUTO_EMPTY_MAX:
            issues.append("make a FIST (enough empty pictures)")
        blocking = [i for i in issues if any(k in i for k in config.AUTO_BLOCKING)]
        warnings = [i for i in issues if i not in blocking]

        sig = None
        if self.paused:
            reason = "paused (SPACE to resume)"
        elif now < self.switch_until:
            reason = "switching"
        elif blocking:
            reason = blocking[0]
        elif self.steady < config.STEADY_FRAMES:
            reason = "hold still"
        elif now - self.last_save < config.AUTO_INTERVAL_S:
            reason = "wait"
        else:
            sig = signature(crop)
            # empty pictures all look alike; they're capped by AUTO_EMPTY_MAX instead
            too_similar = (not empty and self.last_sig is not None
                           and float(np.abs(sig - self.last_sig).mean()) < config.AUTO_MIN_CHANGE)
            reason = "move your hand more" if too_similar else None
        if reason:
            return {"event": None, "reason": reason, "metrics": m, "warnings": warnings}

        self.last_sig = sig
        self.last_save = now
        self.counts[label] += 1
        self.in_burst += 1
        if label == 0 and empty:
            self.empties += 1
        ev = {"event": "capture", "reason": None, "metrics": m, "label": label, "empty": empty,
              "warnings": warnings}
        if self.in_burst >= self.burst or self.counts[label] >= self.targets[label]:
            self._pick(now)
            ev["next"] = self.label
        return ev

    # ---------------------------------------------------------------- controls
    def undo(self, label, was_empty=False):
        self.counts[label] = max(0, self.counts[label] - 1)
        if was_empty:
            self.empties = max(0, self.empties - 1)
        self.last_sig = None
        self.label = label                 # ask for that count again
        self.in_burst = max(0, self.in_burst - 1)
        self.switch_until = 0.0

    def skip(self, now):
        if not self.done:
            self._pick(now)
