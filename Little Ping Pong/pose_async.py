'''
MediaPipe Pose in LIVE_STREAM (async) mode: frames are handed over without waiting,
MediaPipe runs inference on its own thread, skips frames while busy, and calls back
with results. So the red tracker and the game never wait for pose.

    pose = AsyncPose()
    pose.submit(raw_small_bgr, t)     # un-mirrored frame, any size (small = faster)
    latest = pose.latest()            # PoseInfo or None
    pose.close()
'''

import threading
import time

import cv2

from pose_logic import pose_from_landmarks
from tracker import MODEL


class AsyncPose:

    def __init__(self):
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions, vision
        self._mp = mp
        self._lock = threading.Lock()
        self._latest = None
        self._frame_t = {}              # timestamp_ms -> capture time, to stamp results
        self._last_ms = -1
        self._results = 0
        self._t0 = time.perf_counter()
        options = vision.PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=MODEL),
            running_mode=vision.RunningMode.LIVE_STREAM,
            num_poses=1,
            result_callback=self._on_result)
        self._landmarker = vision.PoseLandmarker.create_from_options(options)

    def submit(self, raw_bgr, t):
        '''Hand a frame to MediaPipe. Returns immediately.'''
        ms = max(self._last_ms + 1, int((t - self._t0) * 1000))
        self._last_ms = ms
        with self._lock:
            self._frame_t[ms] = t
            if len(self._frame_t) > 64:             # forget stamps of frames MediaPipe dropped
                for k in sorted(self._frame_t)[:-32]:
                    del self._frame_t[k]
        rgb = cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2RGB)
        self._landmarker.detect_async(self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb), ms)

    def _on_result(self, result, _image, timestamp_ms):
        with self._lock:
            t = self._frame_t.pop(timestamp_ms, time.perf_counter())
        info = pose_from_landmarks(result.pose_landmarks[0], t) if result.pose_landmarks else None
        with self._lock:
            self._results += 1
            if info is not None:            # keep the last person seen; users check its age
                self._latest = info

    def latest(self):
        with self._lock:
            return self._latest

    def result_count(self):
        with self._lock:
            return self._results

    def close(self):
        self._landmarker.close()
