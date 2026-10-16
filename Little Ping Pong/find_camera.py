'''
Find the webcams on this computer and choose which one the game uses.

    python find_camera.py              # list cameras by name, suggest the best live one, ask to save it
    python find_camera.py --use laptop # just switch: 'laptop', 'usb' or a camera number
    python find_camera.py --save       # save the suggestion without asking
    python find_camera.py --preview    # also show each live camera for 2 s so you can see which is which

Camera numbers change when webcams are plugged in or out, and virtual cameras
(NVIDIA Broadcast, OBS) also take numbers but show a dark/blank picture. A camera
counts as "live" if its picture isn't black and it delivers at least 15 fps.
The choice is saved as "camera" in vision_config.json ('laptop' / 'usb' when possible,
so it survives the numbers shifting), which play.py, drill_play.py, live_red.py and
calibrate_vision.py use when --camera isn't given.
'''

import argparse
import contextlib
import os
import time
from dataclasses import dataclass

import cv2
import numpy as np

from camera_select import list_cameras, pick
from paddle_vision import VisionConfig

MAX_INDEX = 6
MIN_BRIGHTNESS = 15
MIN_FPS = 15


@dataclass
class CameraInfo:
    index: int
    width: int
    height: int
    fps: float
    brightness: float

    @property
    def live(self):
        return self.brightness >= MIN_BRIGHTNESS and self.fps >= MIN_FPS


def pick_camera(cameras):
    '''Best live camera: highest frame rate, then resolution. None if none is live.'''
    live = [c for c in cameras if c.live]
    if not live:
        return None
    return max(live, key=lambda c: (round(c.fps), c.width * c.height, -c.index))


@contextlib.contextmanager
def quiet_stdout():
    '''Some virtual-camera drivers print pages of log lines when opened; hide them.'''
    fd = os.dup(1)
    with open(os.devnull, 'w') as null:
        os.dup2(null.fileno(), 1)
        try:
            yield
        finally:
            os.dup2(fd, 1)
            os.close(fd)


def probe(index, seconds=1.5, preview=False):
    with quiet_stdout():
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if not cap.isOpened():
            return None
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 60)
        frame = None
        for _ in range(5):
            ok, f = cap.read()
            frame = f if ok else frame
        t, n = time.perf_counter(), 0
        while time.perf_counter() - t < seconds:
            ok, f = cap.read()
            if ok:
                n += 1
                frame = f
        cap.release()
    if frame is None:
        return CameraInfo(index, 0, 0, 0.0, 0.0)
    info = CameraInfo(index, frame.shape[1], frame.shape[0], n / seconds, float(frame.mean()))
    if preview and info.live:
        view = cv2.flip(frame, 1)
        cv2.putText(view, f'camera {index}', (12, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
        cv2.imshow('find_camera', view)
        cv2.waitKey(2000)
    return info


def main():
    parser = argparse.ArgumentParser(description='Find and choose the webcam.')
    parser.add_argument('--save', action='store_true', help='save the suggestion without asking')
    parser.add_argument('--preview', action='store_true', help='show each live camera for 2 s')
    parser.add_argument('--use', default=None, help="switch straight to 'laptop', 'usb' or a camera number")
    args = parser.parse_args()
    cfg = VisionConfig.load()

    if args.use:                                    # quick switch, no probing
        pick(args.use)                              # checks it exists right now (prints which one)
        cfg.camera = int(args.use) if args.use.isdigit() else args.use
        cfg.save()
        print(f'Saved: the game will use the {args.use!r} camera.')
        return

    names = {c.index: c for c in list_cameras()}
    print('Checking cameras (a few seconds each)...')
    cameras = []
    for i in range(MAX_INDEX):
        dev = names.get(i)
        if dev is not None and dev.infrared:
            print(f'  camera {i}: {dev.name}  (infrared: skipped)')
            continue
        info = probe(i, preview=args.preview)
        if info is None:
            continue
        cameras.append(info)
        status = 'LIVE' if info.live else 'dark/blank (virtual camera, or lens covered)'
        name = dev.name if dev else '?'
        print(f'  camera {i}: {name:40s} {info.width}x{info.height}  {info.fps:4.1f} fps  '
              f'brightness {info.brightness:5.1f}  {status}')
    cv2.destroyAllWindows()

    best = pick_camera(cameras)
    if best is None:
        print('\nNo live camera found. Is a webcam plugged in, with its cover open?')
        return
    dev = names.get(best.index)
    choice = best.index                             # save by kind when we can: numbers shift
    if dev is not None and dev.physical:
        choice = 'laptop' if dev.builtin else 'usb'
    label = f'{dev.name} ({choice!r})' if dev else f'camera {best.index}'
    print(f'\nSuggested: {label}   (currently saved: {cfg.camera!r})')
    if args.save or input('Use it for the game? [Y/n] ').strip().lower() != 'n':
        cfg.camera = choice
        cfg.save()
        print(f'Saved: the game will use {label}. Switch any time: python find_camera.py --use laptop|usb')


if __name__ == '__main__':
    main()
