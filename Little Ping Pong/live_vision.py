'''
Step 3 hardware test: check paddle tracking and forehand/backhand on the webcam.

    python live_vision.py
    python live_vision.py --camera 1 --forehand black --hand left

Window shows the mirrored webcam with: the racket forearm (white, labelled
"racket"; the other arm is grey and ignored), the grey hand cut-out, the yellow
search circle, a dot on the detected paddle face, and FOREHAND/BACKHAND.
Keys:  m = show the red/black colour masks (for checking lighting)
       p = save a snapshot (camera + masks) to snapshots/ to share when asking for help
       q = quit
If detection is poor, run calibrate_vision.py first (it saves vision_config.json).

Expected: red side to the camera -> FOREHAND (with the default --forehand red),
black side -> BACKHAND, edge-on -> keeps the last side briefly, then NO PADDLE FACE.
'''

import argparse
import os
import time
from datetime import datetime

import cv2
import numpy as np

from paddle_vision import VisionConfig
from tracker import VisionTracker


def main():
    parser = argparse.ArgumentParser(description='Live paddle tracking test.')
    parser.add_argument('--camera', default=None,
                        help="'laptop', 'usb' or a camera number (default: the one saved by find_camera.py)")
    parser.add_argument('--forehand', choices=('red', 'black'), default='red',
                        help='which rubber is your forehand side')
    parser.add_argument('--hand', choices=('left', 'right'), default=None,
                        help='your racket arm (default: from vision_config.json, else right)')
    args = parser.parse_args()

    from camera_select import pick
    camera = pick(VisionConfig.load().camera if args.camera is None else args.camera)
    tracker = VisionTracker(camera, args.forehand, keep_masks=True, racket_hand=args.hand)
    tracker.start()
    show_masks = False
    last_id = -1
    last_print = 0.0
    try:
        while True:
            state = tracker.get()
            if state.frame is not None and state.frame_id != last_id:
                last_id = state.frame_id
                view = cv2.cvtColor(state.frame, cv2.COLOR_RGB2BGR)
                if show_masks and state.masks is not None:
                    red, black = state.masks
                    masks = np.zeros_like(view)
                    masks[black > 0] = (90, 90, 90)
                    masks[red > 0] = (40, 40, 230)
                    view = np.hstack([view, masks])
                cv2.putText(view, f'{state.fps:4.1f} fps   red {state.red_frac:4.0%}   black {state.black_frac:4.0%}',
                            (12, view.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.imshow('Paddle tracking - m: masks, q: quit', view)
            now = time.perf_counter()
            if now - last_print > 0.5 and state.paddle:
                last_print = now
                print(f'\r{(state.stroke or "-"):9s} side={str(state.side):5s} hand={state.hand}  '
                      f'paddle=({state.paddle[0]:+.2f}, {state.paddle[1]:+.2f}) m   {state.fps:4.1f} fps   ',
                      end='', flush=True)
            key = cv2.waitKey(5) & 0xFF
            if key == ord('q'):
                break
            if key == ord('m'):
                show_masks = not show_masks
            if key == ord('p') and state.frame is not None and state.masks is not None:
                os.makedirs('snapshots', exist_ok=True)
                red, black = state.masks
                masks = np.zeros_like(state.frame)
                masks[black > 0] = (90, 90, 90)
                masks[red > 0] = (230, 40, 40)
                path = os.path.join('snapshots', f'vision_{datetime.now():%Y%m%d_%H%M%S}.png')
                cv2.imwrite(path, cv2.cvtColor(np.hstack([state.frame, masks]), cv2.COLOR_RGB2BGR))
                print(f'\nSaved {path}')
    finally:
        tracker.stop()
        cv2.destroyAllWindows()
        print()


if __name__ == '__main__':
    main()
