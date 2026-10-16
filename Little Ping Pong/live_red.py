'''
Live test of the fast red-only tracker used by play.py --vision.

    python live_red.py              (--camera 1 for another webcam)

Shows the mirrored, downscaled camera: red pixels tinted, a white line where the
paddle is, yellow lines at the play-area edges (paddle there = table edge).
Prints tracker fps, processing time and the paddle's table x. q = quit.
If the red side isn't picked up, run calibrate_vision.py (red sample only matters).
'''

import argparse

import cv2

from red_tracker import RedTracker


def main():
    parser = argparse.ArgumentParser(description='Live red paddle tracking test.')
    parser.add_argument('--camera', default=None,
                        help="'laptop', 'usb' or a camera number (default: the one saved by find_camera.py)")
    args = parser.parse_args()

    tracker = RedTracker(args.camera)
    tracker.start()
    last_id = -1
    try:
        while True:
            s = tracker.get()
            if s.frame is not None and s.frame_id != last_id:
                last_id = s.frame_id
                view = cv2.resize(cv2.cvtColor(s.frame, cv2.COLOR_RGB2BGR), None, fx=2, fy=2,
                                  interpolation=cv2.INTER_NEAREST)
                cv2.imshow('Red tracking - q to quit', view)
                x = f'{s.paddle_x:+.2f} m' if s.paddle_x is not None else '  -   '
                print(f'\r{s.fps:4.1f} fps  {s.proc_ms:4.1f} ms  paddle x {x}  '
                      f'{"TRACKING" if s.seen else "LOST    "}', end='', flush=True)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        tracker.stop()
        cv2.destroyAllWindows()
        print()


if __name__ == '__main__':
    main()
