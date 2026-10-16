'''
Tune the camera tracking for YOUR paddle, lighting and room. Saves vision_config.json,
which live_vision.py and play.py --vision load automatically.

    python calibrate_vision.py
    python calibrate_vision.py --camera 1

Part 1, colour sampling (about 30 seconds):
    1. Hold the RED side flat to the camera so it covers the green box -> SPACE
    2. Hold your open HAND over the box (so skin is never taken as red) -> SPACE
    3. Same with the BLACK side                                         -> SPACE
    4. Put the paddle out of sight, stand where you play               -> SPACE
       (the background check tightens the colours so your room isn't mistaken for the paddle)

Part 2, live tuning: play-test with the sliders in the "Tuning" window.
    Left: camera with tracking overlay (racket arm in white, the grey circle is the
    hand cut-out that is never treated as paddle).  Right: what the computer sees as red / black.
    s = save    r = redo colour sampling    q = quit
'''

import argparse
from dataclasses import asdict

import cv2
import numpy as np

from paddle_vision import (VisionConfig, calibrate_black, calibrate_red,
                           tighten_red_against_skin,
                           false_positive_rates, tighten_for_background)
from tracker import PoseRunner, TrackerCore, apply_exposure, open_camera

BOX = 70                    # half-size of the sampling box, pixels
SAMPLE_FRAMES = 15
MAIN = 'Calibrate paddle tracking'
TUNE = 'Tuning'

# slider name -> (config field, slider max, slider value -> config value, config value -> slider value)
SLIDERS = [
    ('red hue lo',        'red_hue_lo',         179, int,                       int),
    ('red hue hi',        'red_hue_hi',         179, int,                       int),
    ('red sat min',       'red_sat_min',        255, int,                       int),
    ('red val min',       'red_val_min',        255, int,                       int),
    ('black sat max',     'black_sat_max',      255, int,                       int),
    ('black val max',     'black_val_max',      255, int,                       int),
    ('racket arm 0=L 1=R', 'racket_hand',       1,   lambda v: 'right' if v else 'left',
                                                     lambda c: 0 if c == 'left' else 1),
    ('hand cut-out %',    'hand_exclude',       200, lambda v: v / 100,         lambda c: round(c * 100)),
    ('min red %',         'min_red_frac',       30,  lambda v: v / 100,         lambda c: round(c * 100)),
    ('min black %',       'min_black_frac',     30,  lambda v: v / 100,         lambda c: round(c * 100)),
    ('black solidity %',  'min_black_solidity', 100, lambda v: v / 100,         lambda c: round(c * 100)),
    ('search ahead %',    'roi_ahead',          150, lambda v: v / 100,         lambda c: round(c * 100)),
    ('search size %',     'roi_radius',         200, lambda v: max(20, v) / 100, lambda c: round(c * 100)),
    ('noise filter px',   'kernel_px',          15,  lambda v: max(1, v),       int),
    ('steadiness x10',    'min_cutoff',         50,  lambda v: max(1, v) / 10,  lambda c: round(c * 10)),
    ('fast response x1000', 'beta',             100, lambda v: v / 1000,        lambda c: round(c * 1000)),
    ('exposure (0=auto)', 'exposure',           12,  lambda v: -v,              lambda c: -c),
]


def text(img, msg, y, scale=0.7, color=(255, 255, 255)):
    cv2.putText(img, msg, (12, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4)
    cv2.putText(img, msg, (12, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2)


def read_raw(cap):
    ok, frame = cap.read()
    if not ok:
        raise RuntimeError('camera stopped')
    return frame


def read_frame(cap):
    return cv2.flip(read_raw(cap), 1)


def sample_colours(cap, cfg):
    '''Part 1. Returns (red_hsv, black_hsv) samples, or None if the user quit. Updates cfg.'''
    steps = [('red', 'Hold the RED side flat over the box, then SPACE', (40, 40, 230)),
             ('skin', 'Now your open HAND (palm, no paddle) over the box, then SPACE', (120, 160, 220)),
             ('black', 'Now the BLACK side over the box, then SPACE', (40, 40, 40)),
             ('background', 'Hide the paddle, stand where you play, then SPACE', None)]
    samples = {}
    for kind, prompt, color in steps:
        while True:
            frame = read_frame(cap)
            h, w = frame.shape[:2]
            view = frame.copy()
            if color is not None:
                cv2.rectangle(view, (w // 2 - BOX, h // 2 - BOX), (w // 2 + BOX, h // 2 + BOX), (0, 255, 0), 3)
            text(view, prompt, 30)
            text(view, 'q = quit', h - 15, 0.55)
            cv2.imshow(MAIN, view)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                return None
            if key == ord(' '):
                break
        pixels = []
        for _ in range(SAMPLE_FRAMES):
            frame = read_frame(cap)
            if kind == 'background':
                pixels.append(frame)
            else:
                box = frame[h // 2 - BOX:h // 2 + BOX, w // 2 - BOX:w // 2 + BOX]
                pixels.append(cv2.cvtColor(box, cv2.COLOR_BGR2HSV).reshape(-1, 3))
        samples[kind] = pixels if kind == 'background' else np.vstack(pixels)
        if kind == 'red':
            calibrate_red(samples['red'], cfg)
        elif kind == 'skin':
            before = _skin_rate(samples['skin'], cfg)
            tighten_red_against_skin(cfg, samples['red'], samples['skin'])
            print(f'skin counted as red: {before:.0%} -> {_skin_rate(samples["skin"], cfg):.0%}')
        elif kind == 'black':
            calibrate_black(samples['black'], cfg)

    background = np.vstack(samples['background'][::3])  # a few frames stacked into one tall image
    before = false_positive_rates(background, cfg)
    tighten_for_background(cfg, samples['red'], samples['black'], background)
    after = false_positive_rates(background, cfg)
    print(f'red   hue {cfg.red_hue_lo}..{cfg.red_hue_hi}  sat >= {cfg.red_sat_min}  val >= {cfg.red_val_min}')
    print(f'black sat <= {cfg.black_sat_max}  val <= {cfg.black_val_max}')
    print(f'background wrongly marked: red {before[0]:.1%} -> {after[0]:.1%}, '
          f'black {before[1]:.1%} -> {after[1]:.1%}')
    if after[1] > 0.05:
        print('  Lots of the room still looks "black" (dark clothes / shadows?). The search circle\n'
              '  around the wrist limits the damage, but more light on the player helps most.')
    return samples['red'], samples['black']


def _skin_rate(skin_hsv, cfg):
    from paddle_vision import _hit_rate
    return _hit_rate(skin_hsv, cfg, 'red')


def make_sliders(cfg):
    cv2.namedWindow(TUNE, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(TUNE, 520, 620)
    for name, attr, top, _, to_slider in SLIDERS:
        cv2.createTrackbar(name, TUNE, int(np.clip(to_slider(getattr(cfg, attr)), 0, top)), top, lambda v: None)


def set_sliders(cfg):
    for name, attr, top, _, to_slider in SLIDERS:
        cv2.setTrackbarPos(name, TUNE, int(np.clip(to_slider(getattr(cfg, attr)), 0, top)))


def read_sliders(cfg):
    for name, attr, _, from_slider, _ in SLIDERS:
        setattr(cfg, attr, from_slider(cv2.getTrackbarPos(name, TUNE)))
    return cfg


def mask_view(masks):
    red, black = masks
    out = np.zeros((*red.shape, 3), np.uint8)
    out[black > 0] = (110, 110, 110)
    out[red > 0] = (40, 40, 230)
    return out


def main():
    parser = argparse.ArgumentParser(description='Calibrate paddle colour tracking.')
    parser.add_argument('--camera', default=None,
                        help="'laptop', 'usb' or a camera number (default: the one saved by find_camera.py)")
    parser.add_argument('--forehand', choices=('red', 'black'), default='red')
    parser.add_argument('--hand', choices=('left', 'right'), default=None,
                        help='your racket arm (default: from vision_config.json, else right)')
    args = parser.parse_args()

    cfg = VisionConfig.load()
    if args.hand:
        cfg.racket_hand = args.hand
    from camera_select import pick
    cap = open_camera(pick(cfg.camera if args.camera is None else args.camera), cfg)
    pose = PoseRunner()
    core = TrackerCore(args.forehand, cfg)
    try:
        if sample_colours(cap, cfg) is None:
            return
        make_sliders(cfg)
        saved_cfg = None
        exposure = cfg.exposure
        while True:
            read_sliders(cfg)
            core.set_config(cfg)
            if cfg.exposure != exposure:
                exposure = cfg.exposure
                apply_exposure(cap, exposure)
            raw = read_raw(cap)
            hands = pose.hands(raw)                     # pose on the un-mirrored image
            state = core.step(cv2.flip(raw, 1), hands, cv2.getTickCount() / cv2.getTickFrequency(),
                              keep_masks=True)
            left = cv2.cvtColor(state.frame, cv2.COLOR_RGB2BGR)
            right = mask_view(state.masks)
            text(right, f'red   {state.red_frac:5.1%}  (needs {cfg.min_red_frac:.0%})', 30, 0.65, (90, 90, 255))
            text(right, f'black {state.black_frac:5.1%}  (needs {cfg.min_black_frac:.0%})', 60, 0.65)
            saved = asdict(cfg) == saved_cfg
            text(right, ('saved.  ' if saved else '') + 's = save   r = resample colours   q = quit',
                 right.shape[0] - 15, 0.55, (120, 255, 120) if saved else (255, 255, 255))
            cv2.imshow(MAIN, np.hstack([left, right]))
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            if key == ord('s'):
                cfg.save()
                saved_cfg = asdict(cfg)
                print('Saved vision_config.json')
            elif key == ord('r'):
                if sample_colours(cap, cfg) is None:
                    break
                set_sliders(cfg)
        if asdict(cfg) != saved_cfg:
            print('Quit without saving the latest changes (press s to save next time).')
    finally:
        cap.release()
        pose.close()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
