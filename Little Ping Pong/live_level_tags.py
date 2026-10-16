'''
Live test: show an AprilTag to the webcam and it is labelled Normal / Hard / Pro.

    python live_level_tags.py               (--camera laptop|usb|N)
    python live_level_tags.py --make-tags   write printable tag PNGs to level_tags/ and exit

Each tag gets a green outline and its level; a tag with no level is outlined grey.
The bar under the label fills while the tag is held steady; when it's full the level
is chosen (shown top-left), the same rule the game will use.

Teaching ("training") it which tag is which level: hold a tag up and press
1 = Normal, 2 = Hard, 3 = Pro. Saved to level_tags.json immediately.
Keys: 1/2/3 = assign the biggest tag in view, r = reset to tags 0/1/2, q = quit.
'''

import argparse
import os

import cv2

from level_tags import (DEFAULT_TAGS, LEVEL_NAMES, LevelChooser, LevelTagDetector, assign,
                        save_tags, tag_image)

GREEN, GREY, YELLOW, WHITE = (0, 220, 0), (160, 160, 160), (0, 220, 255), (255, 255, 255)


def make_tags(folder='level_tags'):
    os.makedirs(folder, exist_ok=True)
    for tag_id, level in sorted(DEFAULT_TAGS.items()):
        img = cv2.cvtColor(tag_image(tag_id), cv2.COLOR_GRAY2BGR)
        img = cv2.copyMakeBorder(img, 0, 60, 0, 0, cv2.BORDER_CONSTANT, value=WHITE)
        cv2.putText(img, f'{LEVEL_NAMES[level]}  (tag {tag_id})', (20, img.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
        path = os.path.join(folder, f'tag{tag_id}_{LEVEL_NAMES[level].lower()}.png')
        cv2.imwrite(path, img)
        print(f'wrote {path}')
    print('Print them about 5-10 cm wide; keep the white margin around each tag.')


def draw(frame, sightings, chooser):
    for s in sightings:
        color = GREEN if s.level else GREY
        cv2.polylines(frame, [s.corners.astype(int)], True, color, 3)
        x, y = s.corners[:, 0].min(), s.corners[:, 1].min() - 12
        cv2.putText(frame, s.label.upper(), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1.1, color, 3)
    if sightings and sightings[0].level:
        cx, cy = sightings[0].center.astype(int)
        cv2.rectangle(frame, (cx - 50, cy + 10), (cx - 50 + int(100 * chooser.progress), cy + 20), YELLOW, -1)
    chosen = LEVEL_NAMES[chooser.level] if chooser.level else 'none yet'
    cv2.putText(frame, f'Level: {chosen}', (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, YELLOW, 2)
    cv2.putText(frame, '1/2/3 = assign tag   r = reset   q = quit', (15, frame.shape[0] - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 1)


def main():
    parser = argparse.ArgumentParser(description='Live AprilTag difficulty test.')
    parser.add_argument('--camera', default=None,
                        help="'laptop', 'usb' or a camera number (default: the one saved by find_camera.py)")
    parser.add_argument('--make-tags', action='store_true', help='write printable tags 0-2 and exit')
    args = parser.parse_args()
    if args.make_tags:
        make_tags()
        return

    from camera_select import pick
    from paddle_vision import VisionConfig
    cap = cv2.VideoCapture(pick(args.camera or VisionConfig.load().camera), cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise SystemExit('Could not open the camera (try --camera usb or a number).')

    detector, chooser = LevelTagDetector(), LevelChooser()
    print('Tags:', ', '.join(f'{t} = {LEVEL_NAMES[lv]}' for t, lv in sorted(detector.tags.items())))
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print('Camera stopped sending frames.')
                break
            sightings = detector.detect(frame)      # detect on the unmirrored frame...
            changed = chooser.update(sightings)
            if changed:
                print(f'Level chosen: {LEVEL_NAMES[changed]}')
            draw(frame, sightings, chooser)         # ...labels stay readable (not mirrored)
            cv2.imshow('AprilTag level - q to quit', frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            if key in (ord('1'), ord('2'), ord('3')):
                if not sightings:
                    print('No tag in view: hold one up, then press 1/2/3.')
                    continue
                level, tag_id = key - ord('0'), sightings[0].tag_id
                detector.tags = assign(detector.tags, tag_id, level)
                save_tags(detector.tags)
                chooser.__init__(chooser.stable_frames)
                print(f'Tag {tag_id} is now {LEVEL_NAMES[level]} (saved to level_tags.json)')
            if key == ord('r'):
                detector.tags = dict(DEFAULT_TAGS)
                save_tags(detector.tags)
                print('Reset: tag 0 = Normal, 1 = Hard, 2 = Pro')
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
