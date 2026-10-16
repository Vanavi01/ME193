'''
Spin experiment, step 3: swing forehands and see which hit type is detected.

    python live_strokes.py               (--serial 0994)

Needs models/stroke_model.json from train_strokes.py. Prints one line per
swing: the detected type and a confidence (0 = coin flip between the two
closest types, higher = clearer). Separate from the game on purpose. Ctrl+C quits.
'''

import argparse
import time
from collections import Counter

from paddle_imu import PaddleIMU
from strokes import StrokeCollector, features, load_model
from swing import gyro_mag
from train_strokes import MODEL_PATH


def main():
    parser = argparse.ArgumentParser(description='Live hit-type detection.')
    parser.add_argument('--serial', default=None)
    args = parser.parse_args()

    model = load_model(MODEL_PATH)
    paddle = PaddleIMU()
    print('Connecting to Double Motor...')
    paddle.connect(card_serial=args.serial)
    print(f'Connected. Types: {", ".join(model.labels)}. Swing forehands (red side). Ctrl+C to quit.\n')
    collector = StrokeCollector()
    seen = Counter()
    try:
        while True:
            for s in paddle.drain():
                window = collector.update(s)
                if window is None:
                    continue
                label, confidence = model.predict(features(window))
                seen[label] += 1
                bar = '#' * round(confidence * 20)
                peak = max(gyro_mag(w) for w in window)
                print(f'{label.upper():9s} confidence {confidence:4.2f} [{bar:<20}]   {peak:5.0f} deg/s')
            time.sleep(0.005)
    except KeyboardInterrupt:
        print('\nDetected: ' + ', '.join(f'{k} {v}' for k, v in seen.most_common()))
    finally:
        paddle.disconnect()


if __name__ == '__main__':
    main()
