'''
Spin experiment, step 2: does the IMU tell the hit types apart?

    python train_strokes.py                 # uses every recordings/strokes/*.csv

Prints an honest accuracy (leave-one-out: each swing is classified by a model
that never saw it), a confusion table (which types get mixed up), and which
measurements separate the types best. Saves models/stroke_model.json for
live_strokes.py.
'''

import argparse
from collections import Counter

import numpy as np

from strokes import (NearestCentroid, features, leave_one_out, load_swings,
                     save_model, separation_scores)

MODEL_PATH = 'models/stroke_model.json'


def main():
    parser = argparse.ArgumentParser(description='Check and train the hit-type classifier.')
    parser.add_argument('--data', default='recordings/strokes/*.csv')
    args = parser.parse_args()

    swings = load_swings(args.data)
    counts = Counter(s.label for s in swings)
    if len(counts) < 2:
        raise SystemExit('Need recorded swings of at least 2 types: run record_strokes.py first.')
    print('Swings per type: ' + ', '.join(f'{k} {v}' for k, v in sorted(counts.items())))

    X = np.array([features(s.samples) for s in swings])
    y = [s.label for s in swings]
    accuracy, confusion = leave_one_out(X, y)
    labels = sorted(counts)
    chance = max(counts.values()) / len(y)

    print(f'\nLeave-one-out accuracy: {accuracy:.0%}   (guessing the most common type: {chance:.0%})\n')
    print('true \\ predicted ' + ''.join(f'{p:>10s}' for p in labels))
    for t in labels:
        print(f'{t:16s} ' + ''.join(f'{confusion.get((t, p), 0):10d}' for p in labels))

    print('\nMost useful measurements (between-type / within-type spread):')
    for name, score in sorted(separation_scores(X, y).items(), key=lambda kv: -kv[1])[:5]:
        print(f'  {name:18s} {score:6.2f}')

    if accuracy >= 0.85:
        verdict = 'Reliable: the IMU separates these hit types well.'
    elif accuracy >= 0.65:
        verdict = 'Promising: record more swings (or make the strokes more distinct) and re-check.'
    else:
        verdict = "The IMU doesn't separate these types reliably yet: don't use it for spin in the game."
    print(f'\n{verdict}')

    save_model(MODEL_PATH, NearestCentroid().fit(X, y))
    print(f'Saved {MODEL_PATH}. Try it live: python live_strokes.py')


if __name__ == '__main__':
    main()
