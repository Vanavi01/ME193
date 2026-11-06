# Step counter: uncertainty analysis

Leave-one-recording-out on 5 recordings (310 walking steps). Every recording is counted by networks that never saw it; each recording is counted by an ensemble of 5 networks started from different random weights.

## Headline

- **Mean absolute error: 13.9% (95% CI 8.0% to 23.0%)**, bootstrap over the 5 recordings.
- Bias (mean predicted − true): -2.5 steps; RMSE 10.0 steps.
- All recordings together: 298 predicted vs 310 true (-4.0%).
- Within the hand-count error (±2 steps): 0 of 4 recordings that have hand counts.

## Per recording

| session | activity | truth | net (mean ± sd) | 95% range of one net | error | % error | peak detector | pedometer |
|---|---|---|---|---|---|---|---|---|
| S18 | walk_normal | 47.0* | 52.1 ± 2.0 | 46.5 to 57.7 | +5.1 | +10.9% | 47 (+0.0) | 49 (+2.0) |
| S19 | walk_fast | 51.5 | 47.7 ± 1.9 | 42.6 to 52.8 | -3.8 | -7.4% | 52 (+0.5) | 54 (+2.5) |
| S21 | walk | 76.0 | 70.5 ± 2.5 | 63.6 to 77.4 | -5.5 | -7.2% | 75 (-1.0) | 77 (+1.0) |
| S22 | stop_go | 79.0 | 88.8 ± 2.7 | 81.3 to 96.2 | +9.8 | +12.4% | 75 (-4.0) | 78 (-1.0) |
| S23 | stop_go | 57.0 | 38.9 ± 2.0 | 33.2 to 44.5 | -18.1 | -31.8% | 59 (+2.0) | 62 (+5.0) |

\* truth is the script's step labels, not a hand count (the hand count was loose).

## Method comparison

| method | mean abs % error | 95% CI | mean signed error (steps) |
|---|---|---|---|
| net: ankle+phone (all channels) | 50.2% | 24.1% to 77.3% | -31.2 |
| net: ankle only | 13.9% | 8.0% to 23.0% | -2.5 |
| net: phone only (all channels) | 44.0% | 16.7% to 77.2% | -25.0 |
| net: ankle+phone motion | 32.5% | 19.2% to 44.9% | -19.4 |
| net: phone motion only | 29.8% | 17.1% to 40.9% | -18.6 |
| peak detector | 2.2% | 0.7% to 3.9% | -0.5 |
| phone pedometer | 4.1% | 1.9% to 6.5% | +1.9 |

Difference in mean absolute % error, net minus baseline (positive = net is worse), paired over recordings:

- vs peak detector: +11.8 points (95% CI +6.4 to +20.4): net is worse.
- vs phone pedometer: +9.8 points (95% CI +4.7 to +16.6): net is worse.

## By activity

| group | recordings | mean abs % error | bias (steps) | mean seed sd |
|---|---|---|---|---|
| stop_go | 2 | 22.1% | -4.2 | 2.4 |
| walk | 3 | 8.5% | -1.4 | 2.1 |

## How to read the uncertainty

1. **Training noise vs. error.** Nets trained from different random starts disagree by about 2.2 steps (mean sd), while the typical error is 8.5 steps (ratio 0.26). Training noise is small next to the error, so averaging more nets would not fix it; the error comes from the nets generalizing imperfectly to a recording they have not seen.
2. **The headline interval has width 15.0 points** (8.0% to 23.0%), resampling only 5 recordings. More recordings narrow it; more seeds do not.
3. **The ground truth is not exact.** Hand counts are good to about ±2 steps (about 3.2% of a typical recording), so error below that cannot be told apart from counting error. S18's hand count was loose, so it is scored against the labels instead; that is circular and flatters the peak detector on that recording.
4. **The labels come from the peak detector.** Training labels were made by the same swing-peak method that serves as the baseline, so the net learns to imitate it. A net that matches the peak detector cannot beat it by much; the hand counts are what keep this honest.
5. **The input set was chosen after seeing results, so the headline is a little optimistic.** Five input sets were tried (see the method table) and the best one on these same recordings, ankle only, is the headline. Adding the phone made the network worse: with all 18 phone channels the error was about 3.6 times higher. A likely cause is that orientation and gravity depend on how the phone sits in the pocket, so with only four training recordings the net can memorize which recording it is looking at. Keeping only gravity-free phone motion helped (about a third lower error) but not enough to beat ankle-only. Only new recordings would give an unbiased check.
6. **Held-out recordings differ from each other** (speed, stops, stomps in the middle), so some folds are hard: a fold that holds out the only recording of a given kind leaves the net nothing to learn that behaviour from.
