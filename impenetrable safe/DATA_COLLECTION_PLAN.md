# Camera Calibration & Data Collection Plan

Goal: get clean, consistent, varied images of 0–5 fingers so the PyTorch model learns *finger count*, not lighting, position, camera shifts or one person's hand.

Setup: USB webcam 9 in above a white background with a red border, connected through the Arduino USB hub (PC → hub → webcam + UnoQ).

---

## 1. Assignment requirements this step must satisfy

| Requirement (from the assignment) | How this step handles it |
|---|---|
| Use the new webcam, **not** the phone or computer camera | `calibrate` picks the camera **that sees the red border**. Only the overhead webcam can see it, so the laptop camera can't be chosen by accident. The index is saved and reused by every script. |
| Same camera on the computer and the UnoQ | Resolution, ROI and the empty-background reference are saved in `data/_calibration/`. The UnoQ uses the same file and the same `preprocess.py`. |
| Fixed distance + red square (hint) | The red border is detected automatically and the crop is taken just **inside** it; collection warns if the camera or paper moves. |
| Model returns the number of fingers | Labels 0–5 (0 = fist or empty square), 6 classes, matches the model and cross-entropy loss. |
| Works for different people's hands (rubric) | Images are stored per person (`data/<person>/<label>/`) so we can validate on a person the model has never seen. |
| 3 is the hardest (hint) | Target for class 3 is 1.5× the others, with prompts to vary which three fingers. |
| PyTorch, built from scratch; train on the computer, not the UnoQ | Our own `FingerNet` (`nn.Module`), no pretrained weights; training only on the PC. |

## 2. Pipeline

```
calibrate  → choose the webcam, find the red square, save the empty-background reference,
             measure baseline sharpness/hand size, and check lighting (PASS/WARN report)
collect    → live view + "what the model sees" panel; every save is quality-checked
audit      → re-check the whole dataset, count classes, find duplicates/outliers,
             write contact sheets of exactly what the model will see
train      → only after the audit passes
```

All four use the same `preprocess.py` (crop → grayscale → 64×64 → [-1, 1]) so training and live use can't drift apart.

## 3. Calibration checks (`python -m src.calibrate`)

| Check | Why | Fix if it fails |
|---|---|---|
| Red square found, tilt ≤ 5° | A tilted square puts table/border in the crop | Rotate the paper or camera |
| Crop has ~no red pixels | The border must not be in the training images | Rerun; thicker border; manual select (`m`) |
| Crop ≥ 100 px per side, roughly square | Enough detail for fingers after resizing to 64×64 | Larger square or camera lower |
| Background brightness in range, no saturation | Too dark = noisy; too bright = light skin washes out | Adjust light |
| Background uniform (low std) | Shadows/glare look like fingers | Diffuse light; move the lamp |
| Stable brightness over time | Flicker or auto-exposure "hunting" changes images | Different light; `--lock-exposure` |
| Open hand covers 10–70% of the crop | Confirms the 9 in distance and square size suit a hand | Adjust camera height or square size |
| Background brightness shift when the hand enters is small | Auto-exposure reacting to the hand makes images inconsistent | `--lock-exposure` |

Calibration also stores **per-setup thresholds** (minimum sharpness, minimum hand size) measured from your own open hand, so the quality gates are tuned to this camera instead of guessed.

### When the camera or lighting changes

| Situation | What to do | Why |
|---|---|---|
| Not sure | `--check` (area empty) | Reports moved border, lighting change or new shadows |
| Lighting changed, camera didn't move | `--relight` | Keeps the same crop (so the model's input geometry is unchanged); redoes the background and hand thresholds |
| Camera bumped | `--view`, line the blue box up with the yellow (saved) box until it says ≥90% match | Putting the camera back is better than recalibrating: the training data stays valid |
| Camera moved/rotated on purpose | full `calibrate` | The report says `[CHANGED]` if the crop shape, orientation or hand coverage differs from before; then collect some new images and retrain |

Every calibration backs up the previous one to `data/_calibration/history/`. `collect` also runs the check at startup and warns on screen.

## 4. Quality gates during collection (every save)

A save is rejected (with the reason shown on screen) if:

| Gate | Detects |
|---|---|
| No hand found for labels 1–5 | Pressed a key with nothing in the square |
| Square mostly covered | Arm/sleeve blocking the view |
| Blurry hand edges | Motion blur; hand moving too fast |
| Too dark / overexposed | Lighting problems |
| Lighting changed vs the reference | Lights turned on/off since calibrating |
| Red border in the crop edge | The camera or paper moved |
| Camera moved (red square checked every 15 frames) | Same, caught even without a save |
| Near-duplicate of the last saved image of that class | Holding a key down without moving; duplicates inflate accuracy |
| Saves faster than 5 per second | Keyboard auto-repeat bursts |

Also: `u` undoes the last save (for mislabels), and every saved image's metrics go to `data/<person>/meta.csv`.

## 5. Collection protocol

**Targets per person:**

| Class | You (main) | Each guest (`--scale 0.4`) |
|---|---|---|
| 0 (fist + empty square) | 80 | 32 |
| 1, 2, 4, 5 | 80 each | 32 each |
| **3** | **120** | 48 |
| Total | ~520 | ~210 |

Guests: at least 2 other people (3 is better) with different hand sizes and skin tones.

**Vary every ~10 shots** (the collect screen shows the next hint):
- position: center, near each edge
- rotation: straight, tilted left, tilted right
- left hand and right hand
- fingers spread vs together; palm flat vs slightly raised
- which fingers: e.g. 3 = index+middle+ring **and** thumb+index+middle **and** middle+ring+pinky; 2 = index+middle and thumb+index; 1 = index and thumb
- class 0: fist in different positions **and** empty square **and** fist with arm partly in view

**Keep constant:** camera height and position, the paper, lighting (if lighting must change, recalibrate first). The arm always enters from the same side (as in real use).

## 6. Audit before training (`python -m src.audit`)

**FAIL** (fix before training): any class under 60 images, unreadable files.

**WARN** (look before training):
- class imbalance over 1.6×
- only 1 person
- over 10% near-duplicates
- many images failing quality checks
- image size not matching the calibration crop
- **hand-size outliers per class** (often a mislabeled image, e.g. a 5 saved as a 1)

Contact sheets in `data/_audit/contact_<label>.png` show 64 random images per class **exactly as the model sees them** (64×64 grayscale). Look at them; mislabels and bad crops are obvious there.

## 7. Automated tests (`python -m unittest discover -s tests -t .`)

These need no camera; they use synthetic images (white background, red border, drawn "hands").

| File | What it proves |
|---|---|
| `test_quality.py` | Red square detection (finds the inside, rejects plain white/solid red, reports tilt); hand detection for darker and lighter skin; blur detection; each quality gate fires on the right bad image and **not** on a good one; duplicate detection; IoU |
| `test_preprocess.py` | Output shape/dtype/range; crop uses the calibration ROI; clear error if not calibrated |
| `test_dataset.py` | Loader skips `_calibration`/`_audit`; held-out-person split has no leakage; audit counts correctly, FAILs on a missing class, finds duplicates; dataset tensors have the right shape/range with augmentation |
| `test_model.py` | Output shape; the model can **overfit a tiny synthetic set** (proves the training loop, loss and labels are wired correctly) |
| `test_lock_policy.py` | Lock + Hall-sensor breach logic (already passing) |

## 8. Acceptance criteria before training

- [ ] Calibration report is all PASS (or each WARN understood)
- [ ] All unit tests pass
- [ ] Audit has no FAIL; WARNs reviewed
- [ ] Contact sheets checked by eye for every class
- [ ] At least 2 people besides me (minimum 1 for a held-out validation person)

## 8b. Update: whole-frame input + guided automatic collection

- The model now sees the **whole camera frame**; the red border is only a guide for where to put the hand. Fingers may cross the border but not the edge of the camera view.
- `python -m src.collect` asks for a name and walks through 0→5. It takes a picture automatically when the hand is steady, sharp, fully in view and slightly moved since the last one. Each count gets 30 pictures (40 for 3) plus up to 5 empty-area pictures for 0. It beeps and shows "SWITCH" with a 3 s pause between counts. Anyone can stop with `q` and resume later with the same name.
- The earlier inside-the-border images and model are archived in `data/_archive/` and `models/_archive/`.

## 9. Commands

```
python -m src.calibrate --view           # live view: border detected? matches the saved one?
python -m src.calibrate --check          # saved calibration still valid? (area empty)
python -m src.calibrate --relight        # lighting changed, camera didn't: keep the crop
python -m src.calibrate                  # full calibration (first time / camera moved)
python -m src.calibrate --probe          # list cameras, say which one sees the red border
python -m src.collect --person victor
python -m src.collect --person guest1 --scale 0.4
python -m src.audit
python -m unittest discover -s tests -t .
```
