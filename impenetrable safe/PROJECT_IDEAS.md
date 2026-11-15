# Impenetrable Safe: Ideas & Implementation

Numbering used here (counting the assignment paragraphs):
1. Notion section: V-data fit with gradient descent
2. **The lock: PyTorch finger-counting CNN + safe logic** (this file starts here)
3. Notion page (video, GitHub, writeup)
4. Stop-action movie with voice-over
5. **Something fun/cool we haven't done in class** (the break-in mode)

---

## Current setup

- Webcam fixed 9 in above the table, pointing straight down, white background.
- Servo locks/unlocks the safe.
- UnoQ must drive: **3 LEDs + 1 button + 1 servo**.
- Plan: train on the PC first, then move the trained model to the UnoQ (no training on the UnoQ).

---

## Task 2: The lock

### 2.1 Hardware / user flow

1. Walk up, press the button → LEDs light up, camera starts looking.
2. Place fingers (1–5) in the marked square, one digit at a time, three digits total.
3. Correct code → servo unlocks. Wrong → stays locked.

**LED roles (3 LEDs):**

| LED | Meaning |
|-----|---------|
| Yellow | Awake / listening for the next digit |
| Green | Unlocked (or "digit accepted" blink) |
| Red | Wrong code / locked out |

Digit progress can be shown by blinking yellow 1, 2, 3 times (no extra LEDs needed).

**Button:** wake the system; long-press (hold 3 s) to re-lock manually.

**UnoQ split of work (it has a Linux side and an MCU side):**
- Linux/Python side: webcam capture + PyTorch/ONNX inference + lock policy.
- MCU sketch side: LED pins, button (`INPUT_PULLUP`), servo PWM, talking to Python through the Arduino Bridge.
- Keep the camera on the same USB webcam in both stages (the assignment requires it). A USB hub will probably be needed for webcam + power.

### 2.2 Data collection (do this first)

- Draw a **red square** on the white background; crop the frame to just that area, so the images are small (e.g. 64×64 grayscale or RGB).
- Classes: **0, 1, 2, 3, 4, 5** fingers (6 outputs; 0 = fist/empty is useful as the "no digit" class).
- Target: ~200+ raw images per class, **multiple people's hands**, different lighting, left and right hands, both orientations of the hand, different skin tones and sleeves/rings. Grading says "worked for different people's hands".
- Hint from the assignment: **3 is the hardest**. Collect extra 3s (thumb-in-or-out variations: index+middle+ring, vs thumb+index+middle).
- Fix the 9 in height and don't move the camera; check the focus and exposure once and lock them (turn off auto-exposure if the webcam allows it) so training and UnoQ images match.
- Write a small `collect.py`: press a key 0–5 to save the current cropped frame into `data/<label>/`.
- Split by **person** (not random frames) for the validation set, otherwise accuracy will look better than it really is.

### 2.3 Model (PyTorch, from scratch)

Architecture, following the hint:

```
input 1×64×64 (or 3×64×64)
 conv1 (3×3, 8 ch)  → ReLU → MaxPool(2)   → 32×32
 conv2 (3×3, 16 ch) → ReLU → MaxPool(2)   → 16×16
 conv3 (3×3, 32 ch) → ReLU → MaxPool(2)   → 8×8
 flatten → Linear → 6 logits
loss: CrossEntropyLoss, optimizer: Adam (or SGD to tie back to gradient descent)
```

Optional twist from the hint: pool the output of each conv layer and concatenate all three before the final linear layer (multi-scale features).

Concepts to cover in the writeup/movie:
- **Width** = channels per layer (8/16/32), **depth** = number of layers (3 conv + 1 linear).
- Convolution = a small learned filter sliding across the image; pooling = shrink and keep the strongest response (adds some shift tolerance).
- Backprop = chain rule computing dLoss/dWeight for every filter weight; gradient descent = step each weight against its gradient.
- Cross entropy + softmax because we classify (not regress) a count.

Augmentation (the hint says it improved accuracy): random rotation (±15°), shifts (±10%), small brightness/contrast jitter, maybe horizontal flip (so left/right hands both work). Do NOT apply flips/rotations that change the finger count meaning (none do, but verify visually).

Outputs worth saving for the Notion page: loss and accuracy curves, a confusion matrix (expect 3 to be confused with 2 and 4), and a grid of misclassified images.

### 2.4 Moving to the UnoQ

- Export the trained weights: `torch.save(state_dict)` and also try `torch.onnx.export`; ONNX Runtime is usually much lighter than full PyTorch on a small ARM board.
- Check early whether PyTorch installs on the UnoQ. If it doesn't, use ONNX Runtime (or TorchScript), since the assignment only requires that the trained model is from PyTorch.
- Use the identical preprocessing (crop, resize, normalization) in a shared `preprocess.py`, and run the same test images on both machines to confirm the logits match.
- Measure for the "UnoQ vs computer" question: inference time per frame, FPS, CPU/RAM use, install difficulty, accuracy on the same 100 test images, any difference in webcam behavior.

### 2.5 Lock policy (how it decides)

Draft policy, to refine in testing:

1. A reading only counts if the **softmax confidence ≥ 0.9** (tune this).
2. The same class must be predicted for **N consecutive frames** (e.g. 10 frames ≈ 1 s) so a hand moving in and out of the square doesn't register.
3. Once a digit is accepted: green blink, and the hand must return to **0 / empty** (fist or hand removed) before the next digit is read. This avoids double-counting and gives a natural separator.
4. Digits must be three distinct numbers from 1–5 (per the assignment), e.g. `2-5-3`.
5. After 3 digits, compare with the stored code → servo unlocks (e.g. 90°) or red LED + stays locked.
6. **Timeout**: if there's no input for ~10 s, reset the sequence and go to sleep.
7. **Lockout**: 3 failed attempts → 30 s lockout with red LED.
8. Store the code as a salted hash, not plain text (nice touch, cheap to do).

### 2.5b Limitations (need at least 4)

1. Lighting changes (shadows, a lamp, sunlight) shift the images away from the training data.
2. Hands the model hasn't seen (different size, skin tone, nails, gloves, rings) may be misclassified; 3 is especially error-prone.
3. Camera must stay at 9 in and in the same position; if it's bumped, accuracy drops.
4. Spoofing: a printed photo of a hand or a model hand could fool it (it counts fingers, it doesn't verify *who*), and the code is visible to anyone watching (shoulder surfing).
5. Only 3 digits from 1–5 → 60 possible codes (5·4·3); easy to brute force without a lockout.
6. Servo lock is mechanically weak; anyone can force the box.
7. The UnoQ might be slower than the PC, so frame rate and responsiveness may drop.

---

## Task 5 (chosen direction): Hall-effect door sensor + forced-entry detection

A magnet on the door and a Hall sensor on the frame tell the safe whether the door is open or closed. Combined with the servo state, that gives the "was it forced?" logic:

| Servo | Door sensor | Meaning |
|---|---|---|
| locked | closed → **open** | **BREACH**: door opened without a valid unlock → red LED fast flash + alarm |
| unlocked (valid code) | closed → open | normal use, no alarm |
| unlocked | opened, then closed for 2 s | auto **re-lock** and go to sleep |
| unlocked | never opened for 10 s | auto re-lock |
| locked | stays closed | idle |

Details (all implemented and unit-tested in `src/lock_policy.py`):
- The raw sensor reading must hold 0.2 s (debounce) so vibration or a magnet wobble doesn't false-alarm.
- No alarm if the door is already open at boot (the first reading is trusted).
- BREACH is cleared by pressing the button and entering the correct code ("disarm"). Failing the disarm 3 times returns to BREACH.
- The same sensor makes the safe re-lock automatically, so the servo is never left unlocked.

**Hardware notes (verify against your parts):**
- Pick a **digital Hall sensor** (e.g. A3144 / KY-003 module) or a reed switch as a cheap fallback. Its output is open-collector: use an `INPUT_PULLUP` pin so it reads LOW when the magnet is near (door closed).
- **The UnoQ GPIO is 3.3 V.** With an open-collector sensor you can power the sensor from 5 V while the pull-up goes to 3.3 V. Don't feed a 5 V push-pull output into a pin. Check the UnoQ docs.
- Mount the magnet on the door and the sensor on the frame, 5 to 10 mm apart when closed; test which magnet pole faces the sensor (some sensors only respond to one pole).
- Pins needed in total: 3 LEDs (outputs), 1 button (input), 1 Hall sensor (input), 1 servo (PWM). Optional buzzer for the alarm (output).

**Movie/Notion angle:** "the neural net is the front door, but the Hall sensor is the tripwire: even if someone fools the net or just pries the lid, the safe knows." Stretch goals on top of this: a buzzer siren and the earlier ideas below (adversarial/backdoor demo).

---

## Project framework (already scaffolded)

```
impenetrable safe/
  PROJECT_IDEAS.md        this file
  requirements.txt
  src/
    config.py             all tunables (ROI, thresholds, timers)
    preprocess.py         frame -> model input (shared by PC and UnoQ)
    model.py              FingerNet (3 conv + pool, multi-scale concat, linear)
    collect.py            capture labeled crops: python -m src.collect --person NAME
    train.py              train + curves + confusion matrix: python -m src.train --val-person NAME
    infer.py              Predictor: frame -> (digit, confidence)
    code_store.py         salted-hash code: python -m src.code_store 2 5 3
    lock_policy.py        state machine incl. Hall-sensor breach logic (no hardware deps)
    io_backend.py         SimIO (keyboard) now; UnoQIO to fill in later
    run_lock.py           live PC demo: python -m src.run_lock
  tests/test_lock_policy.py   python -m unittest discover -s tests -t .
```

Policy states: `SLEEP → (button) LISTEN → (3 correct digits) UNLOCKED → (open, close 2 s) SLEEP`; wrong codes ×3 → `LOCKOUT`; door opens while locked → `BREACH` → (button + code) back to `UNLOCKED`.

In the PC demo, `b` simulates the button and `d` toggles the Hall sensor, so the whole policy can be exercised before any wiring.

---

## Task 5 (earlier brainstorm, kept for reference): Break-in mode / easter egg

Goal: use something we have not done in class. The strongest ideas are the ones that also teach something about neural nets, so they can go in the movie.

### Option A (recommended headline feature): Adversarial attack on my own lock

*The safe can be broken by fooling the neural net.* Use PyTorch autograd to compute the gradient of the loss with respect to the **input image** (instead of the weights) and craft an adversarial image/patch that the network classifies as the right digit even though it isn't a hand with that many fingers.

- Techniques: FGSM (one gradient step on the pixels), then PGD, then an **adversarial patch** printed on paper and placed in the red square.
- The tie-in to class material: it's the same backprop and gradient descent, but we update the *pixels*, not the weights. This is a great movie segment.
- Break-in story: "the safe is impenetrable... unless you know the net's blind spot." I generate a printed "key card" that unlocks 2-5-3 without fingers.
- Defense (bonus): adversarial training or a 0-class/"not a hand" class, then show before/after.
- Risk: printed patches captured by a real webcam are harder than digital ones; fall back to showing the digital attack + a printed attempt as the stretch goal.

### Option B: Safecracker minigame ("stethoscope" mode)

A secret trigger (e.g. holding 5-5-5, or a hidden button long-press) puts the safe in **safecracker mode**, a game where the player can crack the code like a movie safe cracker:
- The servo does a tiny "click" jiggle for each digit that is correct and in the correct position (like a tumbler lock).
- LED brightness (PWM) acts as hot/cold using the softmax confidence or the number of correct digits.
- Cracking it unlocks the safe and plays a victory LED pattern.
- Cheap to build, fun at a demo, shows the policy can leak information, which is also a limitation to discuss.

### Option C: Grad-CAM "what is the net looking at" screen

Show a heat map of what the network focuses on while reading fingers (live on the PC; saved frames from the UnoQ). It doesn't strictly break in, but it's a great visual for the movie and explains why 3 is hard. Could be used to find the break-in (feed it the heatmap to see which pixels matter and attack those).

### Option D: Alarm / honeypot

After lockout, a decoy "fake unlock" (the servo wiggles but doesn't release) while the red LED pulses, and the system logs a photo of the intruder's hand. Fun, but less novel technically.

### Suggested combo

- **Main "new thing": Option A** (adversarial examples), documented as the intentional "break-in".
- **Easter egg for the live demo: Option B** (safecracker mode), because it's engaging for visitors and uses only the 3 LEDs, button and servo already wired.
- **Use Option C** as a figure in the movie/Notion for the "how the net works" section.

### Easter egg trigger ideas

- Hold the button while showing 5-5-5.
- Show fist → 5 → fist → 5 → fist → 5 in a timed sequence.
- Press the button 3 short + 1 long.

---

## Suggested order of work

1. `collect.py` + red-square setup → gather data (include a few other people).
2. Train baseline CNN on the PC, plot loss/accuracy, confusion matrix.
3. Add augmentation, retrain, compare accuracy (good before/after graph).
4. Live webcam inference + lock policy in software on the PC (print instead of servo).
5. Wire the UnoQ: LEDs, button, servo; test the sketch alone.
6. Move the model to the UnoQ; compare timing/accuracy with the PC.
7. Task 5 feature (adversarial attack first, then the safecracker easter egg).
8. Movie, Notion page, GitHub repo.

## Open questions

- Which UnoQ pins for the servo (needs PWM) / LEDs / button? (fill in when wired)
- Can PyTorch install on the UnoQ, or do we go ONNX Runtime?
- Image size/color: grayscale 64×64 vs RGB (decide after the first baseline).
- Which code do we use (3 distinct numbers from 1–5)?
