# UnoQ hardware: servo, 3 LEDs, button

The camera and model run on the PC for now. The UnoQ runs a small app that drives the hardware;
the PC talks to it over Wi-Fi.

```
PC: webcam -> CNN -> lock policy --(Wi-Fi, port 5005)--> UnoQ Linux: python/main.py
                                                              | Bridge
                                                              v
                                               UnoQ microcontroller: sketch/sketch.ino
                                               servo, yellow/green/red LEDs, button
```

## Wiring (change the pin numbers at the top of `sketch.ino` if you use others)

| Part | UnoQ pin | Notes |
|---|---|---|
| Servo signal (orange/yellow wire) | D9 | Servo red wire to 5V, brown/black to GND. A big (metal-gear) servo needs its own 5V supply; connect its GND to the UnoQ GND. |
| Red LED (wrong code, lockout, alarm) | D5 | D5 → 220 Ω resistor → LED long leg; short leg → GND |
| Yellow LED (digit registered) | D6 | same |
| Green LED (correct code, unlocked) | D7 | same (if green is very dim at 3.3 V, use 100 Ω) |
| Button | D2 | one leg to D2, the diagonal leg to GND (internal pull-up, no resistor) |
| Hall sensor (later) | D3 | set `HALL_INSTALLED = true` in the sketch and `HALL_SENSOR = True` in `src/config.py` |

All GNDs (LEDs, button, servo, external supply) go to the same GND rail. The UnoQ's I/O is
3.3 V: fine for LEDs, the button and a hobby servo's signal wire.

**Servo (M5Stack 360 = continuous rotation):** the pulse width sets speed and direction
(1500 µs = stop), not an angle, so locking = spinning one way for a calibrated time and unlocking
= spinning back. Calibrate once (and whenever the latch changes):

```
python -m src.hw_test --host <UnoQ IP> --servo
```
1. Nudge the latch to OPEN (`a` toward open, `d` toward closed, Enter when there).
2. Lock tuning: Enter tries closing with the current time; if it's not right it spins back, you
   nudge it to open again, change the time (type ms, or `+` / `-`), and retry. `f` flips the
   direction, `s 300` changes the speed.
3. Unlock tuning: the same, from closed back to open.

The result is saved in `data/_calibration/servo.json` and sent to the board every time the PC
connects. The servo has no position sensor, so the board only knows "locked/unlocked": power the
board off with the latch **closed** (the app always locks on disconnect), and if it ever drifts,
rerun `--servo`. Power the servo from **5V**.

## Install the app (Arduino App Lab)

1. Open **Arduino App Lab**, connect to the UnoQ, and create a new app (e.g. "impenetrable-safe").
2. Replace the app's sketch with `unoq/safe_app/sketch/sketch.ino` and its Python file with
   `unoq/safe_app/python/main.py`.
3. Run the app. The console should print `safe hardware server listening on port 5005`.
   - The sketch makes the servo pulses itself; the UnoQ core has no `<Servo.h>`.
   - If `Bridge.provide` complains about a function, check the Bridge examples in App Lab for
     your library version and adjust the signature.
4. Find the UnoQ's IP address (App Lab shows it, or run `hostname -I` on the board). The PC and
   the UnoQ must be on the same Wi-Fi.

**Wiring check without the PC:** set `SELF_TEST = True` in `main.py` and run the app; each button
press steps through yellow, green, red, all blinking, servo unlocked, servo locked.

## Test from the PC

```
python -m src.hw_test --host <UnoQ IP>                 # LEDs, servo, button, one at a time
python -m src.code_store 2 5 3                         # set your code (3 different digits 1-5)
python -m src.safe_sim --host <UnoQ IP>                # the real lock, with the guard screen
```

## The flow

1. Press the **button** → all LEDs blink once: show your code.
2. Show each digit until the hold bar fills (**yellow** flashes per digit), make a fist between digits.
3. Correct code → **green** flashes, the servo turns back a quarter turn (**unlocked**), green stays on.
   Wrong code → **red** flashes; 3 wrong codes = 30 s lockout (red on).
4. Press the **button** again → servo turns a quarter turn clockwise (**locked**), LEDs off.

The safe stays open until the button is pressed (`UNLOCK_WINDOW_S = None`; set a number of seconds
for an automatic re-lock). Without the Hall sensor the door always reads "closed", so forced-entry
detection is off. If the PC program stops or the connection drops, the UnoQ locks the servo and
turns the LEDs off.
