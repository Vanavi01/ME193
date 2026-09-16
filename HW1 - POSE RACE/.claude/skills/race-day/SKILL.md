---
name: race-day
description: Use when the user says "race day", "preflight", "preflight check", "before demo", "before I run this", "test session", or asks to run through the checklist before a demo/test of the gesture car. A preflight run through every failure mode that's bitten this project before, done once right before every test session — not a code review.
---

# Race-day preflight

Run this the moment before a test session, out loud, one item at a time — not
as a batch of things to assume are fine. The whole point is catching the dumb
thing you'd normally skip because you're nervous and in a hurry. Don't let the
user skip ahead; if they say "yeah yeah it's fine" on a physical check you
haven't actually seen confirmed, ask for the specific observation (LED color,
number on the card, which gesture printed) rather than accepting "fine."

Go through these in order — later steps assume earlier ones passed.

## 1. Fresh batteries

Ask the user directly: are the batteries in the Double Motor fresh, or have
they been sitting in it since the last session? Low batteries are one of the
quietest failure modes here — the hub can still advertise over BLE and connect
fine, then the car just crawls or refuses to move once real load hits the
motors. If there's any doubt, swap them now, before anything else — it's
cheaper than debugging a "connects fine but won't drive" mystery mid-demo.

## 2. Wake the motor

Press the button on the Double Motor hub and confirm the LED responds
(lights up / changes color). If it doesn't react at all, that's a battery or
hardware problem — stop and fix it before continuing (don't proceed to step 3
on a hub you haven't confirmed is awake). For the full decision tree if this
turns into an actual connection failure rather than a quick wake-up, use the
`lego-ble-debug` skill instead of troubleshooting inline here.

## 3. Confirm the serial is still 0999

`CARD_SERIAL` in [drive.py](../../drive.py) is hardcoded to `"0999"`. Confirm
the number printed on the Connection Card currently seated in the hub actually
reads `0999` — cards get swapped between units between sessions more often
than you'd think, and a mismatched serial makes `connect()` retry against a
device that will never show up, which looks exactly like a dead hub. If it's
not `0999`, either swap in the right card or update `CARD_SERIAL` before
running anything.

Quick objective check instead of eyeballing the tiny print — run:

```bash
source my_env/bin/activate
python scan.py
```

and confirm a Double Motor is listed as advertising at all before moving on.

## 4. Pick the right camera index

`drive.py` prints available camera indices and prompts for one at startup.
Before starting the real run, ask the user which index is the webcam actually
pointed at the driving area (not a laptop lid cam pointed at the ceiling, not
a stale index from a USB webcam that's since been unplugged). Get this
confirmed verbally before they type a number in under pressure.

## 5. Run through all seven gestures, confirm the overlay

Start `drive.py`, and with the motor **not** required to actually be
connected for this part conceptually but running for real is fine — walk
through every gesture in the table below one at a time. For each, ask the
user to hold it and read back exactly what the on-screen overlay shows
(the green command text, the `L:`/`R:` hand-shape lines). Don't move to the
next gesture until the current one is confirmed correct — a wrong overlay
reading now is much cheaper to catch than a wrong turn during the demo.

| # | Gesture | Expect on overlay |
|---|---|---|
| 1 | Both thumbs down | `FORWARD` |
| 2 | Both open palms | `BACKWARD` |
| 3 | Right palm only (left hand anything else) | `RIGHT` |
| 4 | Left palm only (right hand anything else) | `LEFT` |
| 5 | Both closed fists | `STOP` |
| 6 | Both pointing up (held) | `SPEED` ticks up |
| 7 | Both peace signs / Victory (held) | `SPEED` ticks down |

If `L:`/`R:` look swapped from what the user's actual hands are doing, that's
the `SWAP_HANDS` flag in [gestures.py](../../gestures.py) — flip it, don't
rewrite the classifier.

## 6. Verify the failsafe — step out of frame

With the car connected and driving, have the user step fully out of the
camera's view (or cover the lens) and count out loud to 2 seconds. Confirm the
command overlay drops to `STOP` and the motors actually cut — this is the
`LOST_TIMEOUT` (1.5s) failsafe in `drive.py`. This is the single most
important check on this list: if it doesn't fire, the car keeps executing the
last command with nobody in frame to stop it, which is the worst possible
failure mode in front of an audience. Do not proceed to the actual demo until
this has been seen working, not just assumed.

## Done

Only once all six have been explicitly confirmed — not "probably fine" — is
this ready for a real audience. If anything failed, fix that specific thing
and re-run just that step, not the whole checklist from scratch, unless the
fix could plausibly have affected an earlier step (e.g. re-pairing after a
BLE fix means re-confirming step 3's serial).
