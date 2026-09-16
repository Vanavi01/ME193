---
name: lego-ble-debug
description: Use when a LEGO Education BLE device in this project (Double Motor, Single Motor, Controller, Color Sensor) won't connect — connect() raises, hangs, exhausts the "not ready" retries in lelib.py, or the script just sits at "Connecting to..." forever. Also use for "the car won't connect", "motor not found", "BLE timeout", "no devices found" style reports. Walks through scanning, checking the Connection Card serial, waking the motor, and checking for a stale pairing.
---

# Debugging a LEGO Education BLE connection failure

Work through these in order — each step rules out one layer, and the fix is
usually obvious once you know which layer is broken. Don't jump straight to
"reinstall legoeducation" or start editing `lelib.py`'s retry loop; nearly
every real-world failure here is one of the four things below, not a code bug.

## 1. Scan — is the device even advertising?

Run the scanner:

```bash
source my_env/bin/activate
python scan.py
```

- **Device shows up** (e.g. `LEGO Move Motor   AA:BB:CC:DD:EE:FF`) → advertising
  fine, skip to step 2.
- **Nothing shows up** → it's not advertising. Almost always step 3 (asleep) or
  the device is out of BLE range / behind another room's walls. Re-run `scan.py`
  right after pressing the button on the hub (step 3) — the advertising window
  is short after a fresh boot but stays open once it's up.
- **`bleak` errors immediately** (e.g. Bluetooth adapter not found) → this is a
  host Bluetooth problem, not a LEGO problem. Check macOS Bluetooth is on
  (System Settings → Bluetooth) before going further.

## 2. Card serial — are you filtering for the wrong device?

`connect()` in this project almost always passes `card_serial=` (see
`CARD_SERIAL` in [drive.py](../../drive.py) = `"0999"`, and `spin_motor.py` =
`"1142"`). If that serial doesn't match the number printed on the physical
Connection Card actually plugged into the hub you're trying to reach,
`connect()` will retry against a device that will never appear and eventually
give up — this looks identical to a real connection failure.

- Check the number printed on the Connection Card in the hub right now against
  the `card_serial` value in the script you're running.
- If you don't know which card is in there, or don't care which specific unit
  you get, drop the filter and let it grab the first advertiser of that type
  (see `test_devices.py`, which calls `connect(card_serial=None)`):

  ```python
  dm.connect(card_serial=None)
  ```

- If two Double Motors are powered on at once, `card_serial=None` will connect
  to whichever answers first — nondeterministic. Use the real serial (or power
  off the other unit) if you specifically need one of them.

## 3. Wake it — is the hub just asleep?

LEGO Education hubs (Double Motor, Single Motor, Controller, Color Sensor) stop
advertising BLE after a period of inactivity and need a physical nudge:

- **Press the button on the hub.** Look for its LED to light up/change color —
  that's your confirmation it's awake and should start advertising.
- If the LED doesn't respond to the button at all, it's a battery problem, not
  BLE — check/replace batteries before continuing.
- After waking it, re-run `scan.py` (step 1) within a few seconds to confirm it
  now shows up before retrying the actual connect script.

## 4. Check for a stale pairing — is something else already holding it?

BLE devices like this generally accept only **one active connection at a time**.
If a previous script crashed without calling `.disconnect()`, or another
laptop/process connected and never let go, your `connect()` call will
time out or hit the "not ready" retry path in `lelib.py`
(`singleMotor.connect`, `doubleMotor.connect`, etc. all retry 5× on a
"not ready" message — see [CLAUDE.md](../../CLAUDE.md) for why that retry
exists) even though the device is awake and advertising.

- **Check for a leftover Python process** from a previous run that never hit
  its `finally: dm.disconnect()`:

  ```bash
  ps aux | grep -i python
  ```

  Kill any stale `drive.py`/`spin_double.py`/`test_devices.py` process, wait a
  few seconds for the hub's connection to time out on its end, then retry.

- **Check macOS's own Bluetooth pairing list** — if the hub was ever paired at
  the OS level (not just connected-to by `bleak`/`legoeducation`), macOS can
  hold a connection open in the background:

  System Settings → Bluetooth → look for the device name in the list. If it
  shows "Connected", click the ⓘ next to it and **Disconnect** (or **Forget
  Device** if it's listed as paired at all — these hubs don't need to be
  OS-paired, `legoeducation` talks to them directly over GATT).

- **Another machine on the same room** running the same script can also be
  the culprit if you're in a shared lab — ask around before assuming it's your
  code.

## After fixing the blocking layer

Re-run `python scan.py` first to confirm the device is visible, *then* retry
your actual script. Don't retry the failing script in a loop hoping the retry
logic in `lelib.py` will paper over one of these four issues — it only retries
on the literal "not ready" message and gives up after 5 tries either way.

If all four check out (advertising, correct serial, awake, nothing else
connected) and it still fails, that's the point to actually suspect a code or
library issue — check the exception message from `connect()` verbatim rather
than guessing, and run [test_devices.py](../../test_devices.py) as a minimal
repro isolated from the rest of `drive.py`.
