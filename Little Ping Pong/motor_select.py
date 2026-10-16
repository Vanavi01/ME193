'''
Make sure we connect to OUR Double Motor, not the first one that answers.

In a classroom several motors advertise at once; their Bluetooth names look like
"🟩 0994 Double Motor" (Connection Card colour + serial) or "⬜    Double Motor"
(no card). Connecting without a serial grabs whichever answers first.

    find_motor.py --use 0994   saves the serial in motor.json; every script then uses it
    --serial 0994              still overrides it for one run
'''

import json
import os
import re
from dataclasses import dataclass

SAVE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'motor.json')
CARD_COLORS = {'\U0001f7e9': 'green', '\U0001f7ea': 'purple', '\U0001f7e5': 'red', '\U0001f7e6': 'blue',
               '\U0001f7e8': 'yellow', '\U0001f7e7': 'orange', '⬛': 'black', '⬜': 'none'}


@dataclass
class MotorInfo:
    address: str
    serial: str = None      # Connection Card serial, None if no card
    color: str = None

    def label(self):
        if not self.serial:
            return f'no card ({self.address})'
        known = self.color and self.color != 'none'
        return f'{self.serial} ({self.color} card)' if known else f'{self.serial} (card)'


def parse_motor(address, name):
    name = name or ''
    m = re.search(r'\b(\d{3,6})\b', name)
    color = next((c for e, c in CARD_COLORS.items() if e in name), None)
    return MotorInfo(address, m.group(1) if m else None, color)


def scan(timeout=5):
    import legoeducation as le
    found = le.DoubleMotor().search(timeout=timeout) or []
    return [parse_motor(d.address, d.name) for d in found]


def load_saved_serial(path=SAVE_PATH):
    try:
        with open(path) as f:
            return json.load(f).get('serial') or None
    except (OSError, ValueError):
        return None


def save_serial(serial, path=SAVE_PATH):
    with open(path, 'w') as f:
        json.dump({'serial': str(serial)}, f)


def choose_serial(requested, saved, motors):
    '''Which serial to connect to. requested (--serial) beats saved (motor.json). With
    neither: fine if exactly one motor is around, otherwise refuse to guess.'''
    if requested:
        return str(requested)
    if saved:
        return saved
    if len(motors) == 1:
        return motors[0].serial            # may be None: then connect to "the" motor
    if not motors:
        raise ConnectionError('No Double Motor found. Is it switched on (light blinking) and nearby?')
    listing = '\n'.join(f'  {m.label()}' for m in motors)
    raise ConnectionError(f'{len(motors)} Double Motors are nearby, so I won\'t guess which is yours:\n'
                          f'{listing}\nSave yours once with:  python find_motor.py --use <serial>')


def explain(error, serial=None):
    '''Turn a raw connection failure into advice.'''
    text = str(error)
    if 'firmware' in text.lower():
        return f'{text}\n-> Update the motor at https://code.legoeducation.com/ (Chrome/Edge), then close that tab.'
    who = f'Double Motor {serial}' if serial else 'the Double Motor'
    return (f'Could not connect to {who} ({text or type(error).__name__}).\n'
            '-> Check it is switched on and close any other program using it (another game window,\n'
            '   code.legoeducation.com). If it was just disconnected, wait ~5 s and try again.\n'
            '-> python find_motor.py lists the motors in range.')
