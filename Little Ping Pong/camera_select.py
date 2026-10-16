'''
Choose the webcam by what it is instead of by number.

OpenCV numbers cameras in the order Windows lists them, and that order changes when
a USB webcam is plugged in or out (virtual cameras like NVIDIA Broadcast, OBS or a
phone also take numbers). So the camera setting can be:

    'laptop'   the built-in camera ("Integrated Camera"; never the IR camera)
    'usb'      a plugged-in physical webcam that isn't built in (falls back to the laptop camera)
    3          a camera number, used as-is
    'Full HD'  any other text: the first camera whose name contains it

Names come from the cv2-enumerate-cameras package.
'''

from dataclasses import dataclass

import cv2

VIRTUAL_HINTS = ('virtual', 'broadcast', 'obs', 'snap camera')


@dataclass
class CameraDevice:
    index: int
    name: str
    physical: bool          # a real device with a USB vendor/product id (not a virtual camera)

    @property
    def builtin(self):
        n = self.name.lower()
        return 'integrated' in n or 'built-in' in n or 'internal' in n

    @property
    def infrared(self):
        n = self.name.lower()
        return ' ir ' in f' {n} ' or 'infrared' in n


def list_cameras():
    '''Cameras OpenCV can open with DirectShow, with their Windows names.'''
    try:
        from cv2_enumerate_cameras import enumerate_cameras
    except ImportError:
        return []
    out = []
    for c in enumerate_cameras(cv2.CAP_DSHOW):
        name = c.name or f'camera {c.index}'
        virtual = c.vid is None or any(h in name.lower() for h in VIRTUAL_HINTS)
        out.append(CameraDevice(c.index, name, not virtual))
    return out


def resolve_camera(spec, cameras):
    '''Camera setting -> (index, note). note explains a fallback, else ''. Raises ValueError
    if nothing suitable exists.'''
    if isinstance(spec, int) or (isinstance(spec, str) and spec.strip().lstrip('-').isdigit()):
        return int(spec), ''
    key = str(spec).strip().lower()
    laptop = [c for c in cameras if c.physical and c.builtin and not c.infrared]
    usb = [c for c in cameras if c.physical and not c.builtin and not c.infrared]
    if key in ('laptop', 'integrated', 'builtin', 'built-in', 'internal'):
        if laptop:
            return laptop[0].index, ''
        raise ValueError('no built-in laptop camera found')
    if key in ('usb', 'external', 'webcam'):
        if usb:
            return usb[0].index, ''
        if laptop:
            return laptop[0].index, 'no USB webcam found: using the laptop camera'
        raise ValueError('no USB webcam (or laptop camera) found')
    for c in cameras:
        if key in c.name.lower():
            return c.index, ''
    raise ValueError(f'no camera named like {spec!r}')


def describe(cameras):
    lines = []
    for c in cameras:
        kind = 'laptop' if c.physical and c.builtin else ('usb' if c.physical else 'virtual')
        if c.infrared:
            kind = 'IR (not usable)'
        lines.append(f'  {c.index}: {c.name}   [{kind}]')
    return '\n'.join(lines)


def pick(spec, verbose=True):
    '''Resolve a camera setting against what is plugged in right now; prints what was chosen.'''
    if isinstance(spec, int) or str(spec).strip().lstrip('-').isdigit():
        return int(spec)                                # a number is used as-is
    cameras = list_cameras()
    if not cameras:                                     # no names available
        print(f'Camera names unavailable (pip install cv2-enumerate-cameras): using camera 0 for {spec!r}')
        return 0
    try:
        index, note = resolve_camera(spec, cameras)
    except ValueError as e:
        raise SystemExit(f'Camera {spec!r}: {e}. Cameras found:\n{describe(cameras)}')
    if verbose:
        name = next((c.name for c in cameras if c.index == index), f'camera {index}')
        print(f'Camera: {name} (#{index})' + (f'  [{note}]' if note else ''))
    return index
