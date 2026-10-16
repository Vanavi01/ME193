'''
MediaPipe Pose information used by the game (pure logic, unit tested; no MediaPipe import).

Pose runs asynchronously at a lower rate than the red tracker (pose_async.py), so
everything here works on the *latest* pose and copes with it being a little old.

Coordinates are normalised (0..1) in the MIRRORED image the player sees; names are
the player's own body sides ('right_wrist' is the player's real right wrist).
'''

from dataclasses import dataclass, field

LANDMARKS = {
    'nose': 0,
    'left_shoulder': 11, 'right_shoulder': 12,
    'left_elbow': 13, 'right_elbow': 14,
    'left_wrist': 15, 'right_wrist': 16,
    'left_hip': 23, 'right_hip': 24,
    'left_pinky': 17, 'right_pinky': 18,
    'left_index': 19, 'right_index': 20,
}
SKELETON = [
    ('left_shoulder', 'right_shoulder'), ('left_hip', 'right_hip'),
    ('left_shoulder', 'left_hip'), ('right_shoulder', 'right_hip'),
    ('left_shoulder', 'left_elbow'), ('left_elbow', 'left_wrist'),
    ('right_shoulder', 'right_elbow'), ('right_elbow', 'right_wrist'),
]
MIN_VISIBILITY = 0.5

# Racket-arm gate: where to look for the red paddle face
GATE_AHEAD = 0.8        # paddle centre ~ wrist + GATE_AHEAD * (wrist - elbow)
GATE_SIZE = 1.6         # gate half-size = GATE_SIZE * forearm length...
GATE_MIN = 0.10         # ...but at least this (normalised units)

# Presence
ABSENT_AFTER_S = 1.0    # no shoulders this long -> player gone (pause)
RETURN_AFTER_S = 0.5    # shoulders back this long -> player back (resume)

# Start gesture
HANDS_UP_HOLD_S = 0.6   # both wrists above the nose this long -> start


@dataclass
class PoseInfo:
    t: float                                    # time the frame was captured (seconds)
    points: dict = field(default_factory=dict)  # name -> (u, v), visible points only

    def get(self, name):
        return self.points.get(name)


def pose_from_landmarks(landmarks, t, mirror=True):
    '''MediaPipe landmark list (from the un-mirrored frame) -> PoseInfo in mirrored coordinates.'''
    points = {}
    for name, i in LANDMARKS.items():
        if i >= len(landmarks):
            continue
        lm = landmarks[i]
        if getattr(lm, 'visibility', 1.0) < MIN_VISIBILITY:
            continue
        points[name] = ((1.0 - lm.x) if mirror else lm.x, lm.y)
    return PoseInfo(t, points)


def racket_gate(pose, racket_hand='right'):
    '''(u, v, half_size) box where the red paddle face should be, from the racket arm,
    or None if that arm isn't visible (then the tracker searches the whole frame).'''
    if pose is None or racket_hand not in ('left', 'right'):
        return None
    wrist, elbow = pose.get(f'{racket_hand}_wrist'), pose.get(f'{racket_hand}_elbow')
    if wrist is None or elbow is None:
        return None
    du, dv = wrist[0] - elbow[0], wrist[1] - elbow[1]
    forearm = (du * du + dv * dv) ** 0.5
    return wrist[0] + GATE_AHEAD * du, wrist[1] + GATE_AHEAD * dv, max(GATE_MIN, GATE_SIZE * forearm)


def hand_cutout(pose, racket_hand, width, height):
    '''Pixel circle (x, y, r) covering the racket hand, to remove from the red search:
    skin can look red, and the hand is right next to the paddle. Spans wrist to
    knuckles (index/pinky landmarks); without knuckles, the start of the forearm
    direction beyond the wrist. None if the wrist isn't visible.'''
    if pose is None or racket_hand not in ('left', 'right'):
        return None
    wrist, elbow = pose.get(f'{racket_hand}_wrist'), pose.get(f'{racket_hand}_elbow')
    if wrist is None:
        return None
    wx, wy = wrist[0] * width, wrist[1] * height
    forearm = 0.0
    if elbow is not None:
        fx, fy = wx - elbow[0] * width, wy - elbow[1] * height
        forearm = (fx * fx + fy * fy) ** 0.5
    index, pinky = pose.get(f'{racket_hand}_index'), pose.get(f'{racket_hand}_pinky')
    if index is not None and pinky is not None:
        kx = (index[0] + pinky[0]) / 2 * width
        ky = (index[1] + pinky[1]) / 2 * height
        span = ((kx - wx) ** 2 + (ky - wy) ** 2) ** 0.5
        return (wx + kx) / 2, (wy + ky) / 2, max(1.2 * span, 0.25 * forearm, 6.0)
    if elbow is None:
        return None
    return wx + 0.2 * fx, wy + 0.2 * fy, max(0.3 * forearm, 6.0)


class Presence:
    '''Is the player in front of the camera? Based on both shoulders being visible,
    with hysteresis so a single missed pose doesn't pause the game.'''

    def __init__(self):
        self.present = False
        self._last_seen = None
        self._seen_since = None

    def update(self, pose, t):
        visible = pose is not None and pose.get('left_shoulder') and pose.get('right_shoulder')
        if visible:
            self._last_seen = t
            if self._seen_since is None:
                self._seen_since = t
            if not self.present and t - self._seen_since >= RETURN_AFTER_S:
                self.present = True
        else:
            self._seen_since = None
            if self.present and (self._last_seen is None or t - self._last_seen >= ABSENT_AFTER_S):
                self.present = False
        return self.present


class StartGesture:
    '''Both hands raised above the head, held for HANDS_UP_HOLD_S -> fires once.
    It re-arms only after the hands come back down below the shoulders.'''

    def __init__(self):
        self._up_since = None
        self._armed = True

    @staticmethod
    def hands_up(pose):
        nose = pose.get('nose') if pose else None
        lw, rw = (pose.get('left_wrist'), pose.get('right_wrist')) if pose else (None, None)
        return bool(nose and lw and rw and lw[1] < nose[1] and rw[1] < nose[1])

    @staticmethod
    def hands_down(pose):
        if pose is None:
            return False
        ls, rs = pose.get('left_shoulder'), pose.get('right_shoulder')
        lw, rw = pose.get('left_wrist'), pose.get('right_wrist')
        if not (ls and rs):
            return False
        shoulder_v = (ls[1] + rs[1]) / 2
        return all(w is None or w[1] > shoulder_v for w in (lw, rw))

    def update(self, pose, t):
        '''Returns True on the one update where the gesture completes.'''
        if self.hands_up(pose):
            if self._up_since is None:
                self._up_since = t
            if self._armed and t - self._up_since >= HANDS_UP_HOLD_S:
                self._armed = False
                return True
        else:
            self._up_since = None
            if self.hands_down(pose):
                self._armed = True
        return False


def skeleton_lines(pose, width, height):
    '''Pixel line segments to draw for the visible parts of the skeleton.'''
    lines = []
    for a, b in SKELETON:
        pa, pb = pose.get(a), pose.get(b)
        if pa and pb:
            lines.append(((int(pa[0] * width), int(pa[1] * height)),
                          (int(pb[0] * width), int(pb[1] * height))))
    return lines
