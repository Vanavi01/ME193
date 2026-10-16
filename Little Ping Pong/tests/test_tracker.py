'''Step 3 tests: the per-frame tracker logic (no camera or MediaPipe needed).'''

import unittest
from types import SimpleNamespace

from paddle_vision import image_to_game
from tests.test_paddle_vision import BLACK, RED, RIGHT, W, H, draw_paddle, frame
from tracker import (LEFT_ELBOW, LEFT_WRIST, LOST_AFTER_S, RIGHT_ELBOW,
                     RIGHT_INDEX, RIGHT_PINKY, RIGHT_WRIST, TrackerCore,
                     hands_from_landmarks)

DT = 1 / 30


def landmarks(visible=('left', 'right')):
    lm = [SimpleNamespace(x=0.5, y=0.5, visibility=0.0) for _ in range(33)]
    def put(i, x, y, side):
        lm[i] = SimpleNamespace(x=x, y=y, visibility=0.9 if side in visible else 0.1)
    put(LEFT_WRIST, 0.30, 0.50, 'left')
    put(LEFT_ELBOW, 0.35, 0.70, 'left')
    put(RIGHT_WRIST, 0.65, 0.50, 'right')
    put(RIGHT_ELBOW, 0.60, 0.70, 'right')
    return lm


class TestHandsFromLandmarks(unittest.TestCase):

    def test_converts_to_pixels(self):
        hands = hands_from_landmarks(landmarks(), 640, 480)
        self.assertEqual([h.name for h in hands], ['left', 'right'])
        self.assertEqual(hands[1].wrist, (416, 240))
        self.assertEqual(hands[1].elbow, (384, 336))

    def test_mirror_keeps_arm_names_and_flips_positions(self):
        '''Pose runs on the real image; the game shows it mirrored. The player's right
        arm stays 'right' and appears on the right of the mirrored frame.'''
        lm = landmarks()
        raw_right_x = lm[RIGHT_WRIST].x
        hands = hands_from_landmarks(lm, 640, 480, mirror=True)
        right = next(h for h in hands if h.name == 'right')
        self.assertEqual(right.wrist[0], int((1 - raw_right_x) * 640))

    def test_knuckle_is_between_index_and_pinky(self):
        lm = landmarks()
        lm[RIGHT_INDEX] = SimpleNamespace(x=0.70, y=0.40, visibility=0.9)
        lm[RIGHT_PINKY] = SimpleNamespace(x=0.66, y=0.42, visibility=0.9)
        right = next(h for h in hands_from_landmarks(lm, 640, 480) if h.name == 'right')
        self.assertEqual(right.knuckle, ((448 + 422) // 2, (192 + 201) // 2))
        left = next(h for h in hands_from_landmarks(lm, 640, 480) if h.name == 'left')
        self.assertIsNone(left.knuckle)                     # left knuckles not visible

    def test_skips_hidden_arms(self):
        hands = hands_from_landmarks(landmarks(visible=('right',)), 640, 480)
        self.assertEqual([h.name for h in hands], ['right'])


class TestTrackerCore(unittest.TestCase):

    def run_frames(self, core, color, n, t0=0.0, hands=(RIGHT,)):
        img = frame()
        center = draw_paddle(img, RIGHT, color) if color else None
        state = None
        for i in range(n):
            state = core.step(img, list(hands), t0 + i * DT)
        return state, center

    def test_red_side_is_forehand_and_position_maps_to_game(self):
        core = TrackerCore()
        state, (cx, cy) = self.run_frames(core, RED, 5)
        self.assertEqual((state.side, state.stroke, state.hand), ('red', 'forehand', 'right'))
        ex, ey = image_to_game(cx / W, cy / H)
        self.assertAlmostEqual(state.paddle[0], ex, delta=0.02)
        self.assertAlmostEqual(state.paddle[1], ey, delta=0.02)
        self.assertTrue(state.seen)
        self.assertEqual(state.frame.shape, (H, W, 3))

    def test_flip_to_black_becomes_backhand_after_a_few_frames(self):
        core = TrackerCore()
        self.run_frames(core, RED, 5)
        state, _ = self.run_frames(core, BLACK, 1, t0=1.0)
        self.assertEqual(state.stroke, 'forehand')          # one frame is not enough
        state, _ = self.run_frames(core, BLACK, 3, t0=1.1)
        self.assertEqual(state.stroke, 'backhand')

    def test_forehand_color_can_be_black(self):
        state, _ = self.run_frames(TrackerCore(forehand_color='black'), BLACK, 5)
        self.assertEqual(state.stroke, 'forehand')

    def test_player_leaving_keeps_last_position_then_reports_lost(self):
        core = TrackerCore()
        state, _ = self.run_frames(core, RED, 5)
        last = state.paddle
        state, _ = self.run_frames(core, None, 3, t0=0.2, hands=())
        self.assertTrue(state.seen)                          # brief dropout
        state, _ = self.run_frames(core, None, 3, t0=0.2 + LOST_AFTER_S, hands=())
        self.assertFalse(state.seen)
        self.assertEqual(state.paddle, last)                 # paddle stays put, no jump
        self.assertIsNone(state.stroke)


if __name__ == '__main__':
    unittest.main()
