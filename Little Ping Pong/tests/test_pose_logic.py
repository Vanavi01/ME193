'''Step 5: MediaPipe Pose information (pure logic; fake landmarks, no MediaPipe needed).'''

import unittest
from types import SimpleNamespace

import cv2
import numpy as np

from game import Game
from paddle_vision import VisionConfig
from pose_logic import (ABSENT_AFTER_S, HANDS_UP_HOLD_S, LANDMARKS, RETURN_AFTER_S, PoseInfo,
                        Presence, StartGesture, pose_from_landmarks, racket_gate,
                        skeleton_lines)
from red_tracker import POSE_FRESH_S, PROC_H, PROC_W, RedTrackerCore
from tests.test_red_tracker import RED, H, W, frame


def fake_landmarks(overrides=None, hidden=()):
    '''A person standing in the middle, arms down. Coordinates are un-mirrored (raw camera).'''
    base = {
        'nose': (0.50, 0.20),
        'left_shoulder': (0.58, 0.35), 'right_shoulder': (0.42, 0.35),
        'left_elbow': (0.62, 0.50), 'right_elbow': (0.38, 0.50),
        'left_wrist': (0.63, 0.62), 'right_wrist': (0.37, 0.62),
        'left_hip': (0.56, 0.70), 'right_hip': (0.44, 0.70),
    }
    base.update(overrides or {})
    lms = [SimpleNamespace(x=0.5, y=0.5, visibility=0.0) for _ in range(33)]
    for name, (x, y) in base.items():
        lms[LANDMARKS[name]] = SimpleNamespace(x=x, y=y, visibility=0.1 if name in hidden else 0.95)
    return lms


class TestPoseFromLandmarks(unittest.TestCase):

    def test_mirrors_x_and_keeps_body_side_names(self):
        p = pose_from_landmarks(fake_landmarks(), t=1.0)
        # player's right wrist is at raw x=0.37 -> shown on the right of the mirrored image
        self.assertAlmostEqual(p.get('right_wrist')[0], 0.63)
        self.assertAlmostEqual(p.get('right_wrist')[1], 0.62)
        self.assertEqual(p.t, 1.0)

    def test_hidden_points_are_left_out(self):
        p = pose_from_landmarks(fake_landmarks(hidden=('left_wrist',)), t=0)
        self.assertIsNone(p.get('left_wrist'))
        self.assertIsNotNone(p.get('right_wrist'))

    def test_skeleton_lines_only_for_visible_pairs(self):
        full = skeleton_lines(pose_from_landmarks(fake_landmarks(), 0), 320, 240)
        self.assertEqual(len(full), 8)
        partial = skeleton_lines(pose_from_landmarks(fake_landmarks(hidden=('right_elbow',)), 0), 320, 240)
        self.assertEqual(len(partial), 6)       # shoulder-elbow and elbow-wrist both need the elbow


class TestTrackerWithPose(unittest.TestCase):

    def test_fresh_pose_is_passed_on_and_drawn(self):
        core = RedTrackerCore()
        pose = pose_from_landmarks(fake_landmarks(), t=0.0)
        s = core.step(frame(0.7), 0.1, pose=pose)
        self.assertIs(s.pose, pose)
        green = np.all(s.frame == (90, 255, 120), axis=2)     # RGB skeleton colour... drawn in BGR then converted
        green |= np.all(s.frame == (120, 255, 90), axis=2)
        self.assertGreater(green.sum(), 50)

    def test_stale_pose_is_ignored(self):
        core = RedTrackerCore()
        pose = pose_from_landmarks(fake_landmarks(), t=0.0)
        s = core.step(frame(0.7), POSE_FRESH_S + 0.1, pose=pose)
        self.assertIsNone(s.pose)



# Player's RIGHT arm raised to the right of the mirrored image, paddle at u~0.7, v~0.46
# (where tests.test_red_tracker.frame draws it). Raw (un-mirrored) x = 1 - u.
RACKET_UP = {'right_wrist': (1 - 0.66, 0.56), 'right_elbow': (1 - 0.62, 0.69)}


def racket_pose(t=0.0, **kw):
    return pose_from_landmarks(fake_landmarks(RACKET_UP, **kw), t)


def with_red_shirt(img):
    '''Big red shirt on the torso (bigger than the paddle face).'''
    cv2.rectangle(img, (int(0.30 * W), int(0.62 * H)), (int(0.58 * W), H), RED, -1)
    return img


class TestRacketGate(unittest.TestCase):

    def test_gate_is_beyond_the_racket_wrist(self):
        u, v, half = racket_gate(racket_pose())
        self.assertAlmostEqual(u, 0.66 + 0.8 * 0.04, places=6)
        self.assertLess(v, 0.56)                            # forearm points up -> look higher
        self.assertGreaterEqual(half, 0.10)

    def test_no_gate_without_the_racket_arm(self):
        self.assertIsNone(racket_gate(racket_pose(hidden=('right_wrist',))))
        self.assertIsNone(racket_gate(None))
        self.assertIsNone(racket_gate(racket_pose(), racket_hand='auto'))

    def test_red_shirt_ignored_when_pose_knows_the_racket_arm(self):
        img = with_red_shirt(frame(0.7))
        no_pose = RedTrackerCore().step(img, 0.0)
        self.assertLess(no_pose.u, 0.6)                     # without pose the bigger shirt wins
        s = RedTrackerCore().step(img, 0.0, pose=racket_pose())
        self.assertAlmostEqual(s.u, 0.7, delta=0.03)        # with pose: the paddle
        self.assertIsNotNone(s.gate)

    def test_red_in_the_free_hand_is_ignored(self):
        '''Paddle-coloured thing on the left of the image, racket arm on the right.'''
        s = RedTrackerCore().step(frame(0.25), 0.0, pose=racket_pose())
        self.assertFalse(s.seen)

    def test_left_handed_player(self):
        lefty = {'left_wrist': (1 - 0.66, 0.56), 'left_elbow': (1 - 0.62, 0.69),
                 'right_wrist': (1 - 0.30, 0.62), 'right_elbow': (1 - 0.32, 0.50)}
        pose = pose_from_landmarks(fake_landmarks(lefty), 0.0)
        s = RedTrackerCore(VisionConfig(racket_hand='left')).step(with_red_shirt(frame(0.7)), 0.0, pose=pose)
        self.assertAlmostEqual(s.u, 0.7, delta=0.03)

    def test_stale_pose_falls_back_to_whole_frame_search(self):
        s = RedTrackerCore().step(frame(0.25), POSE_FRESH_S + 0.1, pose=racket_pose(t=0.0))
        self.assertTrue(s.seen)
        self.assertIsNone(s.gate)


class TestPresence(unittest.TestCase):

    def feed(self, p, poses, t0, dt=1 / 30):
        out = None
        for i, pose in enumerate(poses):
            out = p.update(pose, t0 + i * dt)
        return out

    def test_appears_after_being_seen_steadily(self):
        p = Presence()
        self.assertFalse(p.update(racket_pose(), 0.0))
        self.assertTrue(self.feed(p, [racket_pose()] * 20, 0.0))       # 0.63 s > RETURN_AFTER_S

    def test_brief_dropout_does_not_count_as_leaving(self):
        p = Presence()
        self.feed(p, [racket_pose()] * 20, 0.0)
        self.assertTrue(self.feed(p, [None] * int(ABSENT_AFTER_S * 30 * 0.6), 0.7))

    def test_leaving_and_coming_back(self):
        p = Presence()
        self.feed(p, [racket_pose()] * 20, 0.0)
        self.assertFalse(self.feed(p, [None] * 40, 0.7))                # 1.3 s gone
        self.assertFalse(self.feed(p, [racket_pose()] * 5, 2.1))        # back, but only briefly
        self.assertTrue(self.feed(p, [racket_pose()] * 20, 2.3))

    def test_needs_both_shoulders(self):
        p = Presence()
        one_shoulder = racket_pose(hidden=('left_shoulder',))
        self.assertFalse(self.feed(p, [one_shoulder] * 30, 0.0))

    def test_tracker_reports_presence_only_when_pose_is_on(self):
        core = RedTrackerCore()
        self.assertIsNone(core.step(frame(0.7), 0.0).present)
        s = None
        for i in range(20):
            s = core.step(frame(0.7), i / 30, pose=racket_pose(t=i / 30), pose_enabled=True)
        self.assertTrue(s.present)


HANDS_UP = {'left_wrist': (0.62, 0.10), 'right_wrist': (0.38, 0.12),
            'left_elbow': (0.62, 0.25), 'right_elbow': (0.38, 0.25)}
ONE_HAND_UP = {'right_wrist': (0.38, 0.12), 'right_elbow': (0.38, 0.25)}


class TestStartGesture(unittest.TestCase):

    def feed(self, g, overrides, n, t0, dt=1 / 30):
        fired = 0
        for i in range(n):
            fired += g.update(pose_from_landmarks(fake_landmarks(overrides), t0 + i * dt), t0 + i * dt)
        return fired

    def test_hands_up_held_fires_once(self):
        g = StartGesture()
        self.assertEqual(self.feed(g, HANDS_UP, int(HANDS_UP_HOLD_S * 30) + 3, 0.0), 1)
        self.assertEqual(self.feed(g, HANDS_UP, 60, 1.0), 0)           # still up: no repeat

    def test_quick_wave_is_not_a_start(self):
        g = StartGesture()
        self.assertEqual(self.feed(g, HANDS_UP, int(HANDS_UP_HOLD_S * 30 * 0.5), 0.0), 0)
        self.assertEqual(self.feed(g, {}, 5, 0.5), 0)

    def test_one_hand_up_is_not_a_start(self):
        self.assertEqual(self.feed(StartGesture(), ONE_HAND_UP, 60, 0.0), 0)

    def test_rearms_after_hands_come_down(self):
        g = StartGesture()
        self.feed(g, HANDS_UP, 25, 0.0)
        self.feed(g, {}, 5, 1.0)                                        # arms down by the sides
        self.assertEqual(self.feed(g, HANDS_UP, 25, 2.0), 1)

    def test_no_pose_never_fires(self):
        g = StartGesture()
        self.assertFalse(any(g.update(None, i / 30) for i in range(60)))

    def test_tracker_counts_gestures(self):
        core = RedTrackerCore()
        s = None
        for i in range(25):
            t = i / 30
            s = core.step(frame(0.7), t, pose=pose_from_landmarks(fake_landmarks(HANDS_UP), t), pose_enabled=True)
        self.assertEqual(s.start_count, 1)


class TestGamePause(unittest.TestCase):

    def test_pause_freezes_the_ball_and_ignores_swings(self):
        g = Game()
        g.start()
        g.update(1 / 60, (0, None))
        g.pause()
        z, t = g.ball.z, g.time
        for _ in range(60):
            g.update(1 / 60, (0, None))
        g.swing(0.8)
        self.assertEqual((g.ball.z, g.time), (z, t))
        self.assertEqual(g._last_swing_t, float('-inf'))
        g.resume()
        g.update(1 / 60, (0, None))
        self.assertLess(g.ball.z, z)                                   # moving again


if __name__ == '__main__':
    unittest.main()
