'''Step 3 tests: paddle colour/side detection on synthetic frames, and turning it into game input.'''

import math
import os
import tempfile
import unittest

import cv2
import numpy as np

from paddle_vision import (GAME_X_RANGE, GAME_Y_RANGE, Hand, OneEuroFilter,
                           SideFilter, VisionConfig, calibrate_black,
                           calibrate_red, color_masks, false_positive_rates,
                           find_paddle, hand_circle, hue_mask, image_to_game, search_circle,
                           stroke_for_side, tighten_for_background)

DEFAULT = VisionConfig()
U_RANGE = (DEFAULT.u_lo, DEFAULT.u_hi)
V_RANGE = (DEFAULT.v_lo, DEFAULT.v_hi)

H, W = 480, 640
RED = (40, 30, 210)         # BGR
BLACK = (20, 20, 20)
WALL = (170, 175, 165)      # light grey-green background
SKIN = (150, 180, 220)

# A right hand raised: elbow below-left of the wrist, paddle head beyond the wrist.
RIGHT = Hand('right', wrist=(420, 240), elbow=(380, 330))
LEFT = Hand('left', wrist=(200, 260), elbow=(230, 350))


def frame():
    img = np.full((H, W, 3), WALL, np.uint8)
    noise = np.random.default_rng(0).integers(-8, 9, img.shape)
    return np.clip(img.astype(int) + noise, 0, 255).astype(np.uint8)


def paddle_center(hand):
    '''Where a real paddle head sits: handle + hand put it ~0.8 forearm lengths beyond the wrist.'''
    (wx, wy), (ex, ey) = hand.wrist, hand.elbow
    return int(wx + 0.8 * (wx - ex)), int(wy + 0.8 * (wy - ey))


def draw_paddle(img, hand, color, size=(38, 46)):
    cx, cy = paddle_center(hand)
    cv2.line(img, hand.wrist, (cx, cy), (60, 110, 160), 10)          # wooden handle
    cv2.ellipse(img, (cx, cy), size, 20, 0, 360, color, -1)
    cv2.circle(img, hand.wrist, 14, SKIN, -1)                         # the hand
    return cx, cy


class TestFindPaddle(unittest.TestCase):

    def test_red_side(self):
        img = frame()
        cx, cy = draw_paddle(img, RIGHT, RED)
        obs = find_paddle(img, [RIGHT])
        self.assertEqual(obs.side, 'red')
        self.assertEqual(obs.hand, 'right')
        self.assertAlmostEqual(obs.center[0], cx, delta=4)
        self.assertAlmostEqual(obs.center[1], cy, delta=4)

    def test_black_side(self):
        img = frame()
        cx, cy = draw_paddle(img, RIGHT, BLACK)
        obs = find_paddle(img, [RIGHT])
        self.assertEqual(obs.side, 'black')
        self.assertAlmostEqual(obs.center[0], cx, delta=4)

    def test_edge_on_paddle_shows_no_side(self):
        '''Paddle turned edge-on: only a thin sliver of colour is visible.'''
        img = frame()
        draw_paddle(img, RIGHT, RED, size=(40, 2))
        self.assertIsNone(find_paddle(img, [RIGHT]).side)

    def test_no_paddle(self):
        img = frame()
        cv2.circle(img, RIGHT.wrist, 14, SKIN, -1)
        obs = find_paddle(img, [RIGHT])
        self.assertIsNone(obs.side)
        self.assertEqual(obs.center, search_circle(RIGHT)[:2])     # falls back to the hand

    def test_no_hands(self):
        self.assertIsNone(find_paddle(frame(), []))

    def test_red_shirt_away_from_the_hand_is_ignored(self):
        img = frame()
        cv2.rectangle(img, (40, 330), (300, 480), RED, -1)            # red shirt, lower left
        self.assertIsNone(find_paddle(img, [RIGHT]).side)

    def test_dark_hair_and_shadow_are_not_a_black_paddle(self):
        '''Black needs a compact blob: a thin dark streak through the search area is rejected.'''
        img = frame()
        cx, cy = paddle_center(RIGHT)
        cv2.line(img, (cx - 90, cy - 40), (cx + 90, cy + 40), BLACK, 9)
        self.assertIsNone(find_paddle(img, [RIGHT]).side)

    def test_red_beats_a_dark_background(self):
        '''Red paddle in front of a dark jumper: red is the side facing the camera.'''
        img = frame()
        cx, cy = paddle_center(RIGHT)
        cv2.rectangle(img, (cx - 120, cy - 120), (cx + 120, cy + 120), BLACK, -1)
        draw_paddle(img, RIGHT, RED)
        self.assertEqual(find_paddle(img, [RIGHT]).side, 'red')

    def test_auto_mode_picks_the_hand_holding_the_paddle(self):
        img = frame()
        draw_paddle(img, LEFT, BLACK)
        cv2.circle(img, RIGHT.wrist, 14, SKIN, -1)
        obs = find_paddle(img, [RIGHT, LEFT], cfg=VisionConfig(racket_hand='auto'))
        self.assertEqual((obs.hand, obs.side), ('left', 'black'))


class TestRacketArmOnly(unittest.TestCase):

    def test_only_the_racket_arm_is_searched(self):
        '''A paddle-coloured controller in the free (left) hand is ignored.'''
        img = frame()
        draw_paddle(img, LEFT, RED)                                  # e.g. the motor in the other hand
        cv2.circle(img, RIGHT.wrist, 14, SKIN, -1)
        obs = find_paddle(img, [RIGHT, LEFT])                        # default racket_hand = 'right'
        self.assertEqual((obs.hand, obs.side), ('right', None))

    def test_left_handed_player(self):
        img = frame()
        draw_paddle(img, LEFT, BLACK)
        obs = find_paddle(img, [RIGHT, LEFT], cfg=VisionConfig(racket_hand='left'))
        self.assertEqual((obs.hand, obs.side), ('left', 'black'))

    def test_racket_arm_hidden_means_no_observation(self):
        self.assertIsNone(find_paddle(frame(), [LEFT]))


def draw_fist_with_controller(img, hand, color=BLACK):
    '''The hand gripping something dark (the handle with the Double Motor on it).'''
    hx, hy, _ = hand_circle(hand)
    cv2.rectangle(img, (int(hx) - 28, int(hy) - 28), (int(hx) + 28, int(hy) + 28), color, -1)
    cv2.circle(img, hand.wrist, 14, SKIN, -1)


class TestPaddleNotHand(unittest.TestCase):

    def test_dark_controller_in_the_fist_is_not_the_black_side(self):
        img = frame()
        draw_fist_with_controller(img, RIGHT)
        self.assertIsNone(find_paddle(img, [RIGHT]).side)
        # without the hand cut-out it WOULD be mistaken for the black rubber
        self.assertEqual(find_paddle(img, [RIGHT], cfg=VisionConfig(hand_exclude=0)).side, 'black')

    def test_position_is_the_paddle_head_not_the_hand(self):
        img = frame()
        cx, cy = draw_paddle(img, RIGHT, RED)
        draw_fist_with_controller(img, RIGHT)
        obs = find_paddle(img, [RIGHT])
        self.assertEqual(obs.side, 'red')
        self.assertLess(np.hypot(obs.center[0] - cx, obs.center[1] - cy), 5)

    def test_black_paddle_with_dark_controller_in_the_fist(self):
        img = frame()
        cx, cy = draw_paddle(img, RIGHT, BLACK)
        draw_fist_with_controller(img, RIGHT)
        obs = find_paddle(img, [RIGHT])
        self.assertEqual(obs.side, 'black')
        self.assertLess(np.hypot(obs.center[0] - cx, obs.center[1] - cy), 5)

    def test_hand_circle_uses_knuckles_when_visible(self):
        with_knuckle = Hand('right', RIGHT.wrist, RIGHT.elbow, knuckle=(RIGHT.wrist[0] + 10, RIGHT.wrist[1] - 30))
        cx, cy, r = hand_circle(with_knuckle)
        self.assertEqual((cx, cy), (RIGHT.wrist[0] + 5, RIGHT.wrist[1] - 15))
        self.assertAlmostEqual(r, 1.3 * np.hypot(10, 30))

    def test_search_area_follows_the_forearm(self):
        cx, cy, r = search_circle(RIGHT)
        self.assertGreater(cx, RIGHT.wrist[0])      # elbow is left of the wrist -> look right
        self.assertLess(cy, RIGHT.wrist[1])         # elbow is below -> look up
        self.assertGreater(r, 40)


class TestGameInput(unittest.TestCase):

    def test_image_to_game_corners_and_clamping(self):
        for corner in (0, 1):
            x, y = image_to_game(U_RANGE[corner], V_RANGE[corner])
            self.assertAlmostEqual(x, GAME_X_RANGE[corner])
            self.assertAlmostEqual(y, GAME_Y_RANGE[corner])
        self.assertEqual(image_to_game(0.0, 0.0), image_to_game(U_RANGE[0], V_RANGE[0]))   # clamped
        x, y = image_to_game(0.5, 0.5)
        self.assertAlmostEqual(x, 0.0)
        self.assertAlmostEqual(y, sum(GAME_Y_RANGE) / 2)

    def test_moving_right_in_the_mirror_moves_paddle_right_and_up_is_up(self):
        self.assertLess(image_to_game(0.3, 0.5)[0], image_to_game(0.7, 0.5)[0])
        self.assertGreater(image_to_game(0.5, 0.3)[1], image_to_game(0.5, 0.7)[1])

    def test_stroke_for_side(self):
        self.assertEqual(stroke_for_side('red'), 'forehand')
        self.assertEqual(stroke_for_side('black'), 'backhand')
        self.assertEqual(stroke_for_side('black', forehand_color='black'), 'forehand')
        self.assertIsNone(stroke_for_side(None))


class TestSideFilter(unittest.TestCase):

    def test_single_bad_frame_does_not_flip(self):
        f = SideFilter()
        for s in ['red'] * 5 + ['black'] + ['red'] * 3:
            out = f.update(s)
        self.assertEqual(out, 'red')

    def test_switches_after_consistent_frames(self):
        f = SideFilter()
        outs = [f.update(s) for s in ['red'] * 5 + ['black'] * 3]
        self.assertEqual(outs[-1], 'black')
        self.assertEqual(outs[-2], 'red')

    def test_brief_edge_on_keeps_last_side_long_absence_clears(self):
        f = SideFilter()
        for s in ['black'] * 5 + [None] * 2:
            out = f.update(s)
        self.assertEqual(out, 'black')
        for _ in range(5):
            out = f.update(None)
        self.assertIsNone(out)


class TestOneEuroFilter(unittest.TestCase):
    FPS = 20

    def run_filter(self, f, points):
        return [f.update(p, i / self.FPS) for i, p in enumerate(points)]

    def test_first_value_passes_through_and_still_input_converges(self):
        f = OneEuroFilter()
        self.assertEqual(f.update((100, 50), 0.0), (100, 50))
        out = self.run_filter(f, [(200, 80)] * 60)
        self.assertAlmostEqual(out[-1][0], 200, delta=1)

    def test_jitter_is_smoothed_when_still(self):
        rng = np.random.default_rng(1)
        noisy = [(300 + rng.normal(0, 4), 200 + rng.normal(0, 4)) for _ in range(100)]
        out = np.array(self.run_filter(OneEuroFilter(), noisy))[20:]
        self.assertLess(out[:, 0].std(), 0.5 * np.array(noisy)[20:, 0].std())

    def test_fast_swing_has_little_lag(self):
        '''Paddle sweeping 1500 px/s: the filter should stay close behind it.'''
        sweep = [(100 + 1500 * i / self.FPS, 240) for i in range(12)]
        euro = self.run_filter(OneEuroFilter(), sweep)
        lag = sweep[-1][0] - euro[-1][0]
        self.assertLess(lag, 0.25 * 1500 / self.FPS * 4)       # well under 4 frames behind


class TestConfig(unittest.TestCase):

    def test_save_load_round_trip(self):
        path = os.path.join(tempfile.mkdtemp(), 'cfg.json')
        cfg = VisionConfig(red_sat_min=90, black_val_max=85, exposure=-6)
        cfg.save(path)
        self.assertEqual(VisionConfig.load(path), cfg)

    def test_missing_file_gives_defaults_and_unknown_keys_are_ignored(self):
        d = tempfile.mkdtemp()
        self.assertEqual(VisionConfig.load(os.path.join(d, 'none.json')), VisionConfig())
        path = os.path.join(d, 'old.json')
        with open(path, 'w') as f:
            f.write('{"red_sat_min": 99, "removed_setting": 1}')
        self.assertEqual(VisionConfig.load(path).red_sat_min, 99)


def hsv_pixels(h, s, v, n=400, spread=(3, 15, 15), seed=0):
    rng = np.random.default_rng(seed)
    hh = (h + rng.normal(0, spread[0], n)) % 180
    ss = np.clip(s + rng.normal(0, spread[1], n), 0, 255)
    vv = np.clip(v + rng.normal(0, spread[2], n), 0, 255)
    return np.stack([hh, ss, vv], 1).astype(np.uint8)


class TestCalibration(unittest.TestCase):

    def test_red_hue_that_wraps_around_zero(self):
        hsv = np.array([[[175, 200, 200], [5, 200, 200], [90, 200, 200]]], np.uint8)
        m = hue_mask(hsv, 170, 10, 100, 100)
        self.assertEqual(list(m[0] > 0), [True, True, False])
        self.assertEqual(list(hue_mask(hsv, 0, 10, 100, 100)[0] > 0), [False, True, False])

    def test_calibrated_red_covers_samples_on_both_sides_of_zero(self):
        cfg = calibrate_red(hsv_pixels(178, 150, 160), VisionConfig())
        self.assertGreater(cfg.red_hue_lo, cfg.red_hue_hi)      # wraps
        img = hsv_pixels(178, 150, 160, seed=1).reshape(-1, 1, 3)
        m = hue_mask(img, cfg.red_hue_lo, cfg.red_hue_hi, cfg.red_sat_min, cfg.red_val_min)
        self.assertGreater(np.count_nonzero(m) / m.size, 0.9)

    def test_stray_pixels_in_the_box_do_not_widen_red_into_purple_and_blue(self):
        '''A quarter of the sample is skin/background/blue: red must stay red.'''
        samples = np.vstack([hsv_pixels(178, 170, 160, n=300),
                             hsv_pixels(115, 150, 120, n=60, seed=2),      # blue
                             hsv_pixels(15, 90, 200, n=40, seed=3)])       # skin
        cfg = calibrate_red(samples, VisionConfig())
        test = np.array([[[178, 170, 160], [3, 170, 160], [140, 170, 160], [115, 170, 160]]], np.uint8)
        m = hue_mask(test, cfg.red_hue_lo, cfg.red_hue_hi, 0, 0)[0] > 0
        self.assertEqual(list(m), [True, True, False, False])     # red yes; magenta-purple, blue no

    def test_calibrated_black_accepts_a_greyish_black_rubber(self):
        cfg = calibrate_black(hsv_pixels(100, 40, 85), VisionConfig())     # lit black: V ~85
        self.assertGreaterEqual(cfg.black_val_max, 85)

    def test_calibration_fixes_a_dull_red_paddle_the_defaults_miss(self):
        '''Warm, dim lighting makes the red rubber low-saturation: defaults fail, calibrated works.'''
        dull_red = (95, 90, 150)                                    # BGR, saturation ~ 95
        img = frame()
        draw_paddle(img, RIGHT, dull_red)
        self.assertIsNone(find_paddle(img, [RIGHT]).side)
        cx, cy = paddle_center(RIGHT)
        patch = cv2.cvtColor(img[cy - 15:cy + 15, cx - 15:cx + 15], cv2.COLOR_BGR2HSV).reshape(-1, 3)
        cfg = calibrate_red(patch, VisionConfig())
        self.assertEqual(find_paddle(img, [RIGHT], cfg=cfg).side, 'red')

    def test_background_check_tightens_black_but_keeps_the_paddle(self):
        black_rubber = hsv_pixels(100, 30, 40)
        cfg = calibrate_black(black_rubber, VisionConfig())
        cfg.black_val_max = 120                                     # far too loose
        room = frame()
        room[:200] = (75, 75, 80)                                   # dim grey wall, V ~80
        self.assertGreater(false_positive_rates(room, cfg)[1], 0.2)
        tighten_for_background(cfg, hsv_pixels(0, 200, 200), black_rubber, room)
        self.assertLess(false_positive_rates(room, cfg)[1], 0.01)
        img = frame()
        draw_paddle(img, RIGHT, BLACK)
        self.assertEqual(find_paddle(img, [RIGHT], cfg=cfg).side, 'black')


class TestTrackingContinuity(unittest.TestCase):

    def test_bent_wrist_paddle_outside_wrist_circle_is_found_near_last_position(self):
        img = frame()
        cx, cy, r = search_circle(RIGHT)
        far = (int(cx + 1.3 * r), int(cy))                          # just outside the circle
        cv2.ellipse(img, far, (38, 46), 0, 0, 360, RED, -1)
        self.assertIsNone(find_paddle(img, [RIGHT]).side)
        obs = find_paddle(img, [RIGHT], last_center=(far[0] - 20, far[1]), last_hand='right')
        self.assertEqual(obs.side, 'red')

    def test_no_face_keeps_paddle_where_it_was_relative_to_the_hand(self):
        img = frame()
        offset = (70, -60)
        obs = find_paddle(img, [RIGHT], last_offset=offset, last_hand='right')
        self.assertIsNone(obs.side)
        self.assertEqual(obs.center, (RIGHT.wrist[0] + 70, RIGHT.wrist[1] - 60))

    def test_sticks_with_the_paddle_hand_when_both_are_empty(self):
        img = frame()
        obs = find_paddle(img, [RIGHT, LEFT], cfg=VisionConfig(racket_hand='auto'), last_hand='left')
        self.assertEqual(obs.hand, 'left')


if __name__ == '__main__':
    unittest.main()
