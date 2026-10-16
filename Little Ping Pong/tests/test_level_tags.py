'''AprilTags choose the level: detection on synthetic frames, the ID -> level table, and
the steady-sighting rule.'''

import os
import tempfile
import unittest

import cv2
import numpy as np

from drill import Drill
from level_tags import (DEFAULT_TAGS, LevelChooser, LevelTagDetector, TagSighting, assign,
                        choose_level, draw_on_preview, load_tags, save_tags, tag_image)


def scene(*tags, size=(480, 640)):
    '''A grey frame with (tag_id, x, y, px) tags pasted in.'''
    frame = np.full(size, 128, np.uint8)
    for tag_id, x, y, px in tags:
        img = tag_image(tag_id, px)
        frame[y:y + img.shape[0], x:x + img.shape[1]] = img
    return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)


def seen(level, tag_id=0):
    return TagSighting(tag_id, np.zeros((4, 2)), level)


class TestDetection(unittest.TestCase):

    def test_each_default_tag_is_labelled(self):
        det = LevelTagDetector(dict(DEFAULT_TAGS))
        for tag_id, label in [(0, 'Normal'), (1, 'Hard'), (2, 'Pro')]:
            s = det.detect(scene((tag_id, 200, 100, 200)))
            self.assertEqual([x.tag_id for x in s], [tag_id])
            self.assertEqual(s[0].label, label)

    def test_no_tag_no_sightings(self):
        self.assertEqual(LevelTagDetector(dict(DEFAULT_TAGS)).detect(scene()), [])

    def test_unassigned_tag_has_no_level(self):
        s = LevelTagDetector(dict(DEFAULT_TAGS)).detect(scene((7, 200, 100, 200)))
        self.assertIsNone(s[0].level)
        self.assertIn('no level', s[0].label)

    def test_biggest_tag_first(self):
        s = LevelTagDetector(dict(DEFAULT_TAGS)).detect(scene((0, 20, 20, 120), (2, 250, 100, 300)))
        self.assertEqual([x.tag_id for x in s], [2, 0])

    def test_rotated_tag_still_read(self):
        frame = cv2.rotate(scene((1, 200, 100, 200)), cv2.ROTATE_90_CLOCKWISE)
        self.assertEqual(LevelTagDetector(dict(DEFAULT_TAGS)).detect(frame)[0].label, 'Hard')


class TestTagTable(unittest.TestCase):

    def test_missing_file_gives_defaults_and_save_round_trips(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, 'level_tags.json')
            self.assertEqual(load_tags(path), DEFAULT_TAGS)
            save_tags({14: 3, 5: 1}, path)
            self.assertEqual(load_tags(path), {14: 3, 5: 1})

    def test_assign_keeps_one_tag_per_level(self):
        self.assertEqual(assign(DEFAULT_TAGS, 14, 3), {0: 1, 1: 2, 14: 3})
        self.assertEqual(assign(DEFAULT_TAGS, 0, 2), {0: 2, 2: 3})


class TestLevelChooser(unittest.TestCase):

    def test_level_chosen_only_after_steady_frames(self):
        c = LevelChooser(stable_frames=5)
        results = [c.update([seen(2)]) for _ in range(5)]
        self.assertEqual(results, [None, None, None, None, 2])
        self.assertEqual(c.level, 2)

    def test_glimpse_or_flicker_does_not_switch(self):
        c = LevelChooser(stable_frames=5)
        for frame in [[seen(3)], [], [seen(3)], [seen(1)], [seen(3)]] * 3:
            self.assertIsNone(c.update(frame))
        self.assertIsNone(c.level)

    def test_level_stays_after_tag_removed_and_reports_change_once(self):
        c = LevelChooser(stable_frames=3)
        for _ in range(3):
            c.update([seen(1)])
        for _ in range(10):
            self.assertIsNone(c.update([]))             # tag put away: level kept
        self.assertEqual(c.level, 1)
        for _ in range(10):
            self.assertIsNone(c.update([seen(1)]))      # shown again: no repeat change
        self.assertEqual([c.update([seen(3)]) for _ in range(3)], [None, None, 3])

    def test_unassigned_tags_ignored(self):
        c = LevelChooser(stable_frames=2)
        for _ in range(5):
            c.update([seen(None, 7)])
        self.assertIsNone(c.level)

    def test_progress_only_towards_a_change(self):
        c = LevelChooser(stable_frames=4)
        c.update([seen(2)])
        self.assertEqual(c.progress, 0.25)
        for _ in range(3):
            c.update([seen(2)])
        self.assertEqual(c.progress, 0.0)               # accepted: no bar while still held
        c.update([])
        self.assertEqual(c.progress, 0.0)


class TestDrillLevelByTag(unittest.TestCase):
    '''What drill_play does with each webcam frame's tags on the start/results screens.'''

    def test_steady_tag_sets_the_drill_level(self):
        d, c = Drill(level=1), LevelChooser(stable_frames=3)
        self.assertEqual([choose_level(d, c, [seen(3)]) for _ in range(4)], [None, None, 3, None])
        self.assertEqual(d.level_num, 3)
        self.assertEqual(d.level.name, 'Pro')

    def test_tag_for_the_current_level_does_nothing(self):
        d, c = Drill(level=2), LevelChooser(stable_frames=3)
        for _ in range(10):
            self.assertIsNone(choose_level(d, c, [seen(2)]))
            self.assertEqual(c.progress, 0.0)

    def test_tag_works_again_after_a_keyboard_change(self):
        d, c = Drill(level=1), LevelChooser(stable_frames=2)
        for _ in range(2):
            choose_level(d, c, [seen(3)])
        choose_level(d, c, [])                          # tag put away
        d.set_level(1)                                  # 1 key pressed
        choose_level(d, c, [])
        self.assertEqual([choose_level(d, c, [seen(3)]) for _ in range(2)], [None, 3])

    def test_real_tag_image_end_to_end(self):
        d, c, det = Drill(level=1), LevelChooser(stable_frames=3), LevelTagDetector(dict(DEFAULT_TAGS))
        frame = scene((1, 300, 150, 160))
        for _ in range(3):
            choose_level(d, c, det.detect(frame))
        self.assertEqual(d.level.name, 'Hard')


class TestPreview(unittest.TestCase):

    def test_outline_is_scaled_and_mirrored_onto_the_preview(self):
        s = LevelTagDetector(dict(DEFAULT_TAGS)).detect(scene((0, 40, 100, 160)))   # left of a 640 frame
        view = np.zeros((240, 320, 3), np.uint8)
        draw_on_preview(view, s, 640)
        green = np.argwhere(view[:, :, 1] == 220)
        self.assertGreater(green[:, 1].min(), 160)      # mirrored: right half of the preview
        self.assertLess(green[:, 0].max(), 240)


if __name__ == '__main__':
    unittest.main()
