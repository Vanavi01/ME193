'''
Forehand Drill: smash / chop practice mini-game (separate from play.py).

    python drill_play.py                               # mouse = paddle; Space = smash, H = normal hit, C = chop, T = topspin
    python drill_play.py --vision --imu --serial 0994  # the real thing: red paddle + Double Motor
    options: --level 1-3   --balls 20   --camera N   --no-haptics   --no-mqtt   --topic T

With --vision, show a level AprilTag to the webcam on the start or results screen to
choose the level (tag 0 = Normal, 1 = Hard, 2 = Pro; see live_level_tags.py).

After each completed round the results are published over MQTT (score_publisher.py):
ME193/Rogers/Victor = final score (float), .../hits and .../misses = counts.

A ball machine feeds balls to your forehand and calls SMASH (high ball: hit it
hard), HIT (normal ball: steady forehand) or CHOP (low short ball: chop under it). The stroke you actually played is
recognised by the IMU stroke classifier (models/stroke_model.json, from
train_strokes.py). Enter = start / play again, Esc = quit.
'''

import argparse
import os

import pygame

from camera import Camera
from drill import FEED_X, ROUND_BALLS, Drill, tracking_x_range
from haptics import pattern_for_event
from level_tags import LevelChooser, LevelTagDetector, choose_level
from paddle_vision import VisionConfig
from physics import CPU_HIT_Z, NET_Z, PLAYER_HIT_Z
from score_publisher import TOPIC, ScorePublisher
from play import (FPS, HEIGHT, POP_COLORS, STROKE_MODEL, UI_BORDER, UI_TEXT, WIDTH,
                  Renderer, font, mouse_to_paddle, outlined_text, panel)

CALL_STYLE = {
    'smash': ((255, 140, 30), 'SMASH!', 'high ball: hit it hard'),
    'hit': ((70, 190, 90), 'HIT!', 'normal ball: a steady forehand'),
    'chop': ((90, 180, 255), 'CHOP!', 'low ball: brush under it'),
}
GOOD, BAD = (120, 230, 90), (255, 90, 90)
MACHINE, MACHINE_DARK, TUBE = (235, 236, 240), (150, 155, 170), (60, 64, 76)


def drill_camera(sign=1):
    '''Over the forehand shoulder, looking diagonally across the table (mirrored for left-handers).'''
    return Camera(WIDTH, HEIGHT, pos=(0.25 * sign, 0.9, -1.7), pitch_deg=18, yaw_deg=14 * sign, focal=1000)


class DrillRenderer(Renderer):

    def __init__(self, screen, sign=1):
        super().__init__(screen, drill_camera(sign))
        self.sign = sign
        self.cam_box = pygame.Rect(20, HEIGHT - 260, 320, 240)          # bottom-left: out of the way
        self.publish_status = 'off'                                    # MQTT status for the results screen
        self.publish_topic = TOPIC
        self.tags_on = False                                           # AprilTag level choice active
        self.tag_progress = 0.0                                        # 0..1 while a tag is held up
        self.level_flash_t = -10.0                                     # when a tag last changed the level

    def draw_feeder(self, drill):
        '''Ball machine at the far end, on the forehand side.'''
        x, z = FEED_X * self.sign, CPU_HIT_Z + 0.15
        w, d = 0.22, 0.25
        def quad(y0, y1, zz):
            return [self.p(x - w, y0, zz)[:2], self.p(x + w, y0, zz)[:2],
                    self.p(x + w, y1, zz)[:2], self.p(x - w, y1, zz)[:2]]
        pygame.draw.polygon(self.screen, MACHINE_DARK, quad(-0.76, 0.25, z + d))          # back
        pygame.draw.polygon(self.screen, MACHINE, quad(-0.76, 0.25, z))                  # front
        top = [self.p(x - w, 0.25, z)[:2], self.p(x + w, 0.25, z)[:2],
               self.p(x + w, 0.25, z + d)[:2], self.p(x - w, 0.25, z + d)[:2]]
        pygame.draw.polygon(self.screen, (200, 202, 210), top)
        for i in range(5):                                                               # balls in the hopper
            bx, by, sc = self.p(x - 0.12 + 0.06 * i, 0.29, z + 0.1 + 0.03 * (i % 2))
            pygame.draw.circle(self.screen, (255, 250, 238), (bx, by), max(3, 0.02 * sc))
        a, b = self.p(x, 0.3, z), self.p(x, 0.35, z - 0.12)                             # nozzle
        pygame.draw.line(self.screen, TUBE, a[:2], b[:2], max(4, int(0.06 * a[2])))
        pygame.draw.polygon(self.screen, UI_BORDER, quad(0.12, 0.2, z))                  # blue stripe

    def draw(self, drill, swing_meter, input_label):
        self.screen.blit(self.back, (0, 0))
        self.draw_feeder(drill)
        self.screen.blit(self.table, (0, 0))
        far = drill.ball.z > NET_Z
        if drill.state == 'rally' and far:
            self.draw_ball(drill)
        self.screen.blit(self.net, (0, 0))
        if drill.state == 'rally' and not far:
            self.draw_ball(drill)
        if drill.state == 'done':
            self.draw_player_paddle(drill)              # behind the results panel
            self.draw_hud(drill, swing_meter, input_label)
        else:
            self.draw_hud(drill, swing_meter, input_label)
            self.draw_player_paddle(drill)              # on top, so the webcam box never hides it

    def draw_hud(self, drill, swing_meter, input_label):
        panel(self.screen, pygame.Rect(20, 20, 280, 150))
        self.screen.blit(font(20).render('SCORE', True, UI_TEXT), (40, 30))
        self.screen.blit(outlined_text(str(drill.score), 52, UI_BORDER, (255, 255, 255)), (40, 50))
        self.screen.blit(font(18).render(f'STREAK {drill.streak}   BEST {drill.best}', True, UI_TEXT), (40, 112))
        self.screen.blit(font(18).render(f'BALL {min(drill.fed, drill.balls)}/{drill.balls}', True, UI_TEXT), (40, 138))
        panel(self.screen, pygame.Rect(20, 180, 280, 40), radius=20)
        stars = '*' * drill.level_num + '-' * (3 - drill.level_num)
        self.screen.blit(font(18).render(f'{drill.level.name}  {stars}', True, UI_TEXT), (40, 189))

        if self.vision is not None:                         # webcam feed only when --vision is on
            self.draw_camera_box()
        bar_x = self.cam_box.right + 20 if self.vision is not None else 20
        panel(self.screen, pygame.Rect(bar_x, HEIGHT - 76, 260, 56), radius=16)
        self.screen.blit(font(16).render(f'SWING  ({input_label})', True, UI_TEXT), (bar_x + 16, HEIGHT - 70))
        bar = pygame.Rect(bar_x + 16, HEIGHT - 46, 228, 14)
        pygame.draw.rect(self.screen, (220, 228, 240), bar, border_radius=7)
        if swing_meter > 0:
            pygame.draw.rect(self.screen, (255, 160, 40), (bar.x, bar.y, int(bar.width * swing_meter), bar.height),
                             border_radius=7)

        if drill.state == 'ready':
            self._center_text('Forehand Drill', 64, (255, 255, 255), HEIGHT * 0.36)
            self._center_text('SMASH the high balls, HIT the normal ones, CHOP the low ones', 26,
                              (255, 255, 255), HEIGHT * 0.46, 3)
            hint = 'Press ENTER or raise both hands' if self.vision is not None else 'Press ENTER'
            self._center_text(hint, 26, (255, 230, 120), HEIGHT * 0.54, 3)
            if input_label != 'IMU':
                self._center_text('Mouse = paddle   Space = smash   H = hit   C = chop   T = topspin', 20,
                                  (255, 255, 255), HEIGHT * 0.61, 2)
            self.draw_tag_prompt(drill, HEIGHT * 0.70)
        elif drill.state == 'done':
            self.draw_results(drill)
            self.draw_tag_prompt(drill, HEIGHT - 130)
        elif drill.paused:
            self._center_text('Paused', 64, (255, 255, 255), HEIGHT * 0.36)
            self._center_text('step back into view', 30, (255, 255, 255), HEIGHT * 0.45, 3)
        else:
            if drill.call and drill.heading == 'to_player':
                self.draw_call(drill.call)
            if drill.message:
                good = bool(drill.results) and drill.results[-1][1] == 'correct'
                age = pygame.time.get_ticks() / 1000 - self.pop_time
                self._center_text(drill.message, int(64 * min(1.0, 0.6 + age * 3)), GOOD if good else BAD,
                                  HEIGHT * 0.40)

    def draw_tag_prompt(self, drill, y):
        '''Level choice (AprilTag, or 1/2/3): the level and its pace, a bar while a tag is
        held, a flash on change.'''
        flash = pygame.time.get_ticks() / 1000 - self.level_flash_t < 1.5
        if flash:
            self._center_text(f'Level: {drill.level.name.upper()}', 40, GOOD, y)
        else:
            how = 'show a tag: Normal / Hard / Pro' if self.tags_on else '1 / 2 / 3 to change'
            self._center_text(f'Level: {drill.level.name}   ({how})', 22, (255, 255, 255), y, 2)
        pace = drill.pace
        self._center_text(f'ball speed {pace.speed:.0%}   {pace.gap_s:.1f} s between serves', 18,
                          (255, 230, 120), y + (34 if flash else 26), 2)
        if self.tag_progress > 0 and not flash:
            bar = pygame.Rect(WIDTH // 2 - 120, int(y) + 44, 240, 10)
            pygame.draw.rect(self.screen, (220, 228, 240), bar, border_radius=5)
            pygame.draw.rect(self.screen, (255, 160, 40), (bar.x, bar.y, int(bar.width * self.tag_progress),
                                                           bar.height), border_radius=5)

    def draw_call(self, call):
        color, text, hint = CALL_STYLE[call]
        box = pygame.Rect(0, 0, 360, 96)
        box.midtop = (WIDTH // 2, 20)
        pygame.draw.rect(self.screen, color, box, border_radius=24)
        pygame.draw.rect(self.screen, (255, 255, 255), box, width=4, border_radius=24)
        t = outlined_text(text, 50, (255, 255, 255), (30, 30, 50), 3)
        self.screen.blit(t, t.get_rect(center=(box.centerx, box.y + 40)))
        h = font(18).render(hint, True, (255, 255, 255))
        self.screen.blit(h, h.get_rect(center=(box.centerx, box.y + 78)))

    def draw_results(self, drill):
        box = pygame.Rect(0, 0, 520, 368)
        box.center = (WIDTH // 2, HEIGHT // 2 - 20)
        panel(self.screen, box, radius=26)
        self._center_text('Round over!', 48, (255, 215, 60), box.y + 45)
        rows = [('Score', f'{drill.score}'),
                ('Accuracy', f'{drill.accuracy():.0%}'),
                ('Smash', self._ratio(drill, 'smash')),
                ('Hit', self._ratio(drill, 'hit')),
                ('Chop', self._ratio(drill, 'chop')),
                ('Best streak', f'{drill.best}')]
        for i, (k, v) in enumerate(rows):
            y = box.y + 95 + i * 38
            self.screen.blit(font(24).render(k, True, UI_TEXT), (box.x + 60, y))
            val = font(24).render(v, True, UI_BORDER)
            self.screen.blit(val, val.get_rect(topright=(box.right - 60, y)))
        self._center_text('ENTER to play again', 24, (255, 255, 255), box.bottom - 28, 2)
        status = self.publish_status
        if status != 'off':
            text, color = {'sending': ('Sending results...', (255, 230, 120)),
                           'sent': (f'Results sent to {self.publish_topic}', (150, 240, 120))
                           }.get(status, ("Couldn't send results (offline?)", (255, 120, 120)))
            self._center_text(text, 20, color, box.bottom + 22, 2)

    @staticmethod
    def _ratio(drill, call):
        rs = [o for c, o in drill.results if c == call]
        return f'{sum(o == "correct" for o in rs)}/{len(rs)}' if rs else '-'


def main():
    parser = argparse.ArgumentParser(description='Forehand Drill: smash / chop practice')
    parser.add_argument('--imu', action='store_true', help='swings + stroke type from the Double Motor')
    parser.add_argument('--serial', default=None)
    parser.add_argument('--vision', action='store_true', help='track the red paddle with the webcam')
    parser.add_argument('--camera', default=None,
                        help="'laptop', 'usb' or a camera number (default: the one saved by find_camera.py)")
    parser.add_argument('--level', type=int, default=1, choices=(1, 2, 3))
    parser.add_argument('--balls', type=int, default=ROUND_BALLS)
    parser.add_argument('--no-haptics', action='store_true')
    parser.add_argument('--no-mqtt', action='store_true', help="don't publish round results")
    parser.add_argument('--topic', default=TOPIC, help='MQTT topic for the results')
    args = parser.parse_args()

    hand = VisionConfig.load().racket_hand
    sign = -1 if hand == 'left' else 1

    tracker = None
    if args.vision:
        from red_tracker import RedTracker
        tracker = RedTracker(args.camera, x_range=tracking_x_range(hand))   # forehand zone only
        tracker.tag_detector = LevelTagDetector()       # level by AprilTag before each round
        tracker.start()

    paddle_imu = detector = haptics = classifier = None
    if args.imu:
        from haptics import Haptics
        from paddle_imu import PaddleIMU
        from swing import SwingDetector
        paddle_imu = PaddleIMU()
        print('Connecting to Double Motor...')
        paddle_imu.connect(card_serial=args.serial)
        detector = SwingDetector()
        haptics = Haptics(paddle_imu, enabled=not args.no_haptics)
        if os.path.exists(STROKE_MODEL):
            from strokes import LiveStrokeClassifier, load_model
            classifier = LiveStrokeClassifier(load_model(STROKE_MODEL))
        else:
            print('No models/stroke_model.json: every hit counts as flat (run record/train_strokes.py).')

    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption('Forehand Drill')
    clock = pygame.time.Clock()
    renderer = DrillRenderer(screen, sign)
    publisher = ScorePublisher(args.topic, enabled=not args.no_mqtt)
    renderer.publish_topic = args.topic
    drill = Drill(args.level, hand=hand, balls=args.balls)
    swing_meter, start_count = 0.0, 0
    chooser, tag_frame = LevelChooser(), -1
    renderer.tags_on = tracker is not None
    input_label = 'IMU' if paddle_imu else 'keyboard'
    keys = {pygame.K_SPACE: ('flat', 0.9), pygame.K_h: ('flat', 0.4), pygame.K_c: ('backspin', 0.5),
            pygame.K_t: ('topspin', 0.8)}

    try:
        running = True
        while running:
            dt = min(clock.tick(FPS) / 1000, 0.05)
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                    running = False
                elif event.type == pygame.KEYDOWN and event.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
                    drill.start()
                elif event.type == pygame.KEYDOWN and event.key in (pygame.K_1, pygame.K_2, pygame.K_3):
                    drill.set_level(event.key - pygame.K_0)
                elif not paddle_imu and event.type == pygame.KEYDOWN and event.key in keys:
                    kind, strength = keys[event.key]
                    drill.swing(strength)
                    drill.set_stroke(kind)
                    swing_meter = strength
                elif not paddle_imu and event.type == pygame.MOUSEBUTTONDOWN:
                    drill.swing(0.9)
                    drill.set_stroke('flat')
                    swing_meter = 0.9

            if paddle_imu:
                for s in paddle_imu.drain():
                    if classifier:
                        classifier.add(s)
                    e = detector.update(s)
                    if e:
                        drill.swing(e.strength)
                        swing_meter = max(swing_meter, e.strength, 0.15)
                        if classifier:
                            classifier.on_swing(e.t)
                result = classifier.poll() if classifier else None
                if result:
                    drill.set_stroke(result[0])

            if tracker:
                tracker.watch_tags = drill.state in ('ready', 'done')     # never during a round
                renderer.vision = v = tracker.get()
                if v.tags is not None and v.frame_id != tag_frame:    # each webcam frame once
                    tag_frame = v.frame_id
                    if choose_level(drill, chooser, v.tags):
                        renderer.level_flash_t = pygame.time.get_ticks() / 1000
                    renderer.tag_progress = chooser.progress
                elif v.tags is None:
                    renderer.tag_progress = 0.0
                if v.start_count != start_count:                # both hands up: start / play again
                    start_count = v.start_count
                    if drill.state in ('ready', 'done'):
                        drill.start()
                if v.present is False and drill.state in ('rally', 'gap') and not drill.paused:
                    drill.pause()
                elif v.present and drill.paused:
                    drill.resume()
                paddle_x = drill.paddle[0] if v.paddle_x is None else v.paddle_x
            else:
                paddle_x = mouse_to_paddle(renderer.cam, *pygame.mouse.get_pos())[0]

            drill.update(dt, paddle_x)
            for event in drill.drain_events():
                if event[0] in ('judged', 'miss'):
                    renderer.pop()
                if event[0] == 'round_over':
                    publisher.publish_round(drill)      # background thread: never blocks the game
                feedback = haptics and pattern_for_event(event)
                if feedback:
                    haptics.play(*feedback)
                    detector.suppress_until(paddle_imu.sample_time(haptics.quiet_until()))

            swing_meter = max(0.0, swing_meter - dt * 1.2)
            renderer.publish_status = publisher.status
            renderer.draw(drill, swing_meter, input_label)
            pygame.display.flip()
    finally:
        pygame.quit()
        publisher.close()                               # let a just-finished round's results go out
        if tracker:
            tracker.stop()
        if haptics:
            haptics.close()
        if paddle_imu:
            paddle_imu.disconnect()


if __name__ == '__main__':
    main()
