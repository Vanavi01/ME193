'''
The playable 2.5D table tennis screen (Wii Sports Resort style).

    python play.py                          # mouse = paddle, Space/click = swing
    python play.py --imu                    # mouse = paddle position, Double Motor = swing + haptics
                                            # (add --no-haptics to switch the motor pulses off)
    With --imu and models/stroke_model.json (train_strokes.py), forehand hits are
    classified: topspin flies lower and faster, a chop floats shorter and slower
    (--no-spin to switch off).
    python play.py --vision                 # webcam tracks the red paddle (left/right), Space/click = swing
    python play.py --vision --imu --serial 0994     # the full thing
    options: --camera 1  (default: the camera chosen with find_camera.py)

Controls: Enter = start/restart, 1/2/3 = level, Esc = quit.
With --vision the live webcam feed is shown top right. Tracking is red-only and
x-only (red_tracker.py): the paddle follows you left/right and automatically
meets the ball's height. MediaPipe Pose runs alongside (async): the skeleton is
drawn in the camera box, red is only accepted near the racket arm, and the game
pauses when you step out of view.
'''

import argparse
import functools
import math
import os
import random

import pygame
import pygame.gfxdraw

from camera import Camera
from haptics import pattern_for_event
from game import LEVELS, Game
from physics import (BALL_R, CPU_HIT_Z, NET_H, NET_Z, PLAYER_HIT_Z, TABLE_H,
                     TABLE_L, TABLE_W)

STROKE_MODEL = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models', 'stroke_model.json')
WIDTH, HEIGHT = 1280, 720
FPS = 60
KEYBOARD_SWING_STRENGTH = 0.6
PADDLE_R = 0.085            # paddle head radius, metres
CAM_BOX = pygame.Rect(WIDTH - 340, 20, 320, 240)     # 4:3 like the webcam
FLOOR_Y = -TABLE_H

# Palette: bright, saturated, friendly
SKY_TOP, SKY_BOTTOM = (120, 190, 245), (200, 232, 255)
WALL = (235, 240, 248)
FLOOR_NEAR, FLOOR_FAR = (214, 160, 98), (188, 132, 76)
MAT = (200, 88, 72)
TABLE_TOP, TABLE_SIDE, TABLE_LINE = (28, 96, 186), (16, 60, 125), (250, 250, 250)
LEG = (60, 64, 72)
NET_MESH, NET_TAPE = (30, 34, 44, 150), (250, 250, 250)
BALL, BALL_SHADE = (255, 250, 238), (230, 214, 190)
PADDLE_RED, PADDLE_BLACK, PADDLE_EDGE, HANDLE = (220, 36, 44), (34, 34, 38), (40, 20, 20), (196, 150, 96)
CPU_SHIRT, CPU_SKIN, CPU_PADDLE = (40, 170, 90), (250, 214, 176), (30, 80, 200)
UI_PANEL, UI_BORDER, UI_TEXT = (255, 255, 255), (60, 140, 230), (30, 70, 140)
POP_COLORS = {'Nice': (255, 220, 40), 'Smash': (255, 120, 30),
              'Topspin': (255, 150, 40), 'Chop': (120, 200, 255)}     # by first word
SPIN_TRAIL = {'topspin': (255, 150, 40), 'backspin': (120, 200, 255)}


@functools.lru_cache(maxsize=None)
def font(size):
    return pygame.font.SysFont('arialroundedmtbold,verdana,arial', size, bold=True)


@functools.lru_cache(maxsize=256)          # same text every frame -> render once
def outlined_text(text, size, color, outline=(25, 45, 90), width=3):
    f = font(size)
    base = f.render(text, True, color)
    edge = f.render(text, True, outline)
    surf = pygame.Surface((base.get_width() + 2 * width, base.get_height() + 2 * width), pygame.SRCALPHA)
    for dx in range(-width, width + 1):
        for dy in range(-width, width + 1):
            if dx * dx + dy * dy <= width * width:
                surf.blit(edge, (dx + width, dy + width))
    surf.blit(base, (width, width))
    return surf


def panel(surface, rect, radius=18):
    s = pygame.Surface(rect.size, pygame.SRCALPHA)        # shadow only as big as the panel
    pygame.draw.rect(s, (0, 0, 0, 60), s.get_rect(), border_radius=radius)
    surface.blit(s, rect.move(0, 4))
    pygame.draw.rect(surface, UI_PANEL, rect, border_radius=radius)
    pygame.draw.rect(surface, UI_BORDER, rect, width=4, border_radius=radius)


def aa_circle(surface, color, center, r):
    x, y, r = int(center[0]), int(center[1]), max(1, int(r))
    pygame.gfxdraw.filled_circle(surface, x, y, r, color)
    pygame.gfxdraw.aacircle(surface, x, y, r, color)


class Renderer:

    def __init__(self, screen, camera=None):
        self.screen = screen
        font.cache_clear()              # cached fonts die with pygame.quit(); start fresh
        outlined_text.cache_clear()
        self.cam = camera or Camera(WIDTH, HEIGHT)
        self.cam_box = CAM_BOX           # where the webcam feed goes (the drill moves it)
        self.back = self._draw_back()
        self.table = self._draw_table()
        self.net = self._draw_net()
        self.pop_time = 0.0
        self.vision = None              # latest RedState when --vision is on
        self._cam_surface = None
        self._cam_frame = None
        self.backhand_time = 0.0
        self._was_backhand = False

    def p(self, x, y, z):
        return self.cam.project(x, y, z)

    def quad(self, pts):
        return [self.p(*pt)[:2] for pt in pts]

    # ── static layers (drawn once) ───────────────────────────────────────────

    def _draw_back(self):
        s = pygame.Surface((WIDTH, HEIGHT))
        for row in range(HEIGHT):                       # sky gradient
            k = row / HEIGHT
            s.fill([int(a + (b - a) * k) for a, b in zip(SKY_TOP, SKY_BOTTOM)], (0, row, WIDTH, 1))

        back_z = TABLE_L + 6
        wall_top = self.p(0, 3.2, back_z)[1]
        wall_bottom = self.p(0, FLOOR_Y, back_z)[1]
        pygame.draw.rect(s, WALL, (0, wall_top, WIDTH, wall_bottom - wall_top))

        # Bleachers with a colourful crowd
        rng = random.Random(7)
        rows = 5
        for r in range(rows):
            y0 = FLOOR_Y + 0.5 + r * 0.42
            top = self.p(0, y0 + 0.42, back_z)[1]
            bottom = self.p(0, y0, back_z)[1]
            shade = 150 + r * 12
            pygame.draw.rect(s, (shade, shade + 10, shade + 25), (0, top, WIDTH, bottom - top))
            pygame.draw.line(s, (110, 120, 140), (0, bottom), (WIDTH, bottom), 2)
            x = -8.0 + rng.uniform(0, 0.3)
            while x < 8:
                if rng.random() < 0.8:
                    hx, hy, sc = self.p(x, y0 + 0.28, back_z)
                    color = rng.choice([(230, 80, 80), (80, 150, 230), (250, 200, 60), (120, 200, 120),
                                        (200, 120, 220), (250, 150, 80), (240, 240, 240)])
                    pygame.draw.rect(s, color, (hx - 0.13 * sc, hy, 0.26 * sc, 0.2 * sc), border_radius=4)
                    aa_circle(s, rng.choice([(250, 214, 176), (210, 160, 120), (150, 100, 70)]),
                              (hx, hy - 0.04 * sc), 0.09 * sc)
                x += rng.uniform(0.32, 0.45)

        # Banner along the wall
        by = self.p(0, FLOOR_Y + 0.25, back_z)[1]
        bh = self.p(0, FLOOR_Y, back_z)[1] - by
        pygame.draw.rect(s, (40, 110, 210), (0, by, WIDTH, bh))
        label = font(int(bh * 0.6)).render('LITTLE PING PONG  *  ME193  *  LITTLE PING PONG  *  ME193', True, (255, 255, 255))
        s.blit(label, label.get_rect(center=(WIDTH / 2, by + bh / 2)))

        # Wood floor with perspective boards, then the blue court mat
        floor_top = int(wall_bottom)
        for row in range(floor_top, HEIGHT):
            k = (row - floor_top) / max(1, HEIGHT - floor_top)
            s.fill([int(a + (b - a) * k) for a, b in zip(FLOOR_FAR, FLOOR_NEAR)], (0, row, WIDTH, 1))
        for bx in range(-12, 13):
            a = self.p(bx * 0.6, FLOOR_Y, back_z)
            b = self.p(bx * 0.6, FLOOR_Y, -1.9)
            pygame.draw.aaline(s, (170, 118, 66), a[:2], b[:2])
        mat = self.quad([(-3, FLOOR_Y, -1.6), (3, FLOOR_Y, -1.6), (3, FLOOR_Y, TABLE_L + 3), (-3, FLOOR_Y, TABLE_L + 3)])
        pygame.draw.polygon(s, MAT, mat)
        pygame.draw.polygon(s, (240, 240, 240), mat, 3)
        return s

    def _draw_table(self):
        s = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        hw, th = TABLE_W / 2, 0.035
        # legs
        for lx in (-hw + 0.15, hw - 0.15):
            for lz in (0.25, TABLE_L - 0.25):
                top, bottom = self.p(lx, -th, lz), self.p(lx, FLOOR_Y, lz)
                pygame.draw.line(s, LEG, top[:2], bottom[:2], max(3, int(0.04 * top[2])))
        # side faces then top
        pygame.draw.polygon(s, TABLE_SIDE, self.quad([(-hw, 0, 0), (hw, 0, 0), (hw, -th, 0), (-hw, -th, 0)]))
        for sx in (-hw, hw):
            pygame.draw.polygon(s, TABLE_SIDE, self.quad([(sx, 0, 0), (sx, 0, TABLE_L), (sx, -th, TABLE_L), (sx, -th, 0)]))
        top = self.quad([(-hw, 0, 0), (hw, 0, 0), (hw, 0, TABLE_L), (-hw, 0, TABLE_L)])
        pygame.draw.polygon(s, TABLE_TOP, top)
        # white edge lines and centre line
        e = 0.02
        for line in ([(-hw + e, 0, e), (hw - e, 0, e)], [(-hw + e, 0, TABLE_L - e), (hw - e, 0, TABLE_L - e)],
                     [(-hw + e, 0, e), (-hw + e, 0, TABLE_L - e)], [(hw - e, 0, e), (hw - e, 0, TABLE_L - e)],
                     [(0, 0, e), (0, 0, TABLE_L - e)]):
            a, b = self.p(*line[0]), self.p(*line[1])
            pygame.draw.line(s, TABLE_LINE, a[:2], b[:2], max(2, int(0.012 * (a[2] + b[2]) / 2)))
        return s

    def _draw_net(self):
        s = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        hw = TABLE_W / 2 + 0.15
        mesh = self.quad([(-hw, 0, NET_Z), (hw, 0, NET_Z), (hw, NET_H, NET_Z), (-hw, NET_H, NET_Z)])
        pygame.draw.polygon(s, NET_MESH, mesh)
        for i in range(1, 40):                          # mesh threads
            x = -hw + 2 * hw * i / 40
            a, b = self.p(x, 0, NET_Z), self.p(x, NET_H, NET_Z)
            pygame.draw.line(s, (20, 22, 30, 120), a[:2], b[:2], 1)
        a, b = self.p(-hw, NET_H, NET_Z), self.p(hw, NET_H, NET_Z)
        pygame.draw.line(s, NET_TAPE, a[:2], b[:2], max(3, int(0.02 * a[2])))
        for px in (-hw, hw):
            a, b = self.p(px, -0.03, NET_Z), self.p(px, NET_H + 0.01, NET_Z)
            pygame.draw.line(s, LEG, a[:2], b[:2], max(3, int(0.025 * a[2])))
        return s

    # ── dynamic objects ──────────────────────────────────────────────────────

    def draw_cpu(self, game):
        z = CPU_HIT_Z + 0.35
        x = game.cpu_x
        foot = self.p(x, FLOOR_Y, z)
        head = self.p(x, 0.62, z)
        sc = head[2]
        body = pygame.Rect(0, 0, 0.42 * sc, (foot[1] - head[1]) * 0.75)
        body.midtop = (head[0], head[1] + 0.12 * sc)
        pygame.draw.rect(self.screen, CPU_SHIRT, body, border_radius=int(0.15 * sc))
        aa_circle(self.screen, CPU_SKIN, head[:2], 0.13 * sc)
        for ex in (-0.045, 0.045):                     # eyes
            aa_circle(self.screen, (40, 40, 50), (head[0] + ex * sc, head[1] - 0.01 * sc), 0.016 * sc)
        pygame.draw.arc(self.screen, (40, 40, 50), (head[0] - 0.04 * sc, head[1] + 0.01 * sc, 0.08 * sc, 0.05 * sc),
                        math.pi, 2 * math.pi, 2)       # smile
        # paddle: follows the ball height when the ball is coming to the CPU
        py = min(max(game.ball.y, 0.1), 0.5) if game.heading == 'to_cpu' else 0.25
        pp = self.p(x + 0.18, py, CPU_HIT_Z)
        pygame.draw.line(self.screen, CPU_SKIN, (body.right - 0.05 * sc, body.top + 0.1 * sc), pp[:2], max(2, int(0.05 * sc)))
        aa_circle(self.screen, CPU_PADDLE, pp[:2], PADDLE_R * pp[2])

    def draw_ball(self, game):
        b = game.ball
        on_table = abs(b.x) <= TABLE_W / 2 and 0 <= b.z <= TABLE_L
        shadow_y = 0 if on_table else FLOOR_Y
        sx, sy, ssc = self.p(b.x, shadow_y, b.z)
        k = max(0.3, 1 - (b.y - shadow_y) * 0.8)        # shadow fades/grows with height
        sw, sh = 2.4 * BALL_R * ssc / k, 0.9 * BALL_R * ssc / k
        shadow = pygame.Surface((max(2, sw * 2), max(2, sh * 2)), pygame.SRCALPHA)
        pygame.draw.ellipse(shadow, (0, 0, 0, int(110 * k)), shadow.get_rect())
        self.screen.blit(shadow, shadow.get_rect(center=(sx, sy)))
        bx, by, bsc = self.p(b.x, b.y, b.z)
        r = max(5, BALL_R * bsc * 2.0)                  # drawn big so it reads at a distance
        if b.spin in SPIN_TRAIL:                        # coloured trail shows the spin
            for k, alpha in ((3, 50), (2, 90), (1, 140)):
                tx, ty, tsc = self.p(b.x - b.vx * 0.02 * k, b.y - b.vy * 0.02 * k, b.z - b.vz * 0.02 * k)
                tr = max(3, BALL_R * tsc * 2.0 * (1 - 0.15 * k))
                ghost = pygame.Surface((int(2 * tr) + 2, int(2 * tr) + 2), pygame.SRCALPHA)
                pygame.draw.circle(ghost, (*SPIN_TRAIL[b.spin], alpha), (tr + 1, tr + 1), tr)
                self.screen.blit(ghost, (tx - tr - 1, ty - tr - 1))
        aa_circle(self.screen, BALL_SHADE, (bx, by), r)
        aa_circle(self.screen, BALL, (bx - r * 0.15, by - r * 0.15), r * 0.82)

    def draw_player_paddle(self, game):
        lost = self.vision is not None and not self.vision.seen
        alpha = 110 if lost else 190                # faded while the camera can't see the paddle
        px, py = game.paddle[0], game.paddle_y      # x-only tracking: fixed height
        rubber = PADDLE_BLACK if self.vision is not None and self.vision.backhand else PADDLE_RED
        cx, cy, sc = self.p(px, py, PLAYER_HIT_Z)
        r = PADDLE_R * sc
        reach = game.level.reach_m * sc
        size = int(2 * max(reach, 2.2 * r)) + 4     # surface just big enough for paddle + ring
        s = pygame.Surface((size, size), pygame.SRCALPHA)
        c = size // 2
        pygame.gfxdraw.aacircle(s, c, c, int(reach), (255, 255, 255, 90))       # faint reach ring
        pygame.draw.line(s, (*HANDLE, 200), (c, c + r * 0.8), (c + r * 0.35, c + r * 2.0), int(r * 0.45))
        pygame.gfxdraw.filled_circle(s, c, c, int(r), (*PADDLE_EDGE, 200))
        pygame.gfxdraw.filled_circle(s, c, c, int(r * 0.9), (*rubber, alpha))
        pygame.gfxdraw.aacircle(s, c, c, int(r), (*PADDLE_EDGE, 220))
        self.screen.blit(s, (cx - c, cy - c))

    # ── HUD ──────────────────────────────────────────────────────────────────

    def draw_hud(self, game, swing_meter, input_label):
        # score panel
        panel(self.screen, pygame.Rect(20, 20, 260, 120))
        self.screen.blit(font(20).render('STREAK', True, UI_TEXT), (40, 30))
        self.screen.blit(outlined_text(str(game.streak), 56, UI_BORDER, (255, 255, 255)), (40, 52))
        self.screen.blit(font(20).render('BEST', True, UI_TEXT), (170, 30))
        self.screen.blit(outlined_text(str(game.best), 56, (255, 215, 60), UI_TEXT), (170, 52))

        # level badge
        panel(self.screen, pygame.Rect(20, 150, 260, 44), radius=22)
        stars = '*' * game.level_num + '-' * (3 - game.level_num)
        self.screen.blit(font(20).render(f'{game.level.name}  {stars}', True, UI_TEXT), (40, 160))

        self.draw_camera_box()

        # swing meter
        panel(self.screen, pygame.Rect(20, HEIGHT - 76, 260, 56), radius=16)
        self.screen.blit(font(16).render(f'SWING  ({input_label})', True, UI_TEXT), (36, HEIGHT - 70))
        bar = pygame.Rect(36, HEIGHT - 46, 228, 14)
        pygame.draw.rect(self.screen, (220, 228, 240), bar, border_radius=7)
        fill = bar.copy()
        fill.width = int(bar.width * swing_meter)
        if fill.width > 0:
            pygame.draw.rect(self.screen, (255, 120 + int(100 * (1 - swing_meter)), 40), fill, border_radius=7)

        # pop-up message
        if game.state == 'ready':
            self._center_text('Press ENTER to start', 54, (255, 255, 255), HEIGHT * 0.42)
            swing_hint = 'swing the motor to hit' if input_label == 'IMU' else 'Space / click = swing'
            move_hint = 'Move your paddle' if self.vision is not None else 'Mouse = paddle position'
            if self.vision is not None:
                self._center_text('or raise both hands', 30, (255, 255, 255), HEIGHT * 0.48, outline=3)
            self._center_text(f'{move_hint}   {swing_hint}   1-3 = level', 22, (255, 255, 255),
                              HEIGHT * 0.55, outline=2)
        elif game.paused:
            self._center_text('Paused', 64, (255, 255, 255), HEIGHT * 0.36)
            self._center_text('step back into view', 30, (255, 255, 255), HEIGHT * 0.45, outline=3)
        elif game.message:
            age = pygame.time.get_ticks() / 1000 - self.pop_time
            grow = min(1.0, 0.6 + age * 3)
            color = POP_COLORS.get(game.message.split()[0].strip('!'), (255, 90, 90))
            self._center_text(game.message, int(72 * grow), color, HEIGHT * 0.38)

    def draw_camera_box(self):
        pygame.draw.rect(self.screen, (20, 24, 32), self.cam_box, border_radius=14)
        v = self.vision
        if v is not None and v.frame is not None:
            if v.frame is not self._cam_frame:          # only convert new webcam frames
                self._cam_frame = v.frame
                h, w = v.frame.shape[:2]
                surf = pygame.image.frombuffer(v.frame.tobytes(), (w, h), 'RGB')
                if (w, h) != self.cam_box.size:
                    surf = pygame.transform.smoothscale(surf, self.cam_box.size)
                self._cam_surface = surf
            self.screen.blit(self._cam_surface, self.cam_box)
        else:
            text = 'starting camera...' if v is not None else 'camera off (use --vision)'
            t = font(18).render(text, True, (150, 160, 180))
            self.screen.blit(t, t.get_rect(center=self.cam_box.center))
        pygame.draw.rect(self.screen, UI_PANEL, self.cam_box, width=4, border_radius=14)
        if v is None:
            return

        # status chip under the webcam feed (only when the paddle is lost)
        if not v.seen:
            chip = pygame.Rect(self.cam_box.x, self.cam_box.bottom + 10, self.cam_box.width, 40)
            panel(self.screen, chip, radius=20)
            label = font(18).render('show the RED side to the camera', True, UI_TEXT)
            self.screen.blit(label, label.get_rect(center=chip.center))

    def _center_text(self, text, size, color, y, outline=4):
        surf = outlined_text(text, size, color, width=outline)
        self.screen.blit(surf, surf.get_rect(center=(WIDTH / 2, y)))

    def pop(self):
        self.pop_time = pygame.time.get_ticks() / 1000

    # ── frame ────────────────────────────────────────────────────────────────

    def draw(self, game, swing_meter, input_label):
        self.screen.blit(self.back, (0, 0))
        self.draw_cpu(game)
        self.screen.blit(self.table, (0, 0))
        ball_far = game.ball.z > NET_Z
        if game.state == 'rally' and ball_far:
            self.draw_ball(game)
        self.screen.blit(self.net, (0, 0))
        if game.state == 'rally' and not ball_far:
            self.draw_ball(game)
        self.draw_hud(game, swing_meter, input_label)
        self.draw_player_paddle(game)       # on top, so the webcam box never hides it
        self.draw_backhand_label(game)

    def note_side(self, backhand):
        '''Call every frame with the tracker's backhand flag; starts the label when it turns on.'''
        if backhand and not self._was_backhand:
            self.backhand_time = pygame.time.get_ticks() / 1000
        self._was_backhand = backhand

    def draw_backhand_label(self, game):
        '''Simple "Backhand" message above the paddle while the black side is showing.'''
        if not (self.vision is not None and self.vision.backhand):
            return
        cx, cy, sc = self.p(game.paddle[0], game.paddle_y, PLAYER_HIT_Z)
        age = pygame.time.get_ticks() / 1000 - self.backhand_time
        size = 40 if age > 0.15 else 30              # small pop when it first appears
        label = outlined_text('Backhand', size, (255, 255, 255), (20, 20, 30), 3)
        self.screen.blit(label, label.get_rect(midbottom=(cx, cy - PADDLE_R * sc - 8)))


def mouse_to_paddle(cam, mx, my):
    '''Put the paddle exactly under the mouse cursor, on the player's hit plane.'''
    x, y = cam.unproject_to_z(mx, my, PLAYER_HIT_Z)
    return max(-1.0, min(1.0, x)), max(-0.3, min(0.9, y))


def main():
    parser = argparse.ArgumentParser(description='Little Ping Pong')
    parser.add_argument('--imu', action='store_true', help='use the Double Motor for swings')
    parser.add_argument('--serial', default=None, help='Connection Card serial (optional)')
    parser.add_argument('--no-haptics', action='store_true', help='no motor pulses on hit/miss')
    parser.add_argument('--no-spin', action='store_true', help="don't classify hit types (topspin/chop)")
    parser.add_argument('--level', type=int, default=1, choices=sorted(LEVELS))
    parser.add_argument('--vision', action='store_true', help='track the red paddle with the webcam')
    parser.add_argument('--camera', default=None,
                        help="'laptop', 'usb' or a camera number (default: the one saved by find_camera.py)")
    args = parser.parse_args()

    tracker = None
    if args.vision:
        from red_tracker import RedTracker
        tracker = RedTracker(args.camera)                   # prints which camera it picked
        tracker.start()

    paddle_imu = detector = haptics = stroke_classifier = None
    if args.imu:
        from paddle_imu import PaddleIMU
        from swing import SwingDetector
        paddle_imu = PaddleIMU()
        print('Connecting to Double Motor...')
        paddle_imu.connect(card_serial=args.serial)
        print('Connected.')
        detector = SwingDetector()
        from haptics import Haptics
        haptics = Haptics(paddle_imu, enabled=not args.no_haptics)
        if not args.no_spin and os.path.exists(STROKE_MODEL):
            from strokes import LiveStrokeClassifier, load_model
            stroke_classifier = LiveStrokeClassifier(load_model(STROKE_MODEL))
            print('Hit-type detection on (topspin / chop / flat).')

    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption('Little Ping Pong')
    clock = pygame.time.Clock()
    renderer = Renderer(screen)
    game = Game(level=args.level)
    swing_meter = 0.0
    start_count = 0
    input_label = 'IMU' if paddle_imu else 'keyboard'

    try:
        running = True
        while running:
            dt = min(clock.tick(FPS) / 1000, 0.05)
            swings = []
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
                        game.start()
                    elif event.key in (pygame.K_1, pygame.K_2, pygame.K_3):
                        game.set_level(event.key - pygame.K_0)
                    elif event.key == pygame.K_SPACE and not paddle_imu:
                        swings.append(KEYBOARD_SWING_STRENGTH)
                elif event.type == pygame.MOUSEBUTTONDOWN and not paddle_imu:
                    swings.append(KEYBOARD_SWING_STRENGTH)

            if paddle_imu:
                for s in paddle_imu.drain():
                    if stroke_classifier:
                        stroke_classifier.add(s)
                    e = detector.update(s)
                    if e:
                        swings.append(e.strength)
                        if stroke_classifier:
                            stroke_classifier.on_swing(e.t)
                result = stroke_classifier.poll() if stroke_classifier else None
                if result:
                    game.apply_spin(result[0])      # re-aims the shot in flight (ignored on backhand)
            for strength in swings:
                game.swing(strength)
                swing_meter = max(swing_meter, strength, 0.15)

            if tracker:
                renderer.vision = tracker.get()
                if renderer.vision.start_count != start_count:     # MediaPipe: both hands raised
                    start_count = renderer.vision.start_count
                    if game.state == 'ready':
                        game.start()
                present = renderer.vision.present       # MediaPipe: is the player in view?
                if present is False and game.state != 'ready' and not game.paused:
                    game.pause()
                elif present and game.paused:
                    game.resume()
                renderer.note_side(renderer.vision.backhand)
                game.stroke = 'backhand' if renderer.vision.backhand else None   # hit says 'Nice backhand!'
                x = renderer.vision.paddle_x
                paddle = (game.paddle[0] if x is None else x, None)  # x-only; hold until first seen
            else:
                paddle = mouse_to_paddle(renderer.cam, *pygame.mouse.get_pos())
            game.update(dt, paddle)
            for event in game.drain_events():
                if event[0] in ('hit', 'miss'):
                    renderer.pop()
                feedback = haptics and pattern_for_event(event)
                if feedback:
                    haptics.play(*feedback)
                    # the motor shakes the IMU next to it: don't read that as a swing
                    detector.suppress_until(paddle_imu.sample_time(haptics.quiet_until()))

            swing_meter = max(0.0, swing_meter - dt * 1.2)
            renderer.draw(game, swing_meter, input_label)
            pygame.display.flip()
    finally:
        pygame.quit()
        if tracker:
            tracker.stop()
        if haptics:
            haptics.close()
        if paddle_imu:
            paddle_imu.disconnect()


if __name__ == '__main__':
    main()
