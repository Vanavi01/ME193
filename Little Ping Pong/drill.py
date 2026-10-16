'''
Forehand drill: smash / chop practice (separate mini-game; pure logic, unit tested).

A ball machine at the far end feeds balls to the player's forehand side, one at a
time, and calls the stroke to play (mixed evenly):
    SMASH  -> a high, bouncy ball. Play a hard forehand (flat or topspin, strength >= SMASH_MIN).
    HIT    -> a normal mid-height rally ball. Play a normal forehand (flat or topspin, any strength).
    CHOP   -> a low, short, slow ball. Play your chop (backspin).
The hit itself uses the same rule as the main game (paddle x within reach when the
ball arrives + a swing within HIT_WINDOW_S). The stroke type comes from the IMU
classifier ~0.1 s after the swing (or from the keyboard when testing).
Right stroke: points (+ streak bonus). Wrong stroke: no points, streak resets.
Miss: streak resets. A round is ROUND_BALLS balls, then a results screen.

The level (picked with an AprilTag, or 1/2/3) sets the pace (DRILL_PACE): how fast every
ball flies, and how long the machine waits after one ball before feeding the next.
'''

import math
import random
from dataclasses import dataclass

from game import HIT_WINDOW_S, LEVELS, SPINS, X_ONLY_PADDLE_Y
from physics import (BALL_R, CPU_HIT_Z, GRAVITY, PLAYER_HIT_Z, TABLE_H, TABLE_L,
                     Ball, aim, step)

CALLS = ('smash', 'hit', 'chop')
ROUND_BALLS = 20
SMASH_MIN = 0.5             # swing strength needed for a smash
JUDGE_TIMEOUT_S = 0.35      # no stroke type by then -> judged as a flat hit
EARLY_STROKE_S = 0.4        # a stroke type arriving this long before the hit still counts
FEED_X = 0.45               # ball machine position (x, forehand side)
FOREHAND_X_RANGE = (0.0, 0.75)  # webcam play area maps onto just this (balls land at ~0.16..0.58):
                                # 2.4x finer than mapping it onto the whole table (-0.9..0.9)
POINTS, STREAK_BONUS = 100, 20
PHYSICS_DT = 1 / 240

# Balls the machine feeds for each call (game time): (target z range, flight time, spin)
FEEDS = {
    'smash': ((0.55, 0.90), 0.80, None),          # lofted, lands mid-table, bounces up high
    'hit':   ((0.45, 0.85), 0.60, None),          # a normal rally ball, like the CPU's returns
    'chop':  ((0.25, 0.50), 0.60, 'backspin'),    # low, short, floating, checks up after the bounce
}
STROKE_NAMES = {'flat': 'a flat hit', 'topspin': 'a topspin', 'backspin': 'a chop'}


@dataclass(frozen=True)
class Pace:
    speed: float            # ball speed: game-time multiplier for every ball (feeds and returns)
    gap_s: float            # real seconds between one ball finishing and the next feed


DRILL_PACE = {
    1: Pace(0.75, 1.8),     # Normal: slow balls, plenty of time to reset
    2: Pace(1.00, 1.0),     # Hard: real-speed balls, a quick reset
    3: Pace(1.30, 0.5),     # Pro: fast balls, barely time to recover
}


def tracking_x_range(hand='right'):
    '''Paddle x range the webcam play area maps onto, mirrored for left-handers.'''
    lo, hi = FOREHAND_X_RANGE
    return (lo, hi) if hand != 'left' else (-hi, -lo)


def is_correct(call, kind, strength):
    if call == 'smash':
        return kind in ('flat', 'topspin') and strength >= SMASH_MIN
    if call == 'hit':
        return kind in ('flat', 'topspin')
    return kind == 'backspin'


class Drill:

    def __init__(self, level=1, hand='right', balls=ROUND_BALLS, rng=None):
        self.rng = rng or random.Random()
        self.level_num = level
        self.sign = -1 if hand == 'left' else 1     # forehand side of the table
        self.balls = balls
        self.state = 'ready'        # ready -> rally <-> gap -> done
        self.heading = None         # 'to_player' (incoming) / 'away' (after the hit)
        self.ball = Ball(FEED_X * self.sign, 0.35, CPU_HIT_Z)
        self.paddle = (0.0, None)   # x-only
        self.call = None
        self.message = ''
        self.events = []
        self.time = 0.0
        self.paused = False
        self.reset_scores()

    def reset_scores(self):
        self.score = self.streak = self.best = 0
        self.fed = 0
        self.results = []           # (call, outcome) outcome: 'correct' / 'wrong' / 'miss'
        self._next_feed_at = 0.0
        self._crossing = None
        self._last_swing_t = -math.inf
        self._last_swing_strength = 0.0
        self._early_stroke = None
        self._hit = None            # {'t', 'strength', 'judged', 'flight_t'}

    # ── properties the renderer shares with the main game ────────────────────

    @property
    def level(self):
        return LEVELS[self.level_num]

    @property
    def pace(self):
        return DRILL_PACE[self.level_num]

    @property
    def speed(self):
        return self.pace.speed

    @property
    def paddle_y(self):
        return X_ONLY_PADDLE_Y

    def accuracy(self, call=None):
        rs = [o for c, o in self.results if call is None or c == call]
        return (sum(o == 'correct' for o in rs) / len(rs)) if rs else 0.0

    # ── inputs ───────────────────────────────────────────────────────────────

    def set_level(self, level):
        self.level_num = level

    def start(self):
        self.reset_scores()
        self.events.clear()
        self._feed()

    def pause(self):
        self.paused = True

    def resume(self):
        self.paused = False

    def swing(self, strength):
        if self.paused:
            return
        self._last_swing_t = self.time
        self._last_swing_strength = strength

    def set_stroke(self, kind):
        '''Stroke type for the latest swing (IMU classifier or keyboard).'''
        if self.paused or kind not in STROKE_NAMES:
            return
        hit = self._hit
        if hit and not hit['judged'] and self.time - hit['t'] <= JUDGE_TIMEOUT_S:
            self._judge(kind)
        elif self.heading == 'to_player':
            self._early_stroke = (kind, self.time)

    def drain_events(self):
        events, self.events = self.events, []
        return events

    # ── simulation ───────────────────────────────────────────────────────────

    def update(self, dt, paddle_x):
        if self.paused or self.state in ('ready', 'done'):
            return
        self.time += dt
        self.paddle = (paddle_x, None)
        if self.state == 'gap':
            if self.time >= self._next_feed_at:
                self._feed()
            return

        game_dt = dt * self.speed
        n = max(1, math.ceil(game_dt / PHYSICS_DT))
        for _ in range(n):
            step(self.ball, game_dt / n)

        b = self.ball
        if self.heading == 'to_player':
            self._check_hit()
        elif self.heading == 'away':
            hit = self._hit
            if not hit['judged'] and self.time - hit['t'] > JUDGE_TIMEOUT_S:
                self._judge('flat')                     # no stroke type arrived
            hit['flight_t'] += game_dt
            if b.z > TABLE_L + 0.4 or b.y < -TABLE_H or hit['flight_t'] > 2.5:
                self._end_ball()
        if self.state == 'rally' and self.heading == 'to_player' and b.y < -TABLE_H:
            self._miss('Missed!')

    def _feed(self):
        self.state = 'rally'
        self.heading = 'to_player'
        self.fed += 1
        self.call = self.rng.choice(CALLS)
        (z_lo, z_hi), flight, spin = FEEDS[self.call]
        s = self.sign
        start = (FEED_X * s, 0.35, CPU_HIT_Z)
        target_x = s * self.rng.uniform(0.25, 0.55)
        self.ball = Ball(*start)
        if spin:
            sp = SPINS[spin]
            self.ball.set_spin(spin, GRAVITY * sp.gravity, sp.bounce)
        self.ball.vx, self.ball.vy, self.ball.vz = aim(start, target_x, self.rng.uniform(z_lo, z_hi),
                                                       flight, self.ball.g)
        self._crossing = None
        self._hit = None
        self.message = ''
        self.events.append(('feed', self.call))

    def _check_hit(self):
        b = self.ball
        if self._crossing is None and b.z <= PLAYER_HIT_Z:
            self._crossing = (self.time, abs(b.x - self.paddle[0]))
        if self._crossing is None:
            return
        t_cross, dist = self._crossing
        if dist > self.level.reach_m:
            self._miss('Too far!')
        elif abs(self._last_swing_t - t_cross) <= HIT_WINDOW_S:
            self._player_hit()
        elif self.time > t_cross + HIT_WINDOW_S:
            self._miss('No swing!')

    def _player_hit(self):
        s = self._last_swing_strength
        self._last_swing_t = -math.inf
        self.heading = 'away'
        self._hit = {'t': self.time, 'strength': s, 'judged': False, 'flight_t': 0.0}
        self._launch('flat')                                    # re-aimed once judged
        self.events.append(('hit', s, None))
        early = self._early_stroke
        self._early_stroke = None
        if early and self.time - early[1] <= EARLY_STROKE_S:
            self._judge(early[0])

    def _launch(self, kind):
        '''Send the ball back over the net with this stroke's flight (re-aims from where it is).'''
        b, sp = self.ball, SPINS[kind]
        strength = self._hit['strength']
        b.set_spin(None if kind == 'flat' else kind, GRAVITY * sp.gravity, sp.bounce)
        flight = (0.55 - 0.2 * strength) * sp.flight
        target = (-self.sign * self.rng.uniform(0.0, 0.5), min(TABLE_L - 0.1, TABLE_L - 0.6 + sp.depth_m))
        b.vx, b.vy, b.vz = aim((b.x, max(b.y, BALL_R), b.z), *target, flight, b.g)

    def _judge(self, kind):
        hit = self._hit
        hit['judged'] = True
        self._launch(kind)
        if is_correct(self.call, kind, hit['strength']):
            points = POINTS + STREAK_BONUS * self.streak
            self.score += points
            self.streak += 1
            self.best = max(self.best, self.streak)
            self.results.append((self.call, 'correct'))
            self.message = {'smash': 'Great smash!', 'hit': 'Good hit!', 'chop': 'Great chop!'}[self.call]
            self.events.append(('judged', self.call, kind, True, points))
            if self.streak == self.best:
                self.events.append(('new_best', self.best))
        else:
            self.streak = 0
            self.results.append((self.call, 'wrong'))
            weak_smash = self.call == 'smash' and kind in ('flat', 'topspin')
            self.message = 'Too soft!' if weak_smash else f'That was {STROKE_NAMES[kind]}!'
            self.events.append(('judged', self.call, kind, False, 0))

    def _miss(self, reason):
        self.streak = 0
        self.results.append((self.call, 'miss'))
        self.message = reason
        self.events.append(('miss', reason))
        self._end_ball()

    def _end_ball(self):
        self.heading = None
        self._crossing = None
        if self.fed >= self.balls:
            self.state = 'done'
            self.events.append(('round_over',))
        else:
            self.state = 'gap'
            self._next_feed_at = self.time + self.pace.gap_s
