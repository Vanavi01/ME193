'''
Rally logic: CPU serves and returns, the player hits, the streak is counted.
No pygame here, so it can be unit tested. play.py draws it.

The player hits the ball when BOTH are true (the plan's hit detection rule):
  1. location: when the ball reaches the player's hit plane, the paddle is
     within the level's reach of it. With x-only tracking (paddle y = None)
     only the left/right distance counts and the paddle meets the ball's height.
  2. swing:    a swing happens within HIT_WINDOW_S of that moment (before or after,
     because the IMU's Bluetooth delay can make a swing arrive a little late)

Score: `streak` is the current run of consecutive hits, `best` the record.
Pace: each hit in a rally speeds the ball up a little (RALLY_SPEEDUP), so long
rallies get faster; a miss resets it.
'''

import math
import random
from dataclasses import dataclass

from physics import (BALL_R, CPU_HIT_Z, GRAVITY, NET_Z, PLAYER_HIT_Z, TABLE_H,
                     TABLE_L, Ball, aim, predict_crossing, step)

HIT_WINDOW_S = 0.2
POINT_PAUSE_S = 0.9         # pause after a miss before the CPU serves again
PHYSICS_DT = 1 / 240
CPU_PADDLE_SPEED = 3.0      # m/s, for the CPU paddle animation


@dataclass(frozen=True)
class Level:
    name: str
    time_scale: float       # game speed: shots use real physics, slowed down for easier levels
    reach_m: float          # how close the paddle must be to the ball


LEVELS = {
    1: Level('Normal', 0.75, 0.30),
    2: Level('Hard', 0.95, 0.25),
    3: Level('Pro', 1.15, 0.20),
}

CPU_FLIGHT_S = 0.55         # CPU shot flight time (game time) -> a natural ~0.35 m arc
RALLY_SPEEDUP = 0.03        # +3% ball speed per consecutive hit...
RALLY_SPEEDUP_MAX = 0.30    # ...up to +30%
X_ONLY_PADDLE_Y = 0.38      # x-only tracking: paddle drawn at this fixed height (typical ball height)


@dataclass(frozen=True)
class Spin:
    flight: float           # x flight time: <1 faster, >1 slower
    gravity: float          # x gravity: >1 dips (topspin), <1 floats (backspin)
    depth_m: float          # landing point moved deeper (+) or shorter (-)
    bounce: float           # forward speed kept on the bounce: >1 kicks on, <1 checks up
    message: str


# Hit types from the IMU stroke classifier (forehand only). Topspin: lower arc, gets
# to the CPU faster. Chop (backspin): similar arc height, lands shorter, slower.
SPINS = {
    'flat':     Spin(1.00, 1.00, 0.00, 1.00, ''),
    'topspin':  Spin(0.75, 1.40, +0.15, 1.15, 'Topspin!'),
    'backspin': Spin(1.25, 0.65, -0.45, 0.60, 'Chop!'),
}
SPIN_LATE_S = 0.35          # a classification arriving later than this after the hit is ignored
SPIN_EARLY_S = 0.40         # one arriving up to this long BEFORE the hit is used by it
CPU_TAKES_LOW_Y = 0.12      # CPU also returns a ball that bounced on its side and drops below this


class Game:

    def __init__(self, level=1, rng=None):
        self.rng = rng or random.Random()
        self.level_num = level
        self.time = 0.0
        self.state = 'ready'        # 'ready' -> 'rally' <-> 'point'
        self.heading = None         # 'to_player' or 'to_cpu' while in a rally
        self.ball = Ball(0, 0.3, CPU_HIT_Z)
        self.paddle = (0.0, 0.25)   # player paddle (x, y) in the hit plane, metres; y None = x-only
        self.stroke = None          # 'forehand' / 'backhand' from vision (None without a camera)
        self.cpu_x = 0.0
        self._cpu_target_x = 0.0
        self.streak = 0
        self.best = 0
        self.message = ''
        self.events = []            # ('hit', strength, stroke) ('spin', type) ('miss', reason) ('cpu_hit',) ('new_best', n)
        self._last_swing_t = -math.inf
        self._last_swing_strength = 0.0
        self._crossing = None       # (time, paddle distance) when the ball reached the player
        self._resume_at = 0.0
        self.paused = False         # player stepped out of view (MediaPipe presence)
        self.game_time = 0.0        # ball-clock time (real time x speed)
        self._shot = None           # the player's shot in flight: target and timing, for spin re-aim
        self._next_spin = None      # (type, time) classified before the hit landed
        self._cpu_bounced = False

    @property
    def level(self):
        return LEVELS[self.level_num]

    @property
    def speed(self):
        '''Current game-time multiplier: the level's speed, raised as the rally goes on.'''
        return self.level.time_scale * (1 + min(RALLY_SPEEDUP * self.streak, RALLY_SPEEDUP_MAX))

    @property
    def paddle_y(self):
        '''Paddle height to draw: the tracked height, or a fixed height in x-only mode
        (following each shot's height made the paddle jump on every CPU hit).'''
        return X_ONLY_PADDLE_Y if self.paddle[1] is None else self.paddle[1]

    # ── inputs ───────────────────────────────────────────────────────────────

    def set_level(self, level):
        self.level_num = level

    def start(self):
        '''Start (or restart) a game: resets the streak and the CPU serves.'''
        self.streak = 0
        self.events.clear()
        self._serve()

    def pause(self):
        '''Freeze the rally (ball, timers, swings) until resume().'''
        self.paused = True

    def resume(self):
        self.paused = False

    def swing(self, strength):
        '''The player swung (from the IMU, or the keyboard/mouse fallback).'''
        if self.paused:
            return
        self._last_swing_t = self.time
        self._last_swing_strength = strength

    def apply_spin(self, kind):
        '''Hit type from the IMU classifier (arrives ~0.1 s after the swing peak).
        Re-aims the shot already in flight, or is kept for a hit that hasn't happened yet.'''
        if kind not in SPINS or self.paused or self.stroke == 'backhand':
            return                  # the classifier only knows forehands
        shot = self._shot
        if self.heading == 'to_cpu' and shot and shot['spin'] is None and self.time - shot['t'] <= SPIN_LATE_S:
            self._launch(kind)
        elif self.heading == 'to_player':
            self._next_spin = (kind, self.time)

    def drain_events(self):
        events, self.events = self.events, []
        return events

    # ── simulation ───────────────────────────────────────────────────────────

    def update(self, dt, paddle):
        '''Advance dt real seconds. paddle = (x, y) of the player's paddle in metres.'''
        if self.paused:
            return                  # time stands still, so the swing window and timers stay valid
        self.time += dt
        self.paddle = paddle
        self._move_cpu_paddle(dt * self.speed)

        if self.state == 'point':
            if self.time >= self._resume_at:
                self._serve()
            return
        if self.state != 'rally':
            return

        # self.time (used for the swing window) is real time; the ball moves in game time
        game_dt = dt * self.speed
        self.game_time += game_dt
        n = max(1, math.ceil(game_dt / PHYSICS_DT))
        for _ in range(n):
            if step(self.ball, game_dt / n) == 'bounce' and self.heading == 'to_cpu' and self.ball.z > NET_Z:
                self._cpu_bounced = True

        b = self.ball
        if self.heading == 'to_player':
            self._check_player_hit()
        elif b.z >= CPU_HIT_Z or (self._cpu_bounced and b.vy < 0 and b.y < CPU_TAKES_LOW_Y):
            self._cpu_return()              # (the second case: a short, slow chop taken on the drop)

        if self.state == 'rally' and self.ball.y < -TABLE_H:
            self._miss('Missed!')

    def _check_player_hit(self):
        if self._crossing is None and self.ball.z <= PLAYER_HIT_Z:
            px, py = self.paddle
            dy = 0.0 if py is None else self.ball.y - py
            self._crossing = (self.time, math.hypot(self.ball.x - px, dy))
        if self._crossing is None:
            return
        t_cross, dist = self._crossing
        if dist > self.level.reach_m:
            self._miss('Out of reach!')
        elif abs(self._last_swing_t - t_cross) <= HIT_WINDOW_S:
            self._player_hit()
        elif self.time > t_cross + HIT_WINDOW_S:
            self._miss('No swing!')

    def _player_hit(self):
        s = self._last_swing_strength
        b = self.ball
        px, _ = self.paddle
        # Where the ball met the paddle steers it; a harder swing goes deeper and faster.
        self._shot = {'t': self.time, 'gt': self.game_time, 'spin': None,
                      'x': max(-0.6, min(0.6, (b.x - px) * 2.5 + self.rng.uniform(-0.1, 0.1))),
                      'z': TABLE_L - 0.75 + 0.5 * s, 'flight': 0.65 - 0.25 * s}
        self.heading = 'to_cpu'
        self._cpu_bounced = False
        early = self._next_spin
        self._next_spin = None
        self._crossing = None
        self._last_swing_t = -math.inf      # one swing, one hit
        self.streak += 1
        self.events.append(('hit', s, self.stroke))
        if self.streak > self.best:
            self.best = self.streak
            self.events.append(('new_best', self.best))
        if s > 0.75:
            self.message = 'Smash!'
        else:
            self.message = f'Nice {self.stroke}!' if self.stroke else 'Nice!'
        if early and self.time - early[1] <= SPIN_EARLY_S and self.stroke != 'backhand':
            self._launch(early[0])                      # classified before the ball arrived
        else:
            self._launch('flat', provisional=True)      # spin may still arrive and re-aim it

    def _launch(self, kind, provisional=False):
        '''(Re-)aim the player's shot from where the ball is now, with the given hit type.'''
        spin, shot, b = SPINS[kind], self._shot, self.ball
        b.set_spin(None if kind == 'flat' else kind, GRAVITY * spin.gravity, spin.bounce)
        remaining = max(0.12, shot['flight'] * spin.flight - (self.game_time - shot['gt']))
        target_z = min(TABLE_L - 0.1, shot['z'] + spin.depth_m)
        b.vx, b.vy, b.vz = aim((b.x, max(b.y, BALL_R), b.z), shot['x'], target_z, remaining, b.g)
        if not provisional:
            shot['spin'] = kind
            if spin.message:
                self.message = spin.message
                self.events.append(('spin', kind))
        crossing = predict_crossing(b, CPU_HIT_Z)
        self._cpu_target_x = crossing[1].x if crossing else self._cpu_target_x

    def _cpu_return(self):
        b = self.ball
        start = (b.x, max(b.y, 0.15), min(b.z, CPU_HIT_Z))
        target_x = self.rng.uniform(-0.55, 0.55)
        target_z = self.rng.uniform(0.35, 1.0)
        b.x, b.y, b.z = start
        b.set_spin()                                    # the CPU hits it back flat
        b.vx, b.vy, b.vz = aim(start, target_x, target_z, CPU_FLIGHT_S)
        self._shot = None
        self._cpu_bounced = False
        self.heading = 'to_player'
        self._crossing = None
        self.events.append(('cpu_hit',))

    def _serve(self):
        self.state = 'rally'
        self.message = ''
        self.cpu_x = self._cpu_target_x = self.rng.uniform(-0.3, 0.3)
        self.ball = Ball(self.cpu_x, 0.3, CPU_HIT_Z)
        self._cpu_return()

    def _miss(self, reason):
        self.state = 'point'
        self.heading = None
        self.message = reason
        self.streak = 0
        self._crossing = None
        self._next_spin = None
        self._resume_at = self.time + POINT_PAUSE_S
        self.events.append(('miss', reason))

    def _move_cpu_paddle(self, dt):
        step_m = CPU_PADDLE_SPEED * dt
        self.cpu_x += max(-step_m, min(step_m, self._cpu_target_x - self.cpu_x))
