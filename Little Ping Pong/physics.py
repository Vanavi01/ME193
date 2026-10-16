'''
Ball physics for the virtual table (no pygame here, so it can be unit tested).

World coordinates, in metres:
    x  left/right, 0 = table centre line
    y  up, 0 = table surface
    z  away from the player, 0 = player's end of the table, TABLE_L = CPU's end
'''

import math
from dataclasses import dataclass

TABLE_L = 2.74
TABLE_W = 1.525
NET_H = 0.1525
NET_Z = TABLE_L / 2
TABLE_H = 0.76          # table top above the floor
BALL_R = 0.02
GRAVITY = 9.81
RESTITUTION = 0.88      # fraction of vertical speed kept on a bounce
NET_MARGIN = 0.06       # aimed shots clear the net by at least this much

PLAYER_HIT_Z = -0.25    # where the player meets the ball, just behind their end
CPU_HIT_Z = TABLE_L + 0.25


@dataclass
class Ball:
    x: float
    y: float
    z: float
    vx: float = 0.0
    vy: float = 0.0
    vz: float = 0.0
    g: float = GRAVITY          # effective gravity: spin adds a vertical force (topspin dips, backspin floats)
    bounce_vz: float = 1.0      # forward speed kept on a bounce (topspin kicks on, backspin checks up)
    spin: str = None            # 'topspin' / 'backspin' / None, for display

    def copy(self):
        return Ball(self.x, self.y, self.z, self.vx, self.vy, self.vz, self.g, self.bounce_vz, self.spin)

    def set_spin(self, spin=None, g=GRAVITY, bounce_vz=1.0):
        self.spin, self.g, self.bounce_vz = spin, g, bounce_vz


def on_table(x, z):
    return abs(x) <= TABLE_W / 2 and 0 <= z <= TABLE_L


def step(ball, dt):
    '''Advance the ball by dt seconds. Returns 'bounce' if it bounced on the table.'''
    ball.vy -= ball.g * dt
    ball.x += ball.vx * dt
    ball.y += ball.vy * dt
    ball.z += ball.vz * dt
    if ball.y <= BALL_R and ball.vy < 0 and on_table(ball.x, ball.z):
        ball.y = BALL_R
        ball.vy = -RESTITUTION * ball.vy
        ball.vx *= ball.bounce_vz
        ball.vz *= ball.bounce_vz
        return 'bounce'
    return None


def launch_velocity(start, target_x, target_z, flight_time, g=GRAVITY):
    '''Velocity that takes a ball from start (x, y, z) to land on the table at
    (target_x, target_z) after flight_time seconds (under effective gravity g).'''
    x0, y0, z0 = start
    t = flight_time
    vx = (target_x - x0) / t
    vz = (target_z - z0) / t
    vy = (BALL_R - y0 + 0.5 * g * t * t) / t
    return vx, vy, vz


def height_at_net(start, velocity, g=GRAVITY):
    x0, y0, z0 = start
    vx, vy, vz = velocity
    t = (NET_Z - z0) / vz
    return y0 + vy * t - 0.5 * g * t * t


def aim(start, target_x, target_z, flight_time, g=GRAVITY):
    '''Like launch_velocity, but stretches the flight time (a loftier shot) until
    the ball clears the net. A ball already past the net needs no check.'''
    t = flight_time
    going_away = target_z > start[2]
    past_net = start[2] >= NET_Z if going_away else start[2] <= NET_Z
    while True:
        v = launch_velocity(start, target_x, target_z, t, g)
        if past_net or height_at_net(start, v, g) >= NET_H + NET_MARGIN or t > 3:
            return v
        t *= 1.05


def predict_crossing(ball, plane_z, dt=1 / 240, max_time=4.0):
    '''Simulate a copy of the ball until it crosses z = plane_z.
    Returns (seconds until crossing, Ball at crossing) or None if it never does.'''
    b = ball.copy()
    elapsed = 0.0
    going_up = plane_z > b.z
    while elapsed < max_time:
        step(b, dt)
        elapsed += dt
        if (b.z >= plane_z) if going_up else (b.z <= plane_z):
            return elapsed, b
        if b.y < -TABLE_H:
            return None
    return None
