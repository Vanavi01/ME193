'''
Perspective camera for the 2.5D view: sits behind and above the player's end of
the table, tilted down, like the Wii Sports Resort table tennis camera.

yaw_deg turns the camera left (+) or right (-); the forehand drill uses it to look
across the table from the player's forehand side.
'''

import math


class Camera:

    def __init__(self, width, height, pos=(0.0, 0.85, -1.8), pitch_deg=19.0, focal=1000.0,
                 horizon_shift=0.0, yaw_deg=0.0, center_shift=0.0):
        self.w, self.h = width, height
        self.pos = pos
        self.focal = focal
        p, a = math.radians(pitch_deg), math.radians(yaw_deg)
        cp, sp, ca, sa = math.cos(p), math.sin(p), math.cos(a), math.sin(a)
        # camera axes in world space (yaw about the vertical, then pitch down)
        self._right = (ca, 0.0, sa)
        self._up = (-sa * sp, cp, ca * sp)
        self._fwd = (-sa * cp, -sp, ca * cp)
        self._cx = width / 2 + center_shift
        self._cy = height / 2 + horizon_shift

    def to_camera(self, x, y, z):
        '''World point -> camera space (right, up, forward depth).'''
        d = (x - self.pos[0], y - self.pos[1], z - self.pos[2])
        dot = lambda v: d[0] * v[0] + d[1] * v[1] + d[2] * v[2]   # noqa: E731
        return dot(self._right), dot(self._up), dot(self._fwd)

    def project(self, x, y, z):
        '''World point -> (screen x, screen y, scale). scale = pixels per metre there.'''
        right, up, depth = self.to_camera(x, y, z)
        depth = max(depth, 0.05)
        scale = self.focal / depth
        return self._cx + right * scale, self._cy - up * scale, scale

    def unproject_to_z(self, sx, sy, z):
        '''Screen point -> world (x, y) where its viewing ray meets the plane at depth z.'''
        r = (sx - self._cx) / self.focal
        u = (self._cy - sy) / self.focal
        ray = [r * self._right[i] + u * self._up[i] + self._fwd[i] for i in range(3)]
        t = (z - self.pos[2]) / ray[2]
        return self.pos[0] + ray[0] * t, self.pos[1] + ray[1] * t
