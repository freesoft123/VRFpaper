# -*- coding: utf-8 -*-
"""Three-dimensional hexagonal-frustum geometry and opening utilities for Appendix B.

Geometry: an irregular hexagonal base and a top face scaled by 0.65 and
translated to form a convex frustum with eight faces (two hexagonal faces and
six trapezoidal lateral faces).
Opening: a square aperture of side length w is centered at the midpoint of the
largest-area lateral face. Enclosure walls are sampled as obstacle points at
0.5 m intervals.
"""
import numpy as np

from methods3d import unit

SPACING = 0.5
R_SAFE = 0.5
BASE = [(0.0, 0.0), (14.0, -1.5), (21.0, 5.0), (17.0, 14.0),
        (5.0, 16.0), (-3.0, 7.0)]
SCALE = 0.65
SHIFT = np.array([1.5, 0.5, 0.0])
HEIGHT = 12.0


class Face(object):
    def __init__(self, verts, centre_of_solid):
        self.v = [np.array(p, float) for p in verts]
        n = np.zeros(3)
        for i in range(len(self.v)):
            a, b = self.v[i], self.v[(i + 1) % len(self.v)]
            n[0] += (a[1] - b[1]) * (a[2] + b[2])
            n[1] += (a[2] - b[2]) * (a[0] + b[0])
            n[2] += (a[0] - b[0]) * (a[1] + b[1])
        n = unit(n)
        if np.dot(n, self.v[0] - centre_of_solid) < 0:
            n = -n
            self.v = self.v[::-1]
        self.n = n
        self.o = self.v[0]
        e1 = self.v[1] - self.v[0]
        self.u = unit(e1)
        self.w = unit(np.cross(self.n, self.u))
        self.centre = np.mean(self.v, axis=0)
        self.area = 0.5 * sum(abs(np.linalg.norm(np.cross(self.v[i] - self.v[0],
                                                        self.v[(i + 1) % len(self.v)] - self.v[0])))
                              for i in range(1, len(self.v) - 1))

    def local(self, p):
        d = np.asarray(p, float) - self.o
        return np.array([np.dot(d, self.u), np.dot(d, self.w), np.dot(d, self.n)])

    def inside_poly(self, uv):
        c = False
        n = len(self.v)
        for i in range(n):
            a = self.local(self.v[i])[:2]
            b = self.local(self.v[(i + 1) % n])[:2]
            if (a[1] > uv[1]) != (b[1] > uv[1]):
                if uv[0] < a[0] + (uv[1] - a[1]) * (b[0] - a[0]) / (b[1] - a[1]):
                    c = not c
        return c

    def dist(self, p):
        lp = self.local(p)
        if self.inside_poly(lp[:2]):
            return abs(lp[2])
        best = np.inf
        n = len(self.v)
        for i in range(n):
            a, b = self.v[i], self.v[(i + 1) % n]
            ab = b - a
            t = np.clip(np.dot(p - a, ab) / max(np.dot(ab, ab), 1e-12), 0, 1)
            best = min(best, float(np.linalg.norm(p - (a + t * ab))))
        return best

    def dist_hole(self, p, half, c):
        """Distance to the rim when the projection lies inside the square opening."""
        lp = self.local(p)
        if self.inside_poly(lp[:2]):
            du, dv = abs(lp[0] - c[0]), abs(lp[1] - c[1])
            if du <= half and dv <= half:
                rim = min(half - du, half - dv)
                return float(np.hypot(rim, lp[2]))
            return abs(lp[2])
        best = np.inf
        n = len(self.v)
        for i in range(n):
            a, b = self.v[i], self.v[(i + 1) % n]
            ab = b - a
            t = np.clip(np.dot(p - a, ab) / max(np.dot(ab, ab), 1e-12), 0, 1)
            best = min(best, float(np.linalg.norm(p - (a + t * ab))))
        return best

    def sample(self, skip_square=None):
        pts = []
        uv = np.array([self.local(v)[:2] for v in self.v])
        lo, hi = uv.min(axis=0), uv.max(axis=0)
        xs = np.arange(lo[0], hi[0] + 1e-9, SPACING)
        ys = np.arange(lo[1], hi[1] + 1e-9, SPACING)
        for x in xs:
            for y in ys:
                if not self.inside_poly(np.array([x, y])):
                    continue
                if skip_square is not None:
                    half, c = skip_square
                    if abs(x - c[0]) <= half and abs(y - c[1]) <= half:
                        continue
                pts.append(self.o + x * self.u + y * self.w)
        return pts


def build_polyhedron():
    c = np.mean(np.array(BASE), axis=0)
    P = [np.array([x, y, 0.0]) for x, y in BASE]
    Q = [np.array([c[0], c[1], 0.0]) + SCALE * (np.array([x, y, 0.0]) - np.array([c[0], c[1], 0.0]))
         + SHIFT + np.array([0, 0, HEIGHT]) for x, y in BASE]
    solid_centre = np.mean(np.array(P + Q), axis=0)
    faces = [Face(P[::-1], solid_centre), Face(Q, solid_centre)]
    for i in range(len(BASE)):
        j = (i + 1) % len(BASE)
        faces.append(Face([P[i], P[j], Q[j], Q[i]], solid_centre))
    return faces


def clearance(p, faces, side=None, half=None, c=None):
    best = np.inf
    for f in faces:
        if f is side and half is not None:
            best = min(best, f.dist_hole(p, half, c))
        else:
            best = min(best, f.dist(p))
    return best


def cross_opening(prev, cur, side, c, gap):
    d0 = float(np.dot(prev - side.o, side.n))
    d1 = float(np.dot(cur - side.o, side.n))
    if not (d0 < 0 <= d1) or d1 == d0:
        return None
    t = -d0 / (d1 - d0)
    hit = prev + t * (cur - prev)
    uv = side.local(hit)[:2]
    if abs(uv[0] - c[0]) <= gap / 2.0 and abs(uv[1] - c[1]) <= gap / 2.0:
        return hit
    return None
