# -*- coding: utf-8 -*-
"""Two-dimensional kinematics and escape planners used in Appendices A and B.

The kinematic model follows the simplified bicycle model used by BabyRobot
(pathplanner.py) in reference [11] (Kilic et al.):
  state: pos, theta (body heading), v (scalar speed), psi (velocity direction),
         and wheel_angle (front steering angle);
  constraints: speed_max/min, acc_max/brake_max, wheel_angle_max, and wheel_acc_max;
  update: v += acc_engine*dt (saturated); psi += (v/L)*tan(delta)*dt;
          pos += v*[cos(psi), sin(psi)]*dt.
The resultant force F is interpreted as the desired acceleration direction,
a = F/|F|.

Experimental parameters: robot dimension L = 0.1 m (wheelbase), dt = 0.1 s,
speed_max = 1.0 m/s, speed_min = -1.0, acc_max = brake_max = 1.0,
wheel_angle_max = pi/4, and wheel_acc_max = 2pi. A collision is registered
when the minimum distance from the robot center to a wall is below 0.5 m.

This module also provides the stepwise force routines for SOPC-APF, RV-APF,
DIT-APF, and VRF-APF, avoiding separate modules for individual 2D planners.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

R_SAFE = 0.5
DEFAULT = dict(
    ks=15.0,
    alpha=1.1,
    M=2.0,
    M_inner=0.5,
    rho_o=2.5,
    Rp=4.0,
    rs=0.5,
    L=0.1,
    R=0.5,
    goal_tol=0.2,
    max_steps=3000,
    rot_step_deg=3.0,
    fr2_dir="-target",
    att_target="current",
    recompute_vgoal=True,
    ra_target="real",
)
DEFAULT9 = dict(
    p0=1.5,
    gamma=2.02,
    eps=0.1,
    delta=0.1,
    N=200,
    vortex=True,
    use_waypoint=True,
    wp_radius=0.3,
    goal_tol=0.3,
)


# ---------------- Geometry and common utilities ----------------
def u_enclosure(length=20.0, width=10.0, gap_width=None, closed_side="far",
                spacing=0.5):
    """Build the rectangular U-enclosure and sampled wall obstacles."""
    segments = [
        [(0.0, 0.0), (length, 0.0)],
        [(0.0, width), (length, width)],
        [(0.0, 0.0), (0.0, width)],
    ]
    if gap_width is not None and gap_width < width:
        wall_length = width - gap_width
        if closed_side == "far":
            segments.append([(length, width), (length, width - wall_length)])
        else:
            segments.append([(length, 0.0), (length, wall_length)])

    sampled = []
    for start, end in segments:
        sampled.extend(_sample_segment(np.asarray(start, float),
                                       np.asarray(end, float), spacing))
    obstacles = []
    for point in sampled:
        if not any(np.allclose(point, other, atol=1e-9) for other in obstacles):
            obstacles.append(point)
    return segments, np.array(obstacles)


def dist_point_segments(point, segments):
    best = np.inf
    for start, end in segments:
        start = np.asarray(start, float)
        end = np.asarray(end, float)
        edge = end - start
        denominator = float(np.dot(edge, edge))
        fraction = 0.0 if denominator == 0 else float(
            np.clip(np.dot(point - start, edge) / denominator, 0.0, 1.0)
        )
        best = min(best, float(np.linalg.norm(point - (start + fraction * edge))))
    return best


def dist_point_line(point, start, end):
    edge = end - start
    norm = np.linalg.norm(edge)
    if norm == 0:
        return np.linalg.norm(point - start)
    return abs(edge[0] * (point[1] - start[1])
               - edge[1] * (point[0] - start[0])) / norm


def unit(vector):
    norm = np.linalg.norm(vector)
    return vector / norm if norm > 1.0e-12 else np.zeros_like(vector)


# ---------------- SOPC-APF ----------------
def collision_prediction(position, goal, obstacles, config):
    """Select obstacles inside the prediction range and target safety corridor."""
    delta = obstacles - position
    distances = np.linalg.norm(delta, axis=1)
    selected = distances <= config["Rp"]
    if not np.any(selected):
        return []
    direction = goal - position
    norm = np.linalg.norm(direction)
    if norm < 1.0e-12:
        return []
    line_distance = np.abs(
        direction[0] * delta[:, 1] - direction[1] * delta[:, 0]
    ) / norm
    selected &= line_distance <= config["rs"]
    return [
        (obstacles[index], float(distances[index]))
        for index in np.where(selected)[0]
    ]


def deflection_angle(position, goal, restricted, config):
    goal_direction = goal - position
    goal_direction = goal_direction / max(np.linalg.norm(goal_direction), 1.0e-12)

    signed = 0.0
    for obstacle, distance in restricted:
        obstacle_direction = (obstacle - position) / max(distance, 1.0e-12)
        signed += (
            goal_direction[0] * obstacle_direction[1]
            - goal_direction[1] * obstacle_direction[0]
        )
    sign = -1.0 if signed >= 0 else 1.0

    step = np.deg2rad(config["rot_step_deg"])
    index = 1
    while index <= int(round(np.pi / step)):
        angle = sign * index * step
        cosine, sine = np.cos(angle), np.sin(angle)
        direction = np.array([
            goal_direction[0] * cosine - goal_direction[1] * sine,
            goal_direction[0] * sine + goal_direction[1] * cosine,
        ])
        endpoint = position + direction * max(
            np.linalg.norm(goal - position), 1.0e-9
        )
        if all(
            dist_point_line(obstacle, position, endpoint) > config["rs"]
            for obstacle, _ in restricted
        ):
            return angle
        index += 1
    return sign * np.pi / 2.0


def set_virtual_goal(position, goal, restricted, config):
    angle = deflection_angle(position, goal, restricted, config)
    radius = max(distance for _, distance in restricted)
    base = np.arctan2(goal[1] - position[1], goal[0] - position[0])
    direction = base + angle
    return position + radius * np.array([np.cos(direction), np.sin(direction)])


def repulsive_force(position, target, restricted, config, goal=None):
    """Accumulate the two SOPC-APF repulsive components."""
    if not restricted:
        return np.zeros(2)
    reference = (
        target
        if config["ra_target"] == "current"
        else (goal if goal is not None else target)
    )
    obstacles = np.array([obstacle for obstacle, _ in restricted])
    distances = np.array([distance for _, distance in restricted])
    keep = distances <= config["rho_o"]
    if not np.any(keep):
        return np.zeros(2)

    obstacles = obstacles[keep]
    distances = np.maximum(distances[keep], 1.0e-9)
    target_distance = max(float(np.linalg.norm(reference - position)), 1.0e-9)
    exponent = np.where(
        distances > config["rho_o"] / 2.0,
        config["M"],
        config["M_inner"],
    )
    base = 1.0 / distances - 1.0 / config["rho_o"]
    radial = base * target_distance ** exponent / distances ** 3
    tangential = (base ** 2) * target_distance ** exponent / 2.0
    away = (position - obstacles) / distances[:, None]
    toward = (target - position) / max(np.linalg.norm(target - position), 1.0e-12)
    toward = toward if config["fr2_dir"] == "target" else -toward
    force = (
        (config["alpha"] * radial)[:, None] * away
        + (config["alpha"] * exponent * tangential)[:, None] * toward
    )
    return force.sum(axis=0)


# ---------------- RV-APF ----------------
def repulsion_sum(position, obstacles, config):
    delta = position - obstacles
    distances = np.linalg.norm(delta, axis=1)
    selected = distances <= config["p0"]
    if not np.any(selected):
        return np.zeros(2), 0
    return (
        (delta[selected] / distances[selected][:, None]).sum(axis=0),
        int(selected.sum()),
    )


def gap_waypoint(position, obstacles, config):
    """Select a waypoint in the widest angular gap of nearby obstacles."""
    delta = position - obstacles
    distances = np.linalg.norm(delta, axis=1)
    keep = distances > 1.0e-9
    delta, distances = delta[keep], distances[keep]
    if len(distances) < 2:
        return None

    indices = np.argsort(distances)[: config["N"]]
    angles = np.arctan2(delta[indices, 1], delta[indices, 0])
    radii = distances[indices]
    order = np.argsort(angles)
    angles, radii = angles[order], radii[order]
    if len(angles) < 2:
        return None

    chord = np.sqrt(np.maximum(
        radii[:-1] ** 2
        + radii[1:] ** 2
        - 2.0 * radii[:-1] * radii[1:] * np.cos(angles[:-1] - angles[1:]),
        0.0,
    ))
    widest = int(np.argmax(chord))
    radius = 0.5 * chord[widest]
    first = delta[indices][order][widest]
    second = delta[indices][order][widest + 1]
    midpoint = 0.5 * (first + second)
    if radius < 1.0e-6:
        return None
    return position + radius * unit(midpoint)

# ---------------- Simplified bicycle kinematics ----------------
def _wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


class Bicycle:
    def __init__(self, pos, L=0.1, dt=0.1, speed_max=1.0, speed_min=-1.0,
                 acc_max=1.0, brake_max=1.0, wheel_angle_max=np.pi / 4,
                 wheel_acc_max=2 * np.pi, theta=0.0):
        self.pos = np.array(pos, float)
        self.L = L
        self.dt = dt
        self.theta = theta          # body heading
        self.v = 0.0                # scalar speed
        self.psi = theta            # velocity direction
        self.wheel_angle = 0.0
        self.wheel_desired = 0.0
        self.wheel_acc = 0.0
        self.acc_engine = 0.0
        self.speed_max, self.speed_min = speed_max, speed_min
        self.acc_max, self.brake_max = acc_max, brake_max
        self.wheel_angle_max, self.wheel_acc_max = wheel_angle_max, wheel_acc_max

    def step(self, F):
        # Update speed, steering angle, heading, and position using the simplified bicycle model.
        n = np.linalg.norm(F)                  # resultant-force magnitude
        if n < 1e-12:                          # negligible resultant force
            self.acc_engine = 0.0              # no drive or braking acceleration
            return                             # preserve speed, steering, and position
        a = F / n                              # unit desired acceleration direction
        th_d = np.arctan2(a[1], a[0])          # desired heading from the force direction
        turn = _wrap(th_d - self.theta)        # desired heading error normalized to [-pi, pi]
        if self.v < 0:                         # reverse-motion state
            turn = -turn                       # reverse the steering sign while reversing
        engine_push = n                        # force magnitude used as desired drive acceleration
        if abs(turn) > np.pi / 2:              # desired direction lies behind the body
            engine_push = -engine_push         # command reverse motion
        self.wheel_desired = float(np.clip(turn, -self.wheel_angle_max, self.wheel_angle_max))
        # Set the desired front steering angle from the heading error and saturate it.
        need = _wrap(self.wheel_desired - self.wheel_angle) / self.dt
        # Compute the steering angular rate required to reach the desired angle.
        self.wheel_acc = float(np.clip(need, -self.wheel_acc_max, self.wheel_acc_max))
        # Saturate the steering angular rate.
        self.wheel_angle = float(np.clip(self.wheel_angle + self.wheel_acc * self.dt,
                                         -self.wheel_angle_max, self.wheel_angle_max))
        # Integrate the front steering angle and apply the steering limit.
        self.acc_engine = float(np.clip(engine_push, -self.brake_max, self.acc_max))
        # Saturate the drive and braking acceleration.
        self.v = float(np.clip(self.v + self.acc_engine * self.dt, self.speed_min, self.speed_max))
        # Integrate speed and enforce the minimum and maximum speed limits.
        self.psi = _wrap(self.psi + (self.v / self.L) * np.tan(self.wheel_angle) * self.dt)
        # Update the velocity direction according to v/L*tan(delta).
        self.theta = self.psi                  # body heading coincides with velocity direction
        self.pos = self.pos + self.v * np.array([np.cos(self.psi), np.sin(self.psi)]) * self.dt
        # Integrate the robot position along the velocity direction.


def crosses(p, q, g1, g2):
    d = g2 - g1
    s0 = np.cross(d, p - g1)
    s1 = np.cross(d, q - g1)
    if s0 == 0 or s1 == 0 or s0 * s1 < 0:
        seg = q - p
        denom = np.cross(seg, d)
        if abs(denom) < 1e-12:
            return None
        tt = np.cross(g1 - p, d) / denom
        ss = np.cross(g1 - p, seg) / denom
        if 0.0 <= tt <= 1.0 and 0.0 <= ss <= 1.0:
            return p + tt * seg
    return None


# ---------------- Stepwise force computation for the compared planners ----------------
def force_proposed(pos, obs, lam):
    d = pos - obs
    r = np.maximum(np.linalg.norm(d, axis=1), 1e-9)
    rmin = r.min()
    return (d / (r * np.maximum(r - lam * rmin, 1e-3))[:, None]).sum(axis=0)


def force_ref8(pos, goal, obs, cfg, st):
    if cfg["recompute_vgoal"] or st["vgoal"] is None:
        nr = collision_prediction(pos, goal, obs, cfg)
        if nr:
            st["vgoal"] = set_virtual_goal(pos, goal, nr, cfg)
            st["restricted"] = nr
        elif cfg["recompute_vgoal"]:
            st["vgoal"] = None
            st["restricted"] = []
    vg = st["vgoal"]
    target = vg if vg is not None else goal
    att_target = goal if cfg["att_target"] == "real" else target
    F = cfg["ks"] * (att_target - pos) + repulsive_force(pos, target, st["restricted"], cfg, goal=goal)
    if vg is not None and np.linalg.norm(pos - vg) <= cfg["L"]:
        st["vgoal"] = None
        st["restricted"] = []
    return F


def force_ref12(pos, goal, obs, cfg, st):
    xa = (goal - pos) / max(np.linalg.norm(goal - pos), 1e-12)
    Fr, n_obs = repulsion_sum(pos, obs, cfg)
    plain = cfg["gamma"] * xa + Fr
    stuck = bool(n_obs > 0) and np.linalg.norm(plain) <= cfg["eps"] * max(np.linalg.norm(Fr), 1.0)
    wp = st.get("wp")
    if wp is not None and np.linalg.norm(pos - wp) <= cfg["wp_radius"]:
        wp = None
        st["wp"] = None
    if wp is None and stuck and cfg["use_waypoint"]:
        wp = gap_waypoint(pos, obs, cfg)
        st["wp"] = wp
    if wp is not None:
        xt = (wp - pos) / max(np.linalg.norm(wp - pos), 1e-12)
        if cfg["vortex"]:
            t = np.deg2rad(90.0)
            Rm = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
            F = cfg["gamma"] * xt + Rm @ Fr
        else:
            F = cfg["gamma"] * xt + Fr
    elif stuck and cfg["vortex"]:
        t = np.deg2rad(90.0)
        Rm = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
        F = cfg["gamma"] * xa + Rm @ Fr
    else:
        F = plain
    return F


def rect_geom(gap):
    segs, obs = u_enclosure(20.0, 10.0, gap_width=gap, closed_side="near", spacing=0.5)
    obs = np.array(obs, float)
    g1, g2 = np.array([20.0, 10 - gap]), np.array([20.0, 10.0])
    mid = np.array([20.0, 10 - gap / 2.0])
    return segs, obs, g1, g2, mid, np.array([22.0, 25.0])


def _ccw(poly):
    """Return polygon vertices in counter-clockwise order."""
    vertices = [np.asarray(point, float) for point in poly]
    signed_area = sum(
        p[0] * q[1] - q[0] * p[1]
        for p, q in zip(vertices, vertices[1:] + vertices[:1])
    )
    return vertices if signed_area > 0 else vertices[::-1]


def _polygon_edges(poly):
    return [
        (poly[index], poly[(index + 1) % len(poly)])
        for index in range(len(poly))
    ]


def _sample_segment(start, end, spacing=0.5, skip=()):
    count = max(int(np.floor(float(np.linalg.norm(end - start)) / spacing)), 1)
    points = []
    for index in range(count + 1):
        fraction = index / count
        if any(lower <= fraction <= upper for lower, upper in skip):
            continue
        points.append(start + fraction * (end - start))
    return points


def _build_polygon_opening(poly, open_index, gap):
    start, end = poly[open_index], poly[(open_index + 1) % len(poly)]
    edge = (end - start) / np.linalg.norm(end - start)
    midpoint = 0.5 * (start + end)
    first = midpoint - gap / 2.0 * edge
    second = midpoint + gap / 2.0 * edge
    half_fraction = (gap / 2.0) / np.linalg.norm(end - start)
    skip = [(0.5 - half_fraction, 0.5 + half_fraction)]

    walls, obstacles = [], []
    for index, (vertex_a, vertex_b) in enumerate(_polygon_edges(poly)):
        if index == open_index:
            walls.extend([(vertex_a, first), (second, vertex_b)])
            obstacles.extend(_sample_segment(vertex_a, vertex_b, skip=skip))
        else:
            walls.append((vertex_a, vertex_b))
            obstacles.extend(_sample_segment(vertex_a, vertex_b))
    normal = np.array([edge[1], -edge[0]])
    goal = midpoint + 6.0 * normal
    return walls, np.array(obstacles), first, second, midpoint, normal, goal


def hex_geom(gap):
    poly = _ccw([(0, 0), (16, 0), (23, 6), (17, 14), (5, 15), (-3, 7)])
    lengths = [float(np.linalg.norm(end - start)) for start, end in _polygon_edges(poly)]
    open_index = int(np.argmax(lengths))
    walls, obs, g1, g2, mid, normal, goal = _build_polygon_opening(
        poly, open_index, gap
    )
    return walls, np.array(obs, float), g1, g2, mid, goal
