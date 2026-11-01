# -*- coding: utf-8 -*-
"""Five-method 3D U-enclosure benchmark with five fixed exterior goals.

Geometry and random starts match Figures 13 and 14 in the manuscript. Goals
match Tables 5 and 6. Four square opening widths, five methods, and both
holonomic and 3D yaw-pitch nonholonomic (3D-YPN) kinematics are evaluated
for each goal.
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import numpy as np

SIM_DIR = Path(__file__).resolve().parent
PROJECT = SIM_DIR.parent
if str(SIM_DIR) not in sys.path:
    sys.path.insert(0, str(SIM_DIR))

import methods3d as M3  # noqa: E402
import hex_frustum_geometry as H  # noqa: E402
from bicycle_2d_4methods import (  # noqa: E402
    K_ATTR,
    K_DIST,
    K_INER,
    K_REP,
    T_INERTIA,
    DIT_APF_attractive,
    update_coeffs,
)
from cuboid_geometry import build_cuboid  # noqa: E402

GAPS = (3.0, 5.0, 8.0, 10.0)
N_START = 100
MAX_STEPS = 3000
DT = 0.1
V_MAX = 1.0
A_MAX = 1.0
WHEELBASE = 0.1
STEER_MAX = math.pi / 4.0
STEER_RATE_MAX = 2.0 * math.pi
PITCH_STEER_MAX = math.pi / 4.0
PITCH_STEER_RATE_MAX = 2.0 * math.pi
GAMMA_MAX = math.pi / 2.0 - 1.0e-3
R_SAFE = H.R_SAFE
SPACING = H.SPACING

METHODS = ("SOPC-APF", "DIT-APF", "RV-APF", "SA-MAPF", "VRF-APF")
MODELS = ("Holonomic", "3D-YPN")
GEOMETRIES = ("cuboid", "hex_frustum")

CUBOID_GOALS = np.asarray(
    [
        (25.26, 11.39, 2.36),
        (8.97, 17.56, 7.83),
        (-4.74, 8.48, 8.52),
        (1.57, -8.88, 2.45),
        (19.37, -5.89, 7.03),
    ],
    float,
)
HEX_GOALS = np.asarray(
    [
        (17.98, 19.39, 8.68),
        (0.56, 18.12, 4.96),
        (-7.13, 1.32, 4.13),
        (9.15, -7.14, 5.91),
        (23.44, 2.23, 10.87),
    ],
    float,
)
GOALS = {"cuboid": CUBOID_GOALS, "hex_frustum": HEX_GOALS}

SOPC_APF_CONFIG = dict(
    ks=15.0,
    alpha=1.1,
    M=2.0,
    M_inner=0.5,
    rho_o=2.5,
    Rp=4.0,
    rs=0.5,
    L=0.1,
)
RV_APF_CONFIG = dict(
    delta=0.1,
    p0=1.5,
    gamma=2.02,
    eps=0.1,
    N=200,
    vortex=True,
    use_waypoint=True,
    wp_radius=0.3,
)
SA_CFG = dict(
    k_att=1.2,
    k_rep=8.0,
    radius=2.5,
    lookahead=4.0,
    fuzzy_gain=0.65,
)

_CONTEXTS = {}
_START_DATA = {}


def unit(vector):
    vector = np.asarray(vector, float)
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > 1.0e-12 else np.zeros_like(vector)


def wrap_angle(angle):
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def cuboid_starts():
    rng = np.random.default_rng(20261001)
    unit_points = np.empty((N_START, 3), dtype=float)
    for dimension in range(3):
        strata = rng.permutation(N_START)
        unit_points[:, dimension] = (strata + rng.random(N_START)) / N_START
    low = np.array([0.5, 0.5, 0.5])
    high = np.array([19.5, 9.5, 9.5])
    return low + unit_points * (high - low)


def build_context(geometry, gap):
    if geometry == "cuboid":
        faces, opening_index = build_cuboid()
        side = faces[opening_index]
        starts = _START_DATA["cuboid"]
    elif geometry == "hex_frustum":
        faces = H.build_polyhedron()
        side = max(faces[2:], key=lambda face: face.area)
        starts = _START_DATA["hex_frustum"]
    else:
        raise ValueError(geometry)

    center = side.local(side.centre)[:2]
    obstacles = []
    for face in faces:
        obstacles.extend(
            face.sample(
                skip_square=(gap / 2.0, center) if face is side else None
            )
        )
    return {
        "geometry": geometry,
        "gap": float(gap),
        "faces": faces,
        "side": side,
        "center": center,
        "obstacles": np.asarray(obstacles, float),
        "starts": np.asarray(starts, float),
        "targets": GOALS[geometry],
    }


def initialize_contexts(geometries=GEOMETRIES, gaps=GAPS):
    global _CONTEXTS
    _START_DATA["cuboid"] = cuboid_starts()
    starts_path = SIM_DIR / "data" / "appendix_b_hex_frustum_starts.json"
    with open(starts_path, encoding="utf-8") as handle:
        _START_DATA["hex_frustum"] = np.asarray(json.load(handle)["starts"], float)
    _CONTEXTS = {
        (geometry, float(gap)): build_context(geometry, float(gap))
        for geometry in geometries
        for gap in gaps
    }


class HolonomicRobot3D:
    kind = "holonomic"

    def __init__(self, start):
        self.pos = np.asarray(start, float).copy()
        self.heading = np.array([1.0, 0.0, 0.0])
        self.previous_velocity = np.zeros(3)

    def step(self, force):
        force_norm = float(np.linalg.norm(force))
        if force_norm < 1.0e-12:
            return False, self.heading
        velocity = force if force_norm <= V_MAX else V_MAX * force / force_norm
        previous = self.pos.copy()
        self.pos = self.pos + DT * velocity
        self.previous_velocity = self.pos - previous
        self.heading = unit(velocity)
        return True, self.heading


class SimplifiedBicycle3D:
    kind = "bicycle"

    def __init__(self, start):
        self.pos = np.asarray(start, float).copy()
        self.theta = 0.0
        self.gamma = 0.0
        self.delta = 0.0
        self.alpha = 0.0
        self.speed = 0.0
        self.previous_velocity = np.zeros(3)

    def heading_vector(self):
        return np.array(
            [
                math.cos(self.gamma) * math.cos(self.theta),
                math.cos(self.gamma) * math.sin(self.theta),
                math.sin(self.gamma),
            ],
            float,
        )

    def step(self, force):
        force_norm = float(np.linalg.norm(force))
        if force_norm < 1.0e-12:
            return False, self.heading_vector()

        direction = force / force_norm
        theta_d = math.atan2(direction[1], direction[0])
        gamma_d = math.asin(float(np.clip(direction[2], -1.0, 1.0)))
        delta_theta = wrap_angle(theta_d - self.theta)
        delta_gamma = float(
            np.clip(gamma_d - self.gamma, -math.pi / 2.0, math.pi / 2.0)
        )
        if self.speed < 0.0:
            delta_theta = -delta_theta
            delta_gamma = -delta_gamma

        acceleration_d = force_norm
        previous_heading = self.heading_vector()
        if float(np.dot(previous_heading, direction)) < 0.0:
            acceleration_d = -acceleration_d

        delta_d = float(np.clip(delta_theta, -STEER_MAX, STEER_MAX))
        delta_rate = float(
            np.clip(
                wrap_angle(delta_d - self.delta) / DT,
                -STEER_RATE_MAX,
                STEER_RATE_MAX,
            )
        )
        self.delta = float(
            np.clip(
                self.delta + delta_rate * DT,
                -STEER_MAX,
                STEER_MAX,
            )
        )

        alpha_d = float(
            np.clip(delta_gamma, -PITCH_STEER_MAX, PITCH_STEER_MAX)
        )
        alpha_rate = float(
            np.clip(
                (alpha_d - self.alpha) / DT,
                -PITCH_STEER_RATE_MAX,
                PITCH_STEER_RATE_MAX,
            )
        )
        self.alpha = float(
            np.clip(
                self.alpha + alpha_rate * DT,
                -PITCH_STEER_MAX,
                PITCH_STEER_MAX,
            )
        )

        acceleration = float(np.clip(acceleration_d, -A_MAX, A_MAX))
        self.speed = float(
            np.clip(self.speed + acceleration * DT, -V_MAX, V_MAX)
        )
        self.theta = wrap_angle(
            self.theta
            + (self.speed / WHEELBASE) * math.tan(self.delta) * DT
        )
        self.gamma = float(
            np.clip(
                self.gamma
                + (self.speed / WHEELBASE) * math.tan(self.alpha) * DT,
                -GAMMA_MAX,
                GAMMA_MAX,
            )
        )

        previous = self.pos.copy()
        direction_after = self.heading_vector()
        self.pos = self.pos + self.speed * direction_after * DT
        self.previous_velocity = self.pos - previous
        return True, direction_after


def clearance(point, context):
    return H.clearance(
        np.asarray(point, float),
        context["faces"],
        side=context["side"],
        half=context["gap"] / 2.0,
        c=context["center"],
    )


def force_vrf(position, obstacles, lam):
    vectors = position[None, :] - obstacles
    distances = np.maximum(np.linalg.norm(vectors, axis=1), 1.0e-9)
    d_min = float(distances.min())
    denominator = distances * np.maximum(
        distances - lam * d_min,
        1.0e-3,
    ) ** 2
    return np.sum(vectors / denominator[:, None], axis=0)


def detect_local_minimum_3d(position, positions, headings):
    nb = int(2.0 / DT)
    if len(positions) <= nb + 1:
        return False, None
    pos_start = len(positions) - nb - 1
    pos_end = len(positions) - 2
    for index in range(pos_start, pos_end):
        if np.linalg.norm(position - positions[index]) < 0.1 and index < pos_end / 2:
            return True, positions[index]
    mean_position = np.mean(positions[pos_start + 1:pos_end + 1], axis=0)
    total_turn = 0.0
    for index in range(pos_start, pos_end):
        cosine = float(
            np.clip(np.dot(headings[index], headings[index + 1]), -1.0, 1.0)
        )
        total_turn += math.acos(cosine)
    if np.linalg.norm(position - mean_position) < 2.0 and total_turn > 2.0 * math.pi:
        return True, mean_position
    return False, None


def DIT_APF_repulsive_3d(position, obstacles, k_rep, k_dist, heading):
    delta = position[None, :] - obstacles
    distance = np.maximum(np.linalg.norm(delta, axis=1), 1.0e-9)
    effective = distance - 0.1
    mask = (effective > 0.0) & (distance < k_dist + 0.1)
    repulsive = np.zeros(3)
    tangential = np.zeros(3)
    if not np.any(mask):
        return repulsive, tangential

    delta = delta[mask]
    distance = distance[mask]
    effective = effective[mask]
    gap = 1.0 / effective - 1.0 / k_dist
    magnitude = k_rep * gap * (1.0 / effective) ** 2 * (effective / distance)
    repulsive = np.sum(magnitude[:, None] * delta, axis=0)

    heading = unit(heading)
    normal_component = delta - np.outer(delta @ heading, heading)
    normal_length = np.linalg.norm(normal_component, axis=1)
    valid = normal_length > 1.0e-9
    if np.any(valid):
        normal_component = normal_component[valid] / normal_length[valid, None]
        tangent_direction = np.cross(
            np.broadcast_to(heading, normal_component.shape),
            normal_component,
        )
        tangential = np.sum(
            magnitude[valid, None] * tangent_direction,
            axis=0,
        )
    return repulsive, tangential


def DIT_APF_inertia_3d(velocity_history, gain, position, goal):
    if not velocity_history:
        return np.zeros(3)
    inertia = np.mean(velocity_history, axis=0) * gain
    goal_distance = float(np.linalg.norm(goal - position))
    if goal_distance < 4.0:
        inertia = inertia * (goal_distance / 4.0)
    return inertia


def rotate_vectors_about_axis(vectors, axis, angles):
    """Rodrigues rotation for a batch of candidate direction vectors."""
    vectors = np.asarray(vectors, float)
    axis = unit(axis)
    cosines = np.cos(angles)[:, None]
    sines = np.sin(angles)[:, None]
    return (
        vectors[None, :] * cosines
        + np.cross(np.broadcast_to(axis, vectors.shape), vectors)[None, :] * sines
        + np.broadcast_to(axis, vectors.shape)[None, :]
        * np.sum(vectors * axis)
        * (1.0 - cosines)
    )


def force_sopc(position, goal, obstacles, cfg):
    """3D SOPC-APF with vectorized predictive-collision deflection search."""
    delta = obstacles - position
    distance = np.linalg.norm(delta, axis=1)
    goal_vector = goal - position
    goal_distance = float(np.linalg.norm(goal_vector))
    if goal_distance < 1.0e-12:
        return np.zeros(3)
    goal_direction = goal_vector / goal_distance
    candidate_mask = distance <= cfg["Rp"]
    if not np.any(candidate_mask):
        return cfg["ks"] * goal_vector

    restricted_delta = delta[candidate_mask]
    restricted_distance = distance[candidate_mask]
    line_distance = np.linalg.norm(
        np.cross(
            np.broadcast_to(goal_direction, restricted_delta.shape),
            restricted_delta,
        ),
        axis=1,
    )
    restricted_mask = line_distance <= cfg["rs"]
    if not np.any(restricted_mask):
        return cfg["ks"] * goal_vector

    restricted_delta = restricted_delta[restricted_mask]
    restricted_distance = restricted_distance[restricted_mask]
    obstacle_direction = unit(np.mean(restricted_delta, axis=0))
    axis = unit(np.cross(goal_direction, obstacle_direction))
    if np.linalg.norm(axis) < 1.0e-6:
        fallback = np.array([0.0, 0.0, 1.0])
        if abs(goal_direction[2]) >= 0.9:
            fallback = np.array([1.0, 0.0, 0.0])
        axis = unit(np.cross(goal_direction, fallback))

    step_angle = math.radians(3.0)
    steps = np.arange(1, int(round(math.pi / step_angle)) + 1)
    valid_choices = []
    for sign in (-1.0, 1.0):
        directions = rotate_vectors_about_axis(
            goal_direction,
            axis,
            sign * steps * step_angle,
        )
        line_clearance = np.linalg.norm(
            np.cross(directions[:, None, :], restricted_delta[None, :, :]),
            axis=2,
        )
        valid = np.all(line_clearance > cfg["rs"], axis=1)
        if np.any(valid):
            first = int(np.flatnonzero(valid)[0])
            valid_choices.append((first, sign, directions[first]))
    if valid_choices:
        valid_choices.sort(key=lambda item: (item[0], item[1]))
        target_direction = valid_choices[0][2]
    else:
        target_direction = rotate_vectors_about_axis(
            goal_direction,
            axis,
            np.array([math.pi / 2.0]),
        )[0]

    target = position + float(restricted_distance.max()) * target_direction
    attractive = cfg["ks"] * (target - position)

    keep = restricted_distance <= cfg["rho_o"]
    if not np.any(keep):
        return attractive
    r = np.maximum(restricted_distance[keep], 1.0e-9)
    obstacle_points = obstacles[candidate_mask][restricted_mask][keep]
    R_A = max(float(np.linalg.norm(target - position)), 1.0e-9)
    M = np.where(r > cfg["rho_o"] / 2.0, cfg["M"], cfg["M_inner"])
    base = 1.0 / r - 1.0 / cfg["rho_o"]
    FR1 = base * R_A ** M / r ** 3
    FR2 = base ** 2 * R_A ** M / 2.0
    away = (position - obstacle_points) / r[:, None]
    toward = unit(target - position)
    repulsive = np.sum(
        (cfg["alpha"] * FR1)[:, None] * away,
        axis=0,
    ) - np.sum(cfg["alpha"] * M * FR2) * toward
    return attractive + repulsive


def force_dit(position, goal, obstacles, state, heading):
    if state["recent_lm"] < DT:
        found, virtual_position = detect_local_minimum_3d(
            position,
            state["hist_pos"],
            state["hist_head"],
        )
        if found:
            state["recent_lm"] = 2.0
            if virtual_position is not None:
                state["virtual"].append(np.asarray(virtual_position, float))
    else:
        state["recent_lm"] -= DT

    update_coeffs(
        state,
        DT < state["recent_lm"] <= 2.0,
        position,
        goal,
    )
    all_obstacles = (
        np.vstack([obstacles, np.asarray(state["virtual"], float)])
        if state["virtual"]
        else obstacles
    )
    repulsive, tangential = DIT_APF_repulsive_3d(
        position,
        all_obstacles,
        state["k_rep"],
        state["k_dist"],
        heading,
    )
    attractive = DIT_APF_attractive(position, goal, state["k_attr"])
    inertia = DIT_APF_inertia_3d(
        state["vel_hist"],
        state["k_inertia"],
        position,
        goal,
    )
    return attractive + repulsive + tangential + inertia


def sphere_direction(rng):
    direction = rng.normal(size=3)
    return unit(direction)


def sa_energy(point, goal, obstacles, context, cfg):
    delta = point[None, :] - obstacles
    distance = np.linalg.norm(delta, axis=1)
    active = (distance > 1.0e-9) & (distance <= cfg["radius"])
    repulsive = 0.0
    if np.any(active):
        d = distance[active]
        repulsive = 0.5 * cfg["k_rep"] * np.sum(
            (1.0 / d - 1.0 / cfg["radius"]) ** 2
        )
    wall_penalty = 50.0 * max(0.0, R_SAFE + 0.05 - clearance(point, context)) ** 2
    attractive = 0.5 * cfg["k_att"] * float(np.dot(goal - point, goal - point))
    return attractive + repulsive + wall_penalty


def sa_propose(position, goal, obstacles, context, state, cfg):
    rng = state["rng"]
    current_energy = sa_energy(position, goal, obstacles, context, cfg)
    candidates = []
    for _ in range(28):
        candidate = position + rng.uniform(0.25, 1.5) * sphere_direction(rng)
        if clearance(candidate, context) >= R_SAFE + 0.03:
            candidates.append(candidate)
    if not candidates:
        return None
    energies = np.asarray(
        [sa_energy(candidate, goal, obstacles, context, cfg) for candidate in candidates]
    )
    order = np.argsort(energies)
    best_index = int(order[0])
    best_energy = float(energies[best_index])
    delta = best_energy - current_energy
    temperature = max(2.0, float(state["temperature"]))
    if delta <= 0.0 or rng.random() < math.exp(-min(delta / temperature, 700.0)):
        state["temperature"] = max(2.0, temperature * 0.85)
        return candidates[best_index]
    top = order[: min(6, len(order))]
    weights = np.exp(-(energies[top] - energies[top].min()) / temperature)
    weights = weights / weights.sum()
    selected = int(rng.choice(top, p=weights))
    state["temperature"] = max(2.0, temperature * 0.85)
    return candidates[selected]


def sa_waypoint(position, goal, obstacles):
    delta = position[None, :] - obstacles
    distance = np.linalg.norm(delta, axis=1)
    nearest = np.argsort(distance)[:72]
    directions = delta[nearest] / np.maximum(distance[nearest, None], 1.0e-9)
    sphere = M3.sphere_dirs(256)
    max_cosines = np.max(sphere @ directions.T, axis=1)
    goal_direction = unit(goal - position)
    goal_alignment = sphere @ goal_direction
    score = max_cosines - 0.08 * goal_alignment
    direction = sphere[int(np.argmin(score))]
    nearest_distance = float(distance[nearest[0]]) if nearest.size else 3.0
    return position + min(3.0, 0.6 * nearest_distance) * direction


def sa_repulsive(position, obstacles, radius, gain):
    delta = position[None, :] - obstacles
    distance = np.linalg.norm(delta, axis=1)
    active = (distance > 1.0e-9) & (distance <= radius)
    if not np.any(active):
        return np.zeros(3)
    d = distance[active]
    magnitude = gain * np.maximum(1.0 / d - 1.0 / radius, 0.0) / d ** 2
    return np.sum(magnitude[:, None] * delta[active] / d[:, None], axis=0)


def force_sa(position, goal, obstacles, context, cfg, state):
    active_target = state["active_target"] if state["active_target"] is not None else goal
    attractive = cfg["k_att"] * (active_target - position)
    attractive_norm = float(np.linalg.norm(attractive))
    if attractive_norm > cfg["lookahead"]:
        attractive = attractive * cfg["lookahead"] / attractive_norm
    repulsive = sa_repulsive(position, obstacles, cfg["radius"], cfg["k_rep"])
    raw = attractive + repulsive
    direction = unit(raw)

    delta = position[None, :] - obstacles
    distance = np.linalg.norm(delta, axis=1)
    active = (distance > 1.0e-9) & (distance <= cfg["radius"])
    lateral = np.zeros(3)
    if np.any(active):
        obstacle_direction = delta[active] / distance[active, None]
        tangent = np.cross(
            np.broadcast_to(direction, obstacle_direction.shape),
            obstacle_direction,
        )
        weight = np.exp(-distance[active] / cfg["radius"])
        lateral = cfg["fuzzy_gain"] * np.sum(tangent * weight[:, None], axis=0)
    force = raw + lateral

    moving_norm = float(np.linalg.norm(force))
    state["low_motion_steps"] = state["low_motion_steps"] + 1 if moving_norm < 0.035 else 0
    trapped = moving_norm < 0.01 or state["low_motion_steps"] >= 10
    if state["active_target"] is not None:
        state["target_steps"] -= 1
        if (
            np.linalg.norm(position - state["active_target"]) < 0.45
            or state["target_steps"] <= 0
        ):
            state["active_target"] = None
            state["target_steps"] = 0
    if trapped and state["active_target"] is None:
        state["low_motion_steps"] = 0
        waypoint = sa_waypoint(position, goal, obstacles)
        proposal = sa_propose(position, waypoint, obstacles, context, state, cfg)
        if proposal is None:
            proposal = waypoint
        state["active_target"] = np.asarray(proposal, float)
        state["target_steps"] = 24
    return force


def method_force(method, position, goal, context, robot, state):
    obstacles = context["obstacles"]
    heading = robot.heading if robot.kind == "holonomic" else robot.heading_vector()
    if method == "SOPC-APF":
        return force_sopc(position, goal, obstacles, SOPC_APF_CONFIG)
    if method == "DIT-APF":
        return force_dit(position, goal, obstacles, state, heading)
    if method == "RV-APF":
        return M3.RV_APF_step(position, goal, obstacles, RV_APF_CONFIG, state)
    if method == "SA-MAPF":
        return force_sa(position, goal, obstacles, context, SA_CFG, state)
    if method == "VRF-APF":
        return force_vrf(position, obstacles, state["lam"])
    raise ValueError(method)


def make_state(method, geometry, goal_index, model, start_index):
    if method == "DIT-APF":
        return {
            "k_rep": K_REP[0],
            "k_attr": K_ATTR[1],
            "k_inertia": 0.5,
            "k_dist": K_DIST[0],
            "virtual": [],
            "recent_lm": 0.0,
            "hist_pos": [],
            "hist_head": [],
            "vel_hist": [],
        }
    if method == "RV-APF":
        return {"wp": None}
    if method == "SA-MAPF":
        geometry_offset = 0 if geometry == "cuboid" else 100000
        seed = (
            geometry_offset
            + 10000 * goal_index
            + 1000 * METHODS.index(method)
            + start_index
            + (50000 if model == "3D-YPN" else 0)
        )
        return {
            "active_target": None,
            "target_steps": 0,
            "low_motion_steps": 0,
            "temperature": 50.0,
            "rng": np.random.default_rng(seed),
        }
    if method == "VRF-APF":
        return {"lam": 0.8}
    return {}


def simulate_one(task):
    geometry, gap, goal_index, method, model, start_index = task
    context = _CONTEXTS[(geometry, float(gap))]
    start = context["starts"][start_index]
    goal = context["targets"][goal_index]
    robot = (
        HolonomicRobot3D(start)
        if model == "Holonomic"
        else SimplifiedBicycle3D(start)
    )
    state = make_state(method, geometry, goal_index, model, start_index)
    if method == "DIT-APF":
        state["hist_pos"] = [robot.pos.copy()]
        state["hist_head"] = [
            robot.heading if robot.kind == "holonomic" else robot.heading_vector()
        ]

    min_clearance = clearance(robot.pos, context)
    hit = None
    force_steps = 0
    force_time = 0.0
    trail = [robot.pos.copy()] if method == "VRF-APF" else None
    simulation_start = time.perf_counter()
    stop_reason = "max_steps"

    for step_index in range(MAX_STEPS):
        current_heading = (
            robot.heading
            if robot.kind == "holonomic"
            else robot.heading_vector()
        )
        step_start = time.perf_counter()
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            force = method_force(
                method,
                robot.pos,
                goal,
                context,
                robot,
                state,
            )
        if not np.all(np.isfinite(force)):
            stop_reason = "nonfinite_force"
            break
        previous = robot.pos.copy()
        moved, heading_after = robot.step(force)
        force_time += time.perf_counter() - step_start
        if not moved:
            stop_reason = "zero_force"
            break
        if not np.all(np.isfinite(robot.pos)):
            stop_reason = "nonfinite_state"
            break
        force_steps += 1

        new_position = robot.pos.copy()
        current_clearance = clearance(new_position, context)
        min_clearance = min(min_clearance, current_clearance)
        if trail is not None:
            trail.append(new_position.copy())

        hit = H.cross_opening(
            previous,
            new_position,
            context["side"],
            context["center"],
            context["gap"],
        )
        if hit is not None:
            hit = np.asarray(hit, float)
            min_clearance = min(min_clearance, clearance(hit, context))
            stop_reason = "opening_crossed"
            break
        if current_clearance < R_SAFE:
            stop_reason = "collision"
            break

        if method == "DIT-APF":
            state["hist_pos"].append(new_position.copy())
            state["hist_head"].append(unit(heading_after))
            state["vel_hist"].append(new_position - previous)
            if len(state["vel_hist"]) > int(T_INERTIA / DT):
                state["vel_hist"].pop(0)
        if method == "VRF-APF" and step_index > 0 and step_index % 50 == 0:
            if len(trail) > 50 and np.linalg.norm(robot.pos - trail[-51]) < 0.1:
                state["lam"] = max(0.3, state["lam"] - 0.1)

    elapsed = time.perf_counter() - simulation_start
    deviation = (
        None
        if hit is None
        else float(np.linalg.norm(context["side"].local(hit)[:2] - context["center"]))
    )
    collision = bool(min_clearance < R_SAFE)
    escaped = hit is not None
    return {
        "geometry": geometry,
        "gap": float(gap),
        "goal_index": goal_index,
        "goal": goal.tolist(),
        "method": method,
        "model": model,
        "start_index": start_index,
        "escaped": escaped,
        "collision": collision,
        "safe_success": bool(escaped and not collision),
        "deviation": deviation,
        "steps": force_steps,
        "stop_reason": stop_reason,
        "force_time": force_time,
        "elapsed": elapsed,
        "min_clear": float(min_clearance),
    }


def aggregate(details):
    rows = []
    for geometry in GEOMETRIES:
        for gap in GAPS:
            for goal_index in range(5):
                for method in METHODS:
                    for model in MODELS:
                        subset = [
                            item
                            for item in details
                            if item["geometry"] == geometry
                            and item["gap"] == gap
                            and item["goal_index"] == goal_index
                            and item["method"] == method
                            and item["model"] == model
                        ]
                        safe = [item for item in subset if item["safe_success"]]
                        total_steps = sum(item["steps"] for item in subset)
                        total_force_time = sum(item["force_time"] for item in subset)
                        rows.append(
                            {
                                "geometry": geometry,
                                "gap": float(gap),
                                "goal_index": goal_index,
                                "goal": GOALS[geometry][goal_index].tolist(),
                                "method": method,
                                "model": model,
                                "n": len(subset),
                                "success": len(safe),
                                "collisions": sum(item["collision"] for item in subset),
                                "mean_deviation": (
                                    float(np.mean([item["deviation"] for item in safe]))
                                    if safe
                                    else None
                                ),
                                "mean_steps": (
                                    float(np.mean([item["steps"] for item in safe]))
                                    if safe
                                    else None
                                ),
                                "per_step_ms": (
                                    1000.0 * total_force_time / total_steps
                                    if total_steps
                                    else None
                                ),
                            }
                        )
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-starts", type=int, default=N_START)
    parser.add_argument("--workers", type=int, default=max(1, min(10, mp.cpu_count() or 1)))
    parser.add_argument(
        "--outdir",
        default=str(SIM_DIR / "outputs" / "appendix_b"),
    )
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    partial_path = Path(args.outdir) / "three_d_five_methods_partial.json"

    initialize_contexts()
    all_tasks = [
        (geometry, gap, goal_index, method, model, start_index)
        for geometry in GEOMETRIES
        for gap in GAPS
        for goal_index in range(5)
        for method in METHODS
        for model in MODELS
        for start_index in range(args.n_starts)
    ]
    details = []
    if args.resume and partial_path.exists():
        details = json.loads(partial_path.read_text(encoding="utf-8"))
        print(f"resuming with {len(details)} completed simulations", flush=True)
    completed_keys = {
        (
            item["geometry"],
            item["gap"],
            item["goal_index"],
            item["method"],
            item["model"],
            item["start_index"],
        )
        for item in details
    }
    tasks = [task for task in all_tasks if task not in completed_keys]
    print(f"running {len(tasks)} simulations with {args.workers} workers", flush=True)
    context = mp.get_context("fork")
    completed = 0
    with context.Pool(processes=args.workers) as pool:
        for result in pool.imap_unordered(simulate_one, tasks, chunksize=4):
            details.append(result)
            completed += 1
            if completed % 500 == 0 or completed == len(tasks):
                temporary = partial_path.with_suffix(".tmp")
                temporary.write_text(
                    json.dumps(details, ensure_ascii=False),
                    encoding="utf-8",
                )
                temporary.replace(partial_path)
                print(f"completed {completed}/{len(tasks)}", flush=True)

    details.sort(
        key=lambda item: (
            GEOMETRIES.index(item["geometry"]),
            item["goal_index"],
            METHODS.index(item["method"]),
            MODELS.index(item["model"]),
            item["start_index"],
        )
    )
    rows = aggregate(details)
    json_path = os.path.join(args.outdir, "three_d_five_methods_results.json")
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "configuration": {
                    "n_starts": args.n_starts,
                    "opening_widths_m": list(GAPS),
                    "methods": list(METHODS),
                    "models": list(MODELS),
                    "max_steps": MAX_STEPS,
                    "dt_s": DT,
                    "v_max_m_s": V_MAX,
                    "collision_clearance_m": R_SAFE,
                    "targets": {
                        geometry: GOALS[geometry].tolist() for geometry in GEOMETRIES
                    },
                },
                "summary": rows,
                "detail": details,
            },
            handle,
            ensure_ascii=False,
            indent=1,
        )
    print(f"saved {json_path}", flush=True)

    for row in rows:
        print(
            f"{row['geometry']} G{row['goal_index'] + 1} "
            f"{row['method']} {row['model']}: "
            f"{row['success']}/{row['n']} collisions={row['collisions']} "
            f"dev={row['mean_deviation']} steps={row['mean_steps']} "
            f"time={row['per_step_ms']}",
            flush=True,
        )


if __name__ == "__main__":
    main()
