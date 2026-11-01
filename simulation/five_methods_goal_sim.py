# -*- coding: utf-8 -*-
"""Five-method, five-goal U-enclosure benchmark.

Methods
-------
SOPC-APF : Li and Wang predictive-collision / virtual-target APF
DIT-APF  : Kilic et al. dynamic + tangential + inertial APF
RV-APF   : Al Jabari et al. rotating-vector APF
SA-MAPF  : Liu simulated-annealing MAPF with virtual-target and fuzzy adjustment
VRF-APF  : vector-repulsive-force APF proposed in the manuscript

The experiment uses the same 100 interior starts as the manuscript, a 5 m
opening, and five deterministic random goals chosen outside the reachable side
of each opening. Results are generated for both holonomic and simplified
bicycle kinematics.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import multiprocessing as mp
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np


APF_DIR = str(Path(__file__).resolve().parent)
if APF_DIR not in sys.path:
    sys.path.insert(0, APF_DIR)

from bicycle_2d import (  # noqa: E402
    Bicycle,
    DEFAULT,
    DEFAULT9,
    R_SAFE,
    crosses,
    dist_point_segments,
    force_proposed,
    force_ref8,
    force_ref12,
    gap_waypoint,
    hex_geom as bicycle_hex_geom,
    rect_geom as bicycle_rect_geom,
)
from bicycle_2d_4methods import (  # noqa: E402
    K_ATTR,
    K_DIST,
    K_INER,
    K_REP,
    STEP,
    T_INERTIA,
    detect_lm,
    DIT_APF_attractive,
    DIT_APF_inertia,
    DIT_APF_repulsive,
    update_coeffs,
)
GAPS = (3.0, 5.0, 8.0, 10.0)
N_START = 100
MAX_STEPS = 3000
DT = STEP
V_MAX = 1.0
GOAL_SEED = 20261023
METHODS = ("SOPC-APF", "DIT-APF", "RV-APF", "SA-MAPF", "VRF-APF")
MODELS = ("Holonomic", "Simplified bicycle")

_DATA = {}


class HolonomicRobot:
    """Holonomic velocity kinematic model from Algorithm 2."""

    def __init__(self, pos, dt=DT, speed_max=V_MAX):
        self.pos = np.asarray(pos, float).copy()
        self.dt = float(dt)
        self.speed_max = float(speed_max)
        self.heading = 0.0

    def step(self, force):
        n = float(np.linalg.norm(force))
        if n < 1.0e-12:
            return
        vel = force if n <= self.speed_max else self.speed_max * force / n
        self.pos = self.pos + self.dt * vel
        self.heading = math.atan2(float(vel[1]), float(vel[0]))


def _wrap(angle):
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def _unit(vector):
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > 1.0e-12 else np.zeros_like(vector)


def _clearance(point, context):
    """Vectorized minimum distance from a point to all sampled wall segments."""
    point = np.asarray(point, float)
    ab = context["wall_ab"]
    denom = np.einsum("ij,ij->i", ab, ab)
    denom = np.maximum(denom, 1.0e-12)
    t = np.einsum(
        "ij,ij->i",
        point - context["wall_a"],
        ab,
    ) / denom
    t = np.clip(t, 0.0, 1.0)
    nearest = context["wall_a"] + t[:, None] * ab
    return float(np.linalg.norm(point - nearest, axis=1).min())


def _point_segment_clear(p, q, context, margin):
    """Check segment clearance conservatively using dense samples."""
    for alpha in np.linspace(0.0, 1.0, 15):
        if _clearance(p + alpha * (q - p), context) < margin:
            return False
    return True


def _inside(point, polygon):
    x, y = point
    inside = False
    for i, (x1, y1) in enumerate(polygon):
        x2, y2 = polygon[(i + 1) % len(polygon)]
        if (y1 > y) != (y2 > y):
            x_cross = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
            if x < x_cross:
                inside = not inside
    return inside


def _outward_normal(mid, g1, g2, polygon):
    edge = _unit(np.asarray(g2, float) - np.asarray(g1, float))
    normal = np.array([-edge[1], edge[0]], float)
    centroid = np.mean(np.asarray(polygon, float), axis=0)
    if np.dot(mid + normal - centroid, normal) < 0.0:
        normal = -normal
    return _unit(normal)


def _sample_goals(walls, polygon, seed):
    """Sample five fixed goals distributed uniformly around the exterior.

    The five directions are equally spaced in azimuth around the polygon
    centroid, with a common random angular offset. Each radius extends beyond
    the polygon envelope by a random distance.
    """
    rng = np.random.default_rng(seed)
    polygon = np.asarray(polygon, float)
    centroid = polygon.mean(axis=0)
    offset = rng.uniform(0.0, 2.0 * math.pi / 5.0)
    goals = []
    for index in range(5):
        angle = offset + index * 2.0 * math.pi / 5.0
        direction = np.array([math.cos(angle), math.sin(angle)], float)
        projected = (polygon - centroid) @ direction
        envelope_radius = float(projected.max())
        radius = envelope_radius + rng.uniform(3.0, 7.0)
        candidate = centroid + radius * direction
        # Move outward if the selected point is too close to a sampled wall.
        for _ in range(50):
            if dist_point_segments(candidate, walls) >= 1.5 and not _inside(candidate, polygon):
                break
            radius += 0.5
            candidate = centroid + radius * direction
        else:
            raise RuntimeError("Could not place an exterior target point")
        goals.append(candidate)
    return goals


def _load_starts():
    data_dir = Path(__file__).resolve().parent / "data"
    with open(data_dir / "appendix_a_rect_starts.json", encoding="utf-8") as fh:
        rect = np.asarray(json.load(fh)["starts"], float)
    with open(data_dir / "appendix_a_hex_starts.json", encoding="utf-8") as fh:
        hex_starts = np.asarray(json.load(fh)["starts"], float)
    return rect[:N_START], hex_starts[:N_START]


def _rect_context(gap):
    walls, obs, g1, g2, mid, goal_ref = bicycle_rect_geom(gap)
    polygon = np.array([[0.0, 0.0], [20.0, 0.0], [20.0, 10.0], [0.0, 10.0]])
    wall_array = [np.asarray(segment, float) for segment in walls]
    wall_a = np.asarray([segment[0] for segment in wall_array], float)
    wall_b = np.asarray([segment[1] for segment in wall_array], float)
    return dict(
        name="Rectangular enclosure",
        gap=float(gap),
        polygon=polygon,
        walls=walls,
        wall_a=wall_a,
        wall_b=wall_b,
        wall_ab=wall_b - wall_a,
        obs=np.asarray(obs, float),
        g1=np.asarray(g1, float),
        g2=np.asarray(g2, float),
        mid=np.asarray(mid, float),
        goal_ref=np.asarray(goal_ref, float),
    )


def _hex_context(gap):
    walls, obs, g1, g2, mid, goal_ref = bicycle_hex_geom(gap)
    polygon = np.array(
        [[0.0, 0.0], [16.0, 0.0], [23.0, 6.0], [17.0, 14.0], [5.0, 15.0], [-3.0, 7.0]],
        float,
    )
    wall_array = [np.asarray(segment, float) for segment in walls]
    wall_a = np.asarray([segment[0] for segment in wall_array], float)
    wall_b = np.asarray([segment[1] for segment in wall_array], float)
    return dict(
        name="Hexagonal enclosure",
        gap=float(gap),
        polygon=polygon,
        walls=walls,
        wall_a=wall_a,
        wall_b=wall_b,
        wall_ab=wall_b - wall_a,
        obs=np.asarray(obs, float),
        g1=np.asarray(g1, float),
        g2=np.asarray(g2, float),
        mid=np.asarray(mid, float),
        goal_ref=np.asarray(goal_ref, float),
    )


def _sa_repulsive(pos, obs, radius, gain):
    delta = pos - obs
    distance = np.linalg.norm(delta, axis=1)
    active = (distance > 1.0e-9) & (distance <= radius)
    if not np.any(active):
        return np.zeros(2)
    d = distance[active]
    mag = gain * np.maximum(1.0 / d - 1.0 / radius, 0.0) / d ** 2
    force = (mag[:, None] * delta[active] / d[:, None]).sum(axis=0)
    return force


def _sa_energy(point, goal, obs, context, cfg):
    repulsive = 0.0
    d = np.linalg.norm(np.asarray(obs, float) - point, axis=1)
    active = (d > 1.0e-9) & (d <= cfg["radius"])
    if np.any(active):
        repulsive = 0.5 * cfg["k_rep"] * np.sum((1.0 / d[active] - 1.0 / cfg["radius"]) ** 2)
    clearance = _clearance(point, context)
    penalty = 50.0 * max(0.0, R_SAFE + 0.05 - clearance) ** 2
    return 0.5 * cfg["k_att"] * float(np.dot(goal - point, goal - point)) + repulsive + penalty


def _sa_propose(pos, goal, obs, context, state, cfg):
    """SA proposal for a local temporary target."""
    rng = state["rng"]
    current_energy = _sa_energy(pos, goal, obs, context, cfg)
    candidates = []
    for _ in range(28):
        radius = rng.uniform(0.25, 1.5)
        angle = rng.uniform(-math.pi, math.pi)
        candidate = pos + radius * np.array([math.cos(angle), math.sin(angle)])
        if _clearance(candidate, context) < R_SAFE + 0.03:
            continue
        if not _point_segment_clear(pos, candidate, context, R_SAFE + 0.01):
            continue
        candidates.append(candidate)
    if not candidates:
        return None
    energies = np.array([_sa_energy(c, goal, obs, context, cfg) for c in candidates])
    order = np.argsort(energies)
    best = candidates[int(order[0])]
    best_energy = float(energies[int(order[0])])
    delta = best_energy - current_energy
    temperature = max(2.0, float(state["temperature"]))
    if delta <= 0.0 or rng.random() < math.exp(-min(delta / temperature, 700.0)):
        state["temperature"] = max(2.0, temperature * 0.85)
        return best
    # Avoid a wasted local-minimum event: select a low-energy candidate with
    # Metropolis sampling when the globally best candidate is rejected.
    top = order[: min(6, len(order))]
    weights = np.exp(-(energies[top] - energies[top].min()) / temperature)
    weights = weights / weights.sum()
    idx = int(rng.choice(top, p=weights))
    state["temperature"] = max(2.0, temperature * 0.85)
    return candidates[idx]


def _sa_force(pos, heading, goal, obs, context, cfg, state):
    active_target = state["active_target"] if state["active_target"] is not None else goal
    attractive = cfg["k_att"] * (active_target - pos)
    attr_norm = float(np.linalg.norm(attractive))
    if attr_norm > cfg["lookahead"]:
        attractive = attractive * cfg["lookahead"] / attr_norm
    repulsive = _sa_repulsive(pos, obs, cfg["radius"], cfg["k_rep"])
    raw = attractive + repulsive
    direction = _unit(raw)
    if np.linalg.norm(direction) < 1.0e-12:
        direction = _unit(active_target - pos)

    # Two-rule fuzzy correction: turn away from the closer side and reduce the
    # lateral gain as the left/right distances become balanced.
    delta = obs - pos
    distance = np.linalg.norm(delta, axis=1)
    active = (distance > 1.0e-9) & (distance <= cfg["radius"])
    lateral = np.zeros(2)
    if np.any(active):
        side = direction[0] * delta[active, 1] - direction[1] * delta[active, 0]
        left = distance[active][side > 0.0]
        right = distance[active][side <= 0.0]
        d_left = float(left.min()) if left.size else cfg["radius"]
        d_right = float(right.min()) if right.size else cfg["radius"]
        bias = (d_right - d_left) / (d_right + d_left + 1.0e-6)
        perp = np.array([-direction[1], direction[0]], float)
        lateral = -cfg["fuzzy_gain"] * bias * perp * min(float(np.linalg.norm(raw)), 1.0)

    force = raw + lateral
    displacement_steps = state["low_motion_steps"]
    moving_norm = float(np.linalg.norm(force))
    state["low_motion_steps"] = displacement_steps + 1 if moving_norm < 0.035 else 0
    trapped = moving_norm < 0.01 or state["low_motion_steps"] >= 10

    if state["active_target"] is not None:
        state["target_steps"] -= 1
        if np.linalg.norm(pos - state["active_target"]) < 0.45 or state["target_steps"] <= 0:
            state["active_target"] = None
            state["target_steps"] = 0

    if trapped and state["active_target"] is None:
        state["low_motion_steps"] = 0
        gap_cfg = dict(DEFAULT9)
        gap_cfg.update(N=72, p0=3.5, wp_radius=0.3)
        waypoint = gap_waypoint(pos, obs, gap_cfg)
        proposal_goal = goal if waypoint is None else waypoint
        proposal = _sa_propose(pos, proposal_goal, obs, context, state, cfg)
        if proposal is None:
            proposal = waypoint
        if proposal is not None:
            state["active_target"] = np.asarray(proposal, float)
            state["target_steps"] = 24
    return force


def _make_robot(model, start):
    if model == "Holonomic":
        return HolonomicRobot(start, dt=DT, speed_max=V_MAX)
    return Bicycle(start, L=0.1, dt=DT, speed_max=V_MAX, speed_min=-1.0)


def _simulate_one(geometry_key, goal_index, method, model, start_index):
    context = _DATA[geometry_key]
    start = context["starts"][start_index]
    goal = context["goals"][goal_index]
    walls = context["walls"]
    obs = context["obs"]
    g1, g2 = context["g1"], context["g2"]
    mid = context["mid"]
    robot = _make_robot(model, start)
    trail = [robot.pos.copy()] if method == "VRF-APF" else None
    min_clear = _clearance(robot.pos, context)
    hit = None
    simulation_start = time.perf_counter()

    lam = 0.8
    state = {}
    if method == "SOPC-APF":
        cfg = dict(
            DEFAULT,
            Rp=4.0,
            rs=0.5,
            recompute_vgoal=True,
            fr2_dir="-target",
            att_target="current",
            ra_target="real",
        )
        state = {"vgoal": None, "restricted": []}
    elif method == "DIT-APF":
        state = {
            "k_rep": K_REP[0],
            "k_attr": K_ATTR[1],
            "k_inertia": 0.5,
            "k_dist": K_DIST[0],
            "virtual": [],
            "recent_lm": 0.0,
            "hist_pos": [robot.pos.copy()],
            "hist_head": [],
            "vel_hist": [],
        }
    elif method == "RV-APF":
        state = {"wp": None}
        cfg = dict(DEFAULT9)
    elif method == "SA-MAPF":
        seed = (
            (0 if geometry_key == "rect" else 100000)
            + 10000 * goal_index
            + 1000 * METHODS.index(method)
            + start_index
            + (50000 if model == "Simplified bicycle" else 0)
        )
        state = {
            "active_target": None,
            "target_steps": 0,
            "low_motion_steps": 0,
            "temperature": 50.0,
            "rng": np.random.default_rng(seed),
        }
        cfg = dict(
            k_att=1.2,
            k_rep=8.0,
            radius=2.5,
            lookahead=4.0,
            fuzzy_gain=0.65,
        )
    elif method == "VRF-APF":
        pass
    else:
        raise ValueError(method)

    force_steps = 0
    for k in range(MAX_STEPS):
        current_heading = float(
            getattr(robot, "heading", getattr(robot, "theta", 0.0))
        )
        if method == "SOPC-APF":
            force = force_ref8(robot.pos, goal, obs, cfg, state)
        elif method == "DIT-APF":
            if state["recent_lm"] < DT:
                flag, virtual_position = detect_lm(
                    robot.pos, state["hist_pos"], state["hist_head"]
                )
                if flag:
                    state["recent_lm"] = 2.0
                    if virtual_position is not None:
                        state["virtual"].append(np.asarray(virtual_position, float))
            else:
                state["recent_lm"] -= DT
            update_coeffs(
                state,
                DT < state["recent_lm"] <= 2.0,
                robot.pos,
                goal,
            )
            all_obs = (
                np.vstack([obs, np.asarray(state["virtual"])])
                if state["virtual"]
                else obs
            )
            repulsive, tangential = DIT_APF_repulsive(
                robot.pos,
                all_obs,
                state["k_rep"],
                state["k_dist"],
                current_heading,
            )
            attractive = DIT_APF_attractive(robot.pos, goal, state["k_attr"])
            inertia = DIT_APF_inertia(
                state["vel_hist"],
                state["k_inertia"],
                robot.pos,
                goal,
            )
            force = attractive + repulsive + tangential + inertia
        elif method == "RV-APF":
            force = force_ref12(robot.pos, goal, obs, cfg, state)
        elif method == "SA-MAPF":
            force = _sa_force(
                robot.pos,
                current_heading,
                goal,
                obs,
                context,
                cfg,
                state,
            )
        else:
            if k and k % 50 == 0 and len(trail) > 50:
                if np.linalg.norm(robot.pos - trail[-51]) < 0.1 and lam > 0.3:
                    lam = max(0.3, lam - 0.1)
            force = force_proposed(robot.pos, obs, lam)

        previous = robot.pos.copy()
        previous_heading = current_heading
        robot.step(force)
        force_steps += 1
        new_position = robot.pos.copy()
        if trail is not None:
            trail.append(new_position)
        min_clear = min(min_clear, _clearance(new_position, context))

        hit = crosses(previous, new_position, g1, g2)
        if hit is not None:
            break
        if min_clear < R_SAFE:
            break

        if method == "DIT-APF":
            state["hist_pos"].append(new_position.copy())
            state["hist_head"].append(float(previous_heading))
            state["vel_hist"].append(new_position - previous)
            if len(state["vel_hist"]) > int(T_INERTIA / DT):
                state["vel_hist"].pop(0)
    elapsed = time.perf_counter() - simulation_start

    deviation = None if hit is None else float(np.linalg.norm(hit - mid))
    collision = bool(min_clear < R_SAFE)
    escaped = hit is not None
    safe_success = bool(escaped and not collision)
    return {
        "geometry": context["name"],
        "geometry_key": geometry_key,
        "goal_index": goal_index,
        "goal": [float(goal[0]), float(goal[1])],
        "method": method,
        "model": model,
        "start_index": start_index,
        "escaped": escaped,
        "collision": collision,
        "safe_success": safe_success,
        "deviation": deviation,
        "steps": force_steps,
        "elapsed": elapsed,
        "min_clear": float(min_clear),
    }


def _simulate_task(task):
    return _simulate_one(*task)


def _aggregate(detail):
    rows = []
    for geometry_key in _DATA:
        for goal_index in range(5):
            for method in METHODS:
                for model in MODELS:
                    subset = [
                        item
                        for item in detail
                        if item["geometry_key"] == geometry_key
                        and item["goal_index"] == goal_index
                        and item["method"] == method
                        and item["model"] == model
                    ]
                    safe = [item for item in subset if item["safe_success"]]
                    total_steps = sum(item["steps"] for item in subset)
                    total_time = sum(item["elapsed"] for item in subset)
                    rows.append(
                        {
                            "geometry": _DATA[geometry_key]["name"],
                            "geometry_key": geometry_key,
                            "goal_index": goal_index,
                            "goal": _DATA[geometry_key]["goals"][goal_index].tolist(),
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
                                1000.0 * total_time / total_steps if total_steps else None
                            ),
                        }
                    )
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-starts", type=int, default=N_START)
    parser.add_argument("--workers", type=int, default=max(1, min(8, mp.cpu_count() or 1)))
    parser.add_argument(
        "--outdir",
        default=str(
            Path(__file__).resolve().parent
            / "outputs"
            / "appendix_a"
        ),
    )
    args = parser.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    rect_starts, hex_starts = _load_starts()
    rect_seed_context = _rect_context(GAPS[0])
    hex_seed_context = _hex_context(GAPS[0])
    rect_goals = _sample_goals(
        rect_seed_context["walls"],
        rect_seed_context["polygon"],
        GOAL_SEED,
    )
    hex_goals = _sample_goals(
        hex_seed_context["walls"],
        hex_seed_context["polygon"],
        GOAL_SEED + 1,
    )
    for gap_index, gap in enumerate(GAPS):
        rect = _rect_context(gap)
        hx = _hex_context(gap)
        rect["starts"] = rect_starts[: args.n_starts]
        hx["starts"] = hex_starts[: args.n_starts]
        rect["goals"] = [goal.copy() for goal in rect_goals]
        hx["goals"] = [goal.copy() for goal in hex_goals]
        _DATA[f"rect-{int(gap)}"] = rect
        _DATA[f"hex-{int(gap)}"] = hx
    ordered = {
        **{f"rect-{int(gap)}": _DATA[f"rect-{int(gap)}"] for gap in GAPS},
        **{f"hex-{int(gap)}": _DATA[f"hex-{int(gap)}"] for gap in GAPS},
    }
    _DATA.clear()
    _DATA.update(ordered)

    tasks = [
        (geometry_key, goal_index, method, model, start_index)
        for geometry_key in _DATA
        for goal_index in range(5)
        for method in METHODS
        for model in MODELS
        for start_index in range(args.n_starts)
    ]
    print(f"running {len(tasks)} simulations with {args.workers} workers", flush=True)
    context = mp.get_context("fork")
    detail = []
    completed = 0
    with context.Pool(processes=args.workers) as pool:
        for result in pool.imap_unordered(_simulate_task, tasks, chunksize=4):
            detail.append(result)
            completed += 1
            if completed % 500 == 0 or completed == len(tasks):
                print(f"completed {completed}/{len(tasks)}", flush=True)

    detail.sort(
        key=lambda item: (
            item["geometry_key"],
            item["goal_index"],
            METHODS.index(item["method"]),
            MODELS.index(item["model"]),
            item["start_index"],
        )
    )
    rows = _aggregate(detail)

    json_path = os.path.join(args.outdir, "five_methods_goals_results.json")
    csv_path = os.path.join(args.outdir, "five_methods_goals_results.csv")
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(
            {
                "configuration": {
                    "opening_widths_m": list(GAPS),
                    "n_starts": args.n_starts,
                    "max_steps": MAX_STEPS,
                    "dt_s": DT,
                    "v_max_m_s": V_MAX,
                    "collision_clearance_m": R_SAFE,
                    "goal_seed": GOAL_SEED,
                    "methods": list(METHODS),
                    "models": list(MODELS),
                },
                "fixed_goals": {
                    "rect": [goal.tolist() for goal in rect_goals],
                    "hex": [goal.tolist() for goal in hex_goals],
                },
                "goals_by_opening": {
                    key: [goal.tolist() for goal in value["goals"]]
                    for key, value in _DATA.items()
                },
                "summary": rows,
                "detail": detail,
            },
            fh,
            ensure_ascii=False,
            indent=2,
        )
    with open(csv_path, "w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(json_path, flush=True)
    print(csv_path, flush=True)


if __name__ == "__main__":
    main()
