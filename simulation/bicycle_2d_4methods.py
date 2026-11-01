# -*- coding: utf-8 -*-
"""Two-dimensional comparison routines using the simplified bicycle kinematics from reference [11]."""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

STEP = 0.1

# ---- DIT-APF parameters from reference [11] (pathplanner.py main6 / FDAPF) ----
ROBOT_SIZE = 0.1
LOOKAHEAD = 4.0
K_REP = (1.0, 4.0)
K_ATTR = (0.2, 1.0)
K_DIST = (2.0, 4.0)
K_INER = (0.0, 1.0)
T_INERTIA = 1.0
LM_VRANGE = 2.0
LM_TURNING = 2 * np.pi
LM_TIME = 6.0
LM_PREC = 0.1


def _wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def _angle_diff(a1, a2):
    return _wrap(a2 - a1)


def DIT_APF_repulsive(pos, obs, k_rep, k_dist, heading):
    d = pos - obs
    r = np.linalg.norm(d, axis=1)
    eff = r - ROBOT_SIZE
    m = (eff > 0) & (r < k_dist + ROBOT_SIZE)
    rep = np.zeros(2)
    tan = np.zeros(2)
    if np.any(m):
        dm, rm, em = d[m], r[m], eff[m]
        g = (1.0 / em - 1.0 / k_dist)
        nf = k_rep * (g * (1.0 / em) ** 2 * (em / rm))[:, None] * dm
        rep = nf.sum(axis=0)
        cdir = np.array([np.cos(heading), np.sin(heading)])
        cross = cdir[0] * (-dm[:, 1]) - cdir[1] * (-dm[:, 0])
        sgn = np.where(cross >= 0, 1.0, -1.0)
        nt = np.column_stack((-sgn * nf[:, 1], sgn * nf[:, 0]))
        tan = nt.sum(axis=0)
    return rep, tan


def DIT_APF_attractive(pos, goal, k_attr):
    a = k_attr * (goal - pos)
    na = np.linalg.norm(a)
    if na > LOOKAHEAD:
        a = a / na * LOOKAHEAD
    return a


def DIT_APF_inertia(vel_hist, k_inertia, pos, goal):
    if not vel_hist:
        return np.zeros(2)
    inert = np.mean(vel_hist, axis=0) * k_inertia
    dg = np.linalg.norm(goal - pos)
    if dg < LOOKAHEAD:
        inert = inert * (dg / LOOKAHEAD)
    return inert


def detect_lm(pos, hist_pos, hist_head):
    nb = int(LM_TIME / STEP)
    if len(hist_pos) <= nb + 1:
        return False, None
    pos_start = len(hist_pos) - nb - 1
    pos_end = len(hist_pos) - 2
    for i in range(pos_start, pos_end):
        if np.linalg.norm(pos - hist_pos[i]) < LM_PREC and i < pos_end / 2:
            return True, hist_pos[i]
    mean_pose = np.mean(hist_pos[pos_start + 1:pos_end + 1], axis=0)
    sumt = 0.0
    for i in range(pos_start, pos_end):
        sumt += abs(_angle_diff(hist_head[i], hist_head[i + 1]))
    if np.linalg.norm(pos - mean_pose) < LM_VRANGE and sumt > LM_TURNING:
        return True, mean_pose
    return False, None


def update_coeffs(state, in_lm, pos, goal):
    dt = STEP
    if in_lm:
        state["k_rep"] = min(state["k_rep"] * (1 + 0.5 * dt), K_REP[1])
        state["k_attr"] = max(state["k_attr"] * (1 - 0.1 * dt), K_ATTR[0])
        state["k_inertia"] = max(state["k_inertia"] * (1 - 0.02 * dt), K_INER[0])
        state["k_dist"] = min(state["k_dist"] * (1 + 0.1 * dt), K_DIST[1])
    else:
        state["k_rep"] = max(state["k_rep"] * (1 - 0.5 * dt), K_REP[0])
        state["k_attr"] = min(state["k_attr"] * (1 + 0.5 * dt), K_ATTR[1])
        state["k_inertia"] = min(state["k_inertia"] * (1 + 0.02 * dt), K_INER[1])
        state["k_dist"] = max(state["k_dist"] * (1 - 1.0 * dt), K_DIST[0])
    dg = np.linalg.norm(goal - pos)
    if dg < state["k_dist"]:
        state["k_rep"] *= (dg / state["k_dist"]) ** 2


