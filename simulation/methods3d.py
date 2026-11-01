# -*- coding: utf-8 -*-
"""Three escape methods generalised to the 3D U-shaped enclosure."""

import numpy as np


def unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else np.zeros_like(v)


def rot_about(v, axis, deg):
    """Rodrigues rotation of v about a unit axis."""
    a = unit(np.asarray(axis, float))
    t = np.deg2rad(deg)
    return (v * np.cos(t) + np.cross(a, v) * np.sin(t)
            + a * np.dot(a, v) * (1 - np.cos(t)))


def vr_step(pos, obs, cfg):
    """Paper method, 3D form (current manuscript, revised 2026-10-03).

    V_xi = P_cur - P_oi                       (Eq. 2.2, step 1; sign corrected)
    Basic form:     F_repi = V_xi / r_xi^3   (Eq. 2.2, step 3; 3D form)
    Improved form:  F_repi = V_xi / (r_xi * (r_xi - lam*r_min)^2)
                                               (Eq. 2.3; revised 3D form)

    formula="new3d" uses the revised 3D formulation in the current manuscript.
    formula="old" retains the previous r^2*(r-lam*rmin) denominator.
    sign=+1 defines V = P_cur - P_oi, i.e. the repulsive-force direction.
    sign=-1 retains the earlier literal vector direction V = P_oi - P_cur.
    """
    V = (pos - obs) if cfg.get("sign", 1) > 0 else (obs - pos)
    r = np.linalg.norm(V, axis=1)
    r = np.maximum(r, 1e-9)
    lam = cfg["lam_rep"]
    if lam <= 0:
        F = (V / r[:, None] ** 3).sum(axis=0)
    elif cfg.get("formula", "new3d") == "new3d":
        rmin = r.min()
        denom = r * np.maximum(r - lam * rmin, 1e-3) ** 2
        F = (V / denom[:, None]).sum(axis=0)
    else:
        rmin = r.min()
        denom = r ** 2 * np.maximum(r - lam * rmin, 1e-3)
        F = (V / denom[:, None]).sum(axis=0)
    return unit(F)


def SOPC_APF_step(pos, goal, obs, cfg):
    """Reference [5] SOPC-APF, 3D generalisation of the deflection rule."""
    d = obs - pos
    r = np.linalg.norm(d, axis=1)
    ab = goal - pos
    na = np.linalg.norm(ab)
    u = ab / max(na, 1e-12)
    dist_line = np.linalg.norm(np.cross(np.tile(u, (len(d), 1)), d), axis=1)
    mask = (r <= cfg["Rp"]) & (dist_line <= cfg["rs"])
    restricted = np.where(mask)[0]
    target = goal
    if restricted.size:
        odir = unit(d[restricted].mean(axis=0))
        axis = np.cross(u, odir)
        if np.linalg.norm(axis) < 1e-6:
            tmp = np.array([0.0, 0.0, 1.0]) if abs(u[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
            axis = np.cross(u, tmp)
        axis = unit(axis)
        sign = -1.0 if np.dot(np.cross(u, odir), axis) >= 0 else 1.0
        rel = d[restricted]
        k = 1
        while k <= 60:
            cand = rot_about(u, axis, sign * 3.0 * k)
            dn = np.linalg.norm(np.cross(np.tile(cand, (len(rel), 1)), rel), axis=1)
            if np.all(dn > cfg["rs"]):
                break
            k += 1
        cand = rot_about(u, axis, sign * 3.0 * k)
        rmax = float(r[restricted].max())
        target = pos + cand * rmax

    F = cfg["ks"] * (target - pos)
    if restricted.size:
        RA = max(np.linalg.norm(target - pos), 1e-9)
        rr = np.maximum(r[restricted], 1e-9)
        keep = rr <= cfg["rho_o"]
        if np.any(keep):
            Rk = rr[keep]
            Mk = np.where(Rk > cfg["rho_o"] / 2, cfg["M"], cfg["M_inner"])
            base = 1.0 / Rk - 1.0 / cfg["rho_o"]
            fr1 = base * RA ** Mk / Rk ** 3
            fr2 = (base ** 2) * RA ** Mk / 2.0
            u_away = (pos - obs[restricted][keep]) / Rk[:, None]
            u_tgt = unit(target - pos)
            F = F + cfg["alpha"] * (fr1[:, None] * u_away).sum(axis=0)
            F = F + cfg["alpha"] * float((Mk * fr2).sum()) * u_tgt
    return F


_SPHERE = None


def sphere_dirs(n=256):
    global _SPHERE
    if _SPHERE is None:
        i = np.arange(n) + 0.5
        phi = np.arccos(1 - 2 * i / n)
        theta = np.pi * (1 + 5 ** 0.5) * i
        _SPHERE = np.stack([np.cos(theta) * np.sin(phi),
                            np.sin(theta) * np.sin(phi),
                            np.cos(phi)], axis=1)
    return _SPHERE


def RV_APF_step(pos, goal, obs, cfg, state):
    xa = unit(goal - pos)
    d = pos - obs
    r = np.linalg.norm(d, axis=1)
    m = r <= cfg["p0"]
    Fr = np.zeros(3)
    if np.any(m):
        Fr = (d[m] / r[m][:, None]).sum(axis=0)
    plain = cfg["gamma"] * xa + Fr
    stuck = bool(np.any(m)) and np.linalg.norm(plain) <= cfg["eps"] * max(np.linalg.norm(Fr), 1.0)

    wp = state.get("wp")
    if wp is not None and np.linalg.norm(pos - wp) <= cfg["wp_radius"]:
        wp = None
        state["wp"] = None
    if wp is None and stuck and cfg["use_waypoint"] and np.any(m):
        idx = np.argsort(r)[: cfg["N"]]
        idx = idx[r[idx] > 1e-6]
        if idx.size:
            dirs = unit(obs[idx] - pos)
            sphere = sphere_dirs()
            cosmax = (sphere @ dirs.T).max(axis=1)
            u = sphere[int(np.argmin(cosmax))]
            align = float((dirs @ u).max())
            if align > 0.05:
                j = idx[int(np.argmax(dirs @ u))]
                d_obs = float(np.linalg.norm(obs[j] - pos))
            else:
                d_obs = 3.0
            wp = pos + u * min(3.0, 0.6 * d_obs)
            state["wp"] = wp

    if wp is not None:
        xt = unit(wp - pos)
        if cfg["vortex"]:
            axis = np.cross(xt, Fr)
            axis = unit(axis) if np.linalg.norm(axis) > 1e-9 else np.array([0.0, 0.0, 1.0])
            F = cfg["gamma"] * xt + rot_about(Fr, axis, 90.0)
        else:
            F = cfg["gamma"] * xt + Fr
    elif stuck and cfg["vortex"]:
        axis = np.cross(xa, Fr)
        axis = unit(axis) if np.linalg.norm(axis) > 1e-9 else np.array([0.0, 0.0, 1.0])
        F = cfg["gamma"] * xa + rot_about(Fr, axis, 90.0)
    else:
        F = plain
    return F


