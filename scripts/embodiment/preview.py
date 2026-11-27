"""Shaded pictures of a scene without OpenGL: rays through MuJoCo (mj_multiRay), colour = geom colour x a simple light.

  python scripts/embodiment/preview.py --arm marvin_m6ccs [--out DIR]      robot at its home posture, several views
Used by the other tools for figures (render_scene()).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import mujoco
import numpy as np

import common as C

GROUPS = np.array([0, 1, 1, 0, 0, 0], np.uint8)


def look_at(eye, target, up=(0, 0, 1)):
    """Optical frame (x right, y down, z forward) at `eye` looking at `target`."""
    eye, target = np.asarray(eye, float), np.asarray(target, float)
    z = target - eye
    z /= np.linalg.norm(z)
    x = np.cross(z, up)
    x /= np.linalg.norm(x)
    return C.make_T(np.stack([x, np.cross(z, x), z], 1), eye)


def render_scene(model, data, T_world_cam, width=640, height=480, fov_deg=45.0, bg=(1.0, 1.0, 1.0), extra_groups=(), bodyexclude=-1):
    """fov_deg is the horizontal field of view; bodyexclude hides one body (a camera's own housing)."""
    f = width / 2 / np.tan(np.radians(fov_deg) / 2)
    u = (np.arange(width) + 0.5 - width / 2) / f
    v = (np.arange(height) + 0.5 - height / 2) / f
    d = np.stack(np.broadcast_arrays(u[None, :], v[:, None], 1.0), -1).reshape(-1, 3)
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    R, o = T_world_cam[:3, :3], T_world_cam[:3, 3]
    vec = np.ascontiguousarray(d @ R.T)
    n = len(vec)
    geom, dist, normal = np.full(n, -1, np.int32), np.full(n, -1.0), np.zeros((n, 3))
    groups = GROUPS.copy()
    for g in extra_groups:
        groups[g] = 1
    mujoco.mj_multiRay(model, data, o.copy(), vec.reshape(-1), groups, 1, bodyexclude, geom, dist, normal.reshape(-1), n, 20.0)
    hit = geom >= 0
    rgba = model.geom_rgba[np.clip(geom, 0, None)].copy()
    mat = model.geom_matid[np.clip(geom, 0, None)]
    use_mat = (mat >= 0) & np.all(np.isclose(rgba, [0.5, 0.5, 0.5, 1]), axis=1)
    rgba[use_mat] = model.mat_rgba[mat[use_mat]]
    light = np.array([0.35, -0.45, 0.82])
    light /= np.linalg.norm(light)
    nn = np.where(np.einsum("ij,ij->i", normal, vec)[:, None] > 0, -normal, normal)       # face the camera
    shade = 0.42 + 0.58 * np.clip(nn @ light, 0, 1)
    img = np.where(hit[:, None], rgba[:, :3] * shade[:, None], np.asarray(bg))
    return (np.clip(img, 0, 1).reshape(height, width, 3) * 255).astype(np.uint8)


def main():
    from PIL import Image
    from scene import Scene
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    cfg = C.load_arm(args.arm)
    sc = Scene(args.arm)
    q = np.radians(cfg["home_deg"])
    sc.set_arm(q, np.zeros(22))
    sc.fk()
    Tp = sc.palm()
    out = C.model_dir(args.arm) if args.out is None else Path(args.out)
    p = Tp[:3, 3]
    wrist = p - Tp[:3, 2] * 0.03
    views = {
        "overview": (look_at([-0.9, -1.6, 1.9], [0.25, -0.1, 0.95]), 50.0, (900, 700)),
        "wrist_palm_side": (look_at(wrist + Tp[:3, :3] @ np.array([0.30, 0.16, 0.10]), wrist + Tp[:3, :3] @ np.array([0.02, 0, 0.03]), up=-Tp[:3, 0]), 40.0, (800, 640)),
        "wrist_side": (look_at(wrist + Tp[:3, :3] @ np.array([0.03, 0.38, 0.0]), wrist + Tp[:3, :3] @ np.array([0.02, 0, 0.02]), up=-Tp[:3, 0]), 40.0, (800, 640)),
        "wrist_from_fingers": (look_at(p + Tp[:3, :3] @ np.array([0.10, 0.03, 0.42]), p + Tp[:3, :3] @ np.array([0.03, 0, 0.0]), up=-Tp[:3, 0]), 35.0, (800, 640)),
    }
    for name, (T, fov, (w, h)) in views.items():
        Image.fromarray(render_scene(sc.model, sc.data, T, w, h, fov)).save(out / f"view_{name}.png")
        print(out / f"view_{name}.png")


if __name__ == "__main__":
    main()
