"""What the PyRoki IK (ik_pyroki.py) needs from a built model: URDFs and collision spheres.

build_model.py calls export() after writing scene.xml. It writes into the model directory:
  ik_arm.urdf    the robot as placed in the clip world (root link "world"), only the 7 arm joints actuated
  ik_full.urdf   the same with the 22 hand joints actuated too, and a link at each fingertip (tip_thumb ...)
  spheres.json   per link, in the link's frame: `spheres` {centers, radii, tolerance} and `hull_points` {points,
                 cover}; `groups` (which links are the moving arm, the hardware rigid with the palm, the fixed body)

Two collision models, because they are used for different things:

  spheres      robot against robot (self-collision: the arm against the torso, head, parked arm and its own links).
               Each link's geometry is replaced by its convex hull and covered with a handful of spheres: centres are
               points inside the hull, each with radius (its depth below the hull's surface + tolerance), chosen
               greedily until every point of the surface is inside a sphere; the tolerance grows until at most the
               allowed number of spheres is needed. Few spheres keep the pair count small; they overshoot the
               hull by 1-3 cm on the arm, which only makes the arm keep a little more distance from the body.
  hull_points  robot against the table. The clips leave millimetres between the wrist hardware and the table, so
               spheres that overshoot by centimetres would invent collisions. A convex shape is above a plane
               exactly when all corners of its hull are, so each moving link keeps its hull's corners (thinned to
               at most `max_points`; `cover` is how far a dropped corner can be from a kept one, and is added to the
               margin). The camera housing's eight corners are exact.

Fingers get no spheres (they are supposed to touch things), but their hull corners are kept too: when the solver
is allowed to re-bend them it must not push them deeper into the table than FDR's own fingers were.
"""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from scipy.spatial import ConvexHull

import common as C


def _rpy(R):
    """Rotation matrix -> URDF roll-pitch-yaw (R = Rz(yaw) Ry(pitch) Rx(roll)), also at pitch = +-90 deg, where the
    usual formulas divide noise by noise (the Marvin's wrist joints sit exactly there)."""
    if np.hypot(R[2, 1], R[2, 2]) < 1e-9:                           # gimbal lock: put all of the twist into roll
        rpy = np.array([np.arctan2(-R[1, 2], R[1, 1]), -np.sign(R[2, 0]) * np.pi / 2, 0.0])
    else:
        rpy = np.array([np.arctan2(R[2, 1], R[2, 2]), np.arctan2(-R[2, 0], np.hypot(R[2, 1], R[2, 2])), np.arctan2(R[1, 0], R[0, 0])])
    assert np.allclose(C.rpy_mat(rpy), R, atol=1e-8), (rpy, R)
    return rpy


def fit_spheres(P, tol, max_spheres):
    """Cover the convex hull of points P with spheres that exceed it by at most tol (see the module docstring).
    Returns (centres (n, 3), radii (n,), the tolerance actually used)."""
    import trimesh
    hull = ConvexHull(P)
    A, b = hull.equations[:, :3], hull.equations[:, 3]             # inside: A x + b <= 0
    lo, hi = P.min(0), P.max(0)
    ext = hi - lo
    step = float(np.clip(ext.min() / 4, 0.0025, 0.012))
    grid = np.stack(np.meshgrid(*[np.arange(l + step / 2, h, step) for l, h in zip(lo, hi)], indexing="ij"), -1).reshape(-1, 3)
    depth = -(grid @ A.T + b).max(1)
    grid, depth = grid[depth > 0], depth[depth > 0]
    if len(grid) == 0:                                              # thinner than the grid: use the hull's centre
        grid, depth = P.mean(0)[None], np.array([max(float(-(P.mean(0) @ A.T + b).max()), 0.0)])
    tm = trimesh.Trimesh(P[hull.vertices], ConvexHull(P[hull.vertices]).simplices, process=True)
    surf, _ = trimesh.sample.sample_surface(tm, 1200, seed=0)
    targets = np.concatenate([np.asarray(surf), P[hull.vertices]])
    D = np.linalg.norm(targets[:, None] - grid[None], axis=2)       # (targets, candidates)
    while True:
        cover = D <= depth[None] + tol
        left = np.ones(len(targets), bool)
        chosen = []
        while left.any() and len(chosen) < max_spheres:
            gain = cover[left].sum(0)
            k = int(np.argmax(gain + 1e-3 * depth / depth.max()))  # most new points; ties to the deeper centre
            if gain[k] == 0:
                break
            chosen.append(k)
            left &= ~cover[:, k]
        if not left.any():
            return grid[chosen], depth[chosen] + tol, tol
        tol *= 1.35


def hull_corners(P, max_points):
    """Corners of P's convex hull, thinned by farthest-point sampling. Returns (points, cover radius)."""
    V = P[ConvexHull(P).vertices]
    if len(V) <= max_points:
        return V, 0.0
    keep = [int(np.argmax(np.linalg.norm(V - V.mean(0), axis=1)))]
    dist = np.linalg.norm(V - V[keep[0]], axis=1)
    while len(keep) < max_points:
        k = int(np.argmax(dist))
        keep.append(k)
        dist = np.minimum(dist, np.linalg.norm(V - V[k], axis=1))
    return V[keep], float(dist.max())


def body_points(m, d, b, groups=(1,)):
    """Surface points of body b's visible geoms, in the body's frame."""
    import trimesh
    Rb, pb = d.xmat[b].reshape(3, 3), d.xpos[b]
    out = []
    for g in range(m.ngeom):
        if m.geom_bodyid[g] != b or m.geom_group[g] not in groups:
            continue
        t = m.geom_type[g]
        if t == mujoco.mjtGeom.mjGEOM_MESH:
            i = m.geom_dataid[g]
            v = m.mesh_vert[m.mesh_vertadr[i]: m.mesh_vertadr[i] + m.mesh_vertnum[i]].astype(float)
            if len(v) > 6000:
                v = v[np.random.default_rng(0).choice(len(v), 6000, replace=False)]
        elif t == mujoco.mjtGeom.mjGEOM_BOX:
            v = np.array([[a, bb, c] for a in (-1, 1) for bb in (-1, 1) for c in (-1, 1)], float) * m.geom_size[g]
        elif t == mujoco.mjtGeom.mjGEOM_CYLINDER:
            th = np.linspace(0, 2 * np.pi, 24, endpoint=False)
            r, h = m.geom_size[g][:2]
            v = np.array([[r * np.cos(a), r * np.sin(a), z] for a in th for z in (-h, h)])
        else:
            continue
        w = v @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g]
        out.append((w - pb) @ Rb)
    return np.concatenate(out) if out else np.zeros((0, 3))


def link_groups(m):
    """Body ids: `arm` (moved by arm_j1..6 only), `wrist` (rigid with the palm: last arm link, mount, palm, camera),
    `fingers` (below the palm, with joints), `fixed` (everything that does not move)."""
    name = lambda b: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b)  # noqa: E731
    jb = {k: m.jnt_bodyid[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"arm_j{k}")] for k in (1, 7)}
    palm = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, C.PALM_BODY)

    def under(b, top):
        while b > 0:
            if b == top:
                return True
            b = m.body_parentid[b]
        return False
    g = {"arm": [], "wrist": [], "fingers": [], "fixed": []}
    for b in range(1, m.nbody):
        if name(b).endswith("_optical") and not under(b, palm):
            continue                                               # the exo camera is not part of the robot
        if not under(b, jb[1]):
            g["fixed"].append(b)
        elif not under(b, jb[7]):
            g["arm"].append(b)
        elif under(b, palm) and b != palm and (m.body_jntnum[b] > 0 or any(m.body_jntnum[p] > 0 for p in _chain(m, b, palm))):
            g["fingers"].append(b)
        else:
            g["wrist"].append(b)
    return g


def _chain(m, b, top):
    out = []
    while b != top and b > 0:
        out.append(b)
        b = m.body_parentid[b]
    return out


def write_urdf(m, d, path, actuated, tips, vel):
    """The compiled MuJoCo model's kinematic tree as a URDF without geometry. Joints not in `actuated` become fixed."""
    name = lambda b: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) if b else "world"  # noqa: E731
    robot = ET.Element("robot", name=path.stem)
    ET.SubElement(robot, "link", name="world")
    for b in range(1, m.nbody):
        ET.SubElement(robot, "link", name=name(b))
        jname, jtype, axis, rng = f"fixed_{name(b)}", "fixed", None, None
        if m.body_jntnum[b] == 1:
            j = m.body_jntadr[b]
            jn = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j)
            assert np.allclose(m.jnt_pos[j], 0), jn
            if jn in actuated:
                jname, jtype, axis, rng = jn, "revolute", m.jnt_axis[j], m.jnt_range[j]
        J = ET.SubElement(robot, "joint", name=jname, type=jtype)
        ET.SubElement(J, "parent", link=name(m.body_parentid[b]))
        ET.SubElement(J, "child", link=name(b))
        ET.SubElement(J, "origin", xyz=C.fmt(m.body_pos[b], 12), rpy=C.fmt(_rpy(C.quat2mat(m.body_quat[b])), 12))
        if jtype == "revolute":
            ET.SubElement(J, "axis", xyz=C.fmt(axis, 12))
            ET.SubElement(J, "limit", lower=f"{rng[0]:.8f}", upper=f"{rng[1]:.8f}", velocity=f"{vel.get(jname, 10.0):.6f}", effort="100")
    for tip, s in tips.items():
        ET.SubElement(robot, "link", name=tip)
        J = ET.SubElement(robot, "joint", name=f"fixed_{tip}", type="fixed")
        ET.SubElement(J, "parent", link=name(m.site_bodyid[s]))
        ET.SubElement(J, "child", link=tip)
        ET.SubElement(J, "origin", xyz=C.fmt(m.site_pos[s], 12), rpy=C.fmt(_rpy(C.quat2mat(m.site_quat[s])), 12))
    ET.indent(robot, space="  ")
    path.write_text('<?xml version="1.0"?>\n' + ET.tostring(robot, encoding="unicode") + "\n")


def export(arm_name, out):
    cfg = C.load_arm(arm_name)
    m = mujoco.MjModel.from_xml_path(str(out / "scene.xml"))
    d = mujoco.MjData(m)
    mujoco.mj_kinematics(m, d)
    name = lambda b: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b)  # noqa: E731
    arm_j = [f"arm_j{k + 1}" for k in range(7)]
    hand_j = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) for j in range(m.njnt)
              if mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) not in arm_j]
    vel = dict(zip(arm_j, np.radians(cfg["velocity_deg_s"])))
    tips = {f"tip_{f}": mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, f"track_hand_right_{f}_tip") for f in C.FINGERS}
    write_urdf(m, d, out / "ik_arm.urdf", set(arm_j), {}, vel)
    write_urdf(m, d, out / "ik_full.urdf", set(arm_j + hand_j), tips, vel)
    groups = link_groups(m)
    spec = {"wrist": (0.006, 10, 400), "arm": (0.010, 8, 150), "fixed": (0.020, 8, 0)}
    far = ("wheel", "lidar", "base_cam")                             # fixed parts the arm cannot reach
    spheres, hull_points, n = {}, {}, 0
    for gname, (tol, kmax, max_points) in spec.items():
        for b in groups[gname]:
            if gname == "fixed" and any(k in name(b) for k in far):
                continue
            P = body_points(m, d, b)
            if len(P) < 8:
                continue
            c, r, used = fit_spheres(P, tol, kmax)
            spheres[name(b)] = {"centers": c.round(5).tolist(), "radii": r.round(5).tolist(), "tolerance": round(float(used), 4), "group": gname}
            n += len(r)
            if max_points:
                V, cover = hull_corners(P, max_points)
                hull_points[name(b)] = {"points": V.round(5).tolist(), "cover": round(cover, 5), "group": gname}
    for b in groups["fingers"]:                                      # the fingers' collision meshes (geom group 0)
        P = body_points(m, d, b, groups=(0,))
        if len(P) >= 8:
            V, cover = hull_corners(P, 60)
            hull_points[name(b)] = {"points": V.round(5).tolist(), "cover": round(cover, 5), "group": "fingers"}
    info = {"arm": arm_name, "frame": "each link's own frame (the MuJoCo body frame); metres", "spheres": spheres,
            "hull_points": hull_points,
            "groups": {k: [name(b) for b in v] for k, v in groups.items()}, "arm_joints": arm_j, "hand_joints": hand_j,
            "palm_link": C.PALM_BODY, "tip_links": list(tips)}
    json.dump(info, open(out / "spheres.json", "w"), indent=1)
    return info, n


if __name__ == "__main__":
    import sys
    info, n = export(sys.argv[1], C.model_dir(sys.argv[1]))
    for k, v in info["spheres"].items():
        print(f"  {k:26s} {v['group']:6s} {len(v['radii']):3d} spheres, r {min(v['radii']) * 1000:.0f}-{max(v['radii']) * 1000:.0f} mm, tolerance {v['tolerance'] * 1000:.1f} mm")
    print(f"{n} spheres on {len(info['spheres'])} links")
    for k, v in info["hull_points"].items():
        print(f"  {k:26s} {v['group']:6s} {len(v['points']):3d} hull corners, cover {v['cover'] * 1000:.1f} mm")
