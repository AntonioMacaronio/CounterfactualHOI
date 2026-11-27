"""Build the robot the §1.4 policy controls: a 7-DoF arm with the Sharpa Wave right hand on its flange and the D405
on the wrist, as one MuJoCo model.

  python scripts/embodiment/build_model.py --arm marvin_m6ccs      (venv: $PROJECT_STORAGE_ROOT/venvs/arm)
  python scripts/embodiment/build_model.py --arm vega_1p [--no-check]

Inputs: configs/embodiment/arms/<arm>.yaml (the vendor URDF, which joints are the arm, the flange frame, limits),
configs/embodiment/cameras.yaml (the wrist camera in the palm frame, the exo camera in the world), FlashDexRetarget's
Sharpa model (the hand keeps FDR's body, joint, site and actuator names, so FDR joint vectors map 1:1), the Sharpa
flange and wrist meshes, and the camera mount's mesh when d405_mount.py has written it.

Writes $PROJECT_STORAGE_ROOT/build/embodiment/<arm>_sharpa_r/:
  robot.xml      the robot with its root link at the origin: 7 arm joints (arm_j1..arm_j7) + the 22 hand joints
  scene.xml      the same, placed in the clip world (placement in the arm's config) with the table and the exo camera
  frames.json    joint names and limits, the flange -> palm transform, camera frames
  wrist_stub/    what is rigidly behind the palm (the arm's last link, the mount, the D405) in the palm frame, as
                 collision shapes for FDR's floating hand
  ik_arm.urdf, ik_full.urdf, spheres.json    the placed robot as URDF and its collision model, for ik_pyroki.py
                 (spheres.py)
  meshes/        vendor meshes converted to STL

Chain at the wrist:  last arm link -> arm_flange (site, the vendor's flange frame)
                     -> hand_mount (body; the Sharpa flange's frame F, plate_mm further out and turned by yaw_deg)
                     -> R_forearm_rot_y_link (the palm; F + 29.5 mm, from Sharpa's with_flange URDF)
The checks at the end compare the model's forward kinematics with the URDF's numbers and, where the arm's config has
D-H rows, with those.
"""

from __future__ import annotations

import argparse
import copy
import json
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

import common as C

SHARPA_FLANGE_TO_WRIST = 0.0005      # right_hand_flange -> right_hand_wrist (Sharpa's with_flange URDF)
SHARPA_WRIST_TO_PALM = 0.029         # right_hand_wrist -> right_hand_C_MC
SHARPA_FLANGE_TO_PALM = SHARPA_FLANGE_TO_WRIST + SHARPA_WRIST_TO_PALM
SHARPA_UP = C.VENDOR / "sharpa_upstream" / "right_sharpa_wave"


def pose_attrs(el, T):
    el.set("pos", C.fmt(T[:3, 3]))
    q = C.mat2quat(T[:3, :3])
    if not np.allclose(q, [1, 0, 0, 0], atol=1e-9):
        el.set("quat", C.fmt(q, 9))


def convert_mesh(src, dst, scale):
    """Any mesh trimesh reads (STL, OBJ, GLB with its node transforms) -> binary STL(s) in metres, one per colour.

    A GLB keeps its parts' colours: its primitives are grouped by their material's base colour and each group becomes
    one STL (dst for the largest group, dst's stem + __c<k> for the others). STL and OBJ files have no usable colour
    (the SolidWorks exports colour links red / green / blue to tell them apart) and give one part. Returns a list of
    (path, rgba in 0..1 or None, faces)."""
    import trimesh
    sc = trimesh.load(src, process=False)
    groups = {}
    if isinstance(sc, trimesh.Scene):
        for node in sc.graph.nodes_geometry:
            T, gname = sc.graph[node]
            g = sc.geometry[gname]
            mat = getattr(g.visual, "material", None)
            col = getattr(mat, "baseColorFactor", None) if mat is not None else None
            if col is None and mat is not None and getattr(mat, "main_color", None) is not None:
                col = mat.main_color
            key = tuple(int(c) for c in np.asarray(col).ravel()[:4]) if col is not None else None
            groups.setdefault(key, []).append((trimesh.transform_points(np.asarray(g.vertices), T), np.asarray(g.faces)))
    else:
        groups[None] = [(np.asarray(sc.vertices), np.asarray(sc.faces))]
    parts = []
    for key, pieces in groups.items():
        V, F, off = [], [], 0
        for v, f in pieces:
            V.append(v)
            F.append(f + off)
            off += len(v)
        m = trimesh.Trimesh(np.concatenate(V) * np.asarray(scale), np.concatenate(F), process=True)
        if len(m.faces):
            parts.append((key, m))
    parts.sort(key=lambda kv: -len(kv[1].faces))
    dst.parent.mkdir(parents=True, exist_ok=True)
    out = []
    for k, (key, m) in enumerate(parts):
        path = dst if k == 0 else dst.with_name(f"{dst.stem}__c{k}.stl")
        m.export(path)
        rgba = None if key is None else (np.array(key + (255,) * (4 - len(key)), float)[:4] / 255.0)
        out.append((path, rgba, len(m.faces)))
    return out


def resolve_mesh(urdf_path, filename):
    if filename.startswith("package://"):                       # SolidWorks export: package://<pkg>/meshes/<file>
        return urdf_path.parent.parent / "meshes" / filename.split("/")[-1]
    return (urdf_path.parent / filename).resolve()


def arm_tree(cfg, out, world, assets):
    """The vendor URDF as MJCF bodies under `world`. Arm joints become arm_j1..7; every other joint is frozen at
    cfg['fixed_joints'] (its rotation folded into the body pose). Returns {link: body element}."""
    urdf_path = C.VENDOR / cfg["urdf"]
    links, joints, root = C.read_urdf(urdf_path)
    assert root == cfg["root_link"], (root, cfg["root_link"])
    arm = {n: i for i, n in enumerate(cfg["arm_joints"])}
    fixed = cfg.get("fixed_joints") or {}
    lo, hi = np.radians(cfg["limits_deg"]["lower"]), np.radians(cfg["limits_deg"]["upper"])
    by_parent = {}
    for n, j in joints.items():
        by_parent.setdefault(j["parent"], []).append(n)
    bodies, n_faces = {}, 0

    def add_link(parent_el, link, T, joint_name):
        nonlocal n_faces
        b = ET.SubElement(parent_el, "body", name=link)
        pose_attrs(b, T)
        bodies[link] = b
        inert = links[link]["inertial"]
        moving = joint_name in arm
        if inert and inert["I"] and inert["mass"] > 0:
            i = ET.SubElement(b, "inertial", pos=C.fmt(inert["xyz"]), mass=C.fmt(inert["mass"]), fullinertia=C.fmt(inert["I"], 8))
            if np.any(inert["rpy"]):
                i.set("quat", C.fmt(C.mat2quat(C.rpy_mat(inert["rpy"])), 9))
        elif moving:
            ET.SubElement(b, "inertial", pos="0 0 0", mass="0.01", diaginertia="1e-5 1e-5 1e-5")
        if moving:
            j = joints[joint_name]
            k = arm[joint_name]
            ET.SubElement(b, "joint", name=f"arm_j{k + 1}", axis=C.fmt(j["axis"]), range=f"{lo[k]:.6f} {hi[k]:.6f}",
                          **{"class": "arm"})
        for vi, v in enumerate(links[link]["visuals"]):
            name = f"{link}_{vi}" if vi else link
            parts = convert_mesh(resolve_mesh(urdf_path, v["mesh"]), out / "meshes" / "arm" / f"{name}.stl", v["scale"])
            for path, rgba, nf in parts:
                pn = f"arm_{path.stem}"                          # arm_<link>, arm_<link>__c1, ... (one per colour)
                n_faces += nf
                # inertia="shell": the geoms are visual only (density 0), and some colour parts are flat
                ET.SubElement(assets, "mesh", name=pn, file=f"arm/{path.name}", inertia="shell")
                g = ET.SubElement(b, "geom", name=pn, type="mesh", mesh=pn, **{"class": "arm_visual"})
                if rgba is not None:
                    g.set("rgba", C.fmt(rgba, 4))
                pose_attrs(g, C.make_T(C.rpy_mat(v["rpy"]), v["xyz"]))
        for cn in by_parent.get(link, []):
            j = joints[cn]
            rpy = C.snap_rpy(j["rpy"]) if cfg.get("snap_rpy", True) else j["rpy"]
            Tc = C.make_T(C.rpy_mat(rpy), j["xyz"])
            if cn not in arm and j["type"] != "fixed":          # frozen joint: fold its angle into the pose
                v = float(fixed.get(cn, 0.0))
                if j["type"] in ("revolute", "continuous"):
                    Tc = Tc @ C.make_T(C.axis_angle(j["axis"], v))
                else:
                    Tc = Tc @ C.make_T(None, j["axis"] / np.linalg.norm(j["axis"]) * v)
            add_link(b, j["child"], Tc, cn)

    add_link(world, root, np.eye(4), None)
    return bodies, n_faces


def import_hand(mj, bodies_parent, assets, defaults, out):
    """FDR's right Sharpa hand (palm body and below) appended to `bodies_parent`; meshes copied; defaults wrapped in
    class 'sharpa'. Returns (actuator elements, contact exclude elements, hand joint names)."""
    src_dir = C.FDR / "assets" / "robot" / "sharpa"
    root = ET.parse(src_dir / "right.xml").getroot()
    d = ET.SubElement(defaults, "default", **{"class": "sharpa"})
    for ch in root.find("default"):
        d.append(copy.deepcopy(ch))
    for m in root.find("asset").findall("mesh"):
        f = m.get("file")
        dst = out / "meshes" / "hand" / f
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src_dir / "meshes" / f, dst)
        ET.SubElement(assets, "mesh", name=m.get("name"), file=f"hand/{f}")
    palm = next(b for b in root.iter("body") if b.get("name") == C.PALM_BODY)
    palm = copy.deepcopy(palm)
    for a in ("pos", "quat"):
        palm.attrib.pop(a, None)
    palm.remove(palm.find("joint"))                              # the floating wrist's last joint
    palm.set("childclass", "sharpa")
    for b in palm.iter("body"):
        b.attrib.pop("gravcomp", None)                           # FDR's floating hand is partly gravity-compensated
    palm.attrib.pop("gravcomp", None)
    palm.set("pos", C.fmt([0, 0, SHARPA_FLANGE_TO_PALM]))
    bodies_parent.append(palm)
    hand_joints = [j.get("name") for j in palm.iter("joint")]
    acts = []
    for a in root.find("actuator"):
        if a.get("joint") in hand_joints:
            a = copy.deepcopy(a)
            a.set("class", "sharpa")
            acts.append(a)
    excl = [copy.deepcopy(e) for e in root.find("contact")]
    return palm, acts, excl, hand_joints


def camera_el(parent, name, cam, sensor, T_parent_optical, housing=True):
    """A body at the optical frame with the housing box and a MuJoCo camera (which looks along its -z, y up)."""
    b = ET.SubElement(parent, "body", name=f"{name}_optical")
    pose_attrs(b, T_parent_optical)
    k = sensor["intrinsics"]
    fovy = np.degrees(2 * np.arctan(sensor["stream"]["height"] / 2 / k["fy"]))
    ET.SubElement(b, "camera", name=name, quat="0 1 0 0", fovy=f"{fovy:.4f}",
                  resolution=f"{sensor['stream']['width']} {sensor['stream']['height']}")
    h = sensor.get("housing")
    if housing and h:
        ET.SubElement(b, "inertial", pos=C.fmt(np.array(h["centre_mm"]) / 1000), mass=f"{h['mass_g'] / 1000:.4f}",
                      diaginertia="2e-5 2e-5 2e-5")
        ET.SubElement(b, "geom", name=f"{name}_housing", type="box", pos=C.fmt(np.array(h["centre_mm"]) / 1000),
                      size=C.fmt(np.array(h["size_mm"]) / 2000), **{"class": "mount"}, rgba="0.75 0.76 0.78 1")
        for i, sx in enumerate((0.0, sensor["baseline_m"])):     # the two lenses
            ET.SubElement(b, "geom", name=f"{name}_lens{i}", type="cylinder", size="0.0045 0.0006",
                          pos=C.fmt([sx, 0, (h["centre_mm"][2] + h["size_mm"][2] / 2) / 1000 + 0.0003]),
                          **{"class": "mount"}, rgba="0.05 0.05 0.06 1")
    return b


def flange_to_hand_mount(cfg):
    """Sharpa flange frame F in the arm's flange frame: plate_mm further out along z, turned by yaw_deg about z."""
    hm = cfg["hand_mount"]
    return C.make_T(C.rot_z(np.radians(hm["yaw_deg"])), [0, 0, hm.get("plate_mm", 0) / 1000])


def build(arm_name, with_scene=True):
    cfg = C.load_arm(arm_name)
    cams = C.load_cameras()
    out = C.model_dir(arm_name)
    (out / "meshes").mkdir(parents=True, exist_ok=True)
    mj = ET.Element("mujoco", model=f"{arm_name}_sharpa_r")
    ET.SubElement(mj, "compiler", angle="radian", meshdir="meshes", balanceinertia="true", autolimits="true")
    ET.SubElement(mj, "option", timestep="0.002", integrator="implicitfast")
    defaults = ET.SubElement(mj, "default")
    d_arm = ET.SubElement(defaults, "default", **{"class": "arm"})
    ET.SubElement(d_arm, "joint", damping="2", armature="0.1", frictionloss="0.2")
    ET.SubElement(d_arm, "position", kp="400", kv="40", inheritrange="1")
    d_av = ET.SubElement(defaults, "default", **{"class": "arm_visual"})
    ET.SubElement(d_av, "geom", contype="0", conaffinity="0", group="1", density="0", rgba="0.86 0.87 0.9 1")
    d_m = ET.SubElement(defaults, "default", **{"class": "mount"})
    ET.SubElement(d_m, "geom", contype="0", conaffinity="0", group="1", density="0", rgba="0.25 0.27 0.3 1")
    assets = ET.SubElement(mj, "asset")
    world = ET.SubElement(mj, "worldbody")

    bodies, n_faces = arm_tree(cfg, out, world, assets)
    fl = cfg["flange"]
    T_link_flange = C.cfg_T(fl)
    ET.SubElement(bodies[fl["link"]], "site", name="arm_flange", size="0.004", rgba="0 0.6 1 1", group="4",
                  pos=C.fmt(T_link_flange[:3, 3]), quat=C.fmt(C.mat2quat(T_link_flange[:3, :3]), 9))
    T_flange_F = flange_to_hand_mount(cfg)
    hm = ET.SubElement(bodies[fl["link"]], "body", name="hand_mount")
    pose_attrs(hm, T_link_flange @ T_flange_F)
    plate = cfg["hand_mount"].get("plate_mm", 0) / 1000
    mount_stl = C.BUILD / "mount" / arm_name / "d405_mount.stl"
    mount_mass = 0.016 + 0.0505                                  # Sharpa flange + wrist (its URDF)
    if mount_stl.exists():
        (out / "meshes" / "mount").mkdir(exist_ok=True)
        shutil.copyfile(mount_stl, out / "meshes" / "mount" / "d405_mount.stl")
        ET.SubElement(assets, "mesh", name="d405_mount", file="mount/d405_mount.stl")
        ET.SubElement(hm, "geom", name="d405_mount", type="mesh", mesh="d405_mount", **{"class": "mount"})
        info = mount_stl.with_suffix(".json")
        if info.exists():
            mount_mass += json.load(open(info)).get("mass_kg", 0.0)
    elif plate > 0:
        ET.SubElement(hm, "geom", name="mount_plate", type="cylinder", size=f"0.0375 {plate / 2:.5f}",
                      pos=C.fmt([0, 0, -plate / 2]), **{"class": "mount"})
    ET.SubElement(hm, "inertial", pos="0 0 0.012", mass=f"{mount_mass:.4f}", diaginertia="3e-5 3e-5 4e-5")
    for name, f, z in (("sharpa_flange", "flange_B.STL", 0.0), ("sharpa_wrist", "wrist_B.STL", SHARPA_FLANGE_TO_WRIST)):
        dst = out / "meshes" / "hand" / f
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(SHARPA_UP / "meshes" / f, dst)
        ET.SubElement(assets, "mesh", name=name, file=f"hand/{f}")
        ET.SubElement(hm, "geom", name=name, type="mesh", mesh=name, pos=C.fmt([0, 0, z]), **{"class": "arm_visual"},
                      rgba="0.79 0.82 0.93 1")
    ET.SubElement(hm, "site", name="hand_flange", size="0.004", rgba="1 0.5 0 1", group="4")
    palm, acts, excl, hand_joints = import_hand(mj, hm, assets, defaults, out)

    wc = cams["wrist"]
    T_palm_cam = C.cfg_T(wc["pose"])
    camera_el(palm, "wrist_cam", wc, cams["sensors"][wc["sensor"]], T_palm_cam)

    act = ET.SubElement(mj, "actuator")
    for k in range(len(cfg["arm_joints"])):
        ET.SubElement(act, "position", name=f"arm_j{k + 1}_position", joint=f"arm_j{k + 1}", **{"class": "arm"})
    for a in acts:
        act.append(a)
    con = ET.SubElement(mj, "contact")
    for e in excl:
        con.append(e)

    ET.indent(mj, space="  ")
    (out / "robot.xml").write_text(ET.tostring(mj, encoding="unicode") + "\n")

    frames = {
        "arm": arm_name, "description": cfg["description"], "urdf": str(C.VENDOR / cfg["urdf"]),
        "arm_joints": [f"arm_j{k + 1}" for k in range(7)], "arm_joints_vendor": cfg["arm_joints"],
        "hand_joints": hand_joints, "root_body": cfg["root_link"], "palm_body": C.PALM_BODY,
        "limits_rad": {"lower": np.radians(cfg["limits_deg"]["lower"]).round(6).tolist(),
                       "upper": np.radians(cfg["limits_deg"]["upper"]).round(6).tolist()},
        "T_lastlink_flange": T_link_flange.round(9).tolist(), "T_flange_handmount": T_flange_F.round(9).tolist(),
        "T_handmount_palm": C.make_T(None, [0, 0, SHARPA_FLANGE_TO_PALM]).tolist(),
        "T_flange_palm": (T_flange_F @ C.make_T(None, [0, 0, SHARPA_FLANGE_TO_PALM])).round(9).tolist(),
        "T_palm_wristcam": T_palm_cam.round(9).tolist(), "arm_mesh_faces": n_faces,
    }
    if with_scene:
        scene = copy.deepcopy(mj)
        scene.set("model", f"{arm_name}_sharpa_r_scene")
        w = scene.find("worldbody")
        T_w_root = C.cfg_T(cfg["placement"]["root"])
        pose_attrs(w.find("body"), T_w_root)
        sc = cams["scene"]["table"]
        vis = ET.SubElement(scene, "visual")
        ET.SubElement(vis, "global", offwidth="1280", offheight="960")
        ET.SubElement(vis, "headlight", ambient="0.45 0.45 0.45", diffuse="0.6 0.6 0.6", specular="0 0 0")
        ET.SubElement(w, "light", pos="0.2 0 2.5", dir="0 0 -1", diffuse="0.5 0.5 0.5")
        ET.SubElement(w, "geom", name="floor", type="plane", size="3 3 0.05", rgba="0.93 0.93 0.93 1", contype="0",
                      conaffinity="0", group="2")
        cx, cy = sc["centre"]
        sx, sy = sc["size"]
        ET.SubElement(w, "geom", name="table", type="box", size=f"{sx / 2} {sy / 2} {sc['thickness'] / 2}",
                      pos=f"{cx} {cy} {C.TABLE_Z - sc['thickness'] / 2}", rgba="0.72 0.68 0.62 1", contype="0",
                      conaffinity="0", group="2")
        ec = cams["exo"]
        T_w_exo = C.cfg_T(ec["pose"])
        camera_el(w, "exo_cam", ec, cams["sensors"][ec["sensor"]], T_w_exo)
        ET.indent(scene, space="  ")
        (out / "scene.xml").write_text(ET.tostring(scene, encoding="unicode") + "\n")
        frames["T_world_root"] = T_w_root.round(9).tolist()
        frames["T_world_exocam"] = T_w_exo.round(9).tolist()
    json.dump(frames, open(out / "frames.json", "w"), indent=1)
    return out, cfg, frames


def check(arm_name, out, cfg, frames, n=200, seed=0):
    """Forward kinematics of the generated model against (1) the URDF's own numbers and (2) the vendor's D-H rows."""
    import mujoco
    m = mujoco.MjModel.from_xml_path(str(out / "robot.xml"))
    d = mujoco.MjData(m)
    links, joints, root = C.read_urdf(C.VENDOR / cfg["urdf"])
    lo, hi = np.array(frames["limits_rad"]["lower"]), np.array(frames["limits_rad"]["upper"])
    adr = [m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"arm_j{k + 1}")] for k in range(7)]
    fl_link = cfg["flange"]["link"]
    b_fl = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, fl_link)
    s_fl = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "arm_flange")
    b_palm = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, C.PALM_BODY)
    b_root = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, cfg["root_link"])
    rng = np.random.default_rng(seed)
    fixed = cfg.get("fixed_joints") or {}
    err_urdf, err_mdh, err_palm = [], [], []
    mdh = cfg.get("mdh")
    if mdh:
        rows = [(a / 1000, np.radians(al), dd / 1000, np.radians(off)) for a, al, dd, off in mdh]
    T_fp = np.array(frames["T_flange_palm"])
    for _ in range(n):
        q = rng.uniform(lo, hi)
        d.qpos[:] = 0
        d.qpos[adr] = q
        mujoco.mj_kinematics(m, d)
        T_link = C.make_T(d.xmat[b_fl].reshape(3, 3), d.xpos[b_fl])
        T_flange = C.make_T(d.site_xmat[s_fl].reshape(3, 3), d.site_xpos[s_fl])
        T_palm = C.make_T(d.xmat[b_palm].reshape(3, 3), d.xpos[b_palm])
        qd = {name: q[k] for k, name in enumerate(cfg["arm_joints"])}
        qd.update(fixed)
        T_ref = C.urdf_fk(joints, qd, fl_link, root, snap=cfg.get("snap_rpy", True))
        e = C.pose_error(T_ref, T_link)
        err_urdf.append([np.linalg.norm(e[:3]), np.linalg.norm(e[3:])])
        e = C.pose_error(T_flange @ T_fp, T_palm)
        err_palm.append([np.linalg.norm(e[:3]), np.linalg.norm(e[3:])])
        if mdh:
            T_root = C.make_T(d.xmat[b_root].reshape(3, 3), d.xpos[b_root])
            e = C.pose_error(T_root @ C.mdh_fk(rows, q)[-1], T_flange)
            err_mdh.append([np.linalg.norm(e[:3]), np.linalg.norm(e[3:])])
    rep = {"arm": arm_name, "nq": int(m.nq), "nu": int(m.nu), "nbody": int(m.nbody), "samples": n,
           "flange_link_vs_urdf_max": {"mm": float(np.max(err_urdf, 0)[0] * 1000), "deg": float(np.degrees(np.max(err_urdf, 0)[1]))},
           "palm_vs_flange_chain_max": {"mm": float(np.max(err_palm, 0)[0] * 1000), "deg": float(np.degrees(np.max(err_palm, 0)[1]))}}
    if mdh:
        rep["flange_vs_dh_max"] = {"mm": float(np.max(err_mdh, 0)[0] * 1000), "deg": float(np.degrees(np.max(err_mdh, 0)[1]))}
        # how far the vendor URDF's rounded angles (1.5708 for pi/2) are from the D-H rows when not snapped
        e_raw = []
        for _ in range(n):
            q = rng.uniform(lo, hi)
            qd = {name: q[k] for k, name in enumerate(cfg["arm_joints"])}
            T_raw = C.urdf_fk(joints, qd, fl_link, root, snap=False) @ np.array(frames["T_lastlink_flange"])
            e = C.pose_error(C.mdh_fk(rows, q)[-1], T_raw)
            e_raw.append([np.linalg.norm(e[:3]), np.linalg.norm(e[3:])])
        rep["unsnapped_urdf_vs_dh_max"] = {"mm": float(np.max(e_raw, 0)[0] * 1000), "deg": float(np.degrees(np.max(e_raw, 0)[1]))}
    # reach of the palm and joint count
    names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) for j in range(m.njnt)]
    rep["joints"] = names
    assert names[:7] == frames["arm_joints"] and names[7:] == frames["hand_joints"], names
    json.dump(rep, open(out / "build_report.json", "w"), indent=1)
    return rep


def wrist_stub(arm_name, out):
    """Everything that is rigidly attached behind the palm, in the PALM frame: the arm's last link (convex hull), the
    mount plate and the D405's housing. FDR's floating hand has none of it, so its wrist passes where this hardware
    would be inside the table; adding these three shapes to the palm link as collision geometry lets the retargeting
    keep clear. Writes <model dir>/wrist_stub/{arm_last_link_hull.stl, d405_mount.stl, stub.json}."""
    import mujoco
    import trimesh
    m = mujoco.MjModel.from_xml_path(str(out / "robot.xml"))
    d = mujoco.MjData(m)
    mujoco.mj_kinematics(m, d)
    palm = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, C.PALM_BODY)
    T_wp = C.make_T(d.xmat[palm].reshape(3, 3), d.xpos[palm])
    b7 = m.jnt_bodyid[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "arm_j7")]
    hm = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "hand_mount")

    def under(b, top):
        while b > 0:
            if b == top:
                return True
            b = m.body_parentid[b]
        return False

    def palm_pts(g):
        i = m.geom_dataid[g]
        v = m.mesh_vert[m.mesh_vertadr[i]: m.mesh_vertadr[i] + m.mesh_vertnum[i]]
        w = v @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g]
        return (w - T_wp[:3, 3]) @ T_wp[:3, :3], m.mesh_face[m.mesh_faceadr[i]: m.mesh_faceadr[i] + m.mesh_facenum[i]]

    sd = out / "wrist_stub"
    sd.mkdir(exist_ok=True)
    link = [g for g in range(m.ngeom) if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH
            and under(m.geom_bodyid[g], b7) and not under(m.geom_bodyid[g], hm)]
    P = np.concatenate([palm_pts(g)[0] for g in link])
    hull = trimesh.convex.convex_hull(P)
    hull.export(sd / "arm_last_link_hull.stl")
    info = {"frame": "palm (R_forearm_rot_y_link): +z to the fingers, +x out of the palm, +y to the thumb; metres",
            "arm_last_link_hull": {"file": "arm_last_link_hull.stl", "links": sorted({mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[g]) for g in link}),
                                   "bounds": [P.min(0).round(4).tolist(), P.max(0).round(4).tolist()]}}
    g = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "d405_mount")
    if g >= 0:
        V, F = palm_pts(g)
        trimesh.Trimesh(V, F, process=False).export(sd / "d405_mount.stl")
        info["d405_mount"] = {"file": "d405_mount.stl", "bounds": [V.min(0).round(4).tolist(), V.max(0).round(4).tolist()]}
    g = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "wrist_cam_housing")
    Tg = C.inv_T(T_wp) @ C.make_T(d.geom_xmat[g].reshape(3, 3), d.geom_xpos[g])
    info["d405_housing_box"] = {"centre": Tg[:3, 3].round(5).tolist(), "quat_wxyz": C.mat2quat(Tg[:3, :3]).round(6).tolist(),
                                "size": (2 * m.geom_size[g]).round(5).tolist()}
    json.dump(info, open(sd / "stub.json", "w"), indent=1)
    return info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--no-check", action="store_true")
    args = ap.parse_args()
    out, cfg, frames = build(args.arm)
    print(f"[build] {args.arm}: {out}/robot.xml, scene.xml")
    try:                                                         # the web viewer's thinned meshes (needs pymeshlab)
        import viewer_meshes
        _, n0, n1 = viewer_meshes.export(args.arm)
        print(f"[build] viewer_meshes/: {n0} -> {n1} triangles for view_embodiment.py")
    except ImportError as e:
        print(f"[build] no viewer meshes ({e}); view_embodiment.py will send full-resolution meshes")
    import spheres
    _, n_spheres = spheres.export(args.arm, out)
    print(f"[build] ik_arm.urdf, ik_full.urdf, spheres.json ({n_spheres} collision spheres) for ik_pyroki.py")
    stub = wrist_stub(args.arm, out)
    b = stub["arm_last_link_hull"]["bounds"]
    print(f"[build] wrist_stub/: last link {stub['arm_last_link_hull']['links']} spans x {b[0][0] * 1000:.0f}..{b[1][0] * 1000:.0f}, "
          f"y {b[0][1] * 1000:.0f}..{b[1][1] * 1000:.0f}, z {b[0][2] * 1000:.0f}..{b[1][2] * 1000:.0f} mm in the palm frame")
    if not args.no_check:
        rep = check(args.arm, out, cfg, frames)
        print(f"[check] {rep['nq']} joints ({rep['nu']} actuators), {rep['nbody']} bodies")
        for k in ("flange_link_vs_urdf_max", "flange_vs_dh_max", "unsnapped_urdf_vs_dh_max", "palm_vs_flange_chain_max"):
            if k in rep:
                print(f"[check] {k}: {rep[k]['mm']:.2e} mm, {rep[k]['deg']:.2e} deg")


if __name__ == "__main__":
    main()
