"""Shared helpers for scripts/embodiment: storage paths, SE(3) maths, configs, URDF reading and FlashDexRetarget data.

Conventions used by every script here:
  * transforms are 4x4 numpy arrays T_a_b that map points of frame b into frame a (p_a = T_a_b @ p_b);
  * quaternions are (w, x, y, z), as in MuJoCo;
  * "palm" is the body the FlashDexRetarget (FDR) floating wrist moves, R_forearm_rot_y_link, which is the Sharpa
    hand's base link right_hand_C_MC: +z from the wrist to the fingers, +x out of the palm (the fingers curl towards
    +x), +y towards the thumb;
  * camera poses are optical frames (OpenCV): +z along the view, +x to the image's right, +y down the image;
  * the clip world is FDR's: z up, the table plane at z = 0.8 m, the object's first-frame centre above the origin, +y the
    way the source video's camera looked.
"""

from __future__ import annotations

import json
import os
import xml.etree.ElementTree as ET
from pathlib import Path

# The tools here parallelise over processes; without this every worker also starts one BLAS thread per core.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402
import yaml  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
STORAGE = Path(os.environ.get("PROJECT_STORAGE_ROOT", "/data/sky/projects/CounterfactualHOI"))
FDR = Path(os.environ.get("FDR", STORAGE / "build" / "FlashDexRetarget"))
CFG = REPO / "configs" / "embodiment"
VENDOR = STORAGE / "build"                       # vendor model downloads (setup.sh)
BUILD = STORAGE / "build" / "embodiment"         # generated models, meshes and CAD
RUNS = STORAGE / "runs" / "embodiment"           # design searches, IK results, rendered depth
PALM_BODY = "R_forearm_rot_y_link"
FINGERS = ("thumb", "index", "middle", "ring", "pinky")
TABLE_Z = 0.8


# ---------------------------------------------------------------- SE(3)
def rot_x(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], float)


def rot_y(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], float)


def rot_z(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], float)


def rpy_mat(rpy):
    """URDF fixed-axis roll-pitch-yaw: R = Rz(yaw) Ry(pitch) Rx(roll)."""
    r, p, y = rpy
    return rot_z(y) @ rot_y(p) @ rot_x(r)


def axis_angle(axis, angle):
    a = np.asarray(axis, float)
    a = a / np.linalg.norm(a)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * K @ K


def make_T(R=None, p=None):
    T = np.eye(4)
    if R is not None:
        T[:3, :3] = R
    if p is not None:
        T[:3, 3] = p
    return T


def inv_T(T):
    out = np.eye(4)
    out[:3, :3] = T[:3, :3].T
    out[:3, 3] = -T[:3, :3].T @ T[:3, 3]
    return out


def mat2quat(R):
    """Rotation matrix -> (w, x, y, z), w >= 0."""
    R = np.asarray(R, float)
    t = np.trace(R)
    if t > 0:
        s = np.sqrt(t + 1.0) * 2
        q = [0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s]
    else:
        i = int(np.argmax(np.diag(R)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = np.sqrt(1.0 + R[i, i] - R[j, j] - R[k, k]) * 2
        q = [0.0] * 4
        q[0] = (R[k, j] - R[j, k]) / s
        q[1 + i] = 0.25 * s
        q[1 + j] = (R[j, i] + R[i, j]) / s
        q[1 + k] = (R[k, i] + R[i, k]) / s
    q = np.array(q)
    q /= np.linalg.norm(q)
    return q if q[0] >= 0 else -q


def quat2mat(q):
    w, x, y, z = np.asarray(q, float) / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def rot_log(R):
    """Rotation vector of R (axis * angle)."""
    c = np.clip((np.trace(R) - 1) / 2, -1, 1)
    ang = np.arccos(c)
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    if ang < 1e-9:
        return w / 2
    if np.pi - ang < 1e-6:                              # near pi: take the axis from the symmetric part
        A = (R + np.eye(3)) / 2
        a = np.sqrt(np.clip(np.diag(A), 0, None))
        i = int(np.argmax(a))
        a = A[i] / a[i]
        return a / np.linalg.norm(a) * ang
    return w / (2 * np.sin(ang)) * ang


def pose_error(T_target, T_cur):
    """6-vector (translation, rotation vector) that moves T_cur onto T_target, expressed in the world frame."""
    return np.concatenate([T_target[:3, 3] - T_cur[:3, 3], rot_log(T_target[:3, :3] @ T_cur[:3, :3].T)])


def frame_from_axes(x=None, y=None, z=None):
    """Right-handed rotation matrix from two of its (not necessarily exactly orthogonal) axes."""
    if x is not None and z is not None:
        z = np.asarray(z, float) / np.linalg.norm(z)
        x = np.asarray(x, float) - np.dot(x, z) * z
        x /= np.linalg.norm(x)
        return np.stack([x, np.cross(z, x), z], 1)
    if y is not None and z is not None:
        z = np.asarray(z, float) / np.linalg.norm(z)
        y = np.asarray(y, float) - np.dot(y, z) * z
        y /= np.linalg.norm(y)
        return np.stack([np.cross(y, z), y, z], 1)
    raise ValueError("give x and z, or y and z")


def fmt(v, nd=9):
    return " ".join(f"{float(x):.{nd}g}" for x in np.asarray(v).ravel())


# ---------------------------------------------------------------- configs
def load_yaml(path):
    with open(path) as f:
        return yaml.safe_load(f)


def load_arm(name):
    """configs/embodiment/arms/<name>.yaml, with `extends:` resolved (a variant overrides its parent's keys)."""
    cfg = load_yaml(CFG / "arms" / f"{name}.yaml")
    if "extends" in cfg:
        base = load_arm(cfg.pop("extends"))
        base.update(cfg)
        cfg = base
    cfg["name"] = name
    return cfg


def load_cameras():
    return load_yaml(CFG / "cameras.yaml")


def model_dir(arm):
    return BUILD / f"{arm}_sharpa_r"


def cfg_T(d):
    """{pos: [x, y, z], and one of quat (wxyz), rpy_deg, or two of x_axis / y_axis / z_axis} -> 4x4."""
    p = np.asarray(d.get("pos", [0, 0, 0]), float)
    if "quat" in d:
        R = quat2mat(d["quat"])
    elif "rpy_deg" in d:
        R = rpy_mat(np.radians(d["rpy_deg"]))
    elif "z_axis" in d:
        R = frame_from_axes(x=d.get("x_axis"), y=d.get("y_axis"), z=d["z_axis"])
    else:
        R = np.eye(3)
    return make_T(R, p)


# ---------------------------------------------------------------- URDF
def read_urdf(path):
    """-> (links, joints, root): links[name] = {visuals: [...], inertial}, joints[name] = {type, parent, child, T,
    axis, lower, upper, velocity, effort}. T is the joint origin (child frame in the parent's at q = 0)."""
    root = ET.parse(path).getroot()

    def origin(e):
        o = e.find("origin") if e is not None else None
        if o is None:
            return np.zeros(3), np.zeros(3)
        return (np.array([float(x) for x in o.get("xyz", "0 0 0").split()]),
                np.array([float(x) for x in o.get("rpy", "0 0 0").split()]))

    links = {}
    for l in root.findall("link"):
        vis = []
        for v in l.findall("visual"):
            g = v.find("geometry")
            xyz, rpy = origin(v)
            mesh = g.find("mesh") if g is not None else None
            if mesh is not None:
                scale = [float(s) for s in mesh.get("scale", "1 1 1").split()]
                vis.append({"mesh": mesh.get("filename"), "scale": scale, "xyz": xyz, "rpy": rpy})
        inertial = None
        i = l.find("inertial")
        if i is not None and i.find("mass") is not None:
            xyz, rpy = origin(i)
            I = i.find("inertia")
            inertial = {"mass": float(i.find("mass").get("value")), "xyz": xyz, "rpy": rpy,
                        "I": [float(I.get(k, 0)) for k in ("ixx", "iyy", "izz", "ixy", "ixz", "iyz")] if I is not None else None}
        links[l.get("name")] = {"visuals": vis, "inertial": inertial}
    joints, children = {}, set()
    for j in root.findall("joint"):
        xyz, rpy = origin(j)
        ax = j.find("axis")
        lim = j.find("limit")
        joints[j.get("name")] = {
            "type": j.get("type"), "parent": j.find("parent").get("link"), "child": j.find("child").get("link"),
            "xyz": xyz, "rpy": rpy,
            "axis": np.array([float(x) for x in ax.get("xyz").split()]) if ax is not None else np.array([0, 0, 1.0]),
            "lower": float(lim.get("lower")) if lim is not None and lim.get("lower") else None,
            "upper": float(lim.get("upper")) if lim is not None and lim.get("upper") else None,
            "velocity": float(lim.get("velocity")) if lim is not None and lim.get("velocity") else None,
            "effort": float(lim.get("effort")) if lim is not None and lim.get("effort") else None}
        children.add(joints[j.get("name")]["child"])
    roots = [n for n in links if n not in children]
    return links, joints, roots[0]


def snap_rpy(rpy, tol=2e-4):
    """SolidWorks exports write pi/2 as 1.5708: snap angles within `tol` of a multiple of pi/2 onto it."""
    k = np.round(np.asarray(rpy) / (np.pi / 2))
    snapped = k * np.pi / 2
    return np.where(np.abs(np.asarray(rpy) - snapped) < tol, snapped, rpy)


def urdf_fk(joints, q, chain_to, root, snap=True):
    """Pose of link `chain_to` in `root` for joint values q {name: value}, straight from the URDF's numbers (used to
    check the generated MuJoCo model against its source)."""
    by_child = {j["child"]: (n, j) for n, j in joints.items()}
    seq = []
    link = chain_to
    while link != root:
        n, j = by_child[link]
        seq.append((n, j))
        link = j["parent"]
    T = np.eye(4)
    for n, j in reversed(seq):
        rpy = snap_rpy(j["rpy"]) if snap else j["rpy"]
        T = T @ make_T(rpy_mat(rpy), j["xyz"])
        v = q.get(n, 0.0)
        if j["type"] in ("revolute", "continuous"):
            T = T @ make_T(axis_angle(j["axis"], v))
        elif j["type"] == "prismatic":
            T = T @ make_T(None, j["axis"] / np.linalg.norm(j["axis"]) * v)
    return T


def mdh_fk(rows, q):
    """Craig's modified D-H: rows of (a_{i-1} [m], alpha_{i-1} [rad], d_i [m], theta offset [rad]); q for the joints,
    a trailing row without a joint (the flange) takes angle 0. Returns the list of frames 1..n in frame 0."""
    T, out = np.eye(4), []
    for i, (a, alpha, d, off) in enumerate(rows):
        th = (q[i] if i < len(q) else 0.0) + off
        T = T @ make_T(rot_x(alpha), [a, 0, 0]) @ make_T(rot_z(th), None) @ make_T(None, [0, 0, d])
        out.append(T.copy())
    return out


# ---------------------------------------------------------------- FlashDexRetarget data
def load_pack(dataset):
    """A FlashDexRetarget dataset's motion.pt as numpy: per clip the reference joint angles (28: 6 floating-wrist +
    22 hand joints, in right.xml's order), the object's pose, mesh directory and table disk."""
    import torch
    ds = Path(dataset)
    pt = torch.load(ds / "motion.pt", map_location="cpu", weights_only=False)
    clips = []
    for i, name in enumerate(pt["motion_filename"]):
        a0, n = int(pt["length_starts"][i]), int(pt["motion_num_frames"][i])
        slot = int(pt["motion_object_slot_right"][i])
        clips.append({
            "name": str(name).split("#")[-1].removeprefix("cfhoi_"), "motion_id": i, "frames": n,
            "joint_pos": pt["joint_pos"][a0:a0 + n].numpy().astype(np.float64),
            "obj_pos": pt["obj_right_pos"][a0:a0 + n].numpy().astype(np.float64),
            "obj_rot": pt["obj_right_rotmat"][a0:a0 + n].numpy().astype(np.float64),
            "mesh_dir": ds / str(pt["right_object_mesh_dirs"][slot]),
            "mesh_scale": float(pt["right_object_mesh_scales"][slot]),
            "disk": pt["support_disks"][i, 0].numpy().astype(np.float64) if "support_disks" in pt else None})
    return clips


def load_rollouts(archive, clips=None):
    """A rollout archive (RUN/evals/step<N>): {clip name: {joint_pos, obj_pos, obj_rot, ...}} at 60 Hz."""
    archive = Path(archive)
    man = json.load(open(archive / "manifest.json"))
    out = {}
    for e in man["saved"]:
        name = str(e["name"]).split("#")[-1].removeprefix("cfhoi_")
        if clips and name not in clips:
            continue
        z = np.load(archive / e["rollout"])
        d = {k: z[k] for k in z.files}
        op, oq = d["obj_pos"], d["obj_quat"]
        if op.ndim == 3:
            op, oq = op[:, 0], oq[:, 0]
        d["obj_pos"] = op.astype(np.float64)
        d["obj_rot"] = np.stack([quat2mat(q) for q in oq])
        d["joint_pos"] = d["joint_pos"].astype(np.float64)
        d["motion_id"] = int(e["motion_id"])
        out[name] = d
    return out, man


class FloatingHand:
    """FDR's floating-wrist Sharpa model (right.xml, or bimanual.xml for 56-joint vectors): joint vector -> palm pose
    and the 22 hand joint angles by name."""

    def __init__(self):
        import mujoco
        self.mj = mujoco
        d = FDR / "assets" / "robot" / "sharpa"
        self.models = {}
        for f in ("right.xml", "bimanual.xml"):
            m = mujoco.MjModel.from_xml_path(str(d / f))
            self.models[m.nq] = (m, mujoco.MjData(m))
        m = self.models[28][0]
        self.hand_joints = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) for j in range(6, m.njnt)]

    def _fk(self, q):
        m, d = self.models[len(q)]
        d.qpos[:] = q
        self.mj.mj_kinematics(m, d)
        return m, d

    def palm(self, q):
        m, d = self._fk(q)
        b = self.mj.mj_name2id(m, self.mj.mjtObj.mjOBJ_BODY, PALM_BODY)
        return make_T(d.xmat[b].reshape(3, 3).copy(), d.xpos[b].copy())

    def palm_traj(self, Q):
        return np.stack([self.palm(q) for q in Q])

    def fingers(self, Q):
        """(T, 22) hand joint angles in self.hand_joints' order."""
        Q = np.atleast_2d(Q)
        m, _ = self.models[Q.shape[1]]
        adr = [m.jnt_qposadr[self.mj.mj_name2id(m, self.mj.mjtObj.mjOBJ_JOINT, n)] for n in self.hand_joints]
        return Q[:, adr]
