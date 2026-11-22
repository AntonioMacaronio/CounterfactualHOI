"""One clip for the refinement agent: fixed inputs, editable state, and per-frame metrics.

Inputs come from the FoundationPose + Depth Pro 2 run ($PROJECT_STORAGE_ROOT/runs/fprecon/VID) and CHOIR's stage 1
($CHOIR_ROOT/output/VID): RGB frames, SAM3 masks, hand-scaled metric depth, the textured SAM3D mesh, Dyn-HaMR hands
(moved onto the depth map), GeoCalib gravity and the table plane fitted in the frame-0 depth.

State (what the agent edits): per-frame object pose (object -> OpenCV camera, metric), a per-frame rigid motion of the
hand (rotation about its centroid + translation), a per-frame flexion of each finger on top of Dyn-HaMR's articulation
(curl > 0, open < 0; MANO's skinning weights), and phase labels. Every edit makes a new version; nothing is overwritten
(Versions).

Metrics are computed on GPU at 1/4 resolution: object silhouette coverage / spill against the SAM3 mask (as in
scripts/fprecon/verify_track.py), depth residual and bias against the depth map, the hand's silhouette fit, hand-object
contact distance and penetration (from a signed-distance grid of the object), hand triangles crossing the object's
surface (FCL triangle-triangle tests: catches fingers through a thin wall, which a coarse SDF misses), heights above the table surface measured
locally around the object (the global plane fit can be ~2 cm off far from the camera), the hand's distance to its 2D keypoints
(ViTPose, from CHOIR stage 1; MANO joints from Dyn-HaMR's mesh via MANO's joint regressor), height above the table
and motion.
"""

import json
import os
import sys
from dataclasses import dataclass, field

import cv2
import fcl
import numpy as np
import torch
import trimesh
from scipy.spatial.transform import Rotation, Slerp

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = os.path.dirname(HERE)
for p in (os.path.join(SCRIPTS, "fprecon"), os.path.join(SCRIPTS, "choir"),
          os.environ.get("FP_DIR", "/data/sky/projects/CounterfactualHOI/build/FoundationPose")):
    sys.path.insert(0, p)
import nvdiffrast.torch as dr  # noqa: E402
from Utils import make_mesh_tensors, nvdiffrast_render  # noqa: E402
from hand_scale import corrected_hands  # noqa: E402
from masks import hand_mask, object_mask, tracked_mesh  # noqa: E402
from table_fit import fit_table  # noqa: E402

STORAGE = os.environ.get("PROJECT_STORAGE_ROOT", "/data/sky/projects/CounterfactualHOI")
LOW = 4                                     # metrics resolution divisor
FINGERS = ["thumb", "index", "middle", "ring", "pinky"]
CHAINS = {"thumb": (13, 14, 15), "index": (1, 2, 3), "middle": (4, 5, 6), "ring": (10, 11, 12), "pinky": (7, 8, 9)}
TIPS = {"thumb": 745, "index": 317, "middle": 444, "ring": 556, "pinky": 673}
FLEX_SPLIT = (0.4, 0.35, 0.25)              # a finger's flexion shared over its three joints (base to tip)


@dataclass
class State:
    obj: np.ndarray                         # (N, 4, 4) object -> camera
    hand_off: np.ndarray                    # (N, 3) camera-frame translation added to the hand
    hand_rot: np.ndarray                    # (N, 3) rotation vector applied to the hand about its centroid
    phases: list = field(default_factory=list)   # [{"start", "end", "label"}]
    finger: np.ndarray = None               # (N, 5) flexion in degrees per finger (FINGERS order), 0 = Dyn-HaMR's

    def __post_init__(self):
        if self.finger is None:
            self.finger = np.zeros((len(self.obj), len(FINGERS)))

    def copy(self):
        return State(self.obj.copy(), self.hand_off.copy(), self.hand_rot.copy(), [dict(p) for p in self.phases],
                     self.finger.copy())


def load_state(path, phases=None):
    """A State from a versions/vNNN.npz or final.npz (finger flexion optional: older runs have none)."""
    z = np.load(path)
    return State(z["obj"], z["hand_off"], z["hand_rot"], phases or [], z["finger"] if "finger" in z else None)


class Versions:
    """Append-only version history; `cur` is the working version."""

    def __init__(self, init):
        self.states, self.notes, self.parents, self.cur = [init.copy()], ["initial (FoundationPose + Depth Pro 2)"], [None], 0

    @property
    def state(self):
        return self.states[self.cur]

    def add(self, state, note):
        self.states.append(state)
        self.notes.append(note)
        self.parents.append(self.cur)
        self.cur = len(self.states) - 1
        return self.cur


def se3_interp(T0, T1, s):
    """Pose between T0 and T1 (s in [0, 1]): slerp rotation, lerp translation."""
    r = Slerp([0, 1], Rotation.from_matrix([T0[:3, :3], T1[:3, :3]]))([s]).as_matrix()[0]
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = r, (1 - s) * T0[:3, 3] + s * T1[:3, 3]
    return T


class Clip:
    def __init__(self, vd, run, gravity=None):
        self.vd, self.run = os.path.abspath(vd), os.path.abspath(run)
        self.vid = os.path.basename(self.vd)
        self.K = np.array(json.load(open(os.path.join(vd, "inputs", "intrinsics.json")))["intrinsics"], float)
        f0 = cv2.imread(os.path.join(vd, "inputs", "frames", "rgb", "0.png"))
        self.H, self.W = f0.shape[:2]
        hs = json.load(open(os.path.join(run, "hand_scale.json")))
        self.side, self.depth_scale = hs["side"], hs["scale"]
        self.depth = np.load(os.path.join(run, "depth_dp2.npy"), mmap_mode="r")
        poses = np.load(os.path.join(run, "object_poses.npy"))
        hv, self.hand_faces, self.hand_valid, _ = corrected_hands(vd, run)
        self.N = min(len(poses), len(hv), self.depth.shape[0])
        poses, hv, self.hand_valid = poses[:self.N], hv[:self.N], self.hand_valid[:self.N]
        ok = np.nonzero(self.hand_valid)[0]
        for i in np.nonzero(~self.hand_valid)[0]:
            hv[i] = hv[ok[np.argmin(np.abs(ok - i))]]
        self.hand_base = hv.astype(np.float32)
        self.fps = cv2.VideoCapture(os.path.join(vd, "inputs", "video.mp4")).get(cv2.CAP_PROP_FPS) or 30.0

        self.mesh = trimesh.load(tracked_mesh(run), force="mesh", process=False)
        self.mt = make_mesh_tensors(self.mesh)
        v = np.asarray(self.mesh.vertices)
        self.center_obj = 0.5 * (v.min(0) + v.max(0))
        evals, evecs = np.linalg.eigh(np.cov((v - v.mean(0)).T))
        self.axes_obj = evecs[:, ::-1]                       # columns: long, mid, short (object frame)
        self.extent = np.ptp(v @ self.axes_obj, 0)
        self.sub_v = v[np.random.default_rng(0).choice(len(v), min(len(v), 3000), replace=False)]

        g = json.load(open(gravity or os.path.join(STORAGE, "runs", "choir_gravity", self.vid, "gravity.json")))
        up = np.array(g["vec3d"], float)
        self.up = up / np.linalg.norm(up)
        fwd = np.array([0, 0, 1.0]) - self.up * self.up[2]
        self.fwd = fwd / np.linalg.norm(fwd)
        self.right = np.cross(self.fwd, self.up)

        # 2D hand keypoints (ViTPose, OpenPose order, from CHOIR stage 1) and MANO's 21 joints in the same order:
        # 16 regressed joints (wrist, index, middle, pinky, ring, thumb) + 5 fingertip vertices.
        kp_path = os.path.join(vd, "inputs", "keypoints", ("rh" if self.side == "right" else "lh") + "_keypoints.json")
        kp = json.load(open(kp_path)) if os.path.exists(kp_path) else {}
        self.kp2d = np.full((self.N, 21, 2), np.nan)
        for k, v in kp.items():
            if int(k) < self.N and len(v) == 21:
                self.kp2d[int(k)] = np.asarray(v, float)[:, :2]
        J = np.load(os.path.join(STORAGE, "models", "mano", f"joint_regressor_{self.side}.npz"))["J"]
        tips = {"thumb": 745, "index": 317, "middle": 444, "ring": 556, "pinky": 673}
        order = [0, 13, 14, 15, "thumb", 1, 2, 3, "index", 4, 5, 6, "middle", 10, 11, 12, "ring", 7, 8, 9, "pinky"]
        self.kp_reg = np.zeros((21, 778), np.float32)
        for r, o in enumerate(order):
            if isinstance(o, str):
                self.kp_reg[r, tips[o]] = 1
            else:
                self.kp_reg[r] = J[o]
        self.kp_finger = {f: [order.index(j) for j in CHAINS[f]] + [order.index(f)] for f in FINGERS}
        # MANO skinning (scripts: extracted from MANO_{RIGHT,LEFT}.pkl): finger flexion as linear blend skinning
        sk = np.load(os.path.join(STORAGE, "models", "mano", f"skinning_{self.side}.npz"))
        self.skin_W, self.J16 = sk["weights"].astype(np.float64), sk["J"].astype(np.float64)
        lab = np.zeros((778, len(FINGERS) + 1))               # vertex -> palm (0) or finger k (k + 1)
        lab[:, 0] = self.skin_W[:, 0]
        for k, f in enumerate(FINGERS):
            lab[:, k + 1] = self.skin_W[:, list(CHAINS[f])].sum(1)
        self.vert_finger = lab.argmax(1) - 1                   # -1 = palm
        self.face_finger = np.array([np.bincount(self.vert_finger[f] + 1, minlength=6).argmax() - 1 for f in self.hand_faces])
        self.palm_sign = self._palm_sign()
        self.finger_rows = [np.nonzero(self.skin_W[:, list(CHAINS[f])].sum(1) > 1e-4)[0] for f in FINGERS]
        moving = [self.skin_W[:, list(CHAINS[f])].sum(1) > 0.05 for f in FINGERS]      # faces a finger's flexion moves
        self.finger_faces = [np.asarray(self.hand_faces, np.int32)[mv[self.hand_faces].any(1)] for mv in moving]
        self.distal = [np.nonzero(self.skin_W[:, CHAINS[f][2]] > 0.5)[0] for f in FINGERS]   # last segment (pad + tip)

        self.glctx = dr.RasterizeCudaContext()
        self._rgb, self._ltab, self._xyz = {}, {}, None
        self._load_lowres()
        self._fit_table()
        self._sdf_grid()
        V, F = np.asarray(self.mesh.vertices), np.asarray(self.mesh.faces)
        if len(F) > 4000:                                    # collision copy: decimated (crossings stay the same)
            import open3d as o3d
            d = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(V), o3d.utility.Vector3iVector(F))
            d = d.simplify_quadric_decimation(target_number_of_triangles=4000)
            V, F = np.asarray(d.vertices), np.asarray(d.triangles)
        ob = fcl.BVHModel()                                  # FCL BVH of the object, in the object frame
        ob.beginModel(len(V), len(F))
        ob.addSubModel(V.astype(np.float64), F.astype(np.int32))
        ob.endModel()
        self.col_obj = fcl.CollisionObject(ob, fcl.Transform())
        self.hand_faces_i = np.asarray(self.hand_faces, np.int32)
        self.init_state = State(poses.copy(), np.zeros((self.N, 3)), np.zeros((self.N, 3)), [])

    # ---------- inputs ----------
    def rgb(self, i):
        if i not in self._rgb:
            if len(self._rgb) > 64:
                self._rgb.pop(next(iter(self._rgb)))
            self._rgb[i] = cv2.imread(os.path.join(self.vd, "inputs", "frames", "rgb", f"{i}.png"))
        return self._rgb[i]

    def masks_full(self, i):
        om, hm = object_mask(self.vd, self.run, i), hand_mask(self.vd, self.run, i, self.side)
        om = np.zeros((self.H, self.W), bool) if om is None else om
        hm = np.zeros((self.H, self.W), bool) if hm is None or hm.mean() > 0.2 else hm
        return om, hm

    def depth_full(self, i):
        return self.depth[i].astype(np.float32) * self.depth_scale

    def _load_lowres(self):
        h, w = self.H // LOW, self.W // LOW
        self.hl, self.wl = h, w
        self.om = np.zeros((self.N, h, w), bool)
        self.hm = np.zeros((self.N, h, w), bool)
        self.dl = np.zeros((self.N, h, w), np.float32)
        for i in range(self.N):
            om, hm = self.masks_full(i)
            self.om[i] = cv2.resize(om.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0
            self.hm[i] = cv2.resize(hm.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0
            self.dl[i] = cv2.resize(self.depth_full(i), (w, h), interpolation=cv2.INTER_NEAREST)
        self.Kl = self.K.copy()
        self.Kl[:2] /= LOW

    def _fit_table(self):
        d = self.depth_full(0)
        om, hm = self.masks_full(0)
        fg = cv2.dilate((om | hm).astype(np.uint8), np.ones((15, 15), np.uint8)) > 0
        ys, xs = np.nonzero((d > 0.05) & (d < 8) & ~fg)
        sel = np.random.default_rng(0).choice(len(xs), min(len(xs), 200000), replace=False)
        ys, xs = ys[sel], xs[sel]
        z = d[ys, xs]
        pts = np.stack([(xs - self.K[0, 2]) / self.K[0, 0] * z, (ys - self.K[1, 2]) / self.K[1, 1] * z, z], 1)
        t = fit_table(pts, self.up, np.random.default_rng(0))
        self.table_n = t["normal"] if t else self.up
        self.table_p = t["point"] if t else np.zeros(3)

    def _sdf_grid(self, n=56):
        """Signed distance to the object surface on a grid in the object frame (negative inside), cached."""
        path = os.path.join(self.run, "refine_sdf.npz")
        if os.path.exists(path):
            z = np.load(path)
            self.sdf, self.sdf_lo, self.sdf_hi = z["sdf"], z["lo"], z["hi"]
            return
        v = np.asarray(self.mesh.vertices)
        pad = 0.25 * np.ptp(v, 0).max()
        lo, hi = v.min(0) - pad, v.max(0) + pad
        g = np.stack(np.meshgrid(*[np.linspace(lo[k], hi[k], n) for k in range(3)], indexing="ij"), -1).reshape(-1, 3)
        closest, dist, tri = trimesh.proximity.closest_point(self.mesh, g)
        sign = np.sign(np.einsum("ij,ij->i", g - closest, self.mesh.face_normals[tri]))
        sign[sign == 0] = 1
        self.sdf = (sign * dist).reshape(n, n, n).astype(np.float32)
        self.sdf_lo, self.sdf_hi = lo, hi
        np.savez(path, sdf=self.sdf, lo=lo, hi=hi)

    def sdf_at(self, p_obj):
        """Trilinear SDF lookup for points in the object frame (outside the grid -> large positive)."""
        n = self.sdf.shape[0]
        u = (p_obj - self.sdf_lo) / (self.sdf_hi - self.sdf_lo) * (n - 1)
        out = np.any((u < 0) | (u > n - 1), axis=-1)
        u = np.clip(u, 0, n - 1 - 1e-6)
        i0 = np.floor(u).astype(int)
        f = u - i0
        val = np.zeros(len(u))
        for dx in (0, 1):
            for dy in (0, 1):
                for dz in (0, 1):
                    wgt = (f[:, 0] if dx else 1 - f[:, 0]) * (f[:, 1] if dy else 1 - f[:, 1]) * (f[:, 2] if dz else 1 - f[:, 2])
                    val += wgt * self.sdf[i0[:, 0] + dx, i0[:, 1] + dy, i0[:, 2] + dz]
        val[out] = 1.0
        return val

    # ---------- hand model ----------
    def _palm_sign(self):
        """Sign that makes cross(index MCP - wrist, pinky MCP - wrist) point to the palm side: the side the fingers
        bend towards, summed over the clip's frames (people rarely hyperextend their fingers)."""
        s = 0.0
        for i in np.nonzero(self.hand_valid)[0][::5]:
            v = self.hand_base[i].astype(np.float64)
            J = self.J16 @ v
            n = np.cross(J[1] - J[0], J[7] - J[0])
            n /= np.linalg.norm(n) + 1e-12
            for f in ("index", "middle", "ring", "pinky"):
                j1, j2, j3 = CHAINS[f]
                pts = [J[0], J[j1], J[j2], J[j3], v[TIPS[f]]]
                for a, b, c in zip(pts, pts[1:], pts[2:]):
                    u, w = b - a, c - b
                    s += (w - u * (w @ u) / (u @ u + 1e-12)) @ n
        return 1.0 if s >= 0 else -1.0

    def palm_normal(self, J):
        n = self.palm_sign * np.cross(J[1] - J[0], J[7] - J[0])
        return n / (np.linalg.norm(n) + 1e-12)

    def finger_delta(self, v, J, n, k, deg):
        """Displacement of the hand's vertices when finger k flexes by `deg` degrees (curl > 0): each of its three
        joints turns (FLEX_SPLIT of the angle) about the axis across the finger, perpendicular to the palm normal n,
        and the mesh follows by MANO's skinning weights. Fingers are independent (disjoint joint chains), so the
        deltas of several fingers add up."""
        f = FINGERS[k]
        j1, j2, j3 = CHAINS[f]
        rows = self.finger_rows[k]                           # vertices with any weight on this finger's joints
        vr = v[rows].astype(np.float64)
        Ra, ta = np.eye(3), np.zeros(3)
        out = np.zeros_like(vr)
        for j, nxt, share in ((j1, J[j2], FLEX_SPLIT[0]), (j2, J[j3], FLEX_SPLIT[1]), (j3, v[TIPS[f]], FLEX_SPLIT[2])):
            p, q = Ra @ J[j] + ta, Ra @ nxt + ta
            a = np.cross(q - p, n)
            a /= np.linalg.norm(a) + 1e-12
            Rk = Rotation.from_rotvec(np.radians(share * deg) * a).as_matrix()
            Ra, ta = Rk @ Ra, Rk @ (ta - p) + p
            out += self.skin_W[rows, j][:, None] * (vr @ Ra.T + ta - vr)
        d = np.zeros((len(v), 3), np.float32)
        d[rows] = out
        return d

    def articulate(self, v, flex):
        """Hand vertices v (778, 3) with each finger flexed by flex[k] degrees on top of their pose."""
        if not np.any(flex):
            return v
        J = self.J16 @ v.astype(np.float64)
        n = self.palm_normal(J)
        out = v.astype(np.float32).copy()
        for k in range(len(FINGERS)):
            if flex[k]:
                out += self.finger_delta(v, J, n, k, flex[k])
        return out

    def hand_verts_of(self, i, flex, rot, off):
        """Dyn-HaMR's hand with finger flexion `flex`, rotated by `rot` about its (unflexed) centroid, then shifted."""
        base = self.hand_base[i]
        v = self.articulate(base, flex) if flex is not None else base
        if np.any(rot):
            c = base.mean(0)
            v = (v - c) @ Rotation.from_rotvec(rot).as_matrix().T.astype(np.float32) + c
        return v + np.asarray(off, np.float32)

    def hand_verts(self, state, i):
        return self.hand_verts_of(i, state.finger[i], state.hand_rot[i], state.hand_off[i])

    def crossing(self, verts, T, faces=None, cap=4000):
        """Indices (into `faces`, default all hand faces) of hand triangles that cross the object's surface for object
        pose T: FCL triangle-triangle tests against a decimated copy of the mesh (same answers, faster); at most `cap`
        intersecting pairs are listed, so the count saturates for deep crossings."""
        F = self.hand_faces_i if faces is None else faces
        vo = ((verts - T[:3, 3]) @ T[:3, :3]).astype(np.float64)
        m = fcl.BVHModel()
        m.beginModel(len(vo), len(F))
        m.addSubModel(vo, F)
        m.endModel()
        res = fcl.CollisionResult()
        fcl.collide(fcl.CollisionObject(m, fcl.Transform()), self.col_obj,
                    fcl.CollisionRequest(num_max_contacts=cap, enable_contact=False), res)
        return np.unique([c.b1 for c in res.contacts]) if res.contacts else np.zeros(0, int)

    def hand_colors(self, state, i, verts=None):
        """Per-vertex hand colours for 3D views: red = crossing / inside the object, green = touching it (< 3 mm)."""
        v = self.hand_verts(state, i) if verts is None else verts
        T = state.obj[i]
        sd = self.sdf_at((v - T[:3, 3]) @ T[:3, :3])
        col = np.tile([90, 140, 235, 255], (len(v), 1))
        col[np.abs(sd) < 0.003] = [60, 200, 70, 255]
        col[sd < -0.003] = [230, 40, 40, 255]
        cf = self.crossing(v, T) if sd.min() < 0.01 else []
        if len(cf):
            col[np.unique(self.hand_faces[cf])] = [230, 40, 40, 255]
        return col

    # ---------- geometry helpers ----------

    def hand_kp_proj(self, state, i):
        """MANO's 21 joints projected into frame i (pixels, OpenPose order)."""
        p = (self.K @ (self.kp_reg @ self.hand_verts(state, i)).T).T
        return p[:, :2] / p[:, 2:3]

    def hand_kp_err(self, state, i):
        """Median pixel distance between MANO's joints and the detected keypoints (None without detections)."""
        ok = np.isfinite(self.kp2d[i, :, 0])
        if ok.sum() < 8:
            return None
        return float(np.median(np.linalg.norm(self.hand_kp_proj(state, i)[ok] - self.kp2d[i, ok], axis=1)))

    def obj_center(self, state, i):
        return state.obj[i, :3, :3] @ self.center_obj + state.obj[i, :3, 3]

    def axis(self, state, i, name):
        """Unit axis (camera frame) for edits: object_long/mid/short, world_up/right/forward, camera_x/y/z, ray."""
        if name.startswith("object_"):
            return state.obj[i, :3, :3] @ self.axes_obj[:, ["long", "mid", "short"].index(name[7:])]
        if name.startswith("world_"):
            return {"up": self.up, "right": self.right, "forward": self.fwd}[name[6:]]
        if name.startswith("camera_"):
            return np.eye(3)[["x", "y", "z"].index(name[7:])]
        if name == "ray":
            c = self.obj_center(state, i)
            return c / np.linalg.norm(c)
        raise ValueError(f"unknown axis {name}")

    def local_table_c(self, state, i, r_in=0.04, r_out=0.15, band=0.05):
        """Table height (n . p) right around the object's footprint in frame i, from that frame's depth: the
        median over depth points 4-15 cm horizontally from the footprint (widened to 30, then 50 cm when the hand
        hides that ring), within 5 cm of the global plane, outside the object/hand masks. The global plane can be
        2-5 cm off away from the camera; falls back to it."""
        n, c0 = self.table_n, self.table_n @ self.table_p
        key = (i, tuple(np.round(self.obj_center(state, i), 3)))
        if key in self._ltab:
            return self._ltab[key]
        if self._xyz is None:
            ys, xs = np.mgrid[0:self.hl, 0:self.wl]
            self._xyz = np.stack([(xs - self.Kl[0, 2]) / self.Kl[0, 0], (ys - self.Kl[1, 2]) / self.Kl[1, 1]], -1)
        d = self.dl[i]
        P = np.concatenate([self._xyz * d[..., None], d[..., None]], -1)
        h = P @ n
        c = self.obj_center(state, i)
        foot = c - (c @ n - c0) * n
        horiz = (P - foot) - ((P - foot) @ n)[..., None] * n
        rr = np.linalg.norm(horiz, axis=-1)
        ok = (d > 0.05) & (rr > r_in) & (np.abs(h - c0) < band) & ~self.om[i] & ~self.hm[i]
        val = float(c0)
        for ro in (r_out, 2 * r_out, 3.3 * r_out):       # widen the ring when the hand/arm hides the table around
            sel = ok & (rr < ro)                         # the object (the global plane can be ~5 cm off there)
            if sel.sum() > 30:
                val = float(np.median(h[sel]))
                break
        if len(self._ltab) > 4096:
            self._ltab.clear()
        self._ltab[key] = val
        return val

    def height_above_table(self, state, frames):
        """Lowest point of the object above the local table surface (m), per frame."""
        n = self.table_n
        return np.array([(self.sub_v @ state.obj[i, :3, :3].T + state.obj[i, :3, 3]).dot(n).min() - self.local_table_c(state, i)
                         for i in frames])

    # ---------- rendering (low-res, for metrics) ----------
    def render_obj_low(self, state, frames):
        out_d = []
        for s in range(0, len(frames), 48):
            fr = frames[s:s + 48]
            _, d, _ = nvdiffrast_render(K=self.Kl, H=self.hl, W=self.wl, glctx=self.glctx, mesh_tensors=self.mt,
                                        ob_in_cams=torch.as_tensor(state.obj[fr], device="cuda", dtype=torch.float))
            out_d.append(d.cpu().numpy())
        return np.concatenate(out_d)

    def hand_tensors(self, verts, colors=None):
        m = trimesh.Trimesh(verts, self.hand_faces, process=False)
        m.visual.vertex_colors = np.tile([90, 140, 235, 255], (len(verts), 1)) if colors is None else colors
        return make_mesh_tensors(m)

    def render_hand_low(self, state, frames):
        out = np.zeros((len(frames), self.hl, self.wl), np.float32)
        eye = torch.eye(4, device="cuda")[None]
        for k, i in enumerate(frames):
            _, d, _ = nvdiffrast_render(K=self.Kl, H=self.hl, W=self.wl, glctx=self.glctx,
                                        mesh_tensors=self.hand_tensors(self.hand_verts(state, i)), ob_in_cams=eye)
            out[k] = d[0].cpu().numpy()
        return out

    # ---------- metrics ----------
    def metrics(self, state, frames=None):
        frames = list(range(self.N)) if frames is None else list(frames)
        do = self.render_obj_low(state, frames)
        dh = self.render_hand_low(state, frames)
        kern = np.ones((3, 3), np.uint8)
        rows = []
        for k, i in enumerate(frames):
            so, sh = do[k] > 0, dh[k] > 0
            m, hm, d = self.om[i], self.hm[i] & ~self.om[i], self.dl[i]
            hmd = cv2.dilate(hm.astype(np.uint8), kern) > 0
            n_m, n_so = m.sum(), so.sum()
            r = {"frame": i}
            r["obj_mask_px"] = int(n_m) * LOW * LOW
            r["coverage"] = float((so & m).sum() / n_m) if n_m > 12 else None
            r["spill"] = float((so & ~m & ~hmd).sum() / max(n_so, 1)) if n_m > 12 else None
            r["iou"] = float((so & m).sum() / max(((so & ~hmd) | m).sum(), 1)) if n_m > 12 else None   # hand = occluder
            both = so & m & (d > 0) & ~(sh & (dh[k] < do[k]))
            r["depth_bias_cm"] = float(100 * np.median(do[k][both] - d[both])) if both.sum() > 8 else None
            r["depth_res_cm"] = float(100 * np.median(np.abs(do[k][both] - d[both]))) if both.sum() > 8 else None
            n_h = hm.sum()
            r["hand_coverage"] = float((sh & hm).sum() / n_h) if n_h > 12 else None
            r["hand_spill"] = float((sh & ~hm & ~m).sum() / max(sh.sum(), 1)) if n_h > 12 else None
            hb = sh & hm & (d > 0)
            r["hand_depth_bias_cm"] = float(100 * np.median(dh[k][hb] - d[hb])) if hb.sum() > 8 else None
            r["hand_kp_px"] = self.hand_kp_err(state, i)
            T = state.obj[i]
            hv = self.hand_verts(state, i)
            sd = self.sdf_at((hv - T[:3, 3]) @ T[:3, :3])
            r["contact_cm"] = float(100 * max(sd.min(), 0))
            r["penetration_cm"] = float(100 * max(-sd.min(), 0))
            r["n_penetrating"] = int((sd < -0.003).sum())
            cf = self.crossing(hv, T) if sd.min() < 0.01 else np.zeros(0, int)     # the SDF is coarse, not that coarse
            r["n_cross"] = int(len(cf))                   # hand triangles through the object's surface
            ff = np.bincount(self.face_finger[cf] + 1, minlength=6)[1:] if len(cf) else np.zeros(5, int)
            r["cross_fingers"] = [f for f, c in zip(FINGERS, ff) if c >= 3]
            r["tip_gap_cm"] = [round(float(100 * sd[d].min()), 2) for d in self.distal]   # each finger's last segment
            r["obj_height_cm"] = float(100 * self.height_above_table(state, [i])[0])
            r["hand_height_cm"] = float(100 * ((hv @ self.table_n).min() - self.local_table_c(state, i)))
            r["hand_valid"] = bool(self.hand_valid[i])
            rows.append(r)
        idx = {r["frame"]: r for r in rows}
        for i in frames:                                  # motion, from the neighbouring poses
            j = i - 1 if i > 0 else i + 1
            if 0 <= j < self.N:
                dR = Rotation.from_matrix(state.obj[i, :3, :3] @ state.obj[j, :3, :3].T).magnitude()
                idx[i]["ang_speed_deg_per_frame"] = float(np.degrees(dR))
                idx[i]["speed_cm_per_frame"] = float(100 * np.linalg.norm(self.obj_center(state, i) - self.obj_center(state, j)))
        return rows


def phase_of(state, i):
    for p in state.phases:
        if p["start"] <= i <= p["end"]:
            return p["label"]
    return None


def flag(r, phase):
    """Reasons a frame looks wrong, from its metrics (thresholds are deliberately loose: evidence is noisy)."""
    out = []
    if r.get("coverage") is not None and r["obj_mask_px"] > 3000:
        if r["coverage"] < 0.8:
            out.append("low coverage")
        if r["spill"] > 0.2:
            out.append("spill")
    if r.get("depth_res_cm") is not None and r["depth_res_cm"] > 2.0:
        out.append("depth off")
    if r.get("hand_kp_px") is not None and r["hand_kp_px"] > 25:
        out.append("hand off its keypoints")
    if r["penetration_cm"] > 0.6:
        out.append("hand penetrates object")
    if r.get("n_cross", 0) >= 6:
        out.append("hand through the object's surface" + (f" ({', '.join(r['cross_fingers'])})" if r.get("cross_fingers") else ""))
    if phase in ("rest", "approach", "release") and abs(r["obj_height_cm"]) > 1.0:
        out.append("object off table at rest")
    if r["obj_height_cm"] < -1.5:
        out.append("object below table")
    if r["hand_height_cm"] < -1.0:
        out.append("hand below table")
    if phase in ("grasp", "carry", "rotate", "lift", "place") and r["contact_cm"] > 2.0:
        out.append("hand not touching object")
    if r.get("ang_speed_deg_per_frame", 0) > 12:
        out.append("rotation jump")
    return out


def summarize(clip, state, rows, window=None):
    """Compact text summary for the agent: per-window medians and flagged frame runs."""
    window = window or max(10, clip.N // 12)
    keys = ["coverage", "spill", "depth_res_cm", "depth_bias_cm", "hand_kp_px", "hand_depth_bias_cm", "contact_cm",
            "penetration_cm", "n_cross", "obj_height_cm", "ang_speed_deg_per_frame"]
    lines = ["frames    phase      " + " ".join(f"{k[:11]:>11}" for k in keys)]
    for s in range(rows[0]["frame"], rows[-1]["frame"] + 1, window):
        win = [r for r in rows if s <= r["frame"] < s + window]
        if not win:
            continue
        vals = []
        for k in keys:
            x = [r[k] for r in win if r.get(k) is not None]
            vals.append(f"{np.median(x):11.2f}" if x else f"{'-':>11}")
        ph = phase_of(state, win[len(win) // 2]["frame"]) or "-"
        lines.append(f"{win[0]['frame']:>3}-{win[-1]['frame']:<4}  {ph:<9} " + " ".join(vals))
    flagged = {}
    for r in rows:
        for reason in flag(r, phase_of(state, r["frame"])):
            flagged.setdefault(reason, []).append(r["frame"])
    runs = []
    for reason, fr in flagged.items():
        start = prev = fr[0]
        for f in fr[1:] + [None]:
            if f is None or f > prev + 2:
                if prev - start >= 1 or reason in ("rotation jump",):
                    runs.append(f"{reason}: {start}-{prev}")
                if f is not None:
                    start = f
            if f is not None:
                prev = f
    lines.append("flagged: " + ("; ".join(runs) if runs else "none"))
    return "\n".join(lines)
