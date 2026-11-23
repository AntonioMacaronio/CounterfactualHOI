"""One clip for the refinement agent: fixed inputs, editable state, and per-frame metrics.

Inputs come from the FoundationPose + Depth Pro 2 run ($PROJECT_STORAGE_ROOT/runs/fprecon/VID) and CHOIR's stage 1
($CHOIR_ROOT/output/VID): RGB frames, SAM3 masks, hand-scaled metric depth, the textured SAM3D mesh, Dyn-HaMR hands
(moved onto the depth map), GeoCalib gravity and the table plane fitted in the frame-0 depth.

State (what the agent edits): per-frame object pose (object -> OpenCV camera, metric), the hand as MANO parameters
only (per frame: global orientation, translation, 15 finger-joint rotations; one shape per clip; scripts/refine/mano.py),
and phase labels. Every edit makes a new version; nothing is overwritten (Versions).

Metrics are computed on GPU at 1/4 resolution: object silhouette coverage / spill against the SAM3 mask (as in
scripts/fprecon/verify_track.py), depth residual and bias against the depth map, the hand's silhouette fit, hand-object
contact distance and penetration (from a signed-distance grid of the object), hand triangles crossing the object's
surface (FCL triangle-triangle tests: catches fingers through a thin wall, which a coarse SDF misses), heights above the table surface measured
locally around the object (the global plane fit can be ~2 cm off far from the camera), the hand's distance to its 2D keypoints
(ViTPose, from CHOIR stage 1; MANO joints from Dyn-HaMR's mesh via MANO's joint regressor), height above the table
and motion: speeds and accelerations (second differences: jitter) of the object, of the hand mesh (in the camera and
relative to the object) and of the finger angles.
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
sys.path.insert(0, HERE)
import mano  # noqa: E402


@dataclass
class State:
    obj: np.ndarray                         # (N, 4, 4) object -> camera
    go: np.ndarray                          # (N, 3) MANO global orientation (axis-angle, camera frame)
    tr: np.ndarray                          # (N, 3) MANO translation (camera frame, metres)
    hp: np.ndarray                          # (N, 45) MANO finger-joint rotations (axis-angle, flat-hand-mean)
    phases: list = field(default_factory=list)   # [{"start", "end", "label"}]
    verts: np.ndarray = None                # (N, 778, 3) fixed hand vertices: read-only results from before MANO states
    betas: np.ndarray = None                # (10,) MANO shape, one per clip (None = the hand tracker's, Clip.betas)

    def copy(self):
        return State(self.obj.copy(), self.go.copy(), self.tr.copy(), self.hp.copy(), [dict(p) for p in self.phases],
                     None if self.verts is None else self.verts, None if self.betas is None else self.betas.copy())


def load_state(path, phases=None):
    """A State from a versions/vNNN.npz or final.npz. Runs from before the MANO state (rounds 1-5: a rigid motion and
    vertex-level finger flexion) load read-only, with their exported hand vertices (export/hand_verts.npy)."""
    z = np.load(path)
    if "hand_pose" in z.files:
        return State(z["obj"], z["global_orient"], z["transl"], z["hand_pose"], phases or [],
                     betas=z["betas"] if "betas" in z.files else None)
    N = len(z["obj"])
    hv = os.path.join(os.path.dirname(path), "export", "hand_verts.npy")
    if not os.path.exists(hv):
        hv = os.path.join(os.path.dirname(os.path.dirname(path)), "export", "hand_verts.npy")
    return State(z["obj"], np.zeros((N, 3)), np.zeros((N, 3)), np.zeros((N, 45)), phases or [],
                 np.load(hv) if os.path.exists(hv) else None)


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
        _, self.hand_faces, _, _ = corrected_hands(vd, run)          # Dyn-HaMR's (closed-wrist) MANO faces
        P = mano.tracker_params_camera(vd, run, self.side)            # the hand tracker's MANO parameters, our frame
        self.N = min(len(poses), len(P["hand_pose"]), self.depth.shape[0])
        poses = poses[:self.N]
        self.hand_valid = P["valid"][:self.N]
        self.betas = P["betas"]
        self.J0 = mano.root_joint(self.betas, self.side)
        self._hv_cache = {}
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
        self.distal = [np.nonzero(self.skin_W[:, CHAINS[f][2]] > 0.5)[0] for f in FINGERS]   # last segment (pad + tip)
        self.palm_verts = np.nonzero(self.vert_finger == -1)[0]

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
        self.init_state = State(poses.copy(), P["global_orient"][:self.N].copy(), P["transl"][:self.N].copy(),
                                P["hand_pose"][:self.N].copy(), [])
        # per finger joint, its typical bending direction in this clip (the mean of the tracker's rotation vectors,
        # which flexion dominates): curl_finger turns joints about it
        hp = P["hand_pose"][:self.N][self.hand_valid].reshape(-1, 15, 3)
        mean = hp.mean(0)
        self.flex_axis = mean / np.maximum(np.linalg.norm(mean, axis=1, keepdims=True), 1e-6)
        # how smooth the hand tracker's own motion is (median hand-mesh acceleration): the reference for jitter
        acc = self.motion(self.init_state, range(self.N))
        ref = [a["hand_accel_mm"] for a in acc.values() if a.get("hand_accel_mm") is not None]
        self.jitter_ref = float(np.median(ref)) if ref else 2.0

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

    def _sdf_grid(self, n=128):
        """Signed distance to the object surface on a grid in the object frame (negative inside), cached."""
        path = os.path.join(self.run, f"refine_sdf{n}.npz")
        if os.path.exists(path):
            z = np.load(path)
            self.sdf, self.sdf_lo, self.sdf_hi = z["sdf"], z["lo"], z["hi"]
            return
        v = np.asarray(self.mesh.vertices)
        pad = 0.25 * np.ptp(v, 0).max()
        lo, hi = v.min(0) - pad, v.max(0) + pad
        g = np.stack(np.meshgrid(*[np.linspace(lo[k], hi[k], n) for k in range(3)], indexing="ij"), -1).reshape(-1, 3)
        import open3d as o3d                                 # closest points (open3d's BVH: seconds, not minutes)
        scene = o3d.t.geometry.RaycastingScene()
        scene.add_triangles(o3d.core.Tensor(np.asarray(self.mesh.vertices, np.float32)),
                            o3d.core.Tensor(np.asarray(self.mesh.faces, np.uint32)))
        q = scene.compute_closest_points(o3d.core.Tensor(g.astype(np.float32)))
        closest, tri = q["points"].numpy().astype(np.float64), q["primitive_ids"].numpy().astype(np.int64)
        dist = np.linalg.norm(g - closest, axis=1)
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

    # ---------- hand model (MANO parameters) ----------
    def hand_verts_batch(self, state, frames):
        """Hand vertices (F, 778, 3) for `frames` (one batched MANO forward)."""
        frames = list(frames)
        if state.verts is not None:
            return state.verts[frames].astype(np.float64)
        return mano.mano_forward(state.go[frames], state.tr[frames], state.hp[frames], self.betas_of(state), self.side)

    def hand_verts(self, state, i):
        if state.verts is not None:
            return state.verts[i].astype(np.float64)
        key = (i, state.go[i].tobytes(), state.tr[i].tobytes(), state.hp[i].tobytes(),
               None if state.betas is None else state.betas.tobytes())
        v = self._hv_cache.get(key)
        if v is None:
            if len(self._hv_cache) > 20000:
                self._hv_cache.clear()
            v = self._hv_cache[key] = self.hand_verts_batch(state, [i])[0]
        return v

    def betas_of(self, state):
        return self.betas if state.betas is None else state.betas

    def J0_of(self, state):
        return self.J0 if state.betas is None else mano.root_joint(state.betas, self.side)

    def hand_rigid(self, state, i, R=None, d=None, center=None):
        """New (go, tr) of frame i after rotating its hand by R about `center` (default: its vertex centroid) and
        translating it by d (camera frame): MANO rotates about its root joint J0, so the translation absorbs the rest."""
        R = np.eye(3) if R is None else R
        d = np.zeros(3) if d is None else d
        c = self.hand_verts(state, i).mean(0) if center is None else center
        go = Rotation.from_matrix(R @ Rotation.from_rotvec(state.go[i]).as_matrix()).as_rotvec()
        J0 = self.J0_of(state)
        tr = R @ (J0 + state.tr[i] - c) + c + d - J0
        return go, tr

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

    def render_hand_low(self, state, frames, hv=None):
        out = np.zeros((len(frames), self.hl, self.wl), np.float32)
        eye = torch.eye(4, device="cuda")[None]
        for k, i in enumerate(frames):
            v = hv[i] if hv is not None and i in hv else self.hand_verts(state, i)
            _, d, _ = nvdiffrast_render(K=self.Kl, H=self.hl, W=self.wl, glctx=self.glctx,
                                        mesh_tensors=self.hand_tensors(v), ob_in_cams=eye)
            out[k] = d[0].cpu().numpy()
        return out

    def grasp(self, hv, T, sd=None):
        """Is the object gripped? Touching (< 5 mm): the thumb's and each finger's last segment, and the palm. A grip
        is opposed when the thumb or the palm and at least one other finger touch on sides of the object facing
        roughly opposite ways (outward surface normals at the contacts, from the SDF gradient, more than ~110 deg apart);
        one fingertip resting on the object is not a grip."""
        po = (hv - T[:3, 3]) @ T[:3, :3]
        sd = self.sdf_at(po) if sd is None else sd
        parts = {f: self.distal[k] for k, f in enumerate(FINGERS)}
        parts["palm"] = self.palm_verts
        touch, normal = {}, {}
        eps = 0.002
        for name, idx in parts.items():
            j = idx[int(np.argmin(sd[idx]))]
            touch[name] = bool(sd[j] < 0.005)
            if touch[name]:
                g = np.array([(self.sdf_at(po[[j]] + eps * e) - self.sdf_at(po[[j]] - eps * e))[0] / (2 * eps) for e in np.eye(3)])
                normal[name] = g / (np.linalg.norm(g) + 1e-9)
        opposed = any(touch[a] and touch[b] and normal[a] @ normal[b] < -0.35
                      for a in ("thumb", "palm") for b in ("index", "middle", "ring", "pinky"))
        return {"n_touching": int(sum(touch[f] for f in FINGERS)), "thumb_touching": touch["thumb"],
                "palm_touching": touch["palm"], "opposed": bool(opposed), "thumb_gap_cm": min(5.0, float(100 * sd[self.distal[0]].min()))}

    # ---------- motion ----------
    def motion(self, state, frames, hv=None):
        """Per-frame speed and acceleration (second difference) of the object (center, rotation), the hand mesh (mean
        over its vertices, in the camera and relative to the object) and the finger angles; None where a neighbour
        is missing or the hand is not tracked."""
        if hv is None:
            need = sorted({j for i in frames for j in (i - 1, i, i + 1) if 0 <= j < self.N})
            hv = dict(zip(need, self.hand_verts_batch(state, need)))
        get = lambda j: hv.setdefault(j, self.hand_verts(state, j).astype(np.float64))
        out = {}
        for i in frames:
            r = {}
            j = i - 1 if i > 0 else i + 1
            if 0 <= j < self.N:
                r["ang_speed_deg_per_frame"] = float(np.degrees(Rotation.from_matrix(state.obj[i, :3, :3] @ state.obj[j, :3, :3].T).magnitude()))
                r["speed_cm_per_frame"] = float(100 * np.linalg.norm(self.obj_center(state, i) - self.obj_center(state, j)))
            if 0 < i < self.N - 1:
                c = [self.obj_center(state, k) for k in (i - 1, i, i + 1)]
                r["obj_accel_cm"] = float(100 * np.linalg.norm(c[2] - 2 * c[1] + c[0]))
                w1 = Rotation.from_matrix(state.obj[i, :3, :3] @ state.obj[i - 1, :3, :3].T).as_rotvec()
                w2 = Rotation.from_matrix(state.obj[i + 1, :3, :3] @ state.obj[i, :3, :3].T).as_rotvec()
                r["obj_ang_accel_deg"] = float(np.degrees(np.linalg.norm(w2 - w1)))
                if self.hand_valid[i - 1:i + 2].all():
                    v = [get(k) for k in (i - 1, i, i + 1)]
                    r["hand_accel_mm"] = float(1000 * np.linalg.norm(v[2] - 2 * v[1] + v[0], axis=1).mean())
                    rel = [(v[n] - state.obj[k, :3, 3]) @ state.obj[k, :3, :3] for n, k in enumerate((i - 1, i, i + 1))]
                    r["hand_rel_accel_mm"] = float(1000 * np.linalg.norm(rel[2] - 2 * rel[1] + rel[0], axis=1).mean())
                    r["finger_accel_deg"] = float(np.degrees(np.abs(state.hp[i + 1] - 2 * state.hp[i] + state.hp[i - 1])).mean())
            out[i] = r
        return out

    # ---------- metrics ----------
    def metrics(self, state, frames=None):
        frames = list(range(self.N)) if frames is None else list(frames)
        need = sorted({j for i in frames for j in (i - 1, i, i + 1) if 0 <= j < self.N})
        hvc = dict(zip(need, self.hand_verts_batch(state, need)))     # one batched MANO forward
        do = self.render_obj_low(state, frames)
        dh = self.render_hand_low(state, frames, hvc)
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
            hv = hvc.setdefault(i, self.hand_verts(state, i).astype(np.float64))
            sd = self.sdf_at((hv - T[:3, 3]) @ T[:3, :3])
            r["contact_cm"] = float(100 * max(sd.min(), 0))
            r["penetration_cm"] = float(100 * max(-sd.min(), 0))
            r["n_penetrating"] = int((sd < -0.003).sum())
            cf = self.crossing(hv, T) if sd.min() < 0.01 else np.zeros(0, int)     # the SDF is coarse, not that coarse
            r["n_cross"] = int(len(cf))                   # hand triangles through the object's surface
            ff = np.bincount(self.face_finger[cf] + 1, minlength=6)[1:] if len(cf) else np.zeros(5, int)
            r["cross_fingers"] = [f for f, c in zip(FINGERS, ff) if c >= 3]
            r["tip_gap_cm"] = [round(float(100 * sd[d].min()), 2) for d in self.distal]   # each finger's last segment
            r.update(self.grasp(hv, T, sd))
            r["obj_height_cm"] = float(100 * self.height_above_table(state, [i])[0])
            r["hand_height_cm"] = float(100 * ((hv @ self.table_n).min() - self.local_table_c(state, i)))
            r["hand_valid"] = bool(self.hand_valid[i])
            r["_jitter_ref"] = self.jitter_ref
            rows.append(r)
        mo = self.motion(state, frames, hvc)              # motion and jitter, from the neighbouring frames
        for r in rows:
            r.update(mo[r["frame"]])
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
    if phase in ("lift", "carry", "rotate") and r.get("opposed") is False:
        out.append("no opposing grip (thumb or palm vs fingers)")
    if r.get("hand_depth_bias_cm") is not None and abs(r["hand_depth_bias_cm"]) > 1.5:
        out.append("hand off the depth map")
    if r.get("ang_speed_deg_per_frame", 0) > 12:
        out.append("rotation jump")
    if r.get("hand_accel_mm") is not None and r["hand_accel_mm"] > max(4.0, 2.5 * r.get("_jitter_ref", 2.0)):
        out.append("hand jitter")
    if r.get("finger_accel_deg") is not None and r["finger_accel_deg"] > 8.0:
        out.append("finger jitter")
    return out


def summarize(clip, state, rows, window=None):
    """Compact text summary for the agent: per-window medians and flagged frame runs."""
    window = window or max(10, clip.N // 12)
    keys = ["coverage", "spill", "depth_res_cm", "depth_bias_cm", "hand_kp_px", "hand_depth_bias_cm", "contact_cm",
            "penetration_cm", "n_cross", "obj_height_cm", "ang_speed_deg_per_frame", "hand_accel_mm", "finger_accel_deg"]
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
