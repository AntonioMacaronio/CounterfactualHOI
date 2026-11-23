"""Edit operations on a clip's State, for the refinement agent. Each returns (new_state, touched_frames, note).

Geometric edits are stated in frames the agent can reason about from the images (see Clip.axis): rotate about the
object's own long/mid/short axis or gravity's up/right/forward, translate in gravity-aligned world axes (right,
forward = away from the camera, up) or camera axes, along the viewing ray. A range edit applies fully on
[start, end] and fades out over `falloff` frames on each side (cosine), so it blends into the untouched frames.

Search-backed edits: polish_object runs FoundationPose's refiner from the current pose on the hand-scaled depth and
keeps a frame's result only if it fits the SAM3 mask + depth better and stays within a trust region (so it sharpens a
pose without jumping to another solution); fit_mano is a gradient-based fit of the hand's MANO parameters (the only
hand representation: no vertex edits) to its 2D keypoints, SAM3 mask and the depth, with priors, no penetration and a
grip in held phases; propose_object_poses returns FoundationPose's top registration hypotheses at a frame for the agent to choose from
(apply_candidate).
"""

import os

import cv2
import numpy as np
import torch
from scipy.spatial.transform import Rotation

from clip import FINGERS, LOW, TIP_IDS, TIPS, Clip, State, se3_interp  # noqa: F401
import mano  # noqa: E402
from Utils import nvdiffrast_render  # noqa: E402

_EST = {}


def weights(N, start, end, falloff=0):
    start, end = max(0, int(start)), min(N - 1, int(end))
    w = np.zeros(N)
    w[start:end + 1] = 1
    for k in range(1, int(falloff) + 1):
        v = 0.5 * (1 + np.cos(np.pi * k / (falloff + 1)))
        if start - k >= 0:
            w[start - k] = max(w[start - k], v)
        if end + k < N:
            w[end + k] = max(w[end + k], v)
    return w


def _touched(w):
    return [int(i) for i in np.nonzero(w > 0)[0]]


def whittaker(x, window=12.0, w=None):
    """Smooth x (T, d) along time: min sum w_t |x_t - x0_t|^2 + lam sum |second difference|^2, lam from the
    smoothing period `window` (frames; motion slower than that passes). Removes jitter, keeps the motion."""
    x = np.asarray(x, float)
    T = len(x)
    if T < 4 or window <= 1:
        return x.copy()
    lam = (window / (2 * np.pi)) ** 4
    D = np.diff(np.eye(T), 2, axis=0)
    W = np.diag(np.ones(T) if w is None else np.asarray(w, float))
    return np.linalg.solve(W + lam * D.T @ D, W @ x.reshape(T, -1)).reshape(x.shape)


def runs_of(frames):
    """Split sorted frame indices into runs of consecutive frames."""
    out = []
    for i in frames:
        if out and i == out[-1][-1] + 1:
            out[-1].append(i)
        else:
            out.append([i])
    return out


def _pin(n, start, end, N, run):
    """Weights for whittaker(): strong at a run's ends that border untouched frames, so the smoothed range joins them."""
    w = np.ones(len(run))
    if run[0] > 0 and run[0] == start:
        w[0] = 1e4
    if run[-1] < N - 1 and run[-1] == end:
        w[-1] = 1e4
    return w


def world_vec(clip, dx, dy, dz, frame):
    if frame == "world":                       # right, forward (away from camera, level), up (gravity)
        return dx * clip.right + dy * clip.fwd + dz * clip.up
    if frame == "camera":                      # x right, y down, z forward (depth)
        return np.array([dx, dy, dz], float)
    raise ValueError("frame must be world or camera")


def rotate_object(clip, state, start, end, axis, angle_deg, falloff=0):
    s = state.copy()
    w = weights(clip.N, start, end, falloff)
    for i in _touched(w):
        a = clip.axis(state, i, axis)
        R = Rotation.from_rotvec(np.radians(angle_deg) * w[i] * a).as_matrix()
        c = clip.obj_center(state, i)
        s.obj[i, :3, :3] = R @ state.obj[i, :3, :3]
        s.obj[i, :3, 3] = R @ (state.obj[i, :3, 3] - c) + c
    return s, _touched(w), f"rotate object {angle_deg:+.1f} deg about {axis} on {start}-{end} (falloff {falloff})"


def translate_object(clip, state, start, end, dx_cm, dy_cm, dz_cm, frame="world", falloff=0):
    s = state.copy()
    w = weights(clip.N, start, end, falloff)
    d = world_vec(clip, dx_cm, dy_cm, dz_cm, frame) / 100
    for i in _touched(w):
        s.obj[i, :3, 3] += w[i] * d
    return s, _touched(w), f"translate object ({dx_cm:+.1f},{dy_cm:+.1f},{dz_cm:+.1f}) cm [{frame}] on {start}-{end}"


def move_along_ray(clip, state, start, end, dist_cm, falloff=0):
    """Move the object along the camera ray through its center: depth changes, image position does not."""
    s = state.copy()
    w = weights(clip.N, start, end, falloff)
    for i in _touched(w):
        c = clip.obj_center(state, i)
        s.obj[i, :3, 3] += w[i] * dist_cm / 100 * c / np.linalg.norm(c)
    return s, _touched(w), f"move object {dist_cm:+.1f} cm along the viewing ray on {start}-{end}"


def interpolate_object(clip, state, start, end):
    """Replace the poses strictly between start and end by an SE(3) interpolation of the two endpoint poses."""
    s = state.copy()
    for i in range(start + 1, end):
        s.obj[i] = se3_interp(state.obj[start], state.obj[end], (i - start) / (end - start))
    return s, list(range(start + 1, end)), f"interpolate object poses between {start} and {end}"


def hold_object_static(clip, state, start, end, ref):
    s = state.copy()
    s.obj[start:end + 1] = state.obj[ref]
    return s, list(range(start, end + 1)), f"hold object static at frame {ref}'s pose on {start}-{end}"


def snap_object_to_table(clip, state, start, end, mode="per_frame"):
    """Translate along the table normal so the object's lowest point touches the table."""
    s = state.copy()
    fr = list(range(start, end + 1))
    h = clip.height_above_table(state, fr)
    if mode == "constant":
        h = np.full_like(h, np.median(h))
    for i, hi in zip(fr, h):
        s.obj[i, :3, 3] -= hi * clip.table_n
    return s, fr, f"snap object onto the table on {start}-{end} ({mode}; median shift {100 * np.median(h):+.1f} cm)"


def smooth_object(clip, state, start, end, window=5):
    s = state.copy()
    half = max(1, window // 2)
    for i in range(start, end + 1):
        lo, hi = max(0, i - half), min(clip.N - 1, i + half)
        s.obj[i, :3, 3] = state.obj[lo:hi + 1, :3, 3].mean(0)
        s.obj[i, :3, :3] = Rotation.from_matrix(state.obj[lo:hi + 1, :3, :3]).mean().as_matrix()
    return s, list(range(start, end + 1)), f"smooth object poses on {start}-{end} (window {window})"


def move_hand(clip, state, start, end, dx_cm, dy_cm, dz_cm, frame="world", rot_axis=None, rot_deg=0.0, falloff=0):
    """Rigidly move the hand (MANO global orientation + translation): rotate about its centroid, then translate."""
    s = state.copy()
    w = weights(clip.N, start, end, falloff)
    d = world_vec(clip, dx_cm, dy_cm, dz_cm, frame) / 100
    for i in _touched(w):
        R = None
        if rot_axis and rot_deg:
            R = Rotation.from_rotvec(np.radians(rot_deg) * w[i] * clip.axis(state, i, rot_axis)).as_matrix()
        s.go[i], s.tr[i] = clip.hand_rigid(state, i, R, w[i] * d)
    return s, _touched(w), (f"move hand ({dx_cm:+.1f},{dy_cm:+.1f},{dz_cm:+.1f}) cm [{frame}]"
                            + (f", rotate {rot_deg:+.1f} deg about {rot_axis}" if rot_axis and rot_deg else "") + f" on {start}-{end}")


def attach_hand_to_object(clip, state, start, end, ref):
    """Keep the hand's centroid fixed relative to the object over [start, end] (as at frame ref); fingers unchanged."""
    s = state.copy()
    T = state.obj[ref]
    c_rel = T[:3, :3].T @ (clip.hand_verts(state, ref).mean(0) - T[:3, 3])
    for i in range(start, end + 1):
        target = state.obj[i, :3, :3] @ c_rel + state.obj[i, :3, 3]
        s.tr[i] = state.tr[i] + target - clip.hand_verts(state, i).mean(0)
    s, extra = _fade_edges(clip, state, s, start, end)   # no jump at the range's ends
    return s, sorted(set(range(start, end + 1)) | set(extra)), (
        f"attach hand to object on {start}-{end} (relative position of frame {ref}), fading out over {FADE} frames beyond the range")


# ---------- FoundationPose-backed ----------
def _fp(clip):
    if clip.vid not in _EST:
        from estimater import FoundationPose, PoseRefinePredictor, ScorePredictor
        import logging
        logging.disable(logging.INFO)
        m = clip.mesh
        _EST[clip.vid] = FoundationPose(model_pts=m.vertices, model_normals=m.vertex_normals, mesh=m,
                                        scorer=ScorePredictor(), refiner=PoseRefinePredictor(), glctx=clip.glctx,
                                        debug=0, debug_dir=os.path.join(clip.run, "fpose_debug"))
    return _EST[clip.vid]


def _fp_inputs(clip, i):
    rgb = cv2.cvtColor(clip.rgb(i), cv2.COLOR_BGR2RGB)
    d = clip.depth_full(i)
    om, hm = clip.masks_full(i)
    hm = (cv2.dilate(hm.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0) & ~om
    d[hm] = 0
    return rgb, d, om


def _set_last(est, pose):
    est.pose_last = torch.as_tensor(pose @ np.linalg.inv(est.get_tf_to_centered_mesh().cpu().numpy()),
                                    device="cuda", dtype=torch.float)


def frame_quality(r):
    q = 0.0
    if r.get("coverage") is not None:
        q += r["coverage"] - r["spill"]
    if r.get("depth_res_cm") is not None:
        q -= 0.05 * r["depth_res_cm"]
    return q


def polish_object(clip, state, start, end, max_rot_deg=6.0, max_trans_cm=1.5, iterations=3):
    est = _fp(clip)
    s = state.copy()
    kept = []
    for i in range(start, end + 1):
        rgb, d, om = _fp_inputs(clip, i)
        _set_last(est, state.obj[i])
        p = est.track_one(rgb=rgb, depth=d, K=clip.K, iteration=iterations)
        dR = np.degrees(Rotation.from_matrix(p[:3, :3] @ state.obj[i, :3, :3].T).magnitude())
        dt = 100 * np.linalg.norm(p[:3, 3] - state.obj[i, :3, 3])
        if dR > max_rot_deg or dt > max_trans_cm:
            continue
        trial = state.copy()
        trial.obj[i] = p
        if frame_quality(clip.metrics(trial, [i])[0]) > frame_quality(clip.metrics(state, [i])[0]) + 0.003:
            s.obj[i] = p
            kept.append(i)
    return s, kept, f"polish object on {start}-{end}: refined {len(kept)}/{end - start + 1} frames (trust region {max_rot_deg} deg, {max_trans_cm} cm)"


def propose_object_poses(clip, state, i, k=6):
    """FoundationPose registration at frame i: its top-k pose hypotheses (best-scored first)."""
    est = _fp(clip)
    rgb, d, om = _fp_inputs(clip, i)
    if om.sum() < 200:
        raise ValueError(f"frame {i} has no usable object mask to register from")
    est.register(K=clip.K, rgb=rgb, depth=d, ob_mask=om, iteration=5)
    tf = est.get_tf_to_centered_mesh()
    return [(est.poses[j] @ tf).cpu().numpy() for j in range(min(k, len(est.poses)))]


def apply_candidate(clip, state, i, cand_pose, start, end, falloff=0):
    """Set frame i to cand_pose and apply the same change (rotation about the object center + shift) on [start, end]."""
    s = state.copy()
    R_d = cand_pose[:3, :3] @ state.obj[i, :3, :3].T
    c_old, c_new = clip.obj_center(state, i), cand_pose[:3, :3] @ clip.center_obj + cand_pose[:3, 3]
    rv, t_d = Rotation.from_matrix(R_d).as_rotvec(), c_new - c_old
    w = weights(clip.N, start, end, falloff)
    w[i] = 1.0
    for j in _touched(w):
        R = Rotation.from_rotvec(w[j] * rv).as_matrix()
        c = clip.obj_center(state, j)
        s.obj[j, :3, :3] = R @ state.obj[j, :3, :3]
        s.obj[j, :3, 3] = R @ (state.obj[j, :3, 3] - c) + c + w[j] * t_d
    return s, _touched(w), f"apply registration candidate at frame {i} to {start}-{end} (rotation {np.degrees(np.linalg.norm(rv)):.1f} deg, shift {100 * np.linalg.norm(t_d):.1f} cm)"


# ---------- hand (MANO parameters) ----------
HELD = ("grasp", "lift", "carry", "rotate", "place")
TIP_SMOOTH = 2.0            # weight of the fingertips + wrist in fit_mano's smoothness, relative to the joints'
CHAIN_JOINTS = {"thumb": (13, 14, 15), "index": (1, 2, 3), "middle": (4, 5, 6), "ring": (10, 11, 12), "pinky": (7, 8, 9)}
CURL_SHARE = (0.4, 0.35, 0.25)


def _finger_names(fingers):
    if not fingers or fingers == "all" or fingers == ["all"]:
        return list(FINGERS)
    bad = [f for f in fingers if f not in FINGERS]
    if bad:
        raise ValueError(f"unknown fingers {bad}; use {FINGERS} or 'all'")
    return list(fingers)


def curl_finger(clip, state, start, end, fingers, delta_deg, falloff=0):
    """Turn the named fingers' MANO joints (base to tip: 40 / 35 / 25 % of delta_deg) about each joint's typical
    bending direction in this clip (curl > 0, open < 0)."""
    s = state.copy()
    w = weights(clip.N, start, end, falloff)
    names = _finger_names(fingers)
    for i in _touched(w):
        hp = s.hp[i].reshape(15, 3)
        for f in names:
            for j, share in zip(CHAIN_JOINTS[f], CURL_SHARE):
                k = j - 1
                R = Rotation.from_rotvec(np.radians(w[i] * share * delta_deg) * clip.flex_axis[k]) * Rotation.from_rotvec(hp[k])
                hp[k] = R.as_rotvec()
        s.hp[i] = hp.reshape(45)
    return s, _touched(w), f"curl {', '.join(names)} by {delta_deg:+.0f} deg (MANO joints) on {start}-{end} (falloff {falloff})"


def fit_mano(clip, state, start, end, mode="all", contact=None, iters=600, kp_weight=2.0, depth_weight=0.5,
             smooth_weight=1.0, contact_weight=1.0, lr=1.0, shape=False, tip_smooth=TIP_SMOOTH):
    """Gradient-based fit of the MANO parameters of every frame in [start, end] at once (torch, Adam): the 2D
    keypoints (robust, 15 px units), the depth map on the hand's mask (vertices the depth map sees), staying on the
    hand/object masks, MANO's hand-pose prior (energy outside the top 15 PCA components) and a weak pull to the hand
    tracker's pose, smoothness over time (joint angles, and the positions of the joints, the fingertips and the
    wrist: without the fingertips the fit follows the keypoints' frame-to-frame noise there), no penetration into the object
    (its SDF) or below the table, and in held phases (set_phases; contact=False turns it off) a grip: the thumb or palm
    and the nearest finger on the object's surface. mode: "all", "fingers" (joint rotations only) or "rigid".
    The two frames on each side of the range stay fixed and the fit joins them smoothly (the smoothness term spans the
    range's ends, and a jump there costs as much as anywhere, however long the range): no seam at the ends.
    shape=True also fits the clip's one MANO shape (betas, with a prior towards the current one; kept in the state)
    (the automatic fit does this once: a hand of the wrong size cannot match both the keypoints and the depth)."""
    import torch
    import torch.nn.functional as F
    fr = list(range(start, end + 1))
    n = len(fr)
    dev = "cuda"
    T = lambda a: torch.as_tensor(np.asarray(a, np.float32), device=dev)
    go0, tr0, hp0 = T(state.go[fr]), T(state.tr[fr]), T(state.hp[fr])
    hp_trk, betas0 = T(clip.init_state.hp[fr]), T(clip.betas_of(state))
    pre = [i for i in (start - 2, start - 1) if i >= 0]              # the frames just outside the range: fixed
    post = [i for i in (end + 1, end + 2) if i < clip.N]
    tip_ids = torch.as_tensor(TIP_IDS, device=dev)
    pts = lambda v_, J_: torch.cat([v_[:, tip_ids], J_[:, :1]], 1)   # the 5 fingertips and the wrist

    def fixed(ids):
        if not ids:
            return torch.zeros((0, 45), device=dev), torch.zeros((0, 16, 3), device=dev), torch.zeros((0, 6, 3), device=dev)
        v_, J_ = mano.mano_forward_torch(T(state.go[ids]), T(state.tr[ids]), T(state.hp[ids]), betas0, clip.side)
        return T(state.hp[ids]), J_, pts(v_, J_)
    with torch.no_grad():
        (hp_pre, J_pre, p_pre), (hp_post, J_post, p_post) = fixed(pre), fixed(post)
    # second-difference stencils (frames k, k+1, k+2 of pre + range + post) that reach outside the range
    edge = [k for k in range(len(pre) + n + len(post) - 2) if k < len(pre) or k + 2 >= len(pre) + n]
    dbeta = torch.zeros_like(betas0, requires_grad=shape)
    dgo = torch.zeros_like(go0, requires_grad=mode in ("all", "rigid"))
    dtr = torch.zeros_like(tr0, requires_grad=mode in ("all", "rigid"))
    dhp = torch.zeros_like(hp0, requires_grad=mode in ("all", "fingers"))
    groups = [{"params": [p], "lr": lr * a} for p, a in ((dgo, 0.02), (dtr, 0.004), (dhp, 0.03), (dbeta, 0.02)) if p.requires_grad]
    if not groups:
        raise ValueError("mode must be all, fingers or rigid")
    valid = T([float(clip.hand_valid[i]) for i in fr])
    kp = clip.kp2d[fr]
    kp_m = T(np.isfinite(kp[..., 0]).astype(np.float32)) * valid[:, None]
    kp_t = T(np.nan_to_num(kp))
    K = T(clip.K)
    hl, wl = clip.hl, clip.wl
    dl = T(clip.dl[fr])[:, None]
    hm = T((clip.hm[fr] & ~clip.om[fr]).astype(np.float32))[:, None]
    dtm = T(np.stack([cv2.distanceTransform((~(clip.hm[i] | clip.om[i])).astype(np.uint8), cv2.DIST_L2, 3) for i in fr]))[:, None]
    has_mask = T([float((clip.hm[i] | clip.om[i]).sum() > 30) for i in fr]) * valid
    sdf, lo, hi = T(clip.sdf)[None, None], T(clip.sdf_lo), T(clip.sdf_hi)
    Ro, to = T(state.obj[fr, :3, :3]), T(state.obj[fr, :3, 3])
    held = T([float(contact is not False and any(p["start"] <= i <= p["end"] and p["label"] in HELD for p in state.phases))
              for i in fr]) * valid
    reg = T(clip.kp_reg)
    m = mano.model(clip.side)
    comps, hmean = T(m["hands_components"]), T(m["hands_mean"])
    th_idx = torch.as_tensor(clip.distal[0], device=dev)
    fi_idx = [torch.as_tensor(d, device=dev) for d in clip.distal[1:]]
    eye3 = 0.003 * torch.eye(3, device=dev)
    tn, tc = T(clip.table_n), T([clip.local_table_c(state, i) for i in fr])
    faces = torch.as_tensor(m["faces"].astype(np.int64), device=dev)

    def sample(img, uv):                                   # img (n,1,h,w), uv (n,V,2) low-res pixels -> (n,V)
        g = torch.stack([uv[..., 0] / (wl - 1) * 2 - 1, uv[..., 1] / (hl - 1) * 2 - 1], -1)[:, None]
        return F.grid_sample(img, g, align_corners=True, padding_mode="border")[:, 0, 0]

    def sdf_at(p):                                         # object-frame points (n,V,3) -> signed distance (n,V)
        u = (p - lo) / (hi - lo) * 2 - 1
        g = torch.stack([u[..., 2], u[..., 1], u[..., 0]], -1).reshape(1, 1, 1, -1, 3)
        return F.grid_sample(sdf, g, align_corners=True, padding_mode="border").reshape(p.shape[:2])

    def losses(tip_w=1.0):
        go, tr, hp, betas = go0 + dgo, tr0 + dtr, hp0 + dhp, betas0 + dbeta
        v, J = mano.mano_forward_torch(go, tr, hp, betas, clip.side)
        j21 = torch.einsum("jv,nvc->njc", reg, v)
        p = j21 @ K.T
        e = torch.linalg.norm(p[..., :2] / p[..., 2:3] - kp_t, dim=-1) / 15.0
        L = {"kp": (torch.where(e < 1, 0.5 * e ** 2, e - 0.5) * kp_m).sum() / kp_m.sum().clamp_min(1)}
        pv = v @ K.T
        uvv, z = pv[..., :2] / pv[..., 2:3] / LOW, v[..., 2]
        dmap, inm = sample(dl, uvv), sample(hm, uvv)
        with torch.no_grad():                              # vertices facing the camera, on the hand's mask, each
            fn = torch.cross(v[:, faces[:, 1]] - v[:, faces[:, 0]], v[:, faces[:, 2]] - v[:, faces[:, 0]], dim=-1)
            vn = torch.zeros_like(v).index_add_(1, faces.reshape(-1), fn.repeat_interleave(3, 1))
            # weighted by the image area it covers (MANO's vertices are dense on the fingers, sparse on the palm)
            proj = torch.relu(-(vn * v).sum(-1) / torch.linalg.norm(v, dim=-1)) / z ** 2
            seen = proj * (((z - dmap).abs() < 0.15) & (inm > 0.5) & (dmap > 0.05)).float() * valid[:, None]
            w_seen = seen.sum(1)
            seen = seen / w_seen.clamp_min(1e-12)[:, None]
            n_seen = (seen > 0).sum(1)
        r = 100 * (z - dmap)                               # cm; robust per vertex + the hand's mean offset per frame
        ok = (n_seen > 20).float()
        L["depth"] = ((torch.where(r.abs() < 1, 0.5 * r ** 2, r.abs() - 0.5) * seen).sum(1) * ok).sum() / ok.sum().clamp_min(1)
        bias = (r * seen).sum(1)
        L["depth"] = L["depth"] + 0.5 * (bias ** 2 * ok).sum() / ok.sum().clamp_min(1)
        L["mask"] = (sample(dtm, uvv).clamp(max=20) * has_mask[:, None]).mean()
        sd = sdf_at(torch.einsum("nvi,nij->nvj", v - to[:, None], Ro))
        # no penetration, with a 1.5 mm margin, at the vertices and the triangle centres (thin walls, handles)
        sdc = sdf_at(torch.einsum("nvi,nij->nvj", v[:, faces].mean(2) - to[:, None], Ro))
        L["pen"] = 10 * (torch.relu(0.0015 - sd).sum(1) + torch.relu(0.0015 - sdc).sum(1)).mean()
        # grip: the thumb's last segment and the two closest fingers' on the surface, and the thumb's contact facing
        # the closest finger's (outward normals > ~110 deg apart: an opposed grip, not a hand resting on one side)
        po = torch.einsum("nvi,nij->nvj", v - to[:, None], Ro)
        gt, it = sd[:, th_idx].min(1)
        gf = torch.stack([sd[:, f].min(1).values for f in fi_idx], 1)
        g2, k2 = torch.topk(gf, 2, dim=1, largest=False)
        gap = (torch.relu(gt - 0.0035).clamp(max=0.05) + torch.relu(g2[:, 0] - 0.0035).clamp(max=0.05)
               + 0.5 * torch.relu(g2[:, 1] - 0.0035).clamp(max=0.05))
        with torch.no_grad():                              # the closest finger's closest vertex, per frame
            jf = torch.stack([f[sd[:, f].argmin(1)] for f in fi_idx], 1).gather(1, k2[:, :1])[:, 0]
        ar = torch.arange(n, device=dev)
        pt, pf = po[ar, th_idx[it]], po[ar, jf]
        nrm = lambda p_: F.normalize(torch.stack([sdf_at((p_ + e)[:, None]) - sdf_at((p_ - e)[:, None]) for e in eye3], -1)[:, 0], dim=-1)
        opp = torch.relu((nrm(pt) * nrm(pf)).sum(-1) + 0.34)
        L["grip"] = 100 * (gap * held).sum() / held.sum().clamp_min(1) + (opp * held).sum() / held.sum().clamp_min(1)
        L["table"] = 10 * torch.relu(tc[:, None] - v @ tn).sum(1).mean()
        c = (hp - hmean) @ comps.T
        L["prior"] = 10 * (c[:, 15:] ** 2).sum(1).mean() + ((hp - hp_trk) ** 2).sum(1).mean() + 0.1 * (dbeta ** 2).sum()
        hpf, Jf = torch.cat([hp_pre, hp, hp_post]), torch.cat([J_pre, J, J_post])
        d2h, d2J = hpf[2:] - 2 * hpf[1:-1] + hpf[:-2], Jf[2:] - 2 * Jf[1:-1] + Jf[:-2]
        zero = torch.zeros((), device=dev)
        pf = torch.cat([p_pre, pts(v, J), p_post])
        d2p = pf[2:] - 2 * pf[1:-1] + pf[:-2]
        L["smooth"] = (30 * (d2h ** 2).sum(1).mean() + 1e5 * (d2J ** 2).sum(-1).mean()
                       + tip_w * tip_smooth * 1e5 * (d2p ** 2).sum(-1).mean()) if len(hpf) >= 3 else zero
        # the range's ends: summed, not averaged over the range, so a 1 cm jump there costs ~20 whatever its length
        L["join"] = (30 * (d2h[edge] ** 2).sum(1).sum() + 1e5 * (d2J[edge] ** 2).sum(-1).mean(-1).sum()) if edge else zero
        total = (kp_weight * L["kp"] + depth_weight * L["depth"] + 0.1 * L["mask"] + 5 * L["pen"] + contact_weight * L["grip"]
                 + L["table"] + L["prior"] + smooth_weight * L["smooth"] + max(1.0, smooth_weight) * L["join"])
        return total, L

    with torch.no_grad():
        _, L0 = losses()
    opt = torch.optim.Adam(groups)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, iters, eta_min=0.02 * lr * 0.004)
    for it in range(iters):
        opt.zero_grad()
        # the fingertip smoothness comes in once the fit has found its pose (from 40% to 70% of the steps): from the
        # start it couples the frames so stiffly that Adam does not get there in the steps it has
        total, _ = losses(min(1.0, max(0.0, (it / iters - 0.4) / 0.3)))
        total.backward()
        opt.step()
        sched.step()
    with torch.no_grad():
        _, L1 = losses()
    s = state.copy()
    note = ""
    if shape:
        b = (betas0 + dbeta).detach().cpu().numpy().astype(np.float64)
        flat = lambda bb: mano.mano_forward(np.zeros((1, 3)), np.zeros((1, 3)), np.zeros((1, 45)), bb, clip.side)[0]
        size = np.ptp(flat(b), 0).max() / np.ptp(flat(clip.betas_of(state)), 0).max()
        s.tr = s.tr + clip.J0_of(state) - mano.root_joint(b, clip.side)   # other frames keep their root joint
        s.betas = b
        note = f"; hand shape fitted (size x{size:.3f})"
    s.go[fr] = (go0 + dgo).detach().cpu().numpy()
    s.tr[fr] = (tr0 + dtr).detach().cpu().numpy()
    s.hp[fr] = (hp0 + dhp).detach().cpu().numpy()
    show = ", ".join(f"{k} {float(L0[k]):.2f}->{float(L1[k]):.2f}" for k in ("kp", "depth", "pen", "grip", "smooth") + (("join",) if edge else ()))
    return s, fr, f"fit MANO ({mode}{', no contact' if contact is False else ''}) on {start}-{end}: {n} frames, {iters} steps; losses {show}{note}"


def separate_hand(clip, state, start, end, max_cm=1.5, step_cm=0.2, tol=3, max_kp_px=8.0, smooth=8):
    """Per frame, translate the hand out of the object along the surface normal where its palm crosses the surface,
    in small steps, until at most `tol` palm triangles cross; a step that raises the keypoint error by more than
    max_kp_px is not taken. The pushes are then made one smooth profile over time. (MANO translation only.)"""
    s = state.copy()
    moved = []
    eps = 0.002
    palm = clip.hand_faces_i[clip.face_finger == -1]
    for i in range(start, end + 1):
        T = s.obj[i]
        v = clip.hand_verts(s, i)
        cf = clip.crossing(v, T, palm)
        if len(cf) <= tol:
            continue
        p = (v[np.unique(palm[cf])] - T[:3, 3]) @ T[:3, :3]
        g = np.stack([(clip.sdf_at(p + eps * e) - clip.sdf_at(p - eps * e)) / (2 * eps) for e in np.eye(3)], 1)
        nrm = T[:3, :3] @ g.mean(0)
        if np.linalg.norm(nrm) < 1e-6:
            continue
        nrm /= np.linalg.norm(nrm)
        kp0 = clip.hand_kp_err(s, i)
        best = None
        for k in range(1, int(round(max_cm / step_cm)) + 1):
            trial = s.copy()
            trial.tr[i] = s.tr[i] + nrm * k * step_cm / 100
            kp = clip.hand_kp_err(trial, i)
            if kp0 is not None and kp is not None and kp > kp0 + max_kp_px:
                break
            best = trial.tr[i]
            if len(clip.crossing(clip.hand_verts(trial, i), T, palm)) <= tol:
                break
        if best is not None:
            s.tr[i] = best
            moved.append(i)
    if moved and smooth > 1:
        fr = list(range(start, end + 1))
        d = np.array([s.tr[i] - state.tr[i] for i in fr])
        m_ = np.linalg.norm(d, axis=1)
        env = np.zeros_like(d)
        for n_ in range(len(fr)):
            lo, hi = max(0, n_ - 4), min(len(fr), n_ + 5)
            u = d[lo:hi].sum(0)
            if m_[lo:hi].max() > 0 and np.linalg.norm(u) > 0:
                env[n_] = m_[lo:hi].max() * u / np.linalg.norm(u)
        env = whittaker(env, smooth, _pin(len(fr), start, end, clip.N, fr))
        for n_, i in enumerate(fr):
            s.tr[i] = state.tr[i] + env[n_]
        moved = [i for n_, i in enumerate(fr) if np.linalg.norm(env[n_]) > 1e-4]
    sh = [100 * np.linalg.norm(s.tr[i] - state.tr[i]) for i in moved]
    return s, moved, (f"separate hand from the object on {start}-{end}: moved {len(moved)} frames out along the surface "
                      f"normal where the palm crossed it, smoothly (median {np.median(sh) if sh else 0:.1f} cm, "
                      f"max {max(sh) if sh else 0:.1f} cm)")


def smooth_hand(clip, state, start, end, window=12.0, fingers=True, rigid=True, max_extra_cross=5):
    """Remove jitter from the hand on [start, end]: smooth its MANO parameters over time (whittaker: motion slower
    than `window` frames passes; orientations relative to the range's first frame), joined to the frames outside the
    range. Where that pushes the hand through the object (more than max_extra_cross extra crossing triangles), only
    part of the smoothing is applied there (a fraction that is itself smooth in time)."""
    s = state.copy()
    fr = [i for i in range(start, end + 1) if clip.hand_valid[i]]
    for run in runs_of(fr):
        if len(run) < 4:
            continue
        w = _pin(len(run), start, end, clip.N, run)
        if rigid:
            R0 = Rotation.from_rotvec(state.go[run[0]])
            rel = (R0.inv() * Rotation.from_rotvec(state.go[run])).as_rotvec()
            s.go[run] = (R0 * Rotation.from_rotvec(whittaker(rel, window, w))).as_rotvec()
            s.tr[run] = whittaker(state.tr[run], window, w)
        if fingers:
            s.hp[run] = whittaker(state.hp[run], window, w)
    n0 = {i: len(clip.crossing(clip.hand_verts(state, i), state.obj[i])) for i in fr}
    alpha = np.ones(clip.N)
    bad = 0
    for i in fr:
        if len(clip.crossing(clip.hand_verts(s, i), s.obj[i])) <= n0[i] + max_extra_cross:
            continue
        bad += 1
        for a in (0.75, 0.5, 0.25, 0.0):
            t = _blend(state, s, i, a)
            if a == 0.0 or len(clip.crossing(clip.hand_verts(t, i), t.obj[i])) <= n0[i] + max_extra_cross:
                alpha[i] = a
                break
    if bad:
        for run in runs_of(fr):
            a = np.array([alpha[max(run[0], i - 3):min(run[-1], i + 3) + 1].min() for i in run])
            alpha[run] = np.clip(whittaker(a, 6.0), 0, 1)
        out = s.copy()
        for i in fr:
            out = _blend(state, out, i, alpha[i], into=out)
        s = out
    mo0, mo1 = clip.motion(state, fr), clip.motion(s, fr)
    j0 = [mo0[i]["hand_accel_mm"] for i in fr if mo0[i].get("hand_accel_mm") is not None]
    j1 = [mo1[i]["hand_accel_mm"] for i in fr if mo1[i].get("hand_accel_mm") is not None]
    what = " + ".join(x for x, on in (("rigid motion", rigid), ("finger joints", fingers)) if on)
    return s, fr, (f"smooth hand ({what}) on {start}-{end} over {window:.0f} frames: hand jitter median "
                   f"{np.median(j0) if j0 else 0:.1f} -> {np.median(j1) if j1 else 0:.1f} mm/frame^2"
                   + (f"; smoothing reduced on {bad} frames where it pushed the hand through the object" if bad else ""))


def _blend(a, b, i, w, into=None):
    """State `into` (default a copy of b) with frame i's hand = a + w (b - a) (orientation by slerp)."""
    t = b.copy() if into is None else into
    Ra, Rb = Rotation.from_rotvec(a.go[i]), Rotation.from_rotvec(b.go[i])
    t.go[i] = (Ra * Rotation.from_rotvec(w * (Ra.inv() * Rb).as_rotvec())).as_rotvec()
    t.tr[i] = a.tr[i] + w * (b.tr[i] - a.tr[i])
    t.hp[i] = a.hp[i] + w * (b.hp[i] - a.hp[i])
    return t


FADE = 6


def _fade_edges(clip, before, after, start, end, falloff=FADE):
    """Continue a hand edit made on [start, end] beyond its ends: the change at the first / last frame (rotation
    about the root joint, translation, joint rotations) is applied to the `falloff` frames outside, fading out
    (cosine), so the edit leaves no jump at the range's ends. Returns (state, the extra frames touched)."""
    s, extra = after.copy(), []
    for edge, step in ((start, -1), (end, 1)):
        dR = (Rotation.from_rotvec(after.go[edge]) * Rotation.from_rotvec(before.go[edge]).inv()).as_rotvec()
        dt, dh = after.tr[edge] - before.tr[edge], after.hp[edge] - before.hp[edge]
        for k in range(1, falloff + 1):
            i = edge + step * k
            if not 0 <= i < clip.N:
                break
            w = 0.5 * (1 + np.cos(np.pi * k / (falloff + 1)))
            s.go[i] = (Rotation.from_rotvec(w * dR) * Rotation.from_rotvec(before.go[i])).as_rotvec()
            s.tr[i] = before.tr[i] + w * dt
            s.hp[i] = before.hp[i] + w * dh
            extra.append(i)
    return s, sorted(extra)


def seam_frames(clip, state, refs, factor=1.5, floor=3.0):
    """Frames where `state`'s hand jumps more than every reference state does around that frame (its fastest
    fingertip's acceleration > factor x theirs + floor mm/frame^2), with both accelerations per frame."""
    near = lambda a: np.maximum.reduce([a, np.r_[a[1:], 0.0], np.r_[0.0, a[:-1]]])
    ref = np.maximum.reduce([near(np.nan_to_num(clip.tip_accel(r))) for r in refs])
    a = np.nan_to_num(clip.tip_accel(state))
    return [i for i in range(clip.N) if a[i] > factor * ref[i] + floor], a, ref


def smooth_seams(clip, state, refs, pad=6, window=8.0):
    """Smooth the hand at the seams left where versions were joined (seam_frames). Returns (state, notes)."""
    notes = []
    for _ in range(2):
        bad, a, ref = seam_frames(clip, state, refs)
        if not bad:
            break
        groups = [[bad[0], bad[0]]]
        for i in bad[1:]:
            if i - groups[-1][1] <= 4:
                groups[-1][1] = i
            else:
                groups.append([i, i])
        for lo, hi in groups:
            state, _, note = smooth_hand(clip, state, max(0, lo - pad), min(clip.N - 1, hi + pad), window)
            notes.append(f"seam at frames {lo}-{hi} (fingertip acceleration {a[lo:hi + 1].max():.0f} mm/frame^2, the "
                         f"versions joined there have {ref[lo:hi + 1].max():.0f}): {note}")
    return state, notes


def hand_depth_bias(clip, state, i):
    """Median depth of the rendered hand minus the depth map over the hand's mask pixels (m; < 0 = hand too close)."""
    r = clip.metrics(state, [i])[0]
    return None if r.get("hand_depth_bias_cm") is None else r["hand_depth_bias_cm"] / 100


def move_hand_to_depth(clip, state, start, end, smooth=10):
    """Move the hand along the camera ray through it by its measured depth error (rendered hand vs the depth map on
    its mask), per frame, smoothed over time: one jump, so a hand that is on the wrong side of the object (e.g. in
    front of it while the fingers really wrap its far side) gets there without passing through it step by step.
    Follow with fit_mano to settle the grip."""
    s = state.copy()
    fr = [i for i in range(start, end + 1) if clip.hand_valid[i]]
    rows = {r["frame"]: r for r in clip.metrics(state, fr)}
    d = np.zeros((len(fr), 3))
    for n_, i in enumerate(fr):
        b = rows[i].get("hand_depth_bias_cm")
        if b is None:
            continue
        c = clip.hand_verts(state, i).mean(0)
        d[n_] = -b / 100 * c / np.linalg.norm(c)
    for run in runs_of(list(range(len(fr)))):
        d[run] = whittaker(d[run], smooth)
    for n_, i in enumerate(fr):
        s.tr[i] = state.tr[i] + d[n_]
    extra = []
    if fr:                                               # no jump at the range's ends: the end frames' move fades out
        s, extra = _fade_edges(clip, state, s, fr[0], fr[-1])
    mv = 100 * np.linalg.norm(d, axis=1)
    return s, sorted(set(fr) | set(extra)), (
        f"move hand to its depth on {start}-{end}: along the camera ray by the hand's depth error "
        f"(median {np.median(mv) if len(mv) else 0:.1f} cm, max {mv.max() if len(mv) else 0:.1f} cm), smoothed, "
        f"fading out over {FADE} frames beyond the range")


# ---------- object silhouette fit ----------
def _obj_costs(clip, i, poses):
    """Fit cost of several object poses at frame i (one batched render): 1 - IoU with the object mask outside the
    hand (the hand occludes the object there, so neither silhouette counts), plus the depth error."""
    _, do, _ = nvdiffrast_render(K=clip.Kl, H=clip.hl, W=clip.wl, glctx=clip.glctx, mesh_tensors=clip.mt,
                                 ob_in_cams=torch.as_tensor(np.asarray(poses), device="cuda", dtype=torch.float))
    do = do.cpu().numpy()
    m, d = clip.om[i], clip.dl[i]
    free = ~(clip.hm[i] & ~m)                               # pixels the hand does not cover
    out = []
    for k in range(len(poses)):
        so = (do[k] > 0) & free
        inter, union = (so & m).sum(), (so | (m & free)).sum()
        iou = inter / max(union, 1)
        both = so & m & (d > 0)
        dres = np.median(np.abs(do[k][both] - d[both])) if both.sum() > 8 else 0.05
        out.append(1 - iou + 3.0 * dres)
    return np.array(out)


def fit_object_silhouette(clip, state, start, end, max_rot_deg=8.0, max_trans_cm=2.0, smooth=8):
    """Per-frame 6-DoF fit of the object to its SAM3 mask (outside the hand) and the depth: greedy coordinate
    descent with step halving (rotations about the object's own axes through its center, translations along camera
    x/y and the viewing ray), the 12 neighbouring poses rendered in one batch, within a trust region around the
    current pose (it sharpens a pose; it will not fix a wrong orientation); frames without a usable mask are skipped.
    Then the change is smoothed over time (whittaker, `smooth` frames)."""
    s = state.copy()
    fr = [i for i in range(start, end + 1) if clip.om[i].sum() > 40]
    for i in fr:
        P0 = state.obj[i].copy()
        R0 = P0[:3, :3]
        c0 = clip.obj_center(state, i)
        ray = c0 / np.linalg.norm(c0)
        tdirs = [np.array([1.0, 0, 0]), np.array([0, 1.0, 0]), ray]
        cur_rv, cur_t = np.zeros(3), np.zeros(3)             # rotation (object axes, about the center) and shift

        def pose(rv, t):
            R = R0 @ Rotation.from_rotvec(rv).as_matrix()      # rv in object coordinates
            P = np.eye(4)
            P[:3, :3] = R
            P[:3, 3] = c0 + t - R @ clip.center_obj
            return P

        best = _obj_costs(clip, i, [pose(cur_rv, cur_t)])[0]
        for step_t, step_r in ((0.01, 4.0), (0.005, 2.0), (0.0025, 1.0), (0.00125, 0.5)):
            for _ in range(12):
                cands = []
                for k in range(6):
                    for sign in (1, -1):
                        rv, t = cur_rv.copy(), cur_t.copy()
                        if k < 3:
                            t = t + sign * step_t * tdirs[k]
                            if 100 * np.linalg.norm(t) > max_trans_cm:
                                continue
                        else:
                            ax = clip.axes_obj[:, k - 3]
                            rv = (Rotation.from_rotvec(rv) * Rotation.from_rotvec(np.radians(sign * step_r) * ax)).as_rotvec()
                            if np.degrees(np.linalg.norm(rv)) > max_rot_deg:
                                continue
                        cands.append((rv, t))
                if not cands:
                    break
                cs = _obj_costs(clip, i, [pose(rv, t) for rv, t in cands])
                j = int(np.argmin(cs))
                if cs[j] >= best - 1e-4:
                    break
                best, (cur_rv, cur_t) = cs[j], cands[j]
        s.obj[i] = pose(cur_rv, cur_t)
    if smooth > 1 and len(fr) >= 4:                         # smooth the applied change over time (whittaker), not
        out = s.copy()                                      # the poses themselves
        for run in runs_of(fr):
            if len(run) < 4:
                continue
            w = _pin(len(run), start, end, clip.N, run)
            dr = whittaker(np.array([Rotation.from_matrix(s.obj[i, :3, :3] @ state.obj[i, :3, :3].T).as_rotvec() for i in run]), smooth, w)
            dc = whittaker(np.array([clip.obj_center(s, i) - clip.obj_center(state, i) for i in run]), smooth, w)
            for n_, i in enumerate(run):
                out.obj[i, :3, :3] = Rotation.from_rotvec(dr[n_]).as_matrix() @ state.obj[i, :3, :3]
                out.obj[i, :3, 3] = clip.obj_center(state, i) + dc[n_] - out.obj[i, :3, :3] @ clip.center_obj
        s = out
    return s, fr, f"fit object to its mask + depth on {start}-{end}: {len(fr)} frames (trust region {max_rot_deg} deg, {max_trans_cm} cm)"
