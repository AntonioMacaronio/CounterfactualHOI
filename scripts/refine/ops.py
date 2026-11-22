"""Edit operations on a clip's State, for the refinement agent. Each returns (new_state, touched_frames, note).

Geometric edits are stated in frames the agent can reason about from the images (see Clip.axis): rotate about the
object's own long/mid/short axis or gravity's up/right/forward, translate in gravity-aligned world axes (right,
forward = away from the camera, up) or camera axes, along the viewing ray. A range edit applies fully on
[start, end] and fades out over `falloff` frames on each side (cosine), so it blends into the untouched frames.

Search-backed edits: polish_object runs FoundationPose's refiner from the current pose on the hand-scaled depth and
keeps a frame's result only if it fits the SAM3 mask + depth better and stays within a trust region (so it sharpens a
pose without jumping to another solution); polish_hand fits the hand's rigid motion to its 2D keypoints, its SAM3 mask
and the depth by gradient-free coordinate search (penalising penetration into the object); propose_object_poses returns FoundationPose's
top registration hypotheses at a frame for the agent to choose from (apply_candidate).
"""

import os

import cv2
import numpy as np
import torch
from scipy.spatial.transform import Rotation

from clip import LOW, Clip, State, se3_interp  # noqa: F401
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
    s = state.copy()
    w = weights(clip.N, start, end, falloff)
    d = world_vec(clip, dx_cm, dy_cm, dz_cm, frame) / 100
    for i in _touched(w):
        s.hand_off[i] += w[i] * d
        if rot_axis and rot_deg:
            a = clip.axis(state, i, rot_axis)
            R = Rotation.from_rotvec(np.radians(rot_deg) * w[i] * a) * Rotation.from_rotvec(state.hand_rot[i])
            s.hand_rot[i] = R.as_rotvec()
    return s, _touched(w), (f"move hand ({dx_cm:+.1f},{dy_cm:+.1f},{dz_cm:+.1f}) cm [{frame}]"
                            + (f", rotate {rot_deg:+.1f} deg about {rot_axis}" if rot_axis and rot_deg else "") + f" on {start}-{end}")


def attach_hand_to_object(clip, state, start, end, ref):
    """Keep the hand's centroid fixed relative to the object over [start, end] (as at frame ref); fingers unchanged."""
    s = state.copy()
    T = state.obj[ref]
    c_rel = T[:3, :3].T @ (clip.hand_verts(state, ref).mean(0) - T[:3, 3])
    for i in range(start, end + 1):
        target = state.obj[i, :3, :3] @ c_rel + state.obj[i, :3, 3]
        s.hand_off[i] += target - clip.hand_verts(state, i).mean(0)
    return s, list(range(start, end + 1)), f"attach hand to object on {start}-{end} (relative position of frame {ref})"


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


# ---------- hand fit ----------
HELD = ("grasp", "lift", "carry", "rotate", "place")


def _hand_costs(clip, i, v0, c, mt, rots, offs, T_obj, held=False):
    """Fit cost of several rigid hand motions at frame i (one batched render): robust 2D keypoint error, rendered
    hand outside its mask, depth error, penetration into the object, and, while the object is held, the hand's gap
    to the object's surface (keypoints of fingers hidden behind the object would otherwise pull the hand off it)."""
    Rs = Rotation.from_rotvec(np.asarray(rots)).as_matrix()
    Ts = np.tile(np.eye(4), (len(Rs), 1, 1))
    Ts[:, :3, :3] = Rs
    Ts[:, :3, 3] = c - Rs @ c + np.asarray(offs)
    _, dh, _ = nvdiffrast_render(K=clip.Kl, H=clip.hl, W=clip.wl, glctx=clip.glctx, mesh_tensors=mt,
                                 ob_in_cams=torch.as_tensor(Ts, device="cuda", dtype=torch.float))
    dh = dh.cpu().numpy()
    hm, om, d = clip.hm[i] & ~clip.om[i], clip.om[i], clip.dl[i]
    ok = np.isfinite(clip.kp2d[i, :, 0])
    J0 = clip.kp_reg @ v0
    out = []
    for m in range(len(Rs)):
        sh = dh[m] > 0
        n = sh.sum()
        if n < 10:
            out.append(10.0)
            continue
        spill = (sh & ~hm & ~om).sum() / n                    # rendered hand outside its mask (object mask = occluder)
        both = sh & hm & (d > 0)
        dres = np.median(np.abs(dh[m][both] - d[both])) if both.sum() > 8 else 0.05
        v = (v0 - c) @ Rs[m].T + c + offs[m]
        sd = clip.sdf_at((v - T_obj[:3, 3]) @ T_obj[:3, :3])
        pen = max(-sd.min(), 0)
        gap = max(sd.min() - 0.003, 0) if held else 0.0
        kp_term = 0.0
        if ok.sum() >= 8:                                     # robust (Huber-like) keypoint reprojection, 30 px units
            J = (J0 - c) @ Rs[m].T + c + offs[m]
            p = (clip.K @ J.T).T
            e = np.linalg.norm(p[ok, :2] / p[ok, 2:3] - clip.kp2d[i, ok], axis=1) / 30.0
            kp_term = float(np.mean(np.where(e < 1, 0.5 * e ** 2, e - 0.5)))
        out.append(kp_term + 0.5 * spill + 4.0 * dres + 20.0 * pen + 15.0 * gap)
    return np.array(out)


def polish_hand(clip, state, start, end, max_trans_cm=6.0, max_rot_deg=25.0, smooth=5):
    """Rigid hand fit per frame to its 2D keypoints (robust), its SAM3 mask, the depth, no penetration into the
    object and, in held phases (set_phases), contact with it: greedy coordinate descent with step halving, the 12
    neighbouring moves rendered in one batch; then a temporal median of the fitted motion."""
    s = state.copy()
    fr = [i for i in range(start, end + 1)
          if clip.hand_valid[i] and (clip.hm[i].sum() > 30 or np.isfinite(clip.kp2d[i, :, 0]).sum() >= 8)]
    for i in fr:
        v0 = clip.hand_base[i].astype(np.float64)
        c = v0.mean(0)
        mt = clip.hand_tensors(clip.hand_base[i])
        r0, t0 = state.hand_rot[i].copy(), state.hand_off[i].copy()
        cur_r, cur_t = r0.copy(), t0.copy()
        held = any(p["start"] <= i <= p["end"] and p["label"] in HELD for p in state.phases)
        best = _hand_costs(clip, i, v0, c, mt, [cur_r], [cur_t], s.obj[i], held)[0]
        for step_t, step_r in ((0.02, 8.0), (0.01, 4.0), (0.005, 2.0), (0.0025, 1.0)):
            for _ in range(12):
                rots, offs = [], []
                for k in range(6):
                    for sign in (1, -1):
                        r, t = cur_r.copy(), cur_t.copy()
                        if k < 3:
                            t[k] += sign * step_t
                            if 100 * np.linalg.norm(t - t0) > max_trans_cm:
                                continue
                        else:
                            r = (Rotation.from_rotvec(np.radians(sign * step_r) * np.eye(3)[k - 3]) * Rotation.from_rotvec(r)).as_rotvec()
                            if np.degrees((Rotation.from_rotvec(r) * Rotation.from_rotvec(r0).inv()).magnitude()) > max_rot_deg:
                                continue
                        rots.append(r)
                        offs.append(t)
                if not rots:
                    break
                cs = _hand_costs(clip, i, v0, c, mt, rots, offs, s.obj[i], held)
                j = int(np.argmin(cs))
                if cs[j] >= best - 1e-4:
                    break
                best, cur_r, cur_t = cs[j], rots[j], offs[j]
        s.hand_rot[i], s.hand_off[i] = cur_r, cur_t
    if smooth > 1 and len(fr) > smooth:                       # remove frame-to-frame jitter of the fitted motion
        half = smooth // 2
        off, rot = s.hand_off.copy(), s.hand_rot.copy()
        for i in fr:
            lo, hi = max(start, i - half), min(end, i + half)
            s.hand_off[i] = np.median(off[lo:hi + 1], 0)
            s.hand_rot[i] = Rotation.from_rotvec(rot[lo:hi + 1]).mean().as_rotvec()
    return s, fr, f"polish hand on {start}-{end}: fitted {len(fr)} frames to keypoints + mask + depth (trust region {max_trans_cm} cm, {max_rot_deg} deg)"


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


def fit_object_silhouette(clip, state, start, end, max_rot_deg=8.0, max_trans_cm=2.0, smooth=3):
    """Per-frame 6-DoF fit of the object to its SAM3 mask (outside the hand) and the depth: greedy coordinate
    descent with step halving (rotations about the object's own axes through its center, translations along camera
    x/y and the viewing ray), the 12 neighbouring poses rendered in one batch, within a trust region around the
    current pose (it sharpens a pose; it will not fix a wrong orientation); frames without a usable mask are skipped.
    Then a temporal smoothing of the change over `smooth` frames."""
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
    if smooth > 1 and len(fr) > smooth:                     # smooth the applied change, not the poses themselves
        half = smooth // 2
        dR = [Rotation.from_matrix(s.obj[i, :3, :3] @ state.obj[i, :3, :3].T) for i in range(clip.N)]
        dc = [clip.obj_center(s, i) - clip.obj_center(state, i) for i in range(clip.N)]
        out = s.copy()
        for i in fr:
            lo, hi = max(start, i - half), min(end, i + half)
            Rm = Rotation.concatenate(dR[lo:hi + 1]).mean()
            cm = np.mean(dc[lo:hi + 1], 0)
            c = clip.obj_center(state, i)
            out.obj[i, :3, :3] = Rm.as_matrix() @ state.obj[i, :3, :3]
            out.obj[i, :3, 3] = c + cm - out.obj[i, :3, :3] @ clip.center_obj
        s = out
    return s, fr, f"fit object to its mask + depth on {start}-{end}: {len(fr)} frames (trust region {max_rot_deg} deg, {max_trans_cm} cm)"
