"""MANO hand model (flat-hand-mean convention, as Dyn-HaMR exports it) in numpy and torch, and the hand tracker's
per-frame MANO parameters expressed in our metric camera frame.

The hand state everywhere downstream (refinement, retargeting) is MANO parameters only, per frame:
  global_orient (3, axis-angle, camera frame), transl (3, camera frame, metres), hand_pose (45 = 15 joints x
  axis-angle, relative to the flat hand), plus one betas (10) per clip.
  verts = R(global_orient) (v_posed - J0) + J0 + transl   (smplx's convention; J0 = the shaped template's root joint)

Model arrays come from $PROJECT_STORAGE_ROOT/models/mano/mano_{side}.npz (scripts/refine/extract_mano_skinning.py).
mano_forward(...) reproduces smplx.MANO(flat_hand_mean=True, use_pca=False) and Dyn-HaMR's exported meshes to
< 0.01 mm.
"""

import json
import os

import numpy as np

STORAGE = os.environ.get("PROJECT_STORAGE_ROOT", "/data/sky/projects/CounterfactualHOI")
NB = 10                                               # shape coefficients used (Dyn-HaMR / smplx default)
_MODELS = {}


def model(side="right"):
    if side not in _MODELS:
        z = np.load(os.path.join(STORAGE, "models", "mano", f"mano_{side}.npz"))
        _MODELS[side] = {k: z[k] for k in z.files}
        _MODELS[side]["shapedirs"] = _MODELS[side]["shapedirs"][:, :, :NB]
    return _MODELS[side]


def _rodrigues_np(r):
    """(..., 3) axis-angle -> (..., 3, 3)."""
    th = np.linalg.norm(r, axis=-1, keepdims=True)
    k = r / np.maximum(th, 1e-8)
    K = np.zeros(r.shape[:-1] + (3, 3))
    K[..., 0, 1], K[..., 0, 2], K[..., 1, 0] = -k[..., 2], k[..., 1], k[..., 2]
    K[..., 1, 2], K[..., 2, 0], K[..., 2, 1] = -k[..., 0], -k[..., 1], k[..., 0]
    s, c = np.sin(th)[..., None], np.cos(th)[..., None]
    return np.eye(3) + s * K + (1 - c) * (K @ K)


def mano_forward(global_orient, transl, hand_pose, betas, side="right", return_joints=False):
    """numpy MANO: (N,3), (N,3), (N,45), (10,) or (N,10) -> vertices (N,778,3) [, joints (N,16,3)]."""
    m = model(side)
    go, tr, hp = (np.atleast_2d(np.asarray(x, np.float64)) for x in (global_orient, transl, hand_pose))
    N = len(go)
    b = np.broadcast_to(np.asarray(betas, np.float64), (N, NB))
    v_shaped = m["v_template"][None] + np.einsum("vck,nk->nvc", m["shapedirs"], b)
    J = np.einsum("jv,nvc->njc", m["J_regressor"], v_shaped)
    R = _rodrigues_np(np.concatenate([go, hp], 1).reshape(N, 16, 3))           # (N,16,3,3)
    pf = (R[:, 1:] - np.eye(3)).reshape(N, 135)
    v_posed = v_shaped + np.einsum("vck,nk->nvc", m["posedirs"], pf)
    par = m["parents"]
    G = np.zeros((N, 16, 4, 4))
    G[:, :, 3, 3] = 1
    G[:, 0, :3, :3], G[:, 0, :3, 3] = R[:, 0], J[:, 0]
    for k in range(1, 16):
        L = np.zeros((N, 4, 4))
        L[:, :3, :3], L[:, :3, 3], L[:, 3, 3] = R[:, k], J[:, k] - J[:, par[k]], 1
        G[:, k] = G[:, par[k]] @ L
    joints = G[:, :, :3, 3].copy()
    A = G.copy()
    A[:, :, :3, 3] -= np.einsum("nkij,nkj->nki", G[:, :, :3, :3], J)
    T = np.einsum("vk,nkij->nvij", m["weights"], A)
    v = np.einsum("nvij,nvj->nvi", T[:, :, :3, :3], v_posed) + T[:, :, :3, 3] + tr[:, None]
    return (v, joints + tr[:, None]) if return_joints else v


def root_joint(betas, side="right"):
    """J0 of the shaped template (the point global_orient rotates about)."""
    m = model(side)
    v = m["v_template"] + m["shapedirs"] @ np.asarray(betas, np.float64)
    return m["J_regressor"][0] @ v


def mano_forward_torch(global_orient, transl, hand_pose, betas, side="right"):
    """torch MANO (differentiable): (N,3), (N,3), (N,45), (10,) -> vertices (N,778,3), joints (N,16,3)."""
    import torch
    m = model(side)
    dev, dt = global_orient.device, global_orient.dtype
    t = lambda a: torch.as_tensor(a, device=dev, dtype=dt)
    N = global_orient.shape[0]
    v_shaped = t(m["v_template"])[None] + torch.einsum("vck,k->vc", t(m["shapedirs"]), betas)[None]
    J = torch.einsum("jv,nvc->njc", t(m["J_regressor"]), v_shaped.expand(N, -1, -1))
    r = torch.cat([global_orient, hand_pose], 1).reshape(N, 16, 3)
    th = torch.linalg.norm(r, dim=-1, keepdim=True).clamp_min(1e-8)
    k = r / th
    z = torch.zeros_like(k[..., 0])
    K = torch.stack([z, -k[..., 2], k[..., 1], k[..., 2], z, -k[..., 0], -k[..., 1], k[..., 0], z], -1).reshape(N, 16, 3, 3)
    I = torch.eye(3, device=dev, dtype=dt)
    R = I + torch.sin(th)[..., None] * K + (1 - torch.cos(th))[..., None] * (K @ K)
    pf = (R[:, 1:] - I).reshape(N, 135)
    v_posed = v_shaped + torch.einsum("vck,nk->nvc", t(m["posedirs"]), pf)
    par = m["parents"]
    Gs = [torch.cat([torch.cat([R[:, 0], J[:, 0, :, None]], 2), t([[0, 0, 0, 1]]).expand(N, 1, 4)], 1)]
    for kk in range(1, 16):
        L = torch.cat([torch.cat([R[:, kk], (J[:, kk] - J[:, par[kk]])[:, :, None]], 2), t([[0, 0, 0, 1]]).expand(N, 1, 4)], 1)
        Gs.append(Gs[par[kk]] @ L)
    G = torch.stack(Gs, 1)                                                     # (N,16,4,4)
    joints = G[:, :, :3, 3]
    Arot, Atr = G[:, :, :3, :3], G[:, :, :3, 3] - torch.einsum("nkij,nkj->nki", G[:, :, :3, :3], J)
    W = t(m["weights"])
    Trot, Ttr = torch.einsum("vk,nkij->nvij", W, Arot), torch.einsum("vk,nki->nvi", W, Atr)
    v = torch.einsum("nvij,nvj->nvi", Trot, v_posed) + Ttr + transl[:, None]
    return v, joints + transl[:, None]


def tracker_params_camera(vd, run, side=None):
    """Dyn-HaMR's per-frame MANO parameters in our metric camera frame: its world->camera transform applied to
    global_orient / transl, and fprecon's per-frame depth correction (scripts/fprecon/hand_scale.corrected_hands: a
    shift along the ray through the hand's centroid) folded into transl. One betas per clip (median). Frames without a
    tracked hand take the nearest tracked frame's parameters (valid = False)."""
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "fprecon"))
    from hand_scale import corrected_hands
    side = side or json.load(open(os.path.join(run, "hand_scale.json")))["side"]
    _, _, valid, factor = corrected_hands(vd, run)
    N = len(valid)
    mdir = os.path.join(vd, "stage1", "hand", "mano_params", side)
    if not os.path.isdir(mdir):
        mdir = os.path.join(vd, "stage1", "hand", "mano_params")
    have = np.array([os.path.exists(os.path.join(mdir, f"{i:05d}.json")) for i in range(N)]) & valid
    ok = np.nonzero(have)[0]
    raw = {}
    for i in ok:
        raw[i] = json.load(open(os.path.join(mdir, f"{i:05d}.json")))
    betas = np.median(np.stack([raw[i]["betas"] for i in ok]), 0)[:NB]
    J0 = root_joint(betas, side)
    go, tr, hp = np.zeros((N, 3)), np.zeros((N, 3)), np.zeros((N, 45))
    from scipy.spatial.transform import Rotation
    for i in range(N):
        p = raw[ok[np.argmin(np.abs(ok - i))]]
        T = np.array(p["T_w2c"], float)
        Rw = Rotation.from_rotvec(np.array(p["root_orient"], float)).as_matrix()
        go[i] = Rotation.from_matrix(T[:3, :3] @ Rw).as_rotvec()
        tr[i] = T[:3, :3] @ (J0 + np.array(p["trans"], float)) + T[:3, 3] - J0
        hp[i] = np.array(p["pose"], float).reshape(45)
    v = mano_forward(go, tr, hp, betas, side)
    c = v.mean(1)
    tr = tr + c * (factor[:N, None] - 1)                                         # fprecon's depth correction
    return {"global_orient": go, "transl": tr, "hand_pose": hp, "betas": betas, "valid": have, "side": side}
