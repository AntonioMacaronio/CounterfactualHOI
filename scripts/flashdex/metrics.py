"""Tracking metrics of a FlashDexRetarget rollout against its reference clip (motion.pt), numpy only.

Object: per-frame position error (cm) and rotation error (deg) of the simulated object vs the reconstructed one.
  progress (AdaDex): the share of the clip tracked before the object first deviates by more than 5 cm or 20 deg
    (1.0: never);
  progress_pos: the same with the position threshold only (comparable with Do-as-I-Do's position-only number);
  success: progress = 1 (the whole clip within 5 cm / 20 deg). FDR's own criterion is also reported
    (fdr_success: time-mean errors within 3 cm and 30 deg, its SR@1.0 "obj").
Hand: wrist position error (cm, the floating wrist's translation joints vs the kinematic retarget) and fingertip
  error (cm, the robot's tracked tip sites vs the MANO fingertips; needs the MuJoCo model).
"""

import numpy as np

POS_TH, ROT_TH = 0.05, 20.0
FINGERS = ("thumb", "index", "middle", "ring", "pinky")


def quat_wxyz_to_mat(q):
    q = np.asarray(q, np.float64)
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
                     2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
                     2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1).reshape(q.shape[:-1] + (3, 3))


def rot_err_deg(Ra, Rb):
    tr = np.einsum("tij,tij->t", Ra, Rb)
    return np.degrees(np.arccos(np.clip((tr - 1) / 2, -1, 1)))


def progress(pos_err_m, rot_err_deg, pos_th=POS_TH, rot_th=ROT_TH):
    bad = (pos_err_m > pos_th) | (rot_err_deg > rot_th)
    return 1.0 if not bad.any() else float(np.argmax(bad)) / len(bad)


def clip_metrics(ro_obj_pos, ro_obj_quat, ref_obj_pos, ref_obj_rot, ro_q=None, ref_q=None, tips=None, ref_tips=None):
    """ro_*: the rollout (T, 3), (T, 4 wxyz); ref_*: the clip (T, 3), (T, 3, 3); optional joints (T, nq) and
    fingertips (T, 5, 3) of rollout / reference. Returns per-clip numbers and per-frame curves."""
    T = min(len(ro_obj_pos), len(ref_obj_pos))
    e = np.linalg.norm(ro_obj_pos[:T] - ref_obj_pos[:T], axis=1)
    r = rot_err_deg(quat_wxyz_to_mat(ro_obj_quat[:T]), ref_obj_rot[:T])
    pg = progress(e, r)
    out = {"frames": int(T), "progress": pg, "success": bool(pg >= 1.0),
           "progress_pos": progress(e, np.zeros_like(r)),          # position only (5 cm), as Do-as-I-Do's json allows
           "fdr_success": bool(e.mean() < 0.03 and r.mean() < 30.0),
           "obj_pos_err_cm_mean": float(100 * e.mean()), "obj_pos_err_cm_median": float(100 * np.median(e)),
           "obj_pos_err_cm_max": float(100 * e.max()), "obj_rot_err_deg_mean": float(r.mean()),
           "obj_rot_err_deg_median": float(np.median(r)), "obj_rot_err_deg_max": float(r.max()),
           "frames_within_2cm": float(np.mean(e < 0.02)),
           "final_obj_pos_err_cm": float(100 * e[-1]), "final_obj_rot_err_deg": float(r[-1])}
    curves = {"obj_pos_err_cm": [round(float(x), 2) for x in 100 * e], "obj_rot_err_deg": [round(float(x), 1) for x in r]}
    if ro_q is not None and ref_q is not None:
        w = np.linalg.norm(ro_q[:T, :3] - ref_q[:T, :3], axis=1)
        out.update(wrist_pos_err_cm_mean=float(100 * w.mean()), wrist_pos_err_cm_max=float(100 * w.max()))
    if tips is not None and ref_tips is not None:
        d = np.linalg.norm(tips[:T] - ref_tips[:T], axis=-1).mean(1)
        out.update(tip_err_cm_mean=float(100 * d.mean()), tip_err_cm_max=float(100 * d.max()))
        curves["tip_err_cm"] = [round(float(x), 2) for x in 100 * d]
    return out, curves
