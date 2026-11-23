"""§1.4 policy terms on FlashDexRetarget's env: deployable actor observations and the AdaDex-style hybrid reward.

Loaded by hydra from FDR's config (`_target_: cfhoi_terms.CfhoiObs` etc.; scripts/flashdex/policy/ on PYTHONPATH,
see train_policy.sh). The env is FDR's (Sharpa Wave on a floating 6-DoF wrist, 60 Hz).

Observations (actor, antzhang-notes/project.md §1.4 minus the two depth encoders; the critic keeps FDR's privileged
observations). The 6 floating-wrist joints stand in for the 7-DoF arm, so N = 28 (right hand):
  p14_proprio        last 3 frames of measured q and of the policy's PD targets q*, each mapped to [-1, 1] by the
                     joint limits                                                                     3 x 2N = 168
  p14_obj_pose       object position (workspace / robot-base frame) + rotation 6D (first two columns)      9
  p14_future_obj     K = 5 reference object waypoints 0.1 .. 0.5 s ahead, each as a change from the current
                     object pose: position delta (3) + 6D of R_future R_now^T (6)                   5 x 9 = 45
  p14_future_joints  reference joint angles at the same 5 times, each minus the current q, scaled like q
                                                                                                    5 x N = 140

Reward (AdaDex: one actor on two env populations, by clip):
  TrackEnv (the original clips): FDR's dense object tracking (object point-cloud match) + hand tracking (wrist
    position / rotation, fingertips, joint-space BC to the kinematic retarget).
  GenEnv (cfhoi_gen<k>_* clips from make_gen_clips.py: new object start and goal): the hand-tracking terms are off
    (the human hand only matters through the TrackEnv prior) and gen_obj_success pays a sparse object-success reward:
    1 when the object is within 5 cm and 20 deg of its goal waypoint, else 0. The dense object term stays on in
    both: FDR's early reset and its object-assist curriculum are keyed on it.
"""

from __future__ import annotations

import functools

import torch

import envs  # noqa: F401  (puts FDR's vendored mjlab on sys.path)
from mjlab.utils.lab_api.math import matrix_from_quat, quat_error_magnitude
from envs.observations.motion_tracking import MotionTrackingObs
from envs.rewards.base import _cmd, _mean_over_sides, _side_idx
from envs.rewards.motion_tracking import MotionTrackingRewards

STRIDE = 6          # 0.1 s at 60 Hz
N_FUTURE = 5


def _right_joint_ids(env, robot):
    ids = getattr(env, "_p14_jids", None)
    if ids is None:
        names = list(robot.joint_names)
        ids = torch.tensor([i for i, n in enumerate(names) if n.startswith(("R_", "right_"))], device=env.device)
        env._p14_jids = ids
    return ids


def _limits(env, robot, ids):
    lim = robot.data.soft_joint_pos_limits[0][ids]
    return lim[:, 0], lim[:, 1]


def _norm(q, lo, hi):
    return (2.0 * q - hi - lo) / (hi - lo + 1e-5)


def _rot6d(R):
    return torch.cat([R[..., :, 0], R[..., :, 1]], dim=-1)


def gen_mask(env, command) -> torch.Tensor:
    """(B,) bool: the env plays a GenEnv clip (name cfhoi_gen<k>_*)."""
    t = getattr(command, "_cfhoi_gen", None)
    if t is None:
        names = [str(f).rsplit("#", 1)[-1] for f in command.motion_lib.motion_files]
        t = torch.tensor(["_gen" in n.rsplit("/", 1)[-1] for n in names], dtype=torch.bool, device=env.device)
        command._cfhoi_gen = t
        print(f"[cfhoi] {int(t.sum())} GenEnv / {len(t)} clips", flush=True)
    return t[command.motion_ids]


class CfhoiObs(MotionTrackingObs):
    @staticmethod
    def p14_proprio(env, command_name: str, history: int = 3) -> torch.Tensor:
        command = _cmd(env, command_name)
        robot = command.robot
        ids = _right_joint_ids(env, robot)
        lo, hi = _limits(env, robot, ids)
        cur = torch.cat([_norm(robot.data.joint_pos[:, ids], lo, hi),
                         _norm(robot.data.joint_pos_target[:, ids], lo, hi)], dim=-1)       # (B, 2N)
        token = int(getattr(env, "_sim_step_counter", 0))
        buf = getattr(env, "_p14_hist", None)
        if buf is None or buf.shape[0] != cur.shape[0] or buf.shape[1] != history:
            buf = cur[:, None].repeat(1, history, 1)
        elif getattr(env, "_p14_hist_token", None) != token:
            buf = torch.cat([buf[:, 1:], cur[:, None]], dim=1)
        fresh = env.episode_length_buf == 0                                             # new episode: no past
        if fresh.any():
            buf[fresh] = cur[fresh, None].expand(-1, history, -1)
        env._p14_hist, env._p14_hist_token = buf, token
        return buf.reshape(buf.shape[0], -1)

    @staticmethod
    def p14_obj_pose(env, command_name: str) -> torch.Tensor:
        command = _cmd(env, command_name)
        si = _side_idx(command, "right")
        p = command.sim_obj_trans_w[:, si] - env.scene.env_origins
        R = matrix_from_quat(command.sim_obj_quat_w[:, si])
        return torch.cat([p, _rot6d(R)], dim=-1)

    @staticmethod
    def p14_future_obj(env, command_name: str, n_future: int = N_FUTURE, stride: int = STRIDE) -> torch.Tensor:
        command = _cmd(env, command_name)
        si = _side_idx(command, "right")
        off = torch.arange(1, n_future + 1, device=env.device) * stride
        fp = command.future_obj_traj_trans_w(off)[:, :, si]                              # (B, K, 3)
        fR = command.future_obj_traj_rotmat_w(off)[:, :, si]                             # (B, K, 3, 3)
        p = command.sim_obj_trans_w[:, si]
        R = matrix_from_quat(command.sim_obj_quat_w[:, si])
        dR = fR @ R.transpose(-1, -2)[:, None]
        feat = torch.cat([fp - p[:, None], _rot6d(dR)], dim=-1)                         # (B, K, 9)
        return feat.reshape(feat.shape[0], -1)

    @staticmethod
    def p14_future_joints(env, command_name: str, n_future: int = N_FUTURE, stride: int = STRIDE) -> torch.Tensor:
        command = _cmd(env, command_name)
        robot = command.robot
        ids = _right_joint_ids(env, robot)
        lo, hi = _limits(env, robot, ids)
        off = torch.arange(1, n_future + 1, device=env.device) * stride
        fq = command.future_ref_joint_pos_noisy(off)                                     # (B, K, n_dofs) all joints
        d = (fq[..., ids] - robot.data.joint_pos[:, None, ids]) * (2.0 / (hi - lo + 1e-5))
        return d.reshape(d.shape[0], -1)


def _track_only(fn):
    """A reward term that only pays in TrackEnv envs (zero in GenEnv)."""
    @functools.wraps(fn)
    def wrapped(env, *args, **kwargs):
        r = fn(env, *args, **kwargs)
        cname = kwargs.get("command_name", args[0] if args else "motion")
        return r * (~gen_mask(env, _cmd(env, cname))).to(r.dtype)
    return staticmethod(wrapped)


class CfhoiRewards(MotionTrackingRewards):
    tracking_ref_mano_wrist_trans_error_exp = _track_only(MotionTrackingRewards.tracking_ref_mano_wrist_trans_error_exp)
    tracking_ref_mano_wrist_rot_error_exp = _track_only(MotionTrackingRewards.tracking_ref_mano_wrist_rot_error_exp)
    tracking_ref_mano_tip_trans_error_exp = _track_only(MotionTrackingRewards.tracking_ref_mano_tip_trans_error_exp)
    tracking_ref_mano_tip_distance_bound_exp = _track_only(MotionTrackingRewards.tracking_ref_mano_tip_distance_bound_exp)
    tracking_joint_pos_bc_exp = _track_only(MotionTrackingRewards.tracking_joint_pos_bc_exp)

    @staticmethod
    def gen_obj_success(env, command_name: str, pos_th: float = 0.05, rot_th_deg: float = 20.0,
                        side: str | None = None) -> torch.Tensor:
        """GenEnv only: 1 when the object is within pos_th and rot_th_deg of its goal waypoint (AdaDex's sparse
        object-success reward), else 0."""
        command = _cmd(env, command_name)

        def one(s):
            si = _side_idx(command, s)
            e = torch.norm(command.sim_obj_trans_w[:, si] - command.ref_obj_trans_w[:, si], dim=-1)
            r = quat_error_magnitude(command.sim_obj_quat_w[:, si], command.ref_obj_quat_w[:, si])
            return ((e < pos_th) & (r < rot_th_deg * torch.pi / 180)).float()

        return _mean_over_sides(env, command_name, side, one) * gen_mask(env, command).float()

