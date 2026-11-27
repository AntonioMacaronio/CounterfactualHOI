"""The IK step of §1.4: FlashDexRetarget's floating wrist -> the arm's seven joint angles.

  python scripts/embodiment/ik.py --arm marvin_m6ccs --dataset DATA [--rollouts RUN/evals/step<N>] [--field joint_pos]
      -> $PROJECT_STORAGE_ROOT/runs/embodiment/ik/<arm>/<tag>/<clip>.npz  +  report.json
         (venv: $PROJECT_STORAGE_ROOT/venvs/arm)

FDR's robot is the 22-DoF Sharpa hand on a floating 6-DoF wrist. Its joint vector gives the palm's pose in the clip
world at every 60 Hz step; the arm has to put its own palm (the same body, R_forearm_rot_y_link) there. Per clip:

  1. First frame: many starts inside the joint limits, each iterated to the target. The arm has seven joints for a
     six-dimensional pose, so the solutions form a family (the elbow can swing about the shoulder-wrist line).
  2. Each of the most different solutions is followed along the clip at 10 Hz. The start whose path stays furthest
     from the joint limits, the wrist's coupled limit and the table is kept (failures count first).
  3. The clip is solved at 60 Hz from that start, every step starting at the previous step's solution, so the arm
     moves continuously. A small pull towards the arm's home posture keeps the elbow from drifting.

A step is a small QP over the joint change dq:  min |J dq - e|^2_W + damping |dq|^2 + posture |q + dq - q_home|^2
subject to the joint limits, a per-iteration step limit and the arm's coupled limits (the Marvin's J6/J7 octagon),
iterated until the palm is within 0.05 mm and 0.01 deg of its target. The result is not clamped to the joint speed
limits: the data has to reproduce FDR's wrist path, so speed above the limit is reported instead.

Output per clip: arm_q (T, 7), hand_q (T, 22), joint names, pos_err_mm / rot_err_deg (T), the margins and speeds.
The same call converts joint *targets* once FDR's rollouts log them: --field joint_target.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
from qpsolvers import solve_qp

import common as C

POS_TOL, ROT_TOL = 5e-5, np.radians(0.01)


class ArmIK:
    def __init__(self, arm, scene=None):
        from scene import Scene
        self.arm = arm
        self.cfg = C.load_arm(arm)
        self.scene = scene or Scene(arm)
        m = self.model = self.scene.model
        self.data = self.scene.data
        self.adr = np.array(self.scene.arm_adr)
        self.dof = np.array([m.jnt_dofadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"arm_j{k + 1}")] for k in range(7)])
        self.lo, self.hi = np.radians(self.cfg["limits_deg"]["lower"]), np.radians(self.cfg["limits_deg"]["upper"])
        self.vmax = np.radians(self.cfg["velocity_deg_s"])
        self.home = np.clip(np.radians(self.cfg["home_deg"]), self.lo, self.hi)
        rows = self.cfg.get("coupled_limits_deg") or []
        self.Ac = np.zeros((len(rows), 7))
        self.bc = np.zeros(len(rows))
        for i, r in enumerate(rows):                              # a . q_deg <= b  ->  a . q_rad <= b in rad
            for j, a in zip(r["joints"], r["a"]):
                self.Ac[i, j - 1] = a
            self.bc[i] = np.radians(r["b"])
        self.palm = self.scene.palm_id
        self._jp, self._jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
        # arm links that move (below arm_j1, above the hand): for the clearance to the table
        b1 = m.jnt_bodyid[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "arm_j1")]
        hm = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "hand_mount")

        def below(b, top):
            while b > 0:
                if b == top:
                    return True
                b = m.body_parentid[b]
            return False
        self.link_geoms = [g for g in self.scene.arm_geoms if below(m.geom_bodyid[g], b1) and not below(m.geom_bodyid[g], hm)]
        self.static_geoms = [g for g in self.scene.arm_geoms if not below(m.geom_bodyid[g], b1)]
        self.mount_geoms = [g for g, n in enumerate(self.scene.geom_names) if n in ("d405_mount", "mount_plate", "wrist_cam_housing")]
        self.W = np.diag([1.0, 1.0, 1.0, 0.2, 0.2, 0.2])          # metres vs radians

    # ---- kinematics
    def set(self, q, q_hand=None):
        self.data.qpos[self.adr] = q
        if q_hand is not None:
            self.data.qpos[self.scene.hand_adr] = q_hand
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_comPos(self.model, self.data)

    def fk(self, q):
        self.set(q)
        return self.scene.palm()

    def jac(self):
        mujoco.mj_jacBody(self.model, self.data, self._jp, self._jr, self.palm)
        return np.vstack([self._jp[:, self.dof], self._jr[:, self.dof]])

    def margins(self, q):
        """Normalised distance to the nearest limit, 0 at a limit, 0.5 at mid-range: (per-joint min, coupled)."""
        mj = float(np.min(np.minimum(q - self.lo, self.hi - q) / (self.hi - self.lo)))
        mc = float(np.min(self.bc - self.Ac @ q)) if len(self.bc) else np.inf
        return mj, mc

    def table_clearance(self):
        """Height of the lowest point of the moving arm links above the table top, at the current configuration [m]
        (inf when no link is over the table)."""
        return self.scene.table_clearance(self.link_geoms)

    # ---- one target
    def solve(self, T, q0, iters=60, damping=1e-4, posture=1e-6, step=0.3, q_ref=None):
        """Iterate from q0 to the palm pose T. Returns (q, position error [m], rotation error [rad])."""
        q = np.clip(np.asarray(q0, float), self.lo, self.hi)
        q_ref = self.home if q_ref is None else q_ref
        ep = er = np.inf
        for _ in range(iters):
            e = C.pose_error(T, self.fk(q))
            ep, er = np.linalg.norm(e[:3]), np.linalg.norm(e[3:])
            if ep < POS_TOL and er < ROT_TOL:
                break
            J = self.jac()
            P = J.T @ self.W @ J + (damping + posture) * np.eye(7)
            g = -J.T @ self.W @ e + posture * (q - q_ref)
            lb, ub = np.maximum(self.lo - q, -step), np.minimum(self.hi - q, step)
            G, h = (self.Ac, self.bc - self.Ac @ q) if len(self.bc) else (None, None)
            dq = solve_qp(P, g, G, h, lb=lb, ub=ub, solver="daqp")
            if dq is None:
                break
            q = q + dq
        return q, ep, er

    def starts(self, T, n=96, seed=0, keep=8):
        """Different solutions for one pose: converged multi-start results, thinned so that no two are within
        0.35 rad (joint-space max norm) of each other, best limit margin first."""
        rng = np.random.default_rng(seed)
        seeds = [self.home] + [rng.uniform(self.lo, self.hi) for _ in range(n - 1)]
        sols = []
        for s in seeds:
            q, ep, er = self.solve(T, s)
            if ep < POS_TOL * 2 and er < ROT_TOL * 2:
                mj, mc = self.margins(q)
                sols.append((min(mj, mc / np.pi), q))
        sols.sort(key=lambda x: -x[0])
        out = []
        for _, q in sols:
            if all(np.max(np.abs(q - o)) > 0.35 for o in out):
                out.append(q)
            if len(out) >= keep:
                break
        return out

    # ---- a trajectory
    def track(self, Ts, q0, clearance=False):
        """Follow palm poses Ts (T, 4, 4) from q0, each step starting at the previous solution."""
        n = len(Ts)
        Q = np.zeros((n, 7))
        ep, er = np.zeros(n), np.zeros(n)
        mj, mc = np.zeros(n), np.zeros(n)
        clr = np.full(n, np.nan)
        q = np.asarray(q0, float)
        for t in range(n):
            q, ep[t], er[t] = self.solve(Ts[t], q, iters=25)
            Q[t] = q
            mj[t], mc[t] = self.margins(q)
            if clearance:
                self.set(q)
                clr[t] = self.table_clearance()
        return Q, {"pos_err": ep, "rot_err": er, "margin_joint": mj, "margin_coupled": mc, "table_clearance": clr}

    def best_start(self, Ts, stride=6, n=96, seed=0):
        """Pick the first-frame solution whose path along the (subsampled) clip is best. Returns (q0, score rows)."""
        cands = self.starts(Ts[0], n=n, seed=seed)
        rows = []
        for q0 in cands:
            Q, info = self.track(Ts[::stride], q0, clearance=True)
            fail = int(((info["pos_err"] > 1e-3) | (info["rot_err"] > np.radians(0.5))).sum())
            hit = int((info["table_clearance"] < 0.0).sum())
            jump = float(np.max(np.abs(np.diff(Q, axis=0)))) if len(Q) > 1 else 0.0
            margin = float(min(info["margin_joint"].min(), info["margin_coupled"].min() / np.pi))
            rows.append({"q0": q0, "fail": fail, "table_hits": hit, "margin": margin, "max_step": jump,
                         "key": (fail, hit, -round(margin, 3), jump)})
        rows.sort(key=lambda r: r["key"])
        return (rows[0]["q0"] if rows else None), rows

    def convert(self, Ts, fps=60.0, q0=None):
        """A whole clip. Returns (Q, info) with speeds and margins; Q is None when the first frame has no solution."""
        if q0 is None:
            q0, rows = self.best_start(Ts)
            if q0 is None:
                return None, {"reason": "no IK solution for the first frame"}
        Q, info = self.track(Ts, q0, clearance=True)
        speed = np.abs(np.diff(Q, axis=0)) * fps if len(Q) > 1 else np.zeros((1, 7))
        info["speed_ratio"] = (speed / self.vmax).max(1) if len(speed) else np.zeros(0)
        return Q, info


def summarise(info):
    ok = (info["pos_err"] < 1e-3) & (info["rot_err"] < np.radians(0.5))
    return {"frames": int(len(ok)), "solved_pct": float(ok.mean() * 100),
            "pos_err_max_mm": float(info["pos_err"].max() * 1000), "rot_err_max_deg": float(np.degrees(info["rot_err"].max())),
            "margin_joint_min": float(info["margin_joint"].min()),
            "margin_coupled_min_deg": float(np.degrees(info["margin_coupled"].min())) if np.isfinite(info["margin_coupled"]).all() else None,
            "table_clearance_min_mm": float(np.nanmin(info["table_clearance"]) * 1000),
            "table_hit_pct": float((info["table_clearance"] < 0).mean() * 100),
            "speed_ratio_max": float(info["speed_ratio"].max()) if len(info["speed_ratio"]) else 0.0,
            "speed_over_limit_pct": float((info["speed_ratio"] > 1).mean() * 100) if len(info["speed_ratio"]) else 0.0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--dataset", default=str(C.STORAGE / "runs" / "flashdex" / "data_r10_flat"))
    ap.add_argument("--rollouts", default=None, help="convert a rollout archive (RUN/evals/step<N>) instead of the reference clips")
    ap.add_argument("--field", default="joint_pos", help="which 28/56-joint array of the source to convert")
    ap.add_argument("--clips", nargs="*", default=None)
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()
    clips = C.load_pack(args.dataset)
    roll = C.load_rollouts(args.rollouts)[0] if args.rollouts else None
    tag = args.tag or (Path(args.dataset).name + ("_rollouts" if args.rollouts else "_reference"))
    out = C.RUNS / "ik" / args.arm / tag
    out.mkdir(parents=True, exist_ok=True)
    hand = C.FloatingHand()
    ik = ArmIK(args.arm)
    report = {"arm": args.arm, "dataset": args.dataset, "rollouts": args.rollouts, "field": args.field, "clips": {}}
    for c in clips:
        if args.clips and c["name"] not in args.clips:
            continue
        src = roll.get(c["name"]) if roll is not None else c
        if src is None:
            continue
        Qf = np.asarray(src[args.field], float)
        Ts = hand.palm_traj(Qf)
        Q, info = ik.convert(Ts)
        if Q is None:
            report["clips"][c["name"]] = info
            print(f"{c['name']:22s} no solution for the first frame")
            continue
        s = summarise(info)
        report["clips"][c["name"]] = s
        np.savez_compressed(out / f"{c['name']}.npz", arm_q=Q.astype(np.float32), hand_q=hand.fingers(Qf).astype(np.float32),
                            arm_joints=np.array([f"arm_j{k + 1}" for k in range(7)]), hand_joints=np.array(hand.hand_joints),
                            palm_T=Ts.astype(np.float32), pos_err_mm=(info["pos_err"] * 1000).astype(np.float32),
                            rot_err_deg=np.degrees(info["rot_err"]).astype(np.float32),
                            margin_joint=info["margin_joint"].astype(np.float32),
                            table_clearance_mm=(info["table_clearance"] * 1000).astype(np.float32),
                            speed_ratio=info["speed_ratio"].astype(np.float32))
        print(f"{c['name']:22s} solved {s['solved_pct']:5.1f}% | err max {s['pos_err_max_mm']:.3f} mm {s['rot_err_max_deg']:.3f} deg | "
              f"limit margin {s['margin_joint_min']:.3f} | arm-table {s['table_clearance_min_mm']:.0f} mm | "
              f"speed max {s['speed_ratio_max']:.2f} x limit ({s['speed_over_limit_pct']:.1f}% of steps over)")
    json.dump(report, open(out / "report.json", "w"), indent=1)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
