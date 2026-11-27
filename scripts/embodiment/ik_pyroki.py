"""The IK step with collision avoidance, on PyRoki (github.com/chungmin99/pyroki; JAX, CPU is enough).

  python scripts/embodiment/ik_pyroki.py --arm vega_1p --mode palm       [--dataset DATA] [--rollouts RUN/evals/step<N>]
  python scripts/embodiment/ik_pyroki.py --arm vega_1p --mode fingertips
      -> $PROJECT_STORAGE_ROOT/runs/embodiment/ik_pyroki/<arm>/<tag>_<mode>/<clip>.npz + report.json
         (venv: $PROJECT_STORAGE_ROOT/venvs/arm)

ik.py reproduces FDR's palm pose and reports collisions; this solver avoids them. Collision model (spheres.py):
coarse spheres for robot-against-robot, the exact corners of each link's hull against the table.

Two modes, because most of what collides with the table is rigid with the palm (the arm's last link, the mount, the
camera) and no solver can move that while the palm stays where FDR put it:

  palm        the 7 arm joints. The palm must reach FDR's palm pose; the arm's other links must stay off the table
              and off the robot's own body. Moves the elbow and forearm out of the way; never changes the hand.
  fingertips  the 7 arm joints and the 22 hand joints. The five fingertips must stay where FDR put them (position,
              and loosely orientation); the palm may lift and tilt, and the fingers re-bend to compensate, as far
              as needed to get the wrist hardware and the palm off the table. FDR's palm pose and finger angles are
              only preferences. The result is a kinematic re-retargeting: the fingertips touch the object where FDR's
              did, but the motion has not been through physics again.

Both solve one frame at a time, starting from the previous frame's solution (with a pull towards it, so the arm moves
continuously). Hard constraints (augmented Lagrangian): joint limits, the Marvin's J6/J7 limit, the table, and the
arm's joint speed limits (no joint moves further in a frame than its speed limit allows; --speed-limit 0 lifts
that). Soft: targets, self-collision, a pull to the previous frame and a pull to ik.py's solution for the same
frame, so that the arm's posture is ik.py's wherever nothing forces it away. Where the hard constraints make a
target unreachable the solution leaves the target, and the report says by how much. In fingertips mode the fingers'
own links may not go lower into the table than FDR's fingers were in that frame.

The report measures the result on the exact meshes in MuJoCo, not on the collision model: palm and fingertip errors
against FDR, lowest points of arm, wrist hardware, palm and fingers above the table, joint speeds.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")                    # never touch the GPUs
# the problems are small (7 or 29 unknowns): one thread is as fast as many and leaves the shared machine alone
os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1")

import common as C  # noqa: E402  (first: it limits the BLAS threads before numpy loads)

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import jax_dataclasses as jdc  # noqa: E402
import jaxlie  # noqa: E402
import jaxls  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402
import pyroki as pk  # noqa: E402
import yourdfpy  # noqa: E402

jax.config.update("jax_enable_x64", True)

W = {  # weights: targets in 1/m and 1/rad
    "palm": {"pose_pos": 1000.0, "pose_ori": 200.0, "smooth": 0.3, "ref": 0.1, "self": 5.0},
    "fingertips": {"tip_pos": 1000.0, "tip_ori": 20.0, "palm_pos": 30.0, "palm_ori": 10.0, "hand": 1.0,
                   "smooth": 0.3, "smooth_hand": 0.1, "ref": 0.1, "self": 5.0},
}
# Starting penalty of the hard constraints. jaxls's own choice starts near 10, far below the targets' weights (1000/m),
# and then needs more iterations per frame than a 60 Hz clip can afford before the constraints win.
PENALTY_INITIAL = 1e5
TABLE_MARGIN = 0.002          # hull corners stay this far above the table (plus each link's corner-thinning cover)
SELF_MARGIN = 0.01


@jaxls.Cost.factory(kind="constraint_leq_zero")
def table_constraint(vals, robot, joint_var, pts, link_idx, margin, table):
    """Every hull corner that is over the table's footprint is at least `margin` above its top."""
    T = robot.forward_kinematics(vals[joint_var])
    p = jaxlie.SE3(T[link_idx]) @ pts
    over = (jnp.abs(p[:, 0] - table[0]) <= table[2]) & (jnp.abs(p[:, 1] - table[1]) <= table[3])
    return jnp.where(over, table[4] + margin - p[:, 2], -1.0)


@jaxls.Cost.factory(kind="constraint_leq_zero")
def linear_constraint(vals, joint_var, A, b):
    return A @ vals[joint_var] - b


@jaxls.Cost.factory(kind="constraint_leq_zero")
def step_constraint(vals, joint_var, q_prev, dmax):
    """No joint moves more than dmax from the previous frame (the joint speed limit times the frame time)."""
    dq = vals[joint_var] - q_prev
    return jnp.concatenate([dq - dmax, -dq - dmax])


@jaxls.Cost.factory
def weighted_rest(vals, joint_var, target, w):
    return (vals[joint_var] - target) * w


def se3(T):
    """4x4 numpy (or a batch) -> jaxlie.SE3."""
    T = np.asarray(T)
    q = np.stack([C.mat2quat(R) for R in T[..., :3, :3].reshape(-1, 3, 3)]).reshape(*T.shape[:-2], 4)
    return jaxlie.SE3(jnp.asarray(np.concatenate([q, T[..., :3, 3]], -1)))


class PyrokiIK:
    def __init__(self, arm, mode="palm", speed_limit=1.0, fps=60.0, max_iterations=60, penalty_initial=PENALTY_INITIAL, verbose=True):
        assert mode in W
        self.speed_limit, self.fps, self.max_iterations, self.penalty_initial = speed_limit, fps, max_iterations, penalty_initial
        self.arm, self.mode, self.cfg = arm, mode, C.load_arm(arm)
        d = C.model_dir(arm)
        info = self.info = json.load(open(d / "spheres.json"))
        self.urdf = yourdfpy.URDF.load(str(d / ("ik_arm.urdf" if mode == "palm" else "ik_full.urdf")), load_meshes=False)
        self.arm_joints, self.hand_joints = info["arm_joints"], info["hand_joints"]
        names = self.arm_joints + (self.hand_joints if mode == "fingertips" else [])
        probe = pk.Robot.from_urdf(self.urdf)
        self.act = list(probe.joints.actuated_names)
        assert sorted(self.act) == sorted(names), (self.act, names)
        self.to_pk = np.array([names.index(n) for n in self.act])            # our order -> PyRoki's
        self.from_pk = np.array([self.act.index(n) for n in names])
        self.nq = len(names)
        lo, hi = np.radians(self.cfg["limits_deg"]["lower"]), np.radians(self.cfg["limits_deg"]["upper"])
        home = np.clip(np.radians(self.cfg["home_deg"]), lo, hi)
        self.home = np.concatenate([home, np.zeros(self.nq - 7)])
        self.robot = pk.Robot.from_urdf(self.urdf, default_joint_cfg=self.home[self.to_pk])
        links = list(self.robot.links.names)
        self.palm_idx = jnp.array(links.index(info["palm_link"]), dtype=jnp.int32)
        self.tip_idx = jnp.array([links.index(t) for t in info["tip_links"]], dtype=jnp.int32) if mode == "fingertips" else None
        # the table (footprint and top), as in the scene
        t = C.load_cameras()["scene"]["table"]
        self.table = jnp.array([t["centre"][0], t["centre"][1], t["size"][0] / 2, t["size"][1] / 2, C.TABLE_Z])
        # hull corners that must stay above the table: the arm's links; in fingertips mode also what is rigid with the palm
        groups = ("arm",) if mode == "palm" else ("arm", "wrist", "fingers")
        pts, idx, margin, finger = [], [], [], []
        for link, h in info["hull_points"].items():
            if h["group"] in groups:
                pts.append(np.array(h["points"]))
                idx.append(np.full(len(h["points"]), links.index(link)))
                margin.append(np.full(len(h["points"]), (0.0 if h["group"] == "fingers" else TABLE_MARGIN) + h["cover"]))
                finger.append(np.full(len(h["points"]), h["group"] == "fingers"))
        self.pts, self.pts_idx, self.pts_margin = jnp.array(np.concatenate(pts)), jnp.array(np.concatenate(idx), dtype=jnp.int32), jnp.array(np.concatenate(margin))
        self.pts_finger = jnp.array(np.concatenate(finger))      # these get a per-frame margin: no lower than FDR's fingers were
        # coupled limits (the Marvin's wrist), in PyRoki's joint order
        rows = self.cfg.get("coupled_limits_deg") or []
        A, b = np.zeros((len(rows), self.nq)), np.zeros(len(rows))
        for i, r in enumerate(rows):
            for j, a in zip(r["joints"], r["a"]):
                A[i, self.act.index(f"arm_j{j}")] = a
            b[i] = np.radians(r["b"])
        self.A, self.b = jnp.array(A), jnp.array(b)
        # self-collision spheres; pairs that never move against each other, or always overlap, are left out
        dec = {k: {"centers": v["centers"], "radii": v["radii"]} for k, v in info["spheres"].items()}
        rigid = [info["groups"]["fixed"], info["groups"]["wrist"]]
        ignore = [(a, b2) for g in rigid for i, a in enumerate(g) for b2 in g[i + 1:] if a in dec and b2 in dec]
        coll = pk.collision.RobotCollision.from_sphere_decomposition(dec, self.urdf, user_ignore_pairs=tuple(ignore))
        always = self._always_colliding(coll, links)
        self.ignored_always = always
        self.coll = pk.collision.RobotCollision.from_sphere_decomposition(dec, self.urdf, user_ignore_pairs=tuple(ignore + always))
        # per-frame step limit: the arm's joint speed limits (the hand's are not known: unlimited)
        vmax = dict(zip(self.arm_joints, np.radians(self.cfg["velocity_deg_s"])))
        self.dmax = jnp.array([vmax[n] * speed_limit / fps if n in vmax and speed_limit > 0 else 1e3 for n in self.act])
        self.w_rest = self._weights()
        self._solve = self._build()
        if verbose:
            print(f"[pyroki] {arm} / {mode}: {self.nq} joints, {len(self.pts)} hull corners against the table, "
                  f"{len(self.coll.active_idx_i)} sphere pairs ({len(always)} link pairs always overlap and are ignored)")

    def _always_colliding(self, coll, links, n=300, frac=0.98):
        rng = np.random.default_rng(0)
        lo, hi = np.array(self.robot.joints.lower_limits), np.array(self.robot.joints.upper_limits)
        Q = jnp.array(rng.uniform(lo, hi, (n, len(lo))))
        D = np.array(jax.vmap(lambda q: coll.compute_self_collision_distance(self.robot, q))(Q))      # (n, pairs)
        g2l = np.array(coll._geom_to_link_idx)
        li, lj = g2l[np.array(coll.active_idx_i)], g2l[np.array(coll.active_idx_j)]
        out = []
        for a, b in sorted(set(zip(li.tolist(), lj.tolist()))):
            sel = (li == a) & (lj == b)
            if (D[:, sel].min(1) < 0).mean() >= frac:
                out.append((links[a], links[b]))
        return out

    def _weights(self):
        w = W[self.mode]
        is_arm = np.array([n.startswith("arm_j") for n in self.act])
        return {"ref": jnp.array(np.where(is_arm, w["ref"], 0.0)),
                "smooth": jnp.array(np.where(is_arm, w["smooth"], w.get("smooth_hand", 0.0))),
                "hand": jnp.array(np.where(is_arm, 0.0, w.get("hand", 0.0)))}

    def _build(self):
        robot, coll, mode, w = self.robot, self.coll, self.mode, W[self.mode]
        palm_idx, tip_idx = self.palm_idx, self.tip_idx
        pts, pts_idx, pts_margin, table, A, b = self.pts, self.pts_idx, self.pts_margin, self.table, self.A, self.b
        pts_finger, w_rest, max_iterations = self.pts_finger, self.w_rest, self.max_iterations
        penalty_initial = self.penalty_initial
        has_coupled = A.shape[0] > 0

        @jdc.jit
        def solve(q_prev, dmax, T_palm: jaxlie.SE3, T_tips, q_ref, finger_margin):
            var = robot.joint_var_cls(0)
            costs = [
                weighted_rest(var, q_ref, w_rest["ref"]),
                weighted_rest(var, q_prev, w_rest["smooth"]),
                pk.costs.self_collision_cost(robot, coll, var, SELF_MARGIN, w["self"]),
                pk.costs.limit_constraint(robot, var),
                table_constraint(robot, var, pts, pts_idx, pts_margin + jnp.where(pts_finger, finger_margin, 0.0), table),
            ]
            if has_coupled:
                costs.append(linear_constraint(var, A, b))
            costs.append(step_constraint(var, q_prev, dmax))
            if mode == "palm":
                costs.append(pk.costs.pose_cost(robot, var, T_palm, palm_idx, w["pose_pos"], w["pose_ori"]))
            else:
                costs.append(pk.costs.pose_cost(robot, var, T_tips, tip_idx, w["tip_pos"], w["tip_ori"]))
                costs.append(pk.costs.pose_cost(robot, var, T_palm, palm_idx, w["palm_pos"], w["palm_ori"]))
                costs.append(weighted_rest(var, q_ref, w_rest["hand"]))
            sol = jaxls.LeastSquaresProblem(costs=costs, variables=[var]).analyze().solve(
                initial_vals=jaxls.VarValues.make([var.with_value(q_prev)]), verbose=False, linear_solver="dense_cholesky",
                termination=jaxls.TerminationConfig(max_iterations=max_iterations),
                augmented_lagrangian=jaxls.AugmentedLagrangianConfig(penalty_initial=penalty_initial))
            return sol[var]

        return solve

    def solve_clip(self, Ts_palm, Q_hand, Ts_tips, Q_arm_ref, finger_clearance=None):
        """Ts_palm (T, 4, 4) and Q_hand (T, 22): FDR's palm poses and finger angles; Ts_tips (T, 5, 4, 4): its
        fingertip frames; Q_arm_ref (T, 7): ik.py's arm solution, the posture this solver stays close to where
        nothing forces it away; finger_clearance (T,): how far FDR's fingers were above the table (negative =
        inside), the level this solver's fingers may not go below. Returns arm_q (T, 7), hand_q (T, 22)."""
        n = len(Ts_palm)
        q = np.concatenate([Q_arm_ref[0], Q_hand[0]])[: self.nq][self.to_pk]
        fm = np.zeros(n) if finger_clearance is None else np.minimum(finger_clearance, 0.0) - 0.001
        out = np.zeros((n, self.nq))
        palm = se3(Ts_palm)
        tips = se3(Ts_tips) if self.mode == "fingertips" else None
        free = jnp.full(self.nq, 1e3)
        for t in range(n):
            ref = np.concatenate([Q_arm_ref[t], Q_hand[t]])[: self.nq][self.to_pk]
            # the first frame has no previous frame: it may move as far from the start as the constraints need
            q = np.array(self._solve(jnp.array(q), free if t == 0 else self.dmax, jaxlie.SE3(palm.wxyz_xyz[t]),
                                     jaxlie.SE3(tips.wxyz_xyz[t]) if tips is not None else jaxlie.SE3(palm.wxyz_xyz[t]),
                                     jnp.array(ref), jnp.array(fm[t])))
            out[t] = q
        out = out[:, self.from_pk]
        return out[:, :7], (out[:, 7:] if self.mode == "fingertips" else Q_hand.copy())


class ObjectParts:
    """The clip object's convex collision parts (FDR's): how deep points are inside the object."""

    def __init__(self, clip):
        import trimesh
        from scipy.spatial import ConvexHull
        self.eq, self.box = [], []
        for f in sorted((Path(clip["mesh_dir"]) / "convex").glob("*.obj")):
            v = np.asarray(trimesh.load(f, force="mesh", process=False).vertices) * clip.get("mesh_scale", 1.0)
            if len(v) >= 4:
                self.eq.append(ConvexHull(v).equations)
                self.box.append((v.min(0), v.max(0)))

    def depth(self, P_world, pos, R):
        """Deepest penetration [m] of world points into the object at pose (pos, R); 0 if none is inside."""
        P = (P_world - pos) @ R
        d = 0.0
        for eq, (lo, hi) in zip(self.eq, self.box):
            Q = P[np.all((P >= lo) & (P <= hi), axis=1)]          # only points inside the part's bounding box can be inside it
            if len(Q):
                d = max(d, float((-(Q @ eq[:, :3].T + eq[:, 3]).max(1)).max()))
        return max(d, 0.0)


def evaluate(scene, ik0, arm_q, hand_q, Ts_palm, tips_ref, hand_ref=None, obj=None, fps=60.0):
    """Measure a solution on the exact model: errors against FDR and clearances from the table."""
    m, sc = scene.model, scene
    names = sc.geom_names
    b7 = m.jnt_bodyid[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "arm_j7")]
    rigid_arm = [g for g in ik0.link_geoms if _under(m, m.geom_bodyid[g], b7)]
    movable = [g for g in ik0.link_geoms if g not in rigid_arm]
    hardware = rigid_arm + [g for g, nme in enumerate(names) if nme in ("d405_mount", "wrist_cam_housing")]
    n = len(arm_q)
    out = {k: np.zeros(n) for k in ("palm_pos", "palm_rot", "tip_err", "arm", "hardware", "palm", "fingers", "object_depth", "hand_dq")}
    hand_geoms = sc.palm_geoms + sc.finger_geoms
    for t in range(n):
        sc.set_arm(arm_q[t], hand_q[t])
        sc.fk()
        if obj is not None:
            parts, OP, OR = obj
            out["object_depth"][t] = parts.depth(sc.world_points(hand_geoms, 60), OP[t], OR[t])
        if hand_ref is not None:
            out["hand_dq"][t] = np.abs(hand_q[t] - hand_ref[t]).max()
        e = C.pose_error(Ts_palm[t], sc.palm())
        out["palm_pos"][t], out["palm_rot"][t] = np.linalg.norm(e[:3]), np.linalg.norm(e[3:])
        out["tip_err"][t] = np.linalg.norm(sc.tips() - tips_ref[t], axis=1).max()
        out["arm"][t] = sc.table_clearance(movable)
        out["hardware"][t] = sc.table_clearance(hardware)
        out["palm"][t] = sc.table_clearance(sc.palm_geoms)
        out["fingers"][t] = sc.table_clearance(sc.finger_geoms)
    speed = np.abs(np.diff(arm_q, axis=0)) * fps / ik0.vmax if n > 1 else np.zeros((1, 7))
    out["speed_ratio"] = speed.max(1)
    out["margin_joint"] = np.array([ik0.margins(q)[0] for q in arm_q])
    return out


def _under(m, b, top):
    while b > 0:
        if b == top:
            return True
        b = m.body_parentid[b]
    return False


def usable_frames(ev, floating_fingers, floating_obj):
    """Frames that could be used as robot data: fingertips where FDR's were (3 mm), arm and wrist hardware above the
    table, fingers no deeper in the table and hand no deeper in the object than FDR's (2 mm slack), joint speeds
    within the arm's limits."""
    ok = ev["tip_err"] <= 0.003
    ok &= (ev["arm"] >= 0) & (ev["hardware"] >= 0)
    ok &= ev["fingers"] >= np.minimum(floating_fingers, 0.0) - 0.002
    ok &= ev["object_depth"] <= floating_obj + 0.002
    speed = np.concatenate([[0.0], ev["speed_ratio"]]) if len(ev["speed_ratio"]) == len(ok) - 1 else np.zeros(len(ok))
    return ok & (speed <= 1.02)


def summarise(ev, floating_fingers, floating_obj):
    clear = np.minimum(ev["arm"], ev["hardware"])
    ok = usable_frames(ev, floating_fingers, floating_obj)
    return {"frames": int(len(clear)), "usable_pct": float(ok.mean() * 100),
            "palm_dev_mm": {"median": float(np.median(ev["palm_pos"]) * 1000), "p95": float(np.percentile(ev["palm_pos"], 95) * 1000), "max": float(ev["palm_pos"].max() * 1000)},
            "palm_dev_deg": {"median": float(np.degrees(np.median(ev["palm_rot"]))), "p95": float(np.degrees(np.percentile(ev["palm_rot"], 95))), "max": float(np.degrees(ev["palm_rot"].max()))},
            "tip_err_mm": {"median": float(np.median(ev["tip_err"]) * 1000), "p95": float(np.percentile(ev["tip_err"], 95) * 1000), "max": float(ev["tip_err"].max() * 1000)},
            "arm_table_min_mm": float(ev["arm"].min() * 1000), "hardware_table_min_mm": float(ev["hardware"].min() * 1000),
            "palm_table_min_mm": float(ev["palm"].min() * 1000), "fingers_table_min_mm": float(ev["fingers"].min() * 1000),
            "fingers_table_min_fdr_mm": float(floating_fingers.min() * 1000),
            "inside_table_pct": float((clear < 0).mean() * 100),
            "finger_change_deg": {"p95": float(np.degrees(np.percentile(ev["hand_dq"], 95))), "max": float(np.degrees(ev["hand_dq"].max()))},
            "hand_in_object_max_mm": float(ev["object_depth"].max() * 1000), "hand_in_object_max_fdr_mm": float(floating_obj.max() * 1000),
            "speed_ratio_max": float(ev["speed_ratio"].max()), "speed_over_limit_pct": float((ev["speed_ratio"] > 1.02).mean() * 100),
            "margin_joint_min": float(ev["margin_joint"].min())}


def main():
    from ik import ArmIK
    from scene import Scene
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--mode", default="palm", choices=list(W))
    ap.add_argument("--dataset", default=str(C.STORAGE / "runs" / "flashdex" / "data_r10_flat"))
    ap.add_argument("--rollouts", default=None)
    ap.add_argument("--clips", nargs="*", default=None)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--max-iter", type=int, default=60, help="solver iterations per frame")
    ap.add_argument("--penalty", type=float, default=PENALTY_INITIAL, help="starting penalty of the hard constraints")
    ap.add_argument("--speed-limit", type=float, default=1.0,
                    help="hard limit on the arm's joint speeds, as a multiple of the arm's limits (0: no limit)")
    args = ap.parse_args()
    clips = C.load_pack(args.dataset)
    roll = C.load_rollouts(args.rollouts)[0] if args.rollouts else None
    tag = (args.tag or (Path(args.dataset).name + ("_rollouts" if args.rollouts else "_reference"))) + f"_{args.mode}"
    out = C.RUNS / "ik_pyroki" / args.arm / tag
    out.mkdir(parents=True, exist_ok=True)
    hand = C.FloatingHand()
    ik0 = ArmIK(args.arm)                                        # first-frame solution and the exact model for the report
    solver = PyrokiIK(args.arm, args.mode, speed_limit=args.speed_limit, max_iterations=args.max_iter, penalty_initial=args.penalty)
    floating = Scene("floating")
    report = {"arm": args.arm, "mode": args.mode, "dataset": args.dataset, "rollouts": args.rollouts, "weights": W[args.mode],
              "table_margin_m": TABLE_MARGIN, "clips": {}}
    for c in clips:
        if args.clips and c["name"] not in args.clips:
            continue
        src = roll.get(c["name"]) if roll is not None else c
        if src is None:
            continue
        Qf = np.asarray(src["joint_pos"], float)
        Ts = hand.palm_traj(Qf)
        Qh = hand.fingers(Qf)
        tips, fing0, obj0 = np.zeros((len(Qf), 5, 4, 4)), np.zeros(len(Qf)), np.zeros(len(Qf))
        parts = ObjectParts(c)
        OP, OR = np.asarray(src["obj_pos"]), np.asarray(src["obj_rot"])
        fl_hand = floating.palm_geoms + floating.finger_geoms
        for t, q in enumerate(Qf):                               # FDR's own fingertips, finger-table and hand-object contact
            floating.set_floating(q)
            floating.fk()
            for k, s in enumerate(floating.tip_sites):
                tips[t, k] = C.make_T(floating.data.site_xmat[s].reshape(3, 3), floating.data.site_xpos[s])
            fing0[t] = floating.table_clearance(floating.finger_geoms)
            obj0[t] = parts.depth(floating.world_points(fl_hand, 60), OP[t], OR[t])
        Q_ref, _ = ik0.convert(Ts)                               # ik.py's solution: the first frame and the posture reference
        if Q_ref is None:
            report["clips"][c["name"]] = {"reason": "no IK solution for the first frame"}
            print(f"{c['name']:22s} no solution for the first frame")
            continue
        t0 = time.time()
        arm_q, hand_q = solver.solve_clip(Ts, Qh, tips, Q_ref, fing0)
        dt = time.time() - t0
        ev = evaluate(ik0.scene, ik0, arm_q, hand_q, Ts, tips[:, :, :3, 3], Qh, (parts, OP, OR))
        s = summarise(ev, fing0, obj0)
        s["ms_per_frame"] = 1000 * dt / len(Qf)
        report["clips"][c["name"]] = s
        np.savez_compressed(out / f"{c['name']}.npz", arm_q=arm_q.astype(np.float32), hand_q=hand_q.astype(np.float32),
                            hand_q_fdr=Qh.astype(np.float32), arm_joints=np.array(solver.arm_joints), hand_joints=np.array(solver.hand_joints),
                            palm_T=Ts.astype(np.float32), pos_err_mm=(ev["palm_pos"] * 1000).astype(np.float32),
                            rot_err_deg=np.degrees(ev["palm_rot"]).astype(np.float32), tip_err_mm=(ev["tip_err"] * 1000).astype(np.float32),
                            arm_table_mm=(ev["arm"] * 1000).astype(np.float32), hardware_table_mm=(ev["hardware"] * 1000).astype(np.float32),
                            speed_ratio=ev["speed_ratio"].astype(np.float32))
        print(f"{c['name'].replace('_anthony_real', ''):6s} usable {s['usable_pct']:5.1f}% | palm moved {s['palm_dev_mm']['median']:5.1f} / {s['palm_dev_mm']['max']:5.1f} mm, "
              f"{s['palm_dev_deg']['median']:4.1f} / {s['palm_dev_deg']['max']:4.1f} deg (median / max) | fingertips off {s['tip_err_mm']['p95']:4.1f} / "
              f"{s['tip_err_mm']['max']:5.1f} mm (p95 / max) | above table: arm {s['arm_table_min_mm']:4.0f}, hardware {s['hardware_table_min_mm']:4.0f}, "
              f"palm {s['palm_table_min_mm']:4.0f}, fingers {s['fingers_table_min_mm']:4.0f} (FDR {s['fingers_table_min_fdr_mm']:4.0f}) mm | "
              f"inside {s['inside_table_pct']:4.1f}% | fingers re-bent {s['finger_change_deg']['max']:4.1f} deg | hand in object {s['hand_in_object_max_mm']:4.1f} "
              f"(FDR {s['hand_in_object_max_fdr_mm']:4.1f}) mm | speed {s['speed_ratio_max']:.1f}x ({s['speed_over_limit_pct']:.0f}% over) | {s['ms_per_frame']:.0f} ms/frame")
    done = [v for v in report["clips"].values() if "usable_pct" in v]
    report["clips_fully_usable"] = int(sum(v["usable_pct"] >= 100 for v in done))
    report["frames_usable_pct"] = float(sum(v["usable_pct"] * v["frames"] for v in done) / max(1, sum(v["frames"] for v in done)))
    json.dump(report, open(out / "report.json", "w"), indent=1)
    print(f"{report['clips_fully_usable']} of {len(done)} clips usable in every frame; {report['frames_usable_pct']:.1f}% of all frames usable")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
