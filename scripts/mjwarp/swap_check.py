"""Checks of FDR's one-object swap mode on MuJoCo Warp (scripts/mjwarp/swap_collision.py).

  # one rollout per mode, the same clips and start frames, zero residual actions (the reference played through the
  # PD controller), FDR's eval settings (no reference noise, no assist):
                 python swap_check.py run --motion D/motion_bimanual.pt --out swap.npz  [overrides]   # swap mode (default)
  FDR_OBJ_SWAP=0 python swap_check.py run --motion D/motion_bimanual.pt --out multi.npz [overrides]   # multi-object mode
  SWAP_NO_INJECT=1 python swap_check.py run ... --out noinject.npz    # orphan meshes without hull data
  python swap_check.py compare multi.npz swap.npz noinject.npz

  python swap_check.py subset --motion D/motion_bimanual.pt --clips 5 --out-dir OUT/single5    # one-clip pack
  python swap_check.py widen --motion D/motion_bimanual.pt --copies 50 --out-dir OUT/wide500     # 500-object pack

`run` in swap mode first checks every world's rows against its clip's object: the geoms' mesh ids point at that
object's parts (names obj_right_swap_o<slot>_h<k>, the rest the null tetrahedron), and the body mass, inertia, CoM,
inertia frame, invweights and the geoms' poses, sizes and bounds are the swap table's (or, with --trials, the
trial's mass, inertia and CoM). In both modes it records the active object's parts, CoM, mass and inertia right
after the reset, then per step and env the object's position and orientation error against the reference, the
done flag, the termination term behind each env's first done and the contact count, and writes them to --out.
`compare` prints per-clip tracking statistics side by side, the reset-state differences, and how far each run's
object path is from the first's. A one-clip pack run in swap mode is the single-object reference for that clip
(the object is its own template); `widen` makes a scene with many objects for build-time, memory and mesh-id checks.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np


def _quat_angle(q1, q2):
    d = (q1 * q2).sum(-1).abs().clamp(max=1.0)
    return 2.0 * torch.arccos(d)


def run(a):
    global torch
    import torch

    if os.environ.get("SWAP_NO_INJECT"):          # the failure case: skip the hull data for the orphan meshes
        import swap_collision as sc

        def skip(sim, dataid, nhull, hulls, **k):
            print("[swap_check] SWAP_NO_INJECT: orphan meshes keep graphadr -1 and no polygons", flush=True)
            return {}
        sc.inject_swap_collision = skip
    from create import compose, make_env

    # one env per (clip, copy), the layout of FDR's eval env
    t0 = time.time()
    n_env = len(torch.load(a.motion, map_location="cpu", weights_only=False)["motion_num_frames"]) * a.per_clip
    cfg = compose(["envs/robot@robot=sharpa_bimanual", f"commands.motion_file={a.motion}", f"num_envs={n_env}",
                   "per_hand_actor_critic=false", *a.overrides])
    env = make_env(cfg, n_env, trials=a.trials)
    cmd = env.command_manager.get_term(cfg.commands.name)
    lib = cmd.motion_lib
    n_clip = int(lib.num_trajectories)
    if n_env != n_clip * a.per_clip:
        sys.exit(f"built {n_env} envs for {n_clip} clips")
    names = [f.rsplit("#", 1)[-1] for f in lib.motion_files]
    swap = getattr(cmd, "_swap_tables", None)
    mjm = env.sim.mj_model
    print(f"[swap_check] {'swap' if swap is not None else 'multi-object'} mode, {n_clip} clips x {a.per_clip}, built "
          f"in {time.time() - t0:.1f} s; nq {mjm.nq}, ngeom {mjm.ngeom}, nbody {mjm.nbody}, nmesh {mjm.nmesh}", flush=True)

    cmd.set_eval_mode(sampling_mode="start", noise_to_initial_level=0.0, start_frame=0)
    cmd.motion_ids[:] = torch.arange(n_env, device=env.device) % n_clip
    env.reset()
    dev = env.device
    W = torch.arange(n_env, device=dev)
    clip = cmd.motion_ids.long().clone()
    slot = lib.traj_obj_slot["right"][clip].long() if hasattr(lib, "traj_obj_slot") else clip

    # ---- per-world rows (swap mode)
    m = env.sim.model
    problems = 0
    if swap is not None:
        gid = cmd._swap_geom_ids
        b = cmd._swap_body_id
        did = m.geom_dataid[W[:, None], gid[None, :]].cpu().numpy()
        nh = swap["nhull"].cpu().numpy()
        for w in range(n_env):
            s = int(slot[w])
            for k in range(len(gid)):
                nm = mjm.mesh(int(did[w, k])).name.rsplit("/", 1)[-1]
                want = f"obj_right_swap_o{s}_h{k}" if k < nh[s] else "obj_right_null_tet"
                if nm != want:
                    problems += 1
                    if problems < 5:
                        print(f"[swap_check] env {w} ({names[int(clip[w])]}) geom {k}: mesh {nm}, want {want}")
        for f, tf in (("body_mass", "bmass"), ("body_inertia", "binertia"), ("body_ipos", "bipos")):
            got = getattr(m, f)[W, b]
            want = swap[tf][slot].to(got.dtype)
            if env.trial_randomizer is not None:
                tr = env.trial_randomizer
                k = tr.trial_id
                want = torch.stack([{"body_mass": tr.mass, "body_inertia": tr.inertia, "body_ipos": tr.ipos}[f][int(c)][int(kk)]
                                    for c, kk in zip(clip.tolist(), k.tolist())]).to(got.dtype)
            err = float((got - want).abs().max())
            print(f"[swap_check] {f}: max |world row - expected| {err:.2e}", flush=True)
            problems += err > 1e-6
        if env.trial_randomizer is None:           # with trials, set_const recomputes these from the trial's mass
            for f, tf in (("body_iquat", "biquat"), ("body_invweight0", "binvw"), ("body_subtreemass", "bsubm")):
                got = getattr(m, f)[W, b]
                err = float((got - swap[tf][slot].to(got.dtype)).abs().max())
                print(f"[swap_check] {f}: max |world row - table| {err:.2e}", flush=True)
                problems += err > 1e-6
            d0 = cmd._swap_dof0
            got = m.dof_invweight0[W, d0:d0 + 6]
            err = float((got - swap["dinvw"][slot].to(got.dtype)).abs().max())
            print(f"[swap_check] dof_invweight0: max |world row - table| {err:.2e}", flush=True)
            problems += err > 1e-6
        for f, tf in (("geom_size", "gsize"), ("geom_rbound", "grbound"), ("geom_pos", "gpos"), ("geom_quat", "gquat"),
                      ("geom_aabb", "gaabb")):
            got = getattr(m, f)[W[:, None], gid[None, :]]
            err = float((got.reshape(n_env, len(gid), -1) - swap[tf][slot].to(got.dtype).reshape(n_env, len(gid), -1))
                        .abs().max())
            print(f"[swap_check] {f}: max |world row - table| {err:.2e}", flush=True)
            problems += err > 1e-6
        masses = 1000 * m.body_mass[W, b]
        for c in range(n_clip):
            print(f"[swap_check]   {names[c]}: slot {int(slot[c])}, {int(nh[int(slot[c])])} parts, mass "
                  f"{float(masses[c]):.0f} g", flush=True)
        print(f"[swap_check] per-world rows: {'OK' if problems == 0 else f'{problems} PROBLEMS'}", flush=True)

    # ---- the active object's collision parts in the world right after the reset, for comparing modes: part k's
    # world position and rotation, the body's CoM, mass and principal inertia
    P = 32
    gx = np.full((n_env, P, 3), np.nan, np.float32)
    gm = np.full((n_env, P, 9), np.nan, np.float32)
    d = env.sim.data
    body_ids = []
    for w in range(n_env):
        s = int(slot[w])
        if swap is not None:
            ids = cmd._swap_geom_ids[: int(swap["nhull"][s])].tolist()
            bid = cmd._swap_body_id
        else:
            ent = env.scene[cmd.obj_entity_names("right")[s]]
            bid = int(ent.indexing.body_ids[0])
            named = {mjm.geom(int(g.id)).name.rsplit("_col_", 1)[-1]: int(g.id) for g in ent.indexing.geoms if g.contype}
            ids = [named[str(k)] for k in range(len(named))]
        gx[w, : len(ids)] = d.geom_xpos[w, ids].cpu().numpy()
        gm[w, : len(ids)] = d.geom_xmat[w, ids].reshape(len(ids), 9).cpu().numpy()
        body_ids.append(bid)
    bw = torch.tensor(body_ids, device=dev)
    com = d.xipos[W, bw].cpu().numpy()
    def per_world(x):                              # a model field's rows, whether or not it was expanded per world
        return x[W if x.shape[0] == n_env else torch.zeros_like(W), bw].cpu().numpy()
    mass, inertia = per_world(m.body_mass), per_world(m.body_inertia)
    ximat = d.ximat[W, bw].reshape(n_env, 9).cpu().numpy()
    binvw = per_world(m.body_invweight0)
    dofs = torch.tensor([int(mjm.body_dofadr[int(b_)]) for b_ in body_ids], device=dev)[:, None] + torch.arange(6, device=dev)
    dw = m.dof_invweight0
    dinvw = dw[(W if dw.shape[0] == n_env else torch.zeros_like(W))[:, None], dofs].cpu().numpy()

    # ---- rollout
    T = min(a.steps, int(lib._motion_num_frames.max()))
    act = torch.zeros(n_env, sum(env.action_manager.action_term_dim), device=dev)
    pos_err = np.zeros((T, n_env), np.float32)
    rot_err = np.zeros((T, n_env), np.float32)
    sim_pos = np.zeros((T, n_env, 3), np.float32)
    ref_pos = np.zeros((T, n_env, 3), np.float32)
    done = np.zeros((T, n_env), bool)
    ncon = np.zeros(T, np.int32)
    tm = env.termination_manager
    terms = list(tm.active_terms)
    first_term = np.full(n_env, "", dtype=object)          # the term(s) behind each env's first done
    torch.cuda.synchronize()
    t1 = time.time()
    for t in range(T):
        _, _, term, trunc, _ = env.step(act)
        fired = {k: tm.get_term(k).cpu().numpy() for k in terms}
        for e in np.nonzero((term | trunc).cpu().numpy() & (first_term == ""))[0]:
            first_term[e] = "+".join(k for k in terms if fired[k][e]) or "?"
        sp, rp = cmd.sim_obj_trans_w[:, 0], cmd.ref_obj_trans_w[:, 0]
        pos_err[t] = (sp - rp).norm(dim=-1).cpu().numpy()
        rot_err[t] = _quat_angle(cmd.sim_obj_quat_w[:, 0], cmd.ref_obj_quat_w[:, 0]).cpu().numpy()
        sim_pos[t] = (sp - env.scene.env_origins).cpu().numpy()
        ref_pos[t] = (rp - env.scene.env_origins).cpu().numpy()
        done[t] = (term | trunc).cpu().numpy()
        ncon[t] = int(torch.as_tensor(env.sim.data.nacon).reshape(-1)[0])
    torch.cuda.synchronize()
    dt = time.time() - t1
    n_frames = lib._motion_num_frames[clip].cpu().numpy()
    print(f"[swap_check] {T} steps in {dt:.0f} s ({1000 * dt / T:.0f} ms/step); contacts per step {ncon.mean():.0f}",
          flush=True)
    np.savez(a.out, names=np.array(names), clip=clip.cpu().numpy(), n_frames=n_frames, pos_err=pos_err,
             rot_err=rot_err, sim_pos=sim_pos, ref_pos=ref_pos, done=done, ncon=ncon, problems=problems,
             first_term=first_term.astype(str), geom_xpos0=gx, geom_xmat0=gm, com0=com, mass0=mass, inertia0=inertia,
             ximat0=ximat, binvw0=binvw, dinvw0=dinvw,
             mode="swap" if swap is not None else "multi", inject=not os.environ.get("SWAP_NO_INJECT"))
    print(f"[swap_check] wrote {a.out}", flush=True)


def widen(a):
    """A copy of a motion pack whose object list repeats every object `copies` times, with each clip pointing at its
    object's last copy: the scene then holds copies x n_clip objects' meshes, as a large dataset would, while the
    clips stay the same. For timing the build and the step, and for checking far-away mesh ids."""
    import torch

    src = Path(a.motion).resolve()
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    d = torch.load(src, map_location="cpu", weights_only=False)
    n = len(d["right_object_mesh_dirs"])
    for k in ("right_object_mesh_dirs", "right_object_mesh_scales", "right_object_masses"):
        d[k] = list(d[k]) * a.copies
    d["motion_object_slot_right"] = d["motion_object_slot_right"] + n * (a.copies - 1)
    torch.save(d, out / src.name)
    if not (out / "objects").exists():
        (out / "objects").symlink_to(src.parent / "objects")
    print(f"[swap_check] {out / src.name}: {n * a.copies} objects, clips on slots "
          f"{d['motion_object_slot_right'].tolist()}")


def subset(a):
    """A copy of a motion pack with only the clips `--clips` (indices) and their objects: a one-clip pack is the
    single-object scene, the reference a swap-mode world should reproduce."""
    import torch

    src = Path(a.motion).resolve()
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    d = torch.load(src, map_location="cpu", weights_only=False)
    keep = [int(c) for c in a.clips.split(",")]
    nf, ls = d["motion_num_frames"], d["length_starts"]
    n_clip, n_frame = len(nf), int(nf.sum())
    frames = torch.cat([torch.arange(int(ls[c]), int(ls[c] + nf[c])) for c in keep])
    slots = d["motion_object_slot_right"][keep]
    objs = sorted(set(slots.tolist()))
    new = {}
    for k, v in d.items():
        if k.startswith("right_object_"):
            new[k] = [v[s] for s in objs]
        elif torch.is_tensor(v) and v.shape[:1] == (n_frame,) and n_frame != n_clip:
            new[k] = v[frames]
        elif (torch.is_tensor(v) or isinstance(v, list)) and len(v) == n_clip and k != "mano_right_joint_names":
            new[k] = v[keep] if torch.is_tensor(v) else [v[c] for c in keep]
        else:
            new[k] = v
    new["motion_object_slot_right"] = torch.tensor([objs.index(int(s)) for s in slots], dtype=slots.dtype)
    new["length_starts"] = torch.cumsum(torch.cat([torch.zeros(1, dtype=nf.dtype), new["motion_num_frames"][:-1]]), 0)
    torch.save(new, out / src.name)
    if not (out / "objects").exists():
        (out / "objects").symlink_to(src.parent / "objects")
    print(f"[swap_check] {out / src.name}: clips {[d['motion_filename'][c] for c in keep]}, objects {objs}")


def _alive(done):
    """True up to and including each env's first done step."""
    first = np.where(done.any(0), done.argmax(0), done.shape[0] - 1)
    return np.arange(done.shape[0])[:, None] <= first[None, :], first


def compare(paths):
    runs = [dict(np.load(p, allow_pickle=False)) for p in paths]
    names = list(runs[0]["names"])
    lab = [f"{str(r['mode'])}{'' if bool(r['inject']) else '-noinject'}" for r in runs]
    print("per clip: steps before the first termination (of the clip's length), mean / max object position error (cm), "
          "mean rotation error (deg), mean contacts per step over all envs")
    for i, r in enumerate(runs):
        print(f"  {lab[i]}: contacts {r['ncon'].mean():.0f}, row problems {int(r['problems'])}")
    hdr = f"{'clip':<16}" + "".join(f"{x:>30}" for x in lab)
    print(hdr)
    for c, nm in enumerate(names):
        row = f"{nm:<16}"
        for r in runs:
            al, first = _alive(r["done"])
            sel = r["clip"] == c
            pe = np.where(al[:, sel], r["pos_err"][:, sel], np.nan)
            re = np.where(al[:, sel], r["rot_err"][:, sel], np.nan)
            row += (f"{int(np.mean(first[sel] + 1)):>6}/{int(r['n_frames'][sel][0]):<4} {100 * np.nanmean(pe):5.1f}/"
                    f"{100 * np.nanmax(pe):5.1f} cm {np.degrees(np.nanmean(re)):5.0f}°")
        print(row)
    if all("first_term" in r for r in runs):
        print("\nwhat ended each env's first episode:")
        for c, nm in enumerate(names):
            print(f"{nm:<16}" + "".join(f"  {lab[i]}: " + ", ".join(sorted(set(r['first_term'][r['clip'] == c])))
                                          for i, r in enumerate(runs)))
    base = runs[0]
    if all("geom_xpos0" in r for r in runs):
        print(f"\nright after the reset, against {lab[0]}: max difference of the parts' world positions (mm) and "
              f"rotations, the CoM (mm), mass (g) and principal inertia (relative)")
        for i, r in enumerate(runs[1:], 1):
            for c, nm in enumerate(names):
                e = np.nonzero(base["clip"] == c)[0][0]
                f = np.nonzero(r["clip"] == c)[0][0]
                dp = np.nanmax(np.abs(base["geom_xpos0"][e] - r["geom_xpos0"][f]))
                dr = np.nanmax(np.abs(base["geom_xmat0"][e] - r["geom_xmat0"][f]))
                dc = np.abs(base["com0"][e] - r["com0"][f]).max()
                dm = abs(float(base["mass0"][e]) - float(r["mass0"][f]))
                di = (np.abs(base["inertia0"][e] - r["inertia0"][f]) / np.abs(base["inertia0"][e]).max()).max()
                extra = ""
                if "ximat0" in base and "ximat0" in r:
                    rel = lambda x, y: float((np.abs(x - y) / max(np.abs(x).max(), 1e-12)).max())
                    extra = (f"   inertia frame {np.abs(base['ximat0'][e] - r['ximat0'][f]).max():.1e}   invweight0 body "
                             f"{rel(base['binvw0'][e], r['binvw0'][f]):.1e} dof {rel(base['dinvw0'][e], r['dinvw0'][f]):.1e}")
                print(f"  {lab[i]:<14} {nm:<26} parts {1000 * dp:7.3f} mm {dr:.1e}   CoM {1000 * dc:7.3f} mm   "
                      f"mass {1000 * dm:6.2f} g   inertia {di:.1e}" + extra)
    print(f"\nobject path distance from {lab[0]} (same env, both alive): mean over the first 30 / 120 / all steps (cm)")
    for i, r in enumerate(runs[1:], 1):
        a0, _ = _alive(base["done"])
        a1, _ = _alive(r["done"])
        T = min(len(a0), len(a1))
        both = a0[:T] & a1[:T]
        d = np.linalg.norm(base["sim_pos"][:T] - r["sim_pos"][:T], axis=-1)
        d = np.where(both, d, np.nan)
        out = []
        for n in (30, 120, T):
            out.append(f"{100 * np.nanmean(d[:n]):.2f}")
        print(f"  {lab[i]}: " + " / ".join(out))
        for c, nm in enumerate(names):
            sel = base["clip"] == c
            print(f"    {nm:<16} " + " / ".join(f"{100 * np.nanmean(d[:n, sel]):.2f}" for n in (30, 120, T)))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--motion", required=True)
    r.add_argument("--per-clip", type=int, default=2)
    r.add_argument("--steps", type=int, default=600)
    r.add_argument("--trials", default=None)
    r.add_argument("--out", required=True)
    r.add_argument("overrides", nargs="*")
    c = sub.add_parser("compare")
    c.add_argument("paths", nargs="+")
    w = sub.add_parser("widen")
    w.add_argument("--motion", required=True)
    w.add_argument("--copies", type=int, default=50)
    w.add_argument("--out-dir", required=True)
    s = sub.add_parser("subset")
    s.add_argument("--motion", required=True)
    s.add_argument("--clips", required=True, help="comma-separated clip indices")
    s.add_argument("--out-dir", required=True)
    a = ap.parse_args()
    {"run": run, "widen": widen, "subset": subset, "compare": lambda a: compare(a.paths)}[a.cmd](a)
