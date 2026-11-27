"""Tasks and trials (antzhang-notes/project.md §1.3): one video -> one task (its agent-refined reconstruction, a clip
of a FlashDexRetarget dataset); a task gets N trials, physics variants of its object that are fixed per trial and
stored, so the §1.4 policy can train on the same trials. Robot randomization is drawn again every episode.

  python scripts/mjwarp/trials.py make DATASET [--n 5] [--seed 0]          -> DATASET/trials.json   (any python3 + numpy)
  python scripts/mjwarp/trials.py show DATASET

Per trial (object, fixed):
  mass           the VLM's estimate (scripts/physprops, DATASET's packed right_object_masses) x U(0.7, 1.3); without an
                 estimate, the density mass (FDR's 800 kg/m^3) x the same factor. Inertia scales with it (uniform density).
  center of mass offset from the mesh's, uniform per axis within +-10% of the object's extent there, at most 1.5 cm
  sliding friction  U(VLM's low, high) for the object against the table (default 0.3-0.8)
  torsional friction  U(0.001, 0.02) m,  rolling friction  U(1e-5, 1e-3) m      (MuJoCo's units; object geoms get condim 6)
  restitution    U(0, min(0.5, VLM's restitution + 0.15)), applied as the contact's damping ratio (below)
Per episode (robot, drawn at every reset; scripts/mjwarp/randomize.py):
  PD gains kp, kd x U(0.8, 1.2); joint damping x U(0.7, 1.3); armature x U(0.8, 1.2); joint friction (frictionloss)
  x U(0.5, 1.5); hand friction U(0.8, 1.5); action delay 0, 1 or 2 control steps; action noise N(0, 0.02) per dim.

Restitution in MuJoCo: a contact is a spring-damper with solref = (timeconst, dampratio); a mass-spring-damper
bounces back with e = exp(-zeta pi / sqrt(1 - zeta^2)), so zeta = -ln(e) / sqrt(pi^2 + ln(e)^2) (e = 0 -> zeta = 1,
no bounce). The object's geoms get a large solmix so their solref decides the contact's.
"""

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np

STORAGE = Path(os.environ.get("PROJECT_STORAGE_ROOT", "/data/sky/projects/CounterfactualHOI"))
RANGES = {
    "mass_scale": [0.7, 1.3],
    "com_offset_frac": 0.10, "com_offset_max_m": 0.015,
    "friction_default": [0.3, 0.8],
    "torsional": [0.001, 0.02], "rolling": [1e-5, 1e-3],
    "restitution_max": 0.5, "restitution_margin": 0.15, "restitution_default": 0.2,
    "contact_timeconst": 0.02,
    # per episode, robot
    "kp_scale": [0.8, 1.2], "kd_scale": [0.8, 1.2], "damping_scale": [0.7, 1.3], "armature_scale": [0.8, 1.2],
    "frictionloss_scale": [0.5, 1.5], "hand_friction": [0.8, 1.5], "action_delay_steps": [0, 2], "action_noise": 0.02,
}


def damping_ratio(e):
    """Contact damping ratio that gives a coefficient of restitution e (0 -> 1, critically damped)."""
    e = float(np.clip(e, 1e-4, 0.95))
    return -math.log(e) / math.sqrt(math.pi ** 2 + math.log(e) ** 2) if e > 1e-3 else 1.0


def extents(obj_dir):
    V = np.array([[float(t) for t in l.split()[1:4]] for l in open(Path(obj_dir) / "visual.obj") if l.startswith("v ")])
    return np.ptp(V, axis=0)


def make(dataset, n=5, seed=0):
    import torch

    D = Path(dataset)
    pack = D / "motion_bimanual.pt" if (D / "motion_bimanual.pt").exists() else D / "motion.pt"
    m = torch.load(pack, map_location="cpu", weights_only=False)
    names = list(m["motion_filename"])
    dirs = list(m["right_object_mesh_dirs"])
    slots = m.get("motion_object_slot_right")
    masses = list(m.get("right_object_masses") or [0.0] * len(dirs))
    rng = np.random.default_rng(seed)
    tasks = {}
    for i, name in enumerate(names):
        slot = int(slots[i]) if slots is not None else i
        vid = name.replace("cfhoi_", "")
        pp = STORAGE / "runs" / "physprops" / vid / "physprops.json"
        est = json.load(open(pp))["answer"] if pp.exists() else None
        ext = extents(D / dirs[slot])
        f_lo, f_hi = (sorted([float(est["friction_low"]), float(est["friction_high"])]) if est else RANGES["friction_default"])
        e_hi = min(RANGES["restitution_max"], (float(est["restitution"]) if est else RANGES["restitution_default"])
                   + RANGES["restitution_margin"])
        base = float(masses[slot]) if slot < len(masses) and masses[slot] > 0 else None
        trials = []
        for _ in range(n):
            s = float(rng.uniform(*RANGES["mass_scale"]))
            lim = np.minimum(RANGES["com_offset_frac"] * ext, RANGES["com_offset_max_m"])
            e = float(rng.uniform(0.0, e_hi))
            trials.append({
                "mass_scale": s, "mass_kg": None if base is None else base * s,
                "com_offset_m": [float(x) for x in rng.uniform(-lim, lim)],
                "friction": [float(rng.uniform(f_lo, f_hi)), float(rng.uniform(*RANGES["torsional"])),
                             float(np.exp(rng.uniform(*np.log(RANGES["rolling"]))))],
                "restitution": e,
                "solref": [RANGES["contact_timeconst"], damping_ratio(e)],
            })
        tasks[name] = {"slot": slot, "object_dir": str(dirs[slot]), "extent_m": [float(x) for x in ext],
                       "mass_kg_base": base, "mass_source": "vlm" if base is not None else "density (800 kg/m^3)",
                       "friction_range": [f_lo, f_hi], "trials": trials}
    out = {"dataset": str(D), "pack": pack.name, "n_trials": n, "seed": seed, "ranges": RANGES, "tasks": tasks}
    json.dump(out, open(D / "trials.json", "w"), indent=1)
    return out


def show(dataset):
    t = json.load(open(Path(dataset) / "trials.json"))
    print(f"{len(t['tasks'])} tasks x {t['n_trials']} trials (seed {t['seed']})")
    for name, task in t["tasks"].items():
        ms = [x["mass_kg"] * 1000 if x["mass_kg"] else float("nan") for x in task["trials"]]
        base = 1000 * task["mass_kg_base"] if task["mass_kg_base"] else float("nan")
        fr = ", ".join("%.2f" % x["friction"][0] for x in task["trials"])
        re_ = ", ".join("%.2f" % x["restitution"] for x in task["trials"])
        print(f"  {name}: mass {task['mass_source']}, base {base:.0f} g -> {', '.join('%.0f' % x for x in ms)} g; "
              f"friction {fr}; restitution {re_}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["make", "show"])
    ap.add_argument("dataset")
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    if a.cmd == "make":
        make(a.dataset, a.n, a.seed)
    show(a.dataset)
