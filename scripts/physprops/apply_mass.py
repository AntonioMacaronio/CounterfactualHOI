"""Write the VLM's mass estimates (scripts/physprops/estimate.py) into a FlashDexRetarget dataset.

  python -I scripts/physprops/apply_mass.py DATASET [--pack motion_bimanual.pt]      (flashdex or fdr-mjwarp env)

- DATASET/human_demo/<clip>.npz gets obj_right_mass (kg), the field FDR's pack step (build_motion_pt.py) reads, so
  a later re-pack keeps it.
- The packed motion files (motion.pt, motion_bimanual.pt, or --pack) get right_object_masses per object slot: FDR's
  Isaac backend uses a given mass instead of its density estimate (simulator/isaacsim/scene.py, "measured mass
  wins"), and scripts/mjwarp/trials.py samples the trials around it.
Clips without an estimate keep FDR's density estimate (mass 0 = none). Prints one line per clip.
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

STORAGE = Path(os.environ.get("PROJECT_STORAGE_ROOT", "/data/sky/projects/CounterfactualHOI"))


def estimate(vid):
    p = STORAGE / "runs" / "physprops" / vid / "physprops.json"
    return json.load(open(p))["answer"] if p.exists() else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("--pack", nargs="*", default=None, help="packed motion files in DATASET (default: every motion*.pt)")
    a = ap.parse_args()
    D = Path(a.dataset)
    masses = {}
    for f in sorted((D / "human_demo").glob("*.npz")):
        vid = f.stem.replace("cfhoi_", "")
        est = estimate(vid)
        if est is None:
            print(f"{f.stem}: no estimate, FDR's density estimate stays")
            continue
        clip = dict(np.load(f, allow_pickle=True))
        clip["obj_right_mass"] = np.float32(est["mass_g"] / 1000.0)
        np.savez(f, **clip)
        masses[f.stem] = float(est["mass_g"]) / 1000.0
        print(f"{f.stem}: {est['mass_g']:.0f} g ({est['mass_low_g']:.0f}-{est['mass_high_g']:.0f}, spread "
              f"{est.get('mass_spread', float('nan')):.2f})")
    packs = [D / p for p in a.pack] if a.pack else sorted(D.glob("motion*.pt"))
    for p in packs:
        m = torch.load(p, map_location="cpu", weights_only=False)
        names, slots = list(m["motion_filename"]), m.get("motion_object_slot_right")
        n_slots = len(m.get("right_object_mesh_dirs") or [])
        if not n_slots:
            print(f"{p.name}: no right-hand object slots, skipped")
            continue
        out = [0.0] * n_slots
        for i, name in enumerate(names):
            s = int(slots[i]) if slots is not None else i
            if name in masses:
                out[s] = masses[name]
        m["right_object_masses"] = out
        torch.save(m, p)
        print(f"{p.name}: right_object_masses = {[round(x * 1000) for x in out]} g")


if __name__ == "__main__":
    main()
