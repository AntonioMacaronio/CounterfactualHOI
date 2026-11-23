"""Learning curves of a FlashDexRetarget / §1.4 policy run from its per-eval rollout archives.

  python scripts/flashdex/eval_curves.py RUN [--dataset DATA]      (flashdex env; RUN = $FDR/logs/flashsac/<run>)

For every RUN/evals/step<N>/ (train_cfhoi.py), scores each clip's rollout against its reference (metrics.py:
AdaDex progress and success, object errors) and writes RUN/curves.json: {step: {clip: numbers}} plus means over
TrackEnv clips (the originals) and GenEnv clips (cfhoi_gen*), and RUN/curves.png (progress and object error vs
env steps, with FDR's logged eval success rate from metrics.jsonl).
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import metrics as MT  # noqa: E402


def score_archive(arch, pt, names):
    man = json.load(open(arch / "manifest.json"))
    out = {}
    for e in man["saved"]:
        m = int(e["motion_id"])
        a0, n = int(pt["length_starts"][m]), int(pt["motion_num_frames"][m])
        z = np.load(arch / e["rollout"])
        op, oq = z["obj_pos"], z["obj_quat"]
        if op.ndim == 3:
            op, oq = op[:, 0], oq[:, 0]
        T = min(len(op), n)
        r, _ = MT.clip_metrics(op[:T], oq[:T], pt["obj_right_pos"][a0:a0 + T].numpy(),
                               pt["obj_right_rotmat"][a0:a0 + T].numpy(), z["joint_pos"][:T],
                               pt["joint_pos"][a0:a0 + T].numpy())
        out[names[m]] = r
    return out


def summarize(per_clip):
    is_gen = lambda k: k.startswith("gen") or "_gen" in k  # noqa: E731  (cfhoi_gen<k>_<vid>, prefix stripped)
    groups = {"track": [v for k, v in per_clip.items() if not is_gen(k)],
              "gen": [v for k, v in per_clip.items() if is_gen(k)]}
    out = {}
    for g, vs in groups.items():
        if vs:
            out[g] = {"n": len(vs), "progress": float(np.mean([v["progress"] for v in vs])),
                      "success": float(np.mean([v["success"] for v in vs])),
                      "fdr_success": float(np.mean([v["fdr_success"] for v in vs])),
                      "obj_pos_err_cm_mean": float(np.mean([v["obj_pos_err_cm_mean"] for v in vs])),
                      "obj_rot_err_deg_mean": float(np.mean([v["obj_rot_err_deg_mean"] for v in vs])),
                      "final_goal_success": float(np.mean([v["final_obj_pos_err_cm"] < 5 and v["final_obj_rot_err_deg"] < 20
                                                           for v in vs]))}
    return out


def plot(curves, run, logged):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    steps = sorted(curves, key=int)
    x = [int(s) / 1e6 for s in steps]
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8), facecolor="#fcfcfb")
    colors = {"track": "#2a78d6", "gen": "#eb6834"}                    # categorical slots 1, 2
    labels = {"track": "TrackEnv clips (the 10 originals)", "gen": "GenEnv clips (new start / goal)"}
    for g in ("track", "gen"):
        ys = [curves[s]["summary"].get(g, {}).get("progress") for s in steps]
        if any(y is not None for y in ys):
            ax[0].plot(x, [np.nan if y is None else 100 * y for y in ys], "-o", ms=4, lw=2, color=colors[g], label=labels[g])
            e = [curves[s]["summary"].get(g, {}).get("obj_pos_err_cm_mean") for s in steps]
            ax[1].plot(x, [np.nan if y is None else y for y in e], "-o", ms=4, lw=2, color=colors[g], label=labels[g])
    if logged:
        lx = [s / 1e6 for s, _ in logged]
        ax[0].plot(lx, [100 * v for _, v in logged], "--", color="#52514e", lw=1.5,
                   label="FDR eval SR@1.0 (obj, all clips)")
    ax[0].set(xlabel="env steps (M)", ylabel="%", title="AdaDex progress (mean over clips)", ylim=(0, 102))
    ax[1].set(xlabel="env steps (M)", ylabel="cm", title="object position error (time mean)")
    for a in ax:
        a.set_facecolor("#fcfcfb")
        a.grid(alpha=0.25, lw=0.6)
        a.spines[["top", "right"]].set_visible(False)
    for a in ax:
        a.legend(frameon=False, fontsize=8, loc="best")
    fig.suptitle(run.name, fontsize=10)
    fig.tight_layout()
    fig.savefig(run / "curves.png", dpi=130)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--dataset", default=None, help="default: the motion file in RUN/config.yaml")
    ap.add_argument("--rescore", action="store_true", help="ignore RUN/curves.json and score every eval again")
    args = ap.parse_args()
    run = Path(args.run)
    motion = Path(args.dataset) / "motion.pt" if args.dataset else None
    if motion is None:
        m = re.search(r"motion_file:\s*(\S+)", (run / "config.yaml").read_text())
        motion = Path(m.group(1))
    pt = torch.load(motion, map_location="cpu", weights_only=False)
    names = [str(x).split("#")[-1].removeprefix("cfhoi_") for x in pt["motion_filename"]]
    cache = run / "curves.json"
    curves = json.load(open(cache)) if cache.exists() and not args.rescore else {}
    for arch in sorted((run / "evals").glob("step*"), key=lambda p: int(p.name[4:])):
        s = arch.name[4:]
        if s in curves or not (arch / "manifest.json").exists():
            continue
        per = score_archive(arch, pt, names)
        curves[s] = {"summary": summarize(per), "clips": per}
        sm = curves[s]["summary"]
        print(f"step {int(s) / 1e6:.0f}M: " + ", ".join(f"{g} progress {v['progress']:.2f} success {v['success']:.2f} "
                                                      f"obj {v['obj_pos_err_cm_mean']:.1f} cm" for g, v in sm.items()))
    json.dump(curves, open(cache, "w"), indent=1)
    logged = []
    if (run / "metrics.jsonl").exists():
        for line in open(run / "metrics.jsonl"):
            d = json.loads(line)
            if "eval/success_rate_obj_1.0" in d:
                logged.append((d["step"], d["eval/success_rate_obj_1.0"]))
    if curves:
        plot(curves, run, logged)


if __name__ == "__main__":
    main()
