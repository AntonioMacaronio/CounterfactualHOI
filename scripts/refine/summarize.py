"""Initial vs refined metrics for every refined clip -> runs/refine/VID/summary.json and a markdown table.

  python scripts/refine/summarize.py [VID ...]      (fpose env; default: every runs/refine/<VID> with a final.npz;
                                                    REFINE_TAG=r2 reads runs/refine_r2 instead)

Per clip: medians over all frames, and over the in-hand frames (phases grasp/lift/carry/rotate/place), of the
metrics in clip.py (object mask IoU / coverage / spill outside the hand, depth residual, hand keypoint error, hand
depth bias, penetration, rest height and its spread across rest phases), the number of flagged frames, the agent's
edit count and the verifier verdicts (from report.md / transcript.jsonl). Also qc.jpg: initial vs refined overlays
and a side view, one row per phase.
"""

import glob
import json
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import render as R  # noqa: E402
from clip import Clip, flag, jump_threshold, load_state, phase_of  # noqa: E402

STORAGE = os.environ.get("PROJECT_STORAGE_ROOT", "/data/sky/projects/CounterfactualHOI")
CHOIR_ROOT = os.environ.get("CHOIR_ROOT", os.path.join(STORAGE, "build", "CHOIR"))
KEYS = ["iou", "coverage", "spill", "depth_res_cm", "hand_kp_px", "hand_depth_bias_cm", "penetration_cm", "contact_cm", "n_cross",
        "hand_accel_mm", "hand_rel_accel_mm", "tip_accel_mm", "finger_accel_deg", "obj_accel_cm", "obj_ang_accel_deg"]
HELD = ("grasp", "lift", "carry", "rotate", "place")


def med(rows, k):
    x = [r[k] for r in rows if r.get(k) is not None]
    return float(np.median(x)) if x else None


def qc_sheet(clip, s0, s1, path, n=6):
    """Per phase (middle frame, up to n phases): initial and refined overlay, initial and refined in 3D from behind
    (red = hand through the object, green = touching), refined from the camera's left."""
    ph = s1.phases
    if len(ph) > n:
        ph = [ph[k] for k in np.linspace(0, len(ph) - 1, n).round().astype(int)]
    ims = []
    for p in ph:
        i = (p["start"] + p["end"]) // 2
        ims += [R.overlay(clip, s0, i, label=f"{i} {p['label']}: initial"), R.overlay(clip, s1, i, label=f"{i} {p['label']}: refined"),
                R.view_orbit(clip, s0, i, 90, label=f"{i} initial, from behind"), R.view_orbit(clip, s1, i, 90, label=f"{i} refined, from behind"),
                R.view_orbit(clip, s1, i, 180, label=f"{i} refined, from the left")]
    cv2.imwrite(path, R.sheet(ims, ncols=5, tile_w=420), [cv2.IMWRITE_JPEG_QUALITY, 85])


def main():
    out_root = os.path.join(STORAGE, "runs", "refine" + ("_" + os.environ["REFINE_TAG"] if os.environ.get("REFINE_TAG") else ""))
    vids = sys.argv[1:] or sorted(os.path.basename(os.path.dirname(p)) for p in glob.glob(os.path.join(out_root, "*", "final.npz"))
                                  if "." not in os.path.basename(os.path.dirname(p)))        # skip kept tries (VID.try1)
    table = []
    for vid in vids:
        o = os.path.join(out_root, vid)
        clip = Clip(os.path.join(CHOIR_ROOT, "output", vid), os.path.join(STORAGE, "runs", "fprecon", vid))
        phases = json.load(open(os.path.join(o, "phases.json")))
        s0 = clip.init_state.copy()
        s0.phases = phases
        s1 = load_state(os.path.join(o, "final.npz"), phases)
        m0, m1 = clip.metrics(s0), clip.metrics(s1)
        held = [r["frame"] for r in m0 if phase_of(s1, r["frame"]) in HELD]
        rest = [r["frame"] for r in m0 if phase_of(s1, r["frame"]) in ("approach", "release", "rest")]
        summ = {"video_id": vid, "frames": clip.N, "phases": phases, "hand_jitter_ref_mm": clip.jitter_ref}
        for name, rows0, rows1 in (("all", m0, m1),
                                   ("held", [m0[i] for i in held], [m1[i] for i in held])):
            summ[name] = {k: [med(rows0, k), med(rows1, k)] for k in KEYS}
        summ["rest_abs_height_cm"] = [med([{"h": abs(m0[i]["obj_height_cm"])} for i in rest], "h"),
                                      med([{"h": abs(m1[i]["obj_height_cm"])} for i in rest], "h")]
        # the simulated table sits at the object's resting height (agent.export), so what matters is whether the
        # object rests at the same height in every rest phase: spread of the per-phase median heights
        rp = [p for p in phases if p["label"] in ("approach", "release", "rest")]
        spread = lambda rows: (lambda h: float(max(h) - min(h)) if len(h) > 1 else 0.0)(
            [float(np.median([rows[i]["obj_height_cm"] for i in range(p["start"], p["end"] + 1)])) for p in rp])
        summ["rest_height_spread_cm"] = [spread(m0), spread(m1)]
        lifted = [r["frame"] for r in m0 if phase_of(s1, r["frame"]) in ("lift", "carry", "rotate")]
        summ["grip_lifted"] = [float(np.mean([m0[i]["opposed"] for i in lifted])) if lifted else None,
                               float(np.mean([m1[i]["opposed"] for i in lifted])) if lifted else None]
        cf = os.path.join(o, "critic_final.json")
        if os.path.exists(cf):
            rv = json.load(open(cf))
            summ["reviewer"] = {"overall": rv["overall"], "problems": rv["problems"], "votes": rv.get("votes", [rv["overall"]]),
                                **{f"n_{sev}": sum(p["severity"] == sev for p in rv["problems"]) for sev in ("high", "medium", "low")}}
        summ["flagged_frames"] = [sum(bool(flag(r, phase_of(s1, r["frame"]))) for r in m0),
                                  sum(bool(flag(r, phase_of(s1, r["frame"]))) for r in m1)]
        # jumps: medians hide them (a seam is 2-3 frames), so count the frames and keep the worst
        thr = jump_threshold(clip.tip_ref)
        tips = [[r["tip_accel_mm"] for r in m if r.get("tip_accel_mm") is not None] for m in (m0, m1)]
        summ["jump_threshold_mm"] = thr
        summ["jump_frames"] = [int(sum(a > thr for a in t)) for t in tips]
        summ["tip_accel_max_mm"] = [float(max(t)) if t else None for t in tips]
        verdicts, n_edits, minutes = [], None, None
        for line in open(os.path.join(o, "transcript.jsonl")):
            r = json.loads(line)
            note = r.get("note", "")
            if note.startswith("verify "):
                verdicts.append(note)
            if note.startswith("agent finished at v"):
                n_edits = int(note.split("agent finished at v")[1].split()[0])
                minutes = float(note.split(" after ")[1].split(" min")[0])
        summ.update(n_versions=n_edits, agent_minutes=minutes, verifier=verdicts,
                    reverted=sum("keep False" in v for v in verdicts))
        json.dump(summ, open(os.path.join(o, "summary.json"), "w"), indent=1)
        qc_sheet(clip, s0, s1, os.path.join(o, "qc.jpg"))
        table.append(summ)
        print(f"{vid}: IoU {summ['all']['iou'][0]:.3f} -> {summ['all']['iou'][1]:.3f}, hand keypoints "
              f"{summ['all']['hand_kp_px'][0]:.1f} -> {summ['all']['hand_kp_px'][1]:.1f} px, flagged frames "
              f"{summ['flagged_frames'][0]} -> {summ['flagged_frames'][1]}, verifier reverted {summ['reverted']}/{len(verdicts)}", flush=True)
    rows = ["| clip | object IoU (in hand) | object IoU (all) | hand keypoint px | hand depth bias cm | penetration cm | "
            "hand triangles through the object (held) | hand jitter mm/f² (held) | finger jitter °/f² (held) | object rotation jitter °/f² | "
            "grip while lifted | rest height cm | rest height spread cm | flagged frames | verifier reverted | reviewer | "
            "finger-jump frames | worst fingertip accel mm/f² |", "|" + "---|" * 18]
    f = lambda p, d=2: f"{p[0]:.{d}f} → {p[1]:.{d}f}" if None not in p else "—"
    for t in table:
        rows.append(f"| {t['video_id']} | {f(t['held']['iou'])} | {f(t['all']['iou'])} | {f(t['all']['hand_kp_px'], 1)} | "
                    f"{f(t['all']['hand_depth_bias_cm'])} | {f(t['all']['penetration_cm'])} | {f(t['held']['n_cross'], 0)} | {f(t['held']['hand_accel_mm'], 1)} | "
                    f"{f(t['held']['finger_accel_deg'], 1)} | {f(t['all']['obj_ang_accel_deg'], 1)} | "
                    f"{f(t['grip_lifted']) if None not in t['grip_lifted'] else '—'} | {f(t['rest_abs_height_cm'])} | {f(t['rest_height_spread_cm'])} | "
                    f"{t['flagged_frames'][0]} → {t['flagged_frames'][1]} | {t['reverted']}/{len(t['verifier'])} | "
                    + (f"{t['reviewer']['overall']} ({t['reviewer']['n_high']} high, {t['reviewer']['n_medium']} medium)" if t.get("reviewer") else "—")
                    + f" | {t['jump_frames'][0]} → {t['jump_frames'][1]} | {f(t['tip_accel_max_mm'], 0)} |")
    open(os.path.join(out_root, "summary.md"), "w").write("\n".join(rows) + "\n")
    print("\n".join(rows))


if __name__ == "__main__":
    main()
