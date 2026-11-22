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
from clip import Clip, State, flag, phase_of  # noqa: E402

STORAGE = os.environ.get("PROJECT_STORAGE_ROOT", "/data/sky/projects/CounterfactualHOI")
CHOIR_ROOT = os.environ.get("CHOIR_ROOT", os.path.join(STORAGE, "build", "CHOIR"))
KEYS = ["iou", "coverage", "spill", "depth_res_cm", "hand_kp_px", "hand_depth_bias_cm", "penetration_cm", "contact_cm"]
HELD = ("grasp", "lift", "carry", "rotate", "place")


def med(rows, k):
    x = [r[k] for r in rows if r.get(k) is not None]
    return float(np.median(x)) if x else None


def qc_sheet(clip, s0, s1, path, n=6):
    """Initial vs refined overlay and the refined side view, at the middle frame of up to n phases."""
    ph = s1.phases
    if len(ph) > n:
        ph = [ph[k] for k in np.linspace(0, len(ph) - 1, n).round().astype(int)]
    ims = []
    for p in ph:
        i = (p["start"] + p["end"]) // 2
        ims += [R.overlay(clip, s0, i, label=f"{i} {p['label']}: initial"), R.overlay(clip, s1, i, label=f"{i} {p['label']}: refined"),
                R.view(clip, s1, i, "side", label=f"{i} refined, side view")]
    cv2.imwrite(path, R.sheet(ims, ncols=3, tile_w=520), [cv2.IMWRITE_JPEG_QUALITY, 85])


def main():
    out_root = os.path.join(STORAGE, "runs", "refine" + ("_" + os.environ["REFINE_TAG"] if os.environ.get("REFINE_TAG") else ""))
    vids = sys.argv[1:] or sorted(os.path.basename(os.path.dirname(p)) for p in glob.glob(os.path.join(out_root, "*", "final.npz"))
                                  if "." not in os.path.basename(os.path.dirname(p)))        # skip kept tries (VID.try1)
    table = []
    for vid in vids:
        o = os.path.join(out_root, vid)
        clip = Clip(os.path.join(CHOIR_ROOT, "output", vid), os.path.join(STORAGE, "runs", "fprecon", vid))
        z = np.load(os.path.join(o, "final.npz"))
        phases = json.load(open(os.path.join(o, "phases.json")))
        s0 = clip.init_state.copy()
        s0.phases = phases
        s1 = State(z["obj"], z["hand_off"], z["hand_rot"], phases)
        m0, m1 = clip.metrics(s0), clip.metrics(s1)
        held = [r["frame"] for r in m0 if phase_of(s1, r["frame"]) in HELD]
        rest = [r["frame"] for r in m0 if phase_of(s1, r["frame"]) in ("approach", "release", "rest")]
        summ = {"video_id": vid, "frames": clip.N, "phases": phases}
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
        summ["flagged_frames"] = [sum(bool(flag(r, phase_of(s1, r["frame"]))) for r in m0),
                                  sum(bool(flag(r, phase_of(s1, r["frame"]))) for r in m1)]
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
            "rest height cm | rest height spread cm | flagged frames | verifier reverted |", "|" + "---|" * 10]
    f = lambda p, d=2: f"{p[0]:.{d}f} → {p[1]:.{d}f}" if None not in p else "—"
    for t in table:
        rows.append(f"| {t['video_id']} | {f(t['held']['iou'])} | {f(t['all']['iou'])} | {f(t['all']['hand_kp_px'], 1)} | "
                    f"{f(t['all']['hand_depth_bias_cm'])} | {f(t['all']['penetration_cm'])} | {f(t['rest_abs_height_cm'])} | {f(t['rest_height_spread_cm'])} | "
                    f"{t['flagged_frames'][0]} → {t['flagged_frames'][1]} | {t['reverted']}/{len(t['verifier'])} |")
    open(os.path.join(out_root, "summary.md"), "w").write("\n".join(rows) + "\n")
    print("\n".join(rows))


if __name__ == "__main__":
    main()
